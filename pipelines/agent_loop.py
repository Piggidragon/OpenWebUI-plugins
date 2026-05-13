"""
title: Agent Loop
author: Piggidragon
version: 2.2.0
description: >
  OpenWebUI-native Agent Loop with modular per-phase tool control.

  Architecture:
  - SINGLE model loop (Plan -> Execute -> Review -> Replan -> Execute...)
  - Per-phase tool filtering via Valves -- only relevant tools exposed to the LLM
  - Internal control tools (terminate, replan, complete_task, fail_task) always available
  - Uses OpenWebUI native tool infrastructure (get_tools, get_builtin_tools, get_terminal_tools)
  - Hard replan reset: new plan, cleared state, direct restart to Execute
  - Context window management with adaptive history truncation

  v2.2.0 changes:
  - Phase-based tool filtering via Valves (PLAN_TOOLS, EXECUTE_TOOLS, etc.)
  - Global tool denylist (TOOLS_DENYLIST)
  - Per-phase custom system prompt overrides
  - Internal tools always injected regardless of phase filters
  - Cleaner tool resolution with phase-aware filtering
requirements: open-webui>=0.9.1
"""

import json
import logging
import re
import copy
import uuid
from typing import AsyncGenerator, Callable, Optional, Any, Set, List
from pydantic import BaseModel, Field

from fastapi import Request

from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.tools import get_tools, get_builtin_tools, get_terminal_tools
from open_webui.utils.middleware import process_tool_result, add_file_context

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
#  DEFAULT PROMPTS (overridable via Valves)
# ──────────────────────────────────────────────────────────────────

DEFAULT_PLAN_PROMPT = """\
You are in PLAN mode. Your job is to understand the user's request, gather context, \
and create a clear task plan.

PHASE: PLAN

Available tools: {tool_names}

What to do:
1. Analyse the user's request thoroughly.
2. Read relevant files, search the web, query knowledge -- use any tools to gather context.
3. Create a numbered task list that covers the entire goal.
4. Each task should be a clear, actionable step.
5. After creating the plan, call ask_user_questions to present the plan and ask for confirmation.

Rules:
- Be thorough -- read files before planning changes.
- Break complex tasks into small, verifiable steps.
- If the request is simple (1-2 tasks), still list them explicitly.
- Call exactly ONE tool per step.
- When done planning, call ask_user_questions with the plan for confirmation.
- NEVER call terminate in PLAN mode -- the user must confirm first.
- NEVER call replan in PLAN mode.
- You may only use the tools listed above.
"""

DEFAULT_EXECUTE_PROMPT = """\
You are in EXECUTE mode. Work through tasks one at a time.

PHASE: EXECUTE

Available tools: {tool_names}

{task_state}

What to do:
1. Pick the next incomplete task from the list.
2. Execute it using the appropriate tool(s).
3. After the task is truly done, call complete_task(index) to mark it finished.
4. If a task fails and cannot be recovered, call fail_task(index, reason) to mark it.
5. Move on to the next task.

Rules:
- Call exactly ONE tool per step.
- NEVER repeat identical failed tool calls (duplicate detection is active).
- When all tasks are done, call terminate with a summary.
- If you realise the plan was completely wrong, call replan with what needs to change.
- You MUST call complete_task(index) or fail_task(index, reason) after working on a task.
- You may only use the tools listed above. Do NOT ask the user questions.
"""

DEFAULT_REVIEW_PROMPT = """\
You are in REVIEW mode. Verify that all original requirements are met.

PHASE: REVIEW

Original goal: {goal}

Available tools: {tool_names}

{task_state}

What to do:
1. Check each original requirement against what was actually done.
2. If everything is complete and correct -> call terminate with the final result.
3. If something is missing or wrong -> call replan with what needs to be redone or added.

Rules:
- Be honest -- don't mark something as done if it isn't.
- If the result is good enough, terminate. Don't gold-plate.
- You may only use the tools listed above.
"""

DEFAULT_REPLAN_PROMPT = """\
You are in RE-PLAN mode. The previous plan needs adjustments.

PHASE: RE-PLAN

Original goal: {goal}

Available tools: {tool_names}

{task_state}

What went wrong or is missing: {replan_reason}

What to do:
1. Assess what worked and what didn't.
2. Create a completely NEW task list that covers ONLY what is still needed.
3. Do NOT include already completed tasks.
4. Focus on fixing the failures and filling the gaps.
5. Then call replan() with the new plan and the reason.

Rules:
- Don't repeat tasks that already succeeded.
- Focus only on what's broken or missing.
- Call replan() to restart execution with the new plan.
- Do NOT call ask_user_questions -- the user already confirmed at the start.
- You may only use the tools listed above.
"""


# ──────────────────────────────────────────────────────────────────
#  SSE STREAM PARSER
# ──────────────────────────────────────────────────────────────────

async def stream_completion(request, body, user):
    """Stream OWUI completion, yielding structured events."""
    body["stream"] = True
    try:
        response = await generate_chat_completion(request, body, user=user)
    except Exception as e:
        logger.error(f"generate_chat_completion failed: {e}")
        yield {"type": "error", "text": str(e)}
        return

    if hasattr(response, "body_iterator"):
        sse_buffer = ""
        async for chunk in response.body_iterator:
            decoded = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            sse_buffer += decoded

            while "\n\n" in sse_buffer:
                raw_event, sse_buffer = sse_buffer.split("\n\n", 1)
                data_lines = []
                for line in raw_event.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("data:"):
                        data_lines.append(stripped[5:].lstrip())

                payload = "\n".join(data_lines).strip()
                if not payload or payload == "[DONE]":
                    continue

                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        choices = parsed.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {}) or {}
                            for rk in ("reasoning", "reasoning_content", "thinking"):
                                rv = delta.get(rk)
                                if rv:
                                    yield {"type": "reasoning", "text": rv}
                            cv = delta.get("content")
                            if cv:
                                yield {"type": "content", "text": cv}
                            tc = delta.get("tool_calls")
                            if tc:
                                yield {"type": "tool_calls", "data": tc}
                except json.JSONDecodeError:
                    pass

        # Flush remaining buffer
        if sse_buffer.strip():
            for line in sse_buffer.strip().splitlines():
                stripped = line.strip()
                if stripped.startswith("data:"):
                    payload = stripped[5:].lstrip()
                    if payload and payload != "[DONE]":
                        try:
                            parsed = json.loads(payload)
                            if isinstance(parsed, dict):
                                choices = parsed.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {}) or {}
                                    for rk in ("reasoning", "reasoning_content", "thinking"):
                                        rv = delta.get(rk)
                                        if rv:
                                            yield {"type": "reasoning", "text": rv}
                                    cv = delta.get("content")
                                    if cv:
                                        yield {"type": "content", "text": cv}
                                    tc = delta.get("tool_calls")
                                    if tc:
                                        yield {"type": "tool_calls", "data": tc}
                        except json.JSONDecodeError:
                            pass

    elif isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            if msg.get("tool_calls"):
                yield {"type": "tool_calls", "data": msg["tool_calls"]}
            if msg.get("content"):
                yield {"type": "content", "text": msg["content"]}


# ──────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────

def smart_truncate(text, max_chars):
    if not text or len(text) <= max_chars:
        return text
    for sep in (". ", ".\n", "\n\n", "\n"):
        idx = text[:max_chars].rfind(sep)
        if idx > max_chars // 2:
            return text[:idx + len(sep)].rstrip() + "\n[truncated]"
    return text[:max_chars].rstrip() + "\n[truncated]"


def strip_thinking(text):
    """Remove <thinking>...</thinking> blocks from model output."""
    text = re.sub(
        r"<(?:think|thinking|reason|reasoning|thought)>.*?</(?:think|thinking|reason|reasoning|thought)>",
        "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"\|begin_of_thought\|.*?\|end_of_thought\|", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _comma_list(val: str) -> List[str]:
    """Convert a comma-separated string to a list of stripped, non-empty strings."""
    if not val or not isinstance(val, str):
        return []
    return [x.strip() for x in val.split(",") if x.strip()]


# ──────────────────────────────────────────────────────────────────
#  AGENT LOOP ENGINE
# ──────────────────────────────────────────────────────────────────

class AgentLoopEngine:
    """Single-model agent loop with per-phase tool filtering."""

    PHASE_PLAN = "plan"
    PHASE_EXECUTE = "execute"
    PHASE_REVIEW = "review"
    PHASE_REPLAN = "replan"

    MAX_HISTORY_MESSAGES = 50
    TRUNCATE_TOOL_RESULTS_AT = 30

    # Internal tools that are ALWAYS available regardless of phase filters
    INTERNAL_TOOLS = {"terminate", "replan", "complete_task", "fail_task"}

    def __init__(self, request, user, body, event_emitter, event_call, metadata, valves):
        self.request = request
        self.user = user
        self.body = body
        self.event_emitter = event_emitter
        self.event_call = event_call
        self.metadata = metadata
        self.valves = valves

        self.pipe_metadata = metadata.get("__metadata__", metadata)
        self.chat_id = metadata.get("__chat_id__", "")
        self.message_id = metadata.get("__message_id__", "")

        self.app_models = getattr(request.app.state, "MODELS", {}) if request else {}

        self.history = []
        self.all_tools_dict = {}          # All resolved tools from OWUI
        self.phase_tools_dict = {}        # Filtered tools for current phase
        self.phase_tools_specs = []       # OpenAI-format specs for current phase
        self.phase = self.PHASE_PLAN
        self.task_list = []
        self.completed_tasks = []
        self.failed_tasks = []
        self._replan_reason = ""
        self.consecutive_json_errors = 0
        self.loop_count = 0
        self.goal = ""

    async def emit_status(self, msg, done=False):
        if self.event_emitter:
            try:
                await self.event_emitter({"type": "status", "data": {"description": msg, "done": done}})
            except Exception:
                pass

    # ── Tool Resolution ──

    async def resolve_tools(self):
        """Resolve ALL tools from OWUI's infrastructure into all_tools_dict."""
        tool_ids = self.pipe_metadata.get("toolIds") or self.pipe_metadata.get("tool_ids") or []
        user_dict = (
            self.user.model_dump() if hasattr(self.user, "model_dump")
            else (self.user if isinstance(self.user, dict) else {})
        )

        extra_params = {
            "chat_id": self.chat_id,
            "tool_ids": tool_ids,
            "__user__": user_dict,
            "__metadata__": self.metadata,
            "__event_emitter__": self.event_emitter,
            "__event_call__": self.event_call,
        }

        self.all_tools_dict = {}

        # 1. External tools (DB + OpenAPI)
        unique_ids = list(dict.fromkeys(tid for tid in tool_ids if tid))
        if unique_ids:
            try:
                resolved = await get_tools(self.request, unique_ids, self.user, extra_params)
                if resolved:
                    self.all_tools_dict.update(resolved)
            except Exception as e:
                logger.error(f"get_tools failed: {e}")

        # 2. Built-in tools
        model_info = self.app_models.get(self.body.get("model", ""), {})
        features = self._get_model_features(model_info)
        if features:
            try:
                builtin = await get_builtin_tools(self.request, extra_params, features=features, model=model_info)
                if builtin:
                    self.all_tools_dict.update(builtin)
            except Exception as e:
                logger.error(f"get_builtin_tools failed: {e}")

        # 3. Terminal tools
        terminal_id = self.pipe_metadata.get("terminal_id")
        if terminal_id:
            try:
                raw_term = await get_terminal_tools(self.request, terminal_id, self.user, extra_params)
                if isinstance(raw_term, tuple) and len(raw_term) == 2:
                    t_tools, _ = raw_term
                else:
                    t_tools = raw_term if isinstance(raw_term, dict) else {}
                if t_tools:
                    self.all_tools_dict.update(t_tools)
            except Exception as e:
                logger.error(f"get_terminal_tools failed: {e}")

        # 4. Add internal control tools (always available, stored in all_tools_dict too)
        self._add_internal_tools()

        return self.all_tools_dict

    def _add_internal_tools(self):
        """Register internal control tools. These are always present."""
        self.all_tools_dict["terminate"] = {
            "spec": {
                "name": "terminate",
                "description": "Signal that all tasks are complete. Provide the final result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string", "description": "The final answer or summary of what was accomplished"},
                        "success": {"type": "boolean", "default": True, "description": "Whether the overall task succeeded"},
                    },
                    "required": ["result"],
                },
            },
            "callable": self._tool_terminate,
            "type": "function",
        }
        self.all_tools_dict["replan"] = {
            "spec": {
                "name": "replan",
                "description": "Signal that the current plan needs a complete restart. Provide the new plan and reason.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "What went wrong or what is missing"},
                        "updated_tasks": {"type": "string", "description": "Updated task list as numbered steps (only what is still needed)"},
                    },
                    "required": ["reason"],
                },
            },
            "callable": self._tool_replan,
            "type": "function",
        }
        self.all_tools_dict["complete_task"] = {
            "spec": {
                "name": "complete_task",
                "description": "Mark a specific task as completed. Call this AFTER the task is truly done.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "0-based index of the task to mark complete"},
                    },
                    "required": ["index"],
                },
            },
            "callable": self._tool_complete_task,
            "type": "function",
        }
        self.all_tools_dict["fail_task"] = {
            "spec": {
                "name": "fail_task",
                "description": "Mark a specific task as failed with a reason. Call this when a task cannot be recovered.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "0-based index of the task that failed"},
                        "reason": {"type": "string", "description": "Why the task failed"},
                    },
                    "required": ["index", "reason"],
                },
            },
            "callable": self._tool_fail_task,
            "type": "function",
        }

    def _get_model_features(self, model_info):
        info = model_info.get("info", {}) or {}
        meta = info.get("meta", {}) or {}
        params = info.get("params", {}) or {}
        features = {}
        for block in (meta, params):
            if isinstance(block.get("features"), dict):
                for fk, fv in block["features"].items():
                    features[fk] = bool(fv)
        return features

    # ── Phase-aware Tool Filtering ──

    def _filter_tools_for_phase(self, phase: str):
        """Build phase_tools_dict from all_tools_dict based on Valves config."""
        # Determine which tool names are allowed for this phase
        allowlist: Set[str] = set()

        if phase == self.PHASE_PLAN:
            allowlist = set(_comma_list(self.valves.PLAN_TOOLS))
        elif phase == self.PHASE_EXECUTE:
            allowlist = set(_comma_list(self.valves.EXECUTE_TOOLS))
        elif phase == self.PHASE_REVIEW:
            allowlist = set(_comma_list(self.valves.REVIEW_TOOLS))
        elif phase == self.PHASE_REPLAN:
            allowlist = set(_comma_list(self.valves.REPLAN_TOOLS))

        # Global denylist
        denylist = set(_comma_list(self.valves.TOOLS_DENYLIST))

        # If allowlist is empty -> allow ALL tools (except denylist and internal overrides)
        # If allowlist has entries -> only those tools (plus internals, minus denylist)
        self.phase_tools_dict = {}

        for name, tool in self.all_tools_dict.items():
            # Internal tools are ALWAYS included regardless of allowlist/denylist
            if name in self.INTERNAL_TOOLS:
                self.phase_tools_dict[name] = tool
                continue

            # Global denylist wins over everything
            if name in denylist:
                continue

            # Allowlist filtering
            if allowlist:
                if name in allowlist:
                    self.phase_tools_dict[name] = tool
            else:
                # No allowlist -> allow all (except denylist, already checked)
                self.phase_tools_dict[name] = tool

        # Build OpenAI-format tool specs
        self.phase_tools_specs = [
            {"type": "function", "function": t["spec"]}
            for t in self.phase_tools_dict.values()
            if isinstance(t, dict) and "spec" in t
        ]

    # ── Internal Tools ──

    async def _tool_terminate(self, **kwargs):
        return json.dumps({"terminated": True, "result": kwargs.get("result", ""), "success": kwargs.get("success", True)})

    async def _tool_replan(self, **kwargs):
        return json.dumps({"replan": True, "reason": kwargs.get("reason", ""), "updated_tasks": kwargs.get("updated_tasks", "")})

    async def _tool_complete_task(self, **kwargs):
        idx = kwargs.get("index", -1)
        if 0 <= idx < len(self.task_list):
            task = self.task_list[idx]
            if task not in self.completed_tasks:
                self.completed_tasks.append(task)
            return json.dumps({"completed": True, "task": task, "index": idx})
        return json.dumps({"completed": False, "error": f"Invalid task index {idx}"})

    async def _tool_fail_task(self, **kwargs):
        idx = kwargs.get("index", -1)
        reason = kwargs.get("reason", "Unknown failure")
        if 0 <= idx < len(self.task_list):
            task = self.task_list[idx]
            entry = {"task": task, "reason": reason}
            if not any(f["task"] == task for f in self.failed_tasks):
                self.failed_tasks.append(entry)
            return json.dumps({"failed": True, "task": task, "index": idx, "reason": reason})
        return json.dumps({"failed": False, "error": f"Invalid task index {idx}"})

    # ── Task State String ──

    def _build_task_state(self):
        lines = []
        lines.append("Current Tasks:")
        for i, task in enumerate(self.task_list):
            if task in self.completed_tasks:
                status = "[done]"
            elif any(f["task"] == task for f in self.failed_tasks):
                status = "[FAIL]"
            else:
                status = "[    ]"
            lines.append(f"  {i}. {status} {task}")
        lines.append(f"\nCompleted: {len(self.completed_tasks)}/{len(self.task_list)}")
        if self.failed_tasks:
            lines.append("Failed:")
            for f in self.failed_tasks:
                lines.append(f"  - {f['task']}: {f['reason']}")
        return "\n".join(lines)

    # ── Execute Tool ──

    async def _execute_tool(self, tool_name, args, call_id):
        """Execute a single resolved tool from phase_tools_dict."""
        target = self.phase_tools_dict.get(tool_name)
        if not target:
            available = list(self.phase_tools_dict.keys())
            return f"Tool '{tool_name}' not found in current phase. Available: {', '.join(available[:20])}", []

        spec_params = target.get("spec", {}).get("parameters", {}).get("properties", {})
        allowed_keys = set(spec_params.keys())
        filtered_args = {k: v for k, v in args.items() if k in allowed_keys}

        import inspect
        callable_fn = target.get("callable")
        if callable_fn and inspect.iscoroutinefunction(callable_fn):
            try:
                sig = inspect.signature(callable_fn)
                context_vars = {
                    "__request__": self.request,
                    "__user__": self.metadata.get("__user__"),
                    "__event_emitter__": self.event_emitter,
                    "__event_call__": self.event_call,
                    "__chat_id__": self.chat_id,
                    "__message_id__": self.message_id,
                    "__files__": self.metadata.get("__files__"),
                    "__metadata__": self.metadata.get("__metadata__"),
                }
                for k, v in context_vars.items():
                    if v is not None and k in sig.parameters and k not in filtered_args:
                        filtered_args[k] = v
            except Exception:
                pass

        files = []
        try:
            result = await callable_fn(**filtered_args)
            try:
                processed = await process_tool_result(
                    self.request,
                    tool_name,
                    result,
                    target.get("type", ""),
                    False,
                    self.metadata.get("__metadata__", {}),
                    self.user,
                )
                result_str = ""
                if isinstance(processed, tuple) and len(processed) >= 1:
                    r_val = processed[0]
                    if isinstance(r_val, dict):
                        result_str = r_val.get("message") or r_val.get("description") or json.dumps(r_val, ensure_ascii=False)
                    else:
                        result_str = str(r_val) if r_val is not None else ""
                    if len(processed) >= 3:
                        f = processed[2] if isinstance(processed[2], list) else []
                        files = f
                elif isinstance(processed, dict):
                    result_str = processed.get("message") or processed.get("description") or json.dumps(processed, ensure_ascii=False)
                else:
                    result_str = str(processed) if processed is not None else ""
            except Exception:
                result_str = str(result) if result is not None else ""
            return result_str, files
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return f"Error executing {tool_name}: {e}", []

    # ── Phase System Prompt ──

    def _build_system_prompt(self):
        """Build system prompt based on current phase using Valves overrides."""
        tool_names = ", ".join(sorted(self.phase_tools_dict.keys()))
        task_state = self._build_task_state()

        # Pick base prompt from Valves or fallback to default
        if self.phase == self.PHASE_PLAN:
            base = self.valves.PLAN_PROMPT or DEFAULT_PLAN_PROMPT
            return base.format(tool_names=tool_names)

        elif self.phase == self.PHASE_EXECUTE:
            base = self.valves.EXECUTE_PROMPT or DEFAULT_EXECUTE_PROMPT
            return base.format(tool_names=tool_names, task_state=task_state)

        elif self.phase == self.PHASE_REVIEW:
            base = self.valves.REVIEW_PROMPT or DEFAULT_REVIEW_PROMPT
            return base.format(goal=self.goal, task_state=task_state, tool_names=tool_names)

        elif self.phase == self.PHASE_REPLAN:
            base = self.valves.REPLAN_PROMPT or DEFAULT_REPLAN_PROMPT
            return base.format(
                goal=self.goal,
                replan_reason=self._replan_reason or "Previous plan was incomplete",
                task_state=task_state,
                tool_names=tool_names,
            )

        return DEFAULT_PLAN_PROMPT.format(tool_names=tool_names)

    # ── Phase Transitions ──

    def _transition_to(self, phase, replan_reason=""):
        """Transition to a new phase: update tools, system prompt, state."""
        self.phase = phase
        self._replan_reason = replan_reason

        # Rebuild filtered tools for new phase
        self._filter_tools_for_phase(phase)

        # Update system prompt in history
        if self.history:
            self.history[0]["content"] = self._build_system_prompt()

    # ── Context Window Management ──

    def _manage_context_window(self, messages):
        if len(messages) <= self.MAX_HISTORY_MESSAGES:
            return messages
        to_remove = len(messages) - self.MAX_HISTORY_MESSAGES
        head = messages[:2]
        tail = messages[2 + to_remove:]
        return head + tail

    def _get_truncation_limit(self):
        if len(self.history) > self.TRUNCATE_TOOL_RESULTS_AT:
            return self.valves.MAX_TOOL_RESULT_CHARS // 3
        return self.valves.MAX_TOOL_RESULT_CHARS

    # ── Main Loop ──

    async def run(self, user_msg, model):
        await self.emit_status("Agent starting...")
        await self.resolve_tools()

        self.goal = user_msg
        self.phase = self.PHASE_PLAN
        self._replan_reason = ""
        self.task_list = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.consecutive_json_errors = 0
        self.loop_count = 0

        # Initial tool filtering for PLAN phase
        self._filter_tools_for_phase(self.PHASE_PLAN)

        system_prompt = self._build_system_prompt()
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        recent_calls = []
        output_parts = []

        while self.loop_count < self.valves.MAX_ITERATIONS:
            self.loop_count += 1
            recent_calls = recent_calls[-30:]

            phase_icons = {
                self.PHASE_PLAN: "[PLAN]",
                self.PHASE_EXECUTE: "[EXEC]",
                self.PHASE_REVIEW: "[REVU]",
                self.PHASE_REPLAN: "[RPLN]",
            }
            phase_name = {
                self.PHASE_PLAN: "Plan",
                self.PHASE_EXECUTE: "Execute",
                self.PHASE_REVIEW: "Review",
                self.PHASE_REPLAN: "Re-Plan",
            }
            icon = phase_icons.get(self.phase, "[LOOP]")
            name = phase_name.get(self.phase, "Loop")

            await self.emit_status(f"{icon} {name} -- step {self.loop_count}/{self.valves.MAX_ITERATIONS}")

            self.history = self._manage_context_window(self.history)
            call_messages = [self.history[0]] + [m for m in self.history[1:] if m.get("role") != "system"]

            completion_body = {
                **self.body,
                "model": model,
                "messages": call_messages,
                "tools": self.phase_tools_specs if self.phase_tools_specs else None,
                "metadata": self.pipe_metadata,
            }

            has_builtin = any(t.get("type") == "builtin" for t in self.phase_tools_dict.values())
            if has_builtin and self.phase_tools_specs:
                try:
                    completion_body["messages"] = await add_file_context(
                        copy.deepcopy(call_messages), self.chat_id, self.user
                    )
                except Exception:
                    pass

            # ── Stream LLM response ──
            tc_dict = {}
            content_chunks = []

            async for event in stream_completion(self.request, completion_body, self.user):
                etype = event.get("type")
                if etype == "error":
                    output_parts.append(f"\n[ERROR] LLM Error: {event.get('text', 'Unknown')}")
                    await self.emit_status("Error", done=True)
                    return "".join(output_parts)
                elif etype == "content":
                    content_chunks.append(event.get("text", ""))
                elif etype == "reasoning":
                    pass
                elif etype == "tool_calls":
                    for tc in event.get("data", []):
                        idx = tc.get("index", 0)
                        if idx not in tc_dict:
                            tc_dict[idx] = {
                                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if "name" in tc.get("function", {}):
                            tc_dict[idx]["function"]["name"] = tc["function"]["name"]
                        if "arguments" in tc.get("function", {}):
                            tc_dict[idx]["function"]["arguments"] += tc["function"]["arguments"]

            content = strip_thinking("".join(content_chunks).strip())

            if not tc_dict:
                if content:
                    output_parts.append(content)
                await self.emit_status("Done", done=True)
                return "".join(output_parts)

            tool_calls_list = list(tc_dict.values())
            self.history.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls_list,
            })

            for tc in tool_calls_list:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                call_id = tc.get("id", str(uuid.uuid4()))

                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(args, dict):
                        args = {}
                    self.consecutive_json_errors = 0
                except json.JSONDecodeError:
                    self.consecutive_json_errors += 1
                    if self.consecutive_json_errors >= 3:
                        output_parts.append("\n[ERROR] JSON parse failed 3 times. Stopping.")
                        await self.emit_status("JSON error", done=True)
                        return "".join(output_parts)
                    args = {}
                    self.history.append({
                        "role": "tool",
                        "content": "Error: Invalid JSON in tool arguments.",
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    continue

                # ── Handle terminate ──
                if tool_name == "terminate":
                    result = args.get("result", "Task complete.")
                    success = args.get("success", True)
                    icon = "[OK]" if success else "[FAIL]"
                    if content:
                        output_parts.append(content + "\n\n")
                    output_parts.append(f"{icon} **Finished:** {result}")
                    await self.emit_status("Finished", done=True)
                    return "".join(output_parts)

                # ── Handle replan ──
                if tool_name == "replan":
                    reason = args.get("reason", "Plan needs adjustment")
                    updated = args.get("updated_tasks", "")

                    new_tasks = []
                    if updated:
                        new_tasks = [t.strip() for t in updated.split("\n") if t.strip()]

                    if new_tasks:
                        self.task_list = new_tasks
                    else:
                        failed_task_names = {f["task"] for f in self.failed_tasks}
                        self.task_list = [
                            t for t in self.task_list
                            if t not in self.completed_tasks and t not in failed_task_names
                        ]
                        if not self.task_list:
                            output_parts.append(f"\n[WARN] Replan requested but no remaining tasks. Terminating.")
                            await self.emit_status("No tasks remaining", done=True)
                            return "".join(output_parts)

                    self.completed_tasks = []
                    self.failed_tasks = []
                    self._replan_reason = reason
                    self.consecutive_json_errors = 0
                    self.loop_count = 0
                    recent_calls = []

                    # If we are NOT in REPLAN phase -> enter REPLAN to think about new plan
                    if self.phase != self.PHASE_REPLAN:
                        self._transition_to(self.PHASE_REPLAN, reason)
                        # Add replan tool result to history so LLM sees it
                        result_json = await self._tool_replan(**args)
                        self.history.append({
                            "role": "tool",
                            "content": result_json,
                            "tool_call_id": call_id,
                            "name": tool_name,
                        })
                        output_parts.append(f"\n[RPLN] **Re-planning:** {reason}\n")
                        await self.emit_status(f"[RPLN] Re-planning: {reason}")
                        continue
                    else:
                        # We ARE in REPLAN phase -> this is the "done thinking" signal
                        # Hard reset and jump to EXECUTE
                        self._transition_to(self.PHASE_EXECUTE)
                        self.history = [
                            {"role": "system", "content": self._build_system_prompt()},
                            {"role": "user", "content": self.goal},
                        ]
                        output_parts.append(f"\n[RPLN] **Re-plan complete:** {reason}\n")
                        await self.emit_status(f"[RPLN] Re-plan complete: {reason}")
                        continue

                # ── Handle complete_task ──
                if tool_name == "complete_task":
                    result_json = await self._tool_complete_task(**args)
                    result_data = json.loads(result_json)
                    status_icon = "[OK]" if result_data.get("completed") else "[WARN]"
                    output_parts.append(f"\n{status_icon} Task {args.get('index', '?')} marked complete.\n")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    if self.completed_tasks and len(self.completed_tasks) >= len(self.task_list):
                        self._transition_to(self.PHASE_REVIEW)
                    continue

                # ── Handle fail_task ──
                if tool_name == "fail_task":
                    result_json = await self._tool_fail_task(**args)
                    result_data = json.loads(result_json)
                    status_icon = "[FAIL]" if result_data.get("failed") else "[WARN]"
                    output_parts.append(f"\n{status_icon} Task {args.get('index', '?')} marked failed: {args.get('reason', '')}\n")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    continue

                # ── Phase transition on ask_user_questions ──
                if self.phase == self.PHASE_PLAN and tool_name == "ask_user_questions":
                    if not self.task_list and content:
                        self.task_list = self._extract_task_list(content)
                    self._transition_to(self.PHASE_EXECUTE)

                # ── Duplicate detection ──
                sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                if recent_calls.count(sig) >= 2:
                    tool_result = f"Error: Identical call to `{tool_name}` repeated. Try a different approach."
                else:
                    recent_calls.append(sig)
                    await self.emit_status(f"Running: {tool_name}...")
                    result_str, result_files = await self._execute_tool(tool_name, args, call_id)
                    truncation_limit = self._get_truncation_limit()
                    tool_result = smart_truncate(result_str, truncation_limit)

                # Render
                args_preview = smart_truncate(json.dumps(args, ensure_ascii=False), 200)
                result_preview = smart_truncate(tool_result, 600)
                phase_tag = phase_icons.get(self.phase, "[LOOP]")
                detail_block = (
                    f'<details type="tool_calls">\n'
                    f'<summary>{phase_tag} {tool_name}</summary>\n'
                    f'<b>Args:</b> <code>{args_preview}</code>\n\n'
                    f'<b>Result:</b> {result_preview}\n'
                    f'</details>'
                )
                output_parts.append(f"\n{detail_block}\n")

                self.history.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": call_id,
                    "name": tool_name,
                })

            # Auto-transition to REVIEW
            if self.phase == self.PHASE_EXECUTE and self.completed_tasks and len(self.completed_tasks) >= len(self.task_list):
                self._transition_to(self.PHASE_REVIEW)

        output_parts.append(f"\n[WARN] Max iterations ({self.valves.MAX_ITERATIONS}) reached.")
        await self.emit_status("Max iterations", done=True)
        return "".join(output_parts)

    def _extract_task_list(self, text):
        tasks = []
        for line in text.split("\n"):
            line = line.strip()
            m = re.match(r"^\d+[\.\)]\s+(.+)", line)
            if m:
                tasks.append(m.group(1).strip())
            elif re.match(r"^[-*]\s+", line):
                tasks.append(re.sub(r"^[-*]\s+", "", line).strip())
        return tasks if tasks else ["Complete the user's request"]


# ──────────────────────────────────────────────────────────────────
#  VALVES
# ──────────────────────────────────────────────────────────────────

class AgentValves(BaseModel):
    AGENT_MODEL: str = Field(
        default="",
        description="Model ID for the agent loop. Leave blank to use the selected model. Must support function calling."
    )
    MAX_ITERATIONS: int = Field(
        default=24,
        description="Maximum agent loop iterations before stopping."
    )
    MAX_TOOL_RESULT_CHARS: int = Field(
        default=4200,
        description="Max characters for tool results before truncation."
    )

    # ── Per-phase tool allowlists (comma-separated tool names) ──
    PLAN_TOOLS: str = Field(
        default="",
        description=(
            "Comma-separated tool names allowed in PLAN phase. "
            "Leave EMPTY to allow ALL tools. "
            "Example: ask_user_questions, read_file, search_web, read_github, search_github"
        )
    )
    EXECUTE_TOOLS: str = Field(
        default="",
        description=(
            "Comma-separated tool names allowed in EXECUTE phase. "
            "Leave EMPTY to allow ALL tools. "
            "Example: github_access, file_write, web_search, read_file"
        )
    )
    REVIEW_TOOLS: str = Field(
        default="",
        description=(
            "Comma-separated tool names allowed in REVIEW phase. "
            "Leave EMPTY to allow ALL tools. "
            "Typically empty or minimal -- review should use few tools."
        )
    )
    REPLAN_TOOLS: str = Field(
        default="",
        description=(
            "Comma-separated tool names allowed in REPLAN phase. "
            "Leave EMPTY to allow ALL tools. "
            "Example: read_file, search_web, read_github (read-only tools only)"
        )
    )

    # ── Global denylist ──
    TOOLS_DENYLIST: str = Field(
        default="",
        description=(
            "Comma-separated tool names that are NEVER available in ANY phase. "
            "Useful to permanently disable dangerous or irrelevant tools. "
            "Example: execute_code, shell_command"
        )
    )

    # ── Custom system prompt overrides ──
    PLAN_PROMPT: str = Field(
        default="",
        description="Custom system prompt for PLAN phase. Leave empty for default. Use {tool_names} placeholder."
    )
    EXECUTE_PROMPT: str = Field(
        default="",
        description="Custom system prompt for EXECUTE phase. Leave empty for default. Use {tool_names} and {task_state} placeholders."
    )
    REVIEW_PROMPT: str = Field(
        default="",
        description="Custom system prompt for REVIEW phase. Leave empty for default. Use {goal}, {task_state}, and {tool_names} placeholders."
    )
    REPLAN_PROMPT: str = Field(
        default="",
        description="Custom system prompt for REPLAN phase. Leave empty for default. Use {goal}, {replan_reason}, {task_state}, and {tool_names} placeholders."
    )


# ──────────────────────────────────────────────────────────────────
#  PIPE
# ──────────────────────────────────────────────────────────────────

class Pipe:
    def __init__(self):
        self.type = "manifold"
        self.valves = AgentValves()

    def pipes(self):
        return [{"id": "agent-loop", "name": "Agent Loop"}]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __metadata__: dict = None,
        __event_emitter__: Callable = None,
        __event_call__: Callable = None,
        __files__: list = None,
        __chat_id__: str = None,
        __message_id__: str = None,
        **kwargs,
    ):
        if __request__ is None:
            raise TypeError("Agent Loop requires __request__.")

        __metadata__ = __metadata__ or body.get("metadata", {})

        from open_webui.models.users import Users
        user_id = __user__.get("id") if isinstance(__user__, dict) else ""
        user_obj = await Users.get_user_by_id(user_id) if user_id else None

        model = self.valves.AGENT_MODEL or body.get("model", "")

        messages = body.get("messages", [])
        user_msg = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "Unknown goal"
        )

        metadata = {
            "__user__": __user__,
            "__request__": __request__,
            "__metadata__": __metadata__,
            "__event_emitter__": __event_emitter,
            "__event_call__": __event_call,
            "__files__": __files__ or [],
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
        }

        engine = AgentLoopEngine(
            request=__request__,
            user=user_obj or __user__,
            body=body,
            event_emitter=__event_emitter,
            event_call=__event_call__,
            metadata=metadata,
            valves=self.valves,
        )

        return await engine.run(user_msg, model)
