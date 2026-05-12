"""
title: Agent Loop
author: Piggidragon
version: 2.1.0
description: >
  Minimal OpenWebUI-native Agent Loop with Plan/Execute/Review phases.

  Architecture:
  - SINGLE model loop (no sub-agents, no parallel LLM calls)
  - Plan phase: analyse context, read files, create task list, ask user for confirmation
  - Execute phase: work through tasks one at a time, mark complete/fail explicitly
  - Review phase: verify all tasks complete, either finish or replan
  - Replan phase: hard reset with a completely new plan based on what remains, then restart Execute
  - All tools/skills/filters come from OpenWebUI (no custom tool executor)
  - LLM calls via OWUI's generate_chat_completion (streaming SSE)
  - Tool resolution via OWUI (get_tools, get_builtin_tools, get_terminal_tools)
  - Tool execution via OWUI-resolved callables + process_tool_result
  - Renders tool executions as <details type="tool_calls"> to avoid OWUI retry loop
  - Internal control tools: terminate, replan, complete_task, fail_task
  - Uses ask_user_questions (Claude-like) for plan confirmation

  v2.1.0 changes:
  - Replan is now a hard reset: new plan, cleared state, direct restart to Execute
  - Added complete_task(index) / fail_task(index, reason) for explicit task tracking
  - Removed fragile keyword-based _mark_completed_tasks
  - Fixed consecutive_json_errors counter (was always reset immediately)
  - Task state injected into every phase system prompt
  - Adaptive history truncation to stay within context window
requirements: open-webui>=0.9.1
"""

import json
import logging
import re
import copy
import uuid
from typing import AsyncGenerator, Callable, Optional, Any
from pydantic import BaseModel, Field

from fastapi import Request

from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.tools import get_tools, get_builtin_tools, get_terminal_tools
from open_webui.utils.middleware import process_tool_result, add_file_context

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
#  PHASE PROMPTS
# ──────────────────────────────────────────────────────────────────

PLAN_PROMPT = """\
You are in PLAN mode. Your job is to understand the user's request, gather context, \
and create a clear task plan.

PHASE: PLAN

Available tools: {tool_names}

What to do:
1. Analyse the user's request thoroughly.
2. Read relevant files, search the web, query knowledge — use any tools to gather context.
3. Create a numbered task list that covers the entire goal.
4. Each task should be a clear, actionable step.
5. After creating the plan, call ask_user_questions to present the plan and ask for confirmation.

Rules:
- Be thorough — read files before planning changes.
- Break complex tasks into small, verifiable steps.
- If the request is simple (1-2 tasks), still list them explicitly.
- Call exactly ONE tool per step.
- When done planning, call ask_user_questions with the plan for confirmation.
- NEVER call terminate in PLAN mode — the user must confirm first.
- NEVER call replan in PLAN mode.
"""

EXECUTE_PROMPT = """\
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
"""

REVIEW_PROMPT = """\
You are in REVIEW mode. Verify that all original requirements are met.

PHASE: REVIEW

Original goal: {goal}

{task_state}

What to do:
1. Check each original requirement against what was actually done.
2. If everything is complete and correct → call terminate with the final result.
3. If something is missing or wrong → call replan with what needs to be redone or added.

Rules:
- Be honest — don't mark something as done if it isn't.
- If the result is good enough, terminate. Don't gold-plate.
"""

REPLAN_PROMPT = """\
You are in RE-PLAN mode. The previous plan needs adjustments.

PHASE: RE-PLAN

Original goal: {goal}

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
- Do NOT call ask_user_questions — the user already confirmed at the start.
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
    """Remove <thinking>...<thinking> blocks from model output."""
    text = re.sub(
        r"<(?:think|thinking|reason|reasoning|thought)>.*?</(?:think|thinking|reason|reasoning|thought)>",
        "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"\|begin_of_thought\|.*?\|end_of_thought\|", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


# ──────────────────────────────────────────────────────────────────
#  AGENT LOOP ENGINE
# ──────────────────────────────────────────────────────────────────

class AgentLoopEngine:
    """Single-model agent loop with Plan/Execute/Review phases."""

    # Phase constants
    PHASE_PLAN = "plan"
    PHASE_EXECUTE = "execute"
    PHASE_REVIEW = "review"
    PHASE_REPLAN = "replan"

    # Context window management
    MAX_HISTORY_MESSAGES = 50  # Keep system + user goal + ~24 assistant/tool pairs
    TRUNCATE_TOOL_RESULTS_AT = 30  # When history > this, aggressively truncate tool results

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

        # Loop state
        self.history = []
        self.tools_dict = {}
        self.tools_specs = []
        self.phase = self.PHASE_PLAN
        self.task_list = []
        self.completed_tasks = []
        self.failed_tasks = []  # list of {"task": str, "reason": str}
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
        """Resolve tools from OWUI's tool infrastructure + our internal tools."""
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

        self.tools_dict = {}

        # 1. External tools (DB + OpenAPI)
        unique_ids = list(dict.fromkeys(tid for tid in tool_ids if tid))
        if unique_ids:
            try:
                resolved = await get_tools(self.request, unique_ids, self.user, extra_params)
                if resolved:
                    self.tools_dict.update(resolved)
            except Exception as e:
                logger.error(f"get_tools failed: {e}")

        # 2. Built-in tools
        model_info = self.app_models.get(self.body.get("model", ""), {})
        features = self._get_model_features(model_info)
        if features:
            try:
                builtin = await get_builtin_tools(self.request, extra_params, features=features, model=model_info)
                if builtin:
                    self.tools_dict.update(builtin)
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
                    self.tools_dict.update(t_tools)
            except Exception as e:
                logger.error(f"get_terminal_tools failed: {e}")

        # 4. Internal control tools
        self.tools_dict["terminate"] = {
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
        self.tools_dict["replan"] = {
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
        self.tools_dict["complete_task"] = {
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
        self.tools_dict["fail_task"] = {
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

        # Build tool specs for LLM
        self.tools_specs = [
            {"type": "function", "function": t["spec"]}
            for t in self.tools_dict.values()
            if isinstance(t, dict) and "spec" in t
        ]
        return self.tools_dict

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
            # Avoid duplicates
            if not any(f["task"] == task for f in self.failed_tasks):
                self.failed_tasks.append(entry)
            return json.dumps({"failed": True, "task": task, "index": idx, "reason": reason})
        return json.dumps({"failed": False, "error": f"Invalid task index {idx}"})

    # ── Task State String ──

    def _build_task_state(self):
        """Build a compact task-state block for injection into prompts."""
        lines = []
        lines.append("Current Tasks:")
        for i, task in enumerate(self.task_list):
            if task in self.completed_tasks:
                status = "[✓]"
            elif any(f["task"] == task for f in self.failed_tasks):
                status = "[✗]"
            else:
                status = "[ ]"
            lines.append(f"  {i}. {status} {task}")
        lines.append(f"\nCompleted: {len(self.completed_tasks)}/{len(self.task_list)}")
        if self.failed_tasks:
            lines.append("Failed:")
            for f in self.failed_tasks:
                lines.append(f"  - {f['task']}: {f['reason']}")
        return "\n".join(lines)

    # ── Execute Tool ──

    async def _execute_tool(self, tool_name, args, call_id):
        """Execute a single resolved tool. Returns (result_string, files_list)."""
        target = self.tools_dict.get(tool_name)
        if not target:
            available = list(self.tools_dict.keys())
            return f"Tool '{tool_name}' not found. Available: {', '.join(available[:20])}", []

        # Filter args to spec
        spec_params = target.get("spec", {}).get("parameters", {}).get("properties", {})
        allowed_keys = set(spec_params.keys())
        filtered_args = {k: v for k, v in args.items() if k in allowed_keys}

        # Inject context vars if callable accepts them
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

            # Process via OWUI middleware
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
        """Build system prompt based on current phase."""
        tool_names = ", ".join(sorted(self.tools_dict.keys()))
        task_state = self._build_task_state()

        if self.phase == self.PHASE_PLAN:
            return PLAN_PROMPT.format(tool_names=tool_names)

        elif self.phase == self.PHASE_EXECUTE:
            return EXECUTE_PROMPT.format(tool_names=tool_names, task_state=task_state)

        elif self.phase == self.PHASE_REVIEW:
            return REVIEW_PROMPT.format(goal=self.goal, task_state=task_state)

        elif self.phase == self.PHASE_REPLAN:
            return REPLAN_PROMPT.format(
                goal=self.goal,
                replan_reason=self._replan_reason or "Previous plan was incomplete",
                task_state=task_state,
            )

        return PLAN_PROMPT.format(tool_names=tool_names)

    # ── Phase Transitions ──

    def _transition_to(self, phase, replan_reason=""):
        """Transition to a new phase and update history."""
        self.phase = phase
        self._replan_reason = replan_reason
        if self.history:
            self.history[0]["content"] = self._build_system_prompt()

    # ── Context Window Management ──

    def _manage_context_window(self, messages):
        """Truncate history to stay within context window. Keep system + user goal + recent pairs."""
        if len(messages) <= self.MAX_HISTORY_MESSAGES:
            return messages

        # Always keep system prompt (index 0) and original user goal (index 1)
        # Remove oldest assistant/tool pairs after index 1, keep the most recent ones
        to_remove = len(messages) - self.MAX_HISTORY_MESSAGES
        # We remove from index 2 onwards (after system + user goal)
        # But we keep the tail (recent interactions)
        head = messages[:2]
        tail = messages[2 + to_remove:]
        return head + tail

    def _get_truncation_limit(self):
        """Return aggressive truncation limit when history is long."""
        if len(self.history) > self.TRUNCATE_TOOL_RESULTS_AT:
            return self.valves.MAX_TOOL_RESULT_CHARS // 3
        return self.valves.MAX_TOOL_RESULT_CHARS

    # ── Main Loop ──

    async def run(self, user_msg, model):
        """Main agent loop with Plan/Execute/Review phases."""

        await self.emit_status("Agent starting…")
        await self.resolve_tools()

        self.goal = user_msg
        self.phase = self.PHASE_PLAN
        self._replan_reason = ""
        self.task_list = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.consecutive_json_errors = 0
        self.loop_count = 0

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
                self.PHASE_PLAN: "📋",
                self.PHASE_EXECUTE: "🔨",
                self.PHASE_REVIEW: "🔍",
                self.PHASE_REPLAN: "🔄",
            }
            phase_name = {
                self.PHASE_PLAN: "Plan",
                self.PHASE_EXECUTE: "Execute",
                self.PHASE_REVIEW: "Review",
                self.PHASE_REPLAN: "Re-Plan",
            }
            icon = phase_icons.get(self.phase, "▶")
            name = phase_name.get(self.phase, "Loop")

            await self.emit_status(f"{icon} {name} — step {self.loop_count}/{self.valves.MAX_ITERATIONS}")

            # Manage context window before calling LLM
            self.history = self._manage_context_window(self.history)

            # Prepare messages — keep only system[0] + non-system rest
            call_messages = [self.history[0]] + [m for m in self.history[1:] if m.get("role") != "system"]

            # Build completion body
            completion_body = {
                **self.body,
                "model": model,
                "messages": call_messages,
                "tools": self.tools_specs if self.tools_specs else None,
                "metadata": self.pipe_metadata,
            }

            # Apply file context if built-in tools present
            has_builtin = any(t.get("type") == "builtin" for t in self.tools_dict.values())
            if has_builtin and self.tools_specs:
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
                    output_parts.append(f"\n❌ LLM Error: {event.get('text', 'Unknown')}")
                    await self.emit_status("Error", done=True)
                    return "".join(output_parts)
                elif etype == "content":
                    content_chunks.append(event.get("text", ""))
                elif etype == "reasoning":
                    pass  # silence reasoning in output
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

            # ── No tool calls → final answer ──
            if not tc_dict:
                if content:
                    output_parts.append(content)
                await self.emit_status("Done", done=True)
                return "".join(output_parts)

            # ── Process tool calls ──
            tool_calls_list = list(tc_dict.values())

            # Add assistant message to history
            self.history.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls_list,
            })

            # Execute each tool call
            for tc in tool_calls_list:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                call_id = tc.get("id", str(uuid.uuid4()))

                # Parse args
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(args, dict):
                        args = {}
                    self.consecutive_json_errors = 0
                except json.JSONDecodeError:
                    self.consecutive_json_errors += 1
                    if self.consecutive_json_errors >= 3:
                        output_parts.append("\n❌ JSON parse failed 3 times. Stopping.")
                        await self.emit_status("JSON error", done=True)
                        return "".join(output_parts)
                    args = {}
                    # Add error to history and continue
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
                    icon = "✅" if success else "❌"
                    if content:
                        output_parts.append(content + "\n\n")
                    output_parts.append(f"{icon} **Finished:** {result}")
                    await self.emit_status("Finished", done=True)
                    return "".join(output_parts)

                # ── Handle replan ──
                if tool_name == "replan":
                    reason = args.get("reason", "Plan needs adjustment")
                    updated = args.get("updated_tasks", "")

                    # Parse updated tasks if provided
                    new_tasks = []
                    if updated:
                        new_tasks = [t.strip() for t in updated.split("\n") if t.strip()]

                    # HARD RESET: new plan only, clear state, restart Execute
                    if new_tasks:
                        self.task_list = new_tasks
                    # If no tasks provided, keep only non-completed non-failed tasks
                    else:
                        failed_task_names = {f["task"] for f in self.failed_tasks}
                        self.task_list = [
                            t for t in self.task_list
                            if t not in self.completed_tasks and t not in failed_task_names
                        ]
                        if not self.task_list:
                            # Nothing left to do — this shouldn't happen, but handle it
                            output_parts.append(f"\n⚠️ Replan requested but no remaining tasks. Terminating.")
                            await self.emit_status("No tasks remaining", done=True)
                            return "".join(output_parts)

                    self.completed_tasks = []
                    self.failed_tasks = []
                    self._replan_reason = reason
                    self.consecutive_json_errors = 0
                    self.loop_count = 0  # Reset iteration counter for fresh start
                    recent_calls = []

                    # Rebuild history with new system prompt + original user goal
                    system_prompt = self._build_system_prompt()
                    self.history = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": self.goal},
                    ]

                    output_parts.append(f"\n🔄 **Re-planning:** {reason}\n")
                    await self.emit_status(f"🔄 Re-planning: {reason}")
                    # Skip adding tool result to history — we already rebuilt it
                    continue

                # ── Handle complete_task ──
                if tool_name == "complete_task":
                    result_json = await self._tool_complete_task(**args)
                    result_data = json.loads(result_json)
                    status_icon = "✅" if result_data.get("completed") else "⚠️"
                    output_parts.append(f"\n{status_icon} Task {args.get('index', '?')} marked complete.\n")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    # Auto-transition to REVIEW when all tasks done
                    if self.completed_tasks and len(self.completed_tasks) >= len(self.task_list):
                        self._transition_to(self.PHASE_REVIEW)
                    continue

                # ── Handle fail_task ──
                if tool_name == "fail_task":
                    result_json = await self._tool_fail_task(**args)
                    result_data = json.loads(result_json)
                    status_icon = "❌" if result_data.get("failed") else "⚠️"
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
                    await self.emit_status(f"Running: {tool_name}…")
                    result_str, result_files = await self._execute_tool(tool_name, args, call_id)
                    truncation_limit = self._get_truncation_limit()
                    tool_result = smart_truncate(result_str, truncation_limit)

                # Render tool execution as <details type="tool_calls">
                args_preview = smart_truncate(json.dumps(args, ensure_ascii=False), 200)
                result_preview = smart_truncate(tool_result, 600)
                phase_tag = phase_icons.get(self.phase, "▶")
                detail_block = (
                    f'<details type="tool_calls">\n'
                    f'<summary>{phase_tag} {tool_name}</summary>\n'
                    f'<b>Args:</b> <code>{args_preview}</code>\n\n'
                    f'<b>Result:</b> {result_preview}\n'
                    f'</details>'
                )
                output_parts.append(f"\n{detail_block}\n")

                # Add tool result to history
                self.history.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": call_id,
                    "name": tool_name,
                })

            # Auto-transition to REVIEW when all tasks completed
            if self.phase == self.PHASE_EXECUTE and self.completed_tasks and len(self.completed_tasks) >= len(self.task_list):
                self._transition_to(self.PHASE_REVIEW)

        # Max iterations
        output_parts.append(f"\n⚠️ Max iterations ({self.valves.MAX_ITERATIONS}) reached.")
        await self.emit_status("Max iterations", done=True)
        return "".join(output_parts)

    def _extract_task_list(self, text):
        """Try to extract a numbered task list from model output."""
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
#  PIPE
# ──────────────────────────────────────────────────────────────────

class Pipe:
    class Valves(BaseModel):
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

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

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

        # Resolve user
        from open_webui.models.users import Users
        user_id = __user__.get("id") if isinstance(__user__, dict) else ""
        user_obj = await Users.get_user_by_id(user_id) if user_id else None

        # Determine model
        model = self.valves.AGENT_MODEL or body.get("model", "")

        # Extract user message
        messages = body.get("messages", [])
        user_msg = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "Unknown goal"
        )

        # Setup metadata
        metadata = {
            "__user__": __user__,
            "__request__": __request__,
            "__metadata__": __metadata__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__files__": __files__ or [],
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
        }

        # Create engine and run
        engine = AgentLoopEngine(
            request=__request__,
            user=user_obj or __user__,
            body=body,
            event_emitter=__event_emitter__,
            event_call=__event_call__,
            metadata=metadata,
            valves=self.valves,
        )

        return await engine.run(user_msg, model)
