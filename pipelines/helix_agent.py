"""
title: Helix Agent
author: Piggidragon
version: 4.4.0
description: >
  Helix Agent — OpenWebUI-native agent loop with modular per-phase tool control.

  Architecture:
  - SINGLE model loop (Plan -> Execute -> Review -> Replan -> Execute...)
  - Per-phase tool filtering via Valves -- only relevant tools exposed to the LLM
  - Internal control tools (terminate, replan, fix_plan, complete_task, fail_task, confirm_plan) always available
  - Uses OpenWebUI native tool infrastructure (get_tools, get_builtin_tools, get_terminal_tools)
  - Replan as internal tool: updates task list and transitions to EXECUTE phase
  - Context window management with adaptive history truncation and tool-call pair integrity
  - Plan confirmation via custom JS UI (UserValves: ENABLE_PLAN_APPROVAL, YOLO_MODE)
  - Native OpenWebUI task progress UI via chat:message:tasks events, finalized on termination
  - System prompt refresh: task mutations (complete, fail, fix_plan) update the LLM's task state context
  - State persistence via [AGENT_STATE] messages; restored on conversation continuation
  - Silent mode: suppresses intermediate noise (tool_call details, reasoning, [PLAN]/[EXEC]/[RPLN]/[FIX] lines)
  - Iteration limit with Continue/Cancel modal; graceful shutdown on CancelledError/GeneratorExit
requirements: open-webui>=0.9.1
"""

import asyncio
import html
import inspect
import json
import logging
import re
import copy
import uuid
from typing import AsyncGenerator, Callable, Optional, Any, Set, List, Dict
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
5. After creating the plan, call confirm_plan with the plan text to present it for review.

Plan format for confirm_plan:
When calling confirm_plan, provide the plan parameter as a numbered list with one task per line:
1. First task description
2. Second task description
3. Third task description

Alternatively, you may provide the plan as JSON: {{"tasks": ["task 1", "task 2", "task 3"]}}

Rules:
- Be thorough -- read files before planning changes.
- Break complex tasks into small, verifiable steps.
- If the request is simple (1-2 tasks), still list them explicitly.
- Call exactly ONE tool per step.
- When done planning, call confirm_plan with the plan for confirmation.
- NEVER call terminate in PLAN mode -- the user must confirm first.
- NEVER call replan in PLAN mode.
- If a tool returns an error during planning (e.g., file not found), try an alternative tool or note the limitation in your plan. Do not call fix_plan for planning-stage errors.
- If the user rejects your plan with feedback, revise the plan based on their feedback and call confirm_plan again with the updated plan. Do NOT repeat the same plan unchanged.
- If the user cancels the plan, acknowledge it and stop.
- You may only use the tools listed above.
"""

DEFAULT_EXECUTE_PROMPT = """\
You are in EXECUTE mode. Work through tasks one at a time.

PHASE: EXECUTE

Available tools: {tool_names}

{task_state}

Task status markers: [done] = completed, [FAIL: reason] = failed, [    ] = not started.

What to do:
1. Pick the next incomplete task (marked [    ]) from the list above.
2. Execute it using the appropriate tool(s).
3. After the task is truly done, call complete_task(index) where index is the task number shown in the list.
4. If a task fails and cannot be recovered, call fail_task(index, reason) to mark it.
5. Move on to the next task.

Rules:
- Call exactly ONE tool per step.
- NEVER repeat identical failed tool calls (duplicate detection is active).
- When all tasks are done, call terminate with a summary.
- If a tool returns an error (timeout, file not found, syntax error, wrong path), analyze the error and retry with corrected parameters. You do NOT need to call fix_plan for trivial errors.
- Only call fix_plan if the same task fails repeatedly (3+ attempts) or if the task design was wrong.
- Only call replan(mode='soft') if the entire approach is wrong. Use replan(mode='hard') only for complete strategy replacement.
- If you need to think step-by-step before acting, do so — reasoning will be captured in a collapsible block.
- You MUST call complete_task(index) or fail_task(index, reason) after working on a task.
- If a tool named `parallel_tools` is available, use it to call multiple independent tools at once for efficiency.
- You may only use the tools listed above. Do NOT ask the user questions.
"""

DEFAULT_REVIEW_PROMPT = """\
You are in REVIEW mode. Your ONLY job is to pick one of three actions.

PHASE: REVIEW
Original goal: {goal}
Available tools: {tool_names}

{task_state}

Task status markers: [done] = completed, [FAIL: reason] = failed with reason, [    ] = not started.

You MUST call exactly ONE of these tools:

1. `terminate(final_answer)` — Everything is done and correct. Provide a concise final answer summarising what was accomplished.
2. `fix_plan(reason, updated_tasks)` — Only minor fixes are needed (a task failed or needs a small correction). List just the new/corrected tasks.
3. `replan(reason, updated_tasks, mode="soft")` — The overall strategy is broken and tasks need to be replaced entirely.

Rules:
- If there are only minor issues with individual tasks, ALWAYS prefer `fix_plan` over `replan`. Only use `replan` if the overall strategy is broken.
- Be honest — don't call `terminate` if something is missing or wrong.
- If the result is good enough, call `terminate`. Don't gold-plate.
- Provide a brief reasoning for your assessment before calling the final tool.
- You may only use the tools listed above.
"""


# ──────────────────────────────────────────────────────────────────
#  SSE STREAM PARSER
# ──────────────────────────────────────────────────────────────────

async def stream_completion(request, body, user):
    """Stream OWUI completion, yielding structured events. Retries once on transient errors."""
    body["stream"] = True
    max_retries = 1
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = await generate_chat_completion(request, body, user=user)
            break
        except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(f"Transient API error (attempt {attempt + 1}), retrying: {e}")
                await asyncio.sleep(2 ** attempt)
                continue
            logger.error(f"LLM API error after {attempt + 1} attempt(s): {e}")
            yield {"type": "error", "text": str(e)}
            return
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
    """Remove thinking/reasoning blocks from model output.
    Handles: paired tags, unclosed tags, pipe-style blocks, and reasoning prefixes."""
    # Remove paired tags: <thinking>...</thinking> etc.
    text = re.sub(
        r"<(?:think|thinking|reason|reasoning|thought)>.*?</(?:think|thinking|reason|reasoning|thought)>",
        "", text, flags=re.DOTALL | re.IGNORECASE
    )
    # Remove pipe-style blocks: |begin_of_thought|...|end_of_thought|
    text = re.sub(r"\|begin_of_thought\|.*?\|end_of_thought\|", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove unclosed tags that run to end of text or before next <
    text = re.sub(
        r"<(?:think|thinking|reason|reasoning|thought)>[^<]*",
        "", text, flags=re.DOTALL | re.IGNORECASE
    )
    # Remove reasoning prefixes on their own line
    text = re.sub(
        r"^(?:Thinking|Thought|Reasoning|Analysis|Reason)\s*:\s*",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )
    return text.strip()


def extract_xml_tool_calls(text):
    """Attempt to extract tool calls from hallucinated XML <ToolCall> blocks."""
    calls = []
    pattern = re.compile(
        r"<ToolCall>\s*<name>\s*(.*?)\s*</name>\s*<arguments>\s*(.*?)\s*</arguments>\s*</ToolCall>",
        re.DOTALL | re.IGNORECASE
    )
    for i, m in enumerate(pattern.finditer(text)):
        name = m.group(1).strip()
        args_str = m.group(2).strip()
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            args = {}
        calls.append({
            "id": f"call_xml_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args) if isinstance(args, dict) else args_str},
            "index": i,
        })
    return calls


def strip_html(text):
    """Remove HTML tags and decode common entities for plain-text display."""
    text = re.sub(r"<details[^>]*>.*?</details>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<summary>(.*?)</summary>", r"\1: ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return re.sub(r"\s+", " ", text).strip()


def _comma_list(val: str) -> List[str]:
    """Convert a comma-separated string to a list of stripped, non-empty strings."""
    if not val or not isinstance(val, str):
        return []
    return [x.strip() for x in val.split(",") if x.strip()]


# ──────────────────────────────────────────────────────────────────
#  AGENT LOOP ENGINE
# ──────────────────────────────────────────────────────────────────

class HelixAgentEngine:
    """Helix Agent — single-model agent loop with per-phase tool filtering."""

    PHASE_PLAN = "plan"
    PHASE_EXECUTE = "execute"
    PHASE_REVIEW = "review"

    MAX_HISTORY_MESSAGES = 50

    # Internal tools that are ALWAYS available regardless of phase filters
    INTERNAL_TOOLS = {"terminate", "replan", "complete_task", "fail_task", "confirm_plan", "fix_plan"}

    def __init__(self, request, user, body, event_emitter, event_call, metadata, valves, user_valves=None):
        self.request = request
        self.user = user
        self.body = body
        self.event_emitter = event_emitter
        self.event_call = event_call
        self.metadata = metadata
        self.valves = valves
        self.user_valves = user_valves

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
        self.consecutive_json_errors = 0
        self._consecutive_tool_misses: Dict[str, int] = {}
        self._seen_file_ids: Set[str] = set()
        self._state_restored = False
        self._output_parts = []
        self.loop_count = 0
        self.goal = ""
        self._stream_queue = None
        self._turn_buffer: list[str] = []
        self._flush_task: Optional[asyncio.Task] = None

    @property
    def is_silent(self):
        return getattr(self.user_valves, "SILENT_MODE", False)

    def _format_output(self):
        if not self.is_silent:
            return "".join(self._output_parts)
        filtered = []
        for part in self._output_parts:
            s = part.strip()
            if not s:
                continue
            if "<details type=\"reasoning\">" in s:
                continue
            if "<details type=\"tool_calls\">" in s:
                continue
            if s.startswith("[EXEC]"):
                continue
            if s.startswith("[RPLN]"):
                continue
            if s.startswith("[FIX]"):
                continue
            if s.startswith("[OK]") and "marked complete" in s:
                continue
            if s.startswith("[FAIL]") and "marked failed" in s:
                continue
            if s.startswith("[PLAN]"):
                continue
            filtered.append(part)
        return "".join(filtered)

    def _save_state_to_history(self):
        state = json.dumps({
            "goal": self.goal,
            "task_list": self.task_list,
            "completed": self.completed_tasks,
            "failed": self.failed_tasks,
            "phase": self.phase,
        }, ensure_ascii=False)

        # Remove old state messages to prevent bloat
        self.history = [
            m for m in self.history
            if not (m.get("role") == "system" and m.get("content", "").startswith("[AGENT_STATE]"))
        ]

        self.history.append({
            "role": "system",
            "content": f"[AGENT_STATE] {state}",
        })

    def _restore_state_from_messages(self, messages):
        for msg in reversed(messages):
            content = msg.get("content", "")
            if msg.get("role") == "system" and content.startswith("[AGENT_STATE]"):
                try:
                    payload = json.loads(content.replace("[AGENT_STATE] ", "", 1))
                    self.task_list = payload.get("task_list", [])
                    self.completed_tasks = payload.get("completed", [])
                    self.failed_tasks = payload.get("failed", [])
                    self.phase = payload.get("phase", self.PHASE_PLAN)
                    self.goal = payload.get("goal", self.goal)
                    self.loop_count = 0
                    self._state_restored = True
                except Exception:
                    pass
                break

    async def emit_status(self, msg, done=False):
        if self.event_emitter:
            try:
                await self.event_emitter({"type": "status", "data": {"description": msg, "done": done}})
            except Exception:
                pass

    async def emit_output(self, text):
        self._output_parts.append(text)
        if self.is_silent:
            return
        self._turn_buffer.append(text)
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = asyncio.create_task(self._schedule_flush(0.3))

    async def _schedule_flush(self, delay: float):
        try:
            await asyncio.sleep(delay)
            await self._flush_turn_buffer()
        except asyncio.CancelledError:
            pass

    async def _flush_turn_buffer(self):
        if not self._turn_buffer:
            return
        combined = "\n\n".join(self._turn_buffer)
        # Ensure blank line padding so OpenWebUI's \n join never swallows <details>
        if not combined.startswith("\n\n"):
            combined = "\n\n" + combined
        if not combined.endswith("\n\n"):
            combined = combined + "\n\n"
        self._turn_buffer.clear()
        q = getattr(self, "_stream_queue", None)
        if q is not None:
            await q.put(combined)

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
                "description": "Adjust the plan. Mode 'soft' (default) compresses history and keeps context for continuity. Mode 'hard' does a full reset with new task list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "What went wrong or what is missing"},
                        "updated_tasks": {"type": "string", "description": "Updated task list as numbered steps (only what is still needed)"},
                        "mode": {"type": "string", "enum": ["soft", "hard"], "description": "Replan mode: 'soft' (default) = compress history and keep context, 'hard' = full reset with new task list"},
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
        self.all_tools_dict["confirm_plan"] = {
            "spec": {
                "name": "confirm_plan",
                "description": "Present the task plan to the user for approval. Call this after creating the plan in PLAN phase. Provide the plan as a numbered list (e.g. '1. Task one\\n2. Task two') or as JSON with a 'tasks' array. The user can accept, provide feedback, or cancel.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {"type": "string", "description": "The full plan text. Use numbered format '1. Task description' or JSON {\"tasks\": [\"task1\", \"task2\"]}. Each task should be a clear, actionable step."},
                    },
                    "required": ["plan"],
                },
            },
            "callable": self._tool_confirm_plan,
            "type": "function",
        }
        self.all_tools_dict["fix_plan"] = {
            "spec": {
                "name": "fix_plan",
                "description": "Add correction tasks for minor issues without resetting history. Use when individual tasks failed but the overall approach is still correct. The new tasks will be appended to the current task list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Why the fix is needed"},
                        "updated_tasks": {"type": "string", "description": "Newline-separated list of correction or replacement tasks to append"},
                    },
                    "required": ["reason", "updated_tasks"],
                },
            },
            "callable": self._tool_fix_plan,
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

        # If allowlist is empty -> allow ALL tools
        # If allowlist has entries -> only those tools (plus internals)
        self.phase_tools_dict = {}

        for name, tool in self.all_tools_dict.items():
            # Internal tools are ALWAYS included regardless of allowlist
            if name in self.INTERNAL_TOOLS:
                self.phase_tools_dict[name] = tool
                continue

            # Allowlist filtering
            if allowlist:
                if name in allowlist:
                    self.phase_tools_dict[name] = tool
            else:
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

    # Soft or hard reset. Prefer soft mode to preserve compressed history.
    async def _tool_replan(self, reason: str, updated_tasks: str, mode: str = "soft", **kwargs):
        """Process a replan: update task list and compress or reset history."""
        new_tasks = self._extract_task_list(updated_tasks) if updated_tasks else []

        if mode == "soft":
            # Soft replan: replace task list (or keep remaining), compress history
            if new_tasks:
                # Replace the task list entirely; old tasks are preserved in compressed history
                self.task_list = new_tasks
            else:
                failed_task_names = {f["task"] for f in self.failed_tasks}
                self.task_list = [
                    t for t in self.task_list
                    if t not in self.completed_tasks and t not in failed_task_names
                ]
            self.completed_tasks = []
            self.failed_tasks = []
            self.consecutive_json_errors = 0
            # Grant 3 extra iterations without fully resetting the safety counter
            self.loop_count = max(0, self.loop_count - 3)
            self.history = self._compress_history()
        else:
            # Hard replan: replace task list entirely, full history reset
            if new_tasks:
                self.task_list = new_tasks
            else:
                failed_task_names = {f["task"] for f in self.failed_tasks}
                self.task_list = [
                    t for t in self.task_list
                    if t not in self.completed_tasks and t not in failed_task_names
                ]
            self.completed_tasks = []
            self.failed_tasks = []
            self.consecutive_json_errors = 0
            self.loop_count = 0
            self.history = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self.goal},
            ]

        if self.phase != self.PHASE_EXECUTE:
            self._transition_to(self.PHASE_EXECUTE)

        self._save_state_to_history()
        await self.emit_task_update()
        return json.dumps({"replan": True, "reason": reason, "updated_tasks": updated_tasks, "mode": mode})

    async def _tool_complete_task(self, **kwargs):
        idx = kwargs.get("index", -1)
        if 0 <= idx < len(self.task_list):
            task = self.task_list[idx]
            if task not in self.completed_tasks:
                self.completed_tasks.append(task)
            self._save_state_to_history()
            self.history[0]["content"] = self._build_system_prompt()
            await self.emit_task_update()
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
            self._save_state_to_history()
            self.history[0]["content"] = self._build_system_prompt()
            await self.emit_task_update()
            return json.dumps({"failed": True, "task": task, "index": idx, "reason": reason})
        return json.dumps({"failed": False, "error": f"Invalid task index {idx}"})

    # Lightweight correction tool. Use this instead of replan for minor issues.
    async def _tool_fix_plan(self, reason: str, updated_tasks: str, **kwargs):
        if not self.task_list:
            return json.dumps({"fix_plan": False, "error": "No task list available"})

        new_tasks = self._extract_task_list(updated_tasks)
        if not new_tasks:
            return json.dumps({"fix_plan": False, "error": "No tasks provided"})

        # Compute insertion index BEFORE removing failed tasks
        failed_names = {f["task"] for f in self.failed_tasks}
        insert_idx = len(self.task_list)  # default: append
        for i, t in enumerate(self.task_list):
            if any(f in t or t in f for f in failed_names):
                insert_idx = i
                break

        # Remove failed tasks from the task list
        self.task_list = [t for t in self.task_list if t not in failed_names]
        if insert_idx > len(self.task_list):
            insert_idx = len(self.task_list)
        self.failed_tasks = []

        self.task_list[insert_idx:insert_idx] = new_tasks

        self._save_state_to_history()
        self.history[0]["content"] = self._build_system_prompt()
        await self.emit_task_update()
        return json.dumps({"fix_plan": True, "inserted_tasks": new_tasks, "reason": reason})

    async def _tool_confirm_plan(self, **kwargs):
        plan_text = kwargs.get("plan", "")
        uv = self.user_valves

        if uv and (getattr(uv, "YOLO_MODE", False) or not getattr(uv, "ENABLE_PLAN_APPROVAL", False)):
            return json.dumps({"action": "accept"})

        if not self.event_call:
            return json.dumps({"action": "accept"})

        tasks = self._extract_task_list(plan_text)
        tasks_data = [{"task_id": f"T{i+1}", "description": t} for i, t in enumerate(tasks)]
        if not tasks_data:
            tasks_data = [{"task_id": "T1", "description": plan_text}]

        js = self._build_plan_approval_js(tasks_data)
        try:
            raw = await self.event_call({"type": "execute", "data": {"code": js}})
        except Exception as e:
            logger.error(f"Plan approval event_call failed: {e}")
            return json.dumps({"action": "accept"})

        raw_str = raw if isinstance(raw, str) else (raw.get("result") or raw.get("value") or "{}") if raw else "{}"
        try:
            res = json.loads(raw_str) if isinstance(raw_str, str) and raw_str.startswith("{") else {"action": "accept"}
        except (json.JSONDecodeError, AttributeError):
            res = {"action": "accept"}

        return json.dumps(res)

    def _base_theme_js(self):
        return """
            const col = {
                overlay: 'rgba(0,0,0,0.55)', panel: '#1e293b', border: '#334155',
                text: '#f1f5f9', sub: '#94a3b8', input: '#0f172a', inputBorder: '#475569',
                btn: '#334155', btnText: '#e2e8f0', btnBorder: '#475569',
                btnPrimary: '#3b82f6', btnPrimaryText: '#ffffff',
            };
            try { const s = getComputedStyle(document.documentElement);
              col.panel = s.getPropertyValue('--color-gray-900').trim() || col.panel;
              col.text = s.getPropertyValue('--color-gray-50').trim() || col.text;
              col.btnPrimary = s.getPropertyValue('--color-blue-500').trim() || col.btnPrimary;
            } catch(e) {}
        """

    def _build_plan_approval_js(self, tasks: list, timeout_s: int = 600) -> str:
        ts = json.dumps(tasks)
        return f"""
    return (function() {{
      return new Promise((resolve) => {{
    {self._base_theme_js()}
        let _timer;
        const overlay = document.createElement('div');
        overlay.style.cssText = `position:fixed;inset:0;z-index:999999;background:${{col.overlay}};display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(4px);`;
        const panel = document.createElement('div');
        panel.style.cssText = `background:${{col.panel}};border:1px solid ${{col.border}};border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);color:${{col.text}};font-family:ui-sans-serif,system-ui,sans-serif;width:100%;max-width:520px;max-height:90vh;padding:32px;display:flex;flex-direction:column;gap:24px;`;

        const header = document.createElement('div');
        header.style.cssText = 'display:flex;align-items:center;gap:12px;flex-shrink:0;';
        const icon = document.createElement('div'); icon.textContent = '\uD83D\uDCCB'; icon.style.cssText = 'font-size:24px;';
        const title = document.createElement('div'); title.textContent = 'Review Proposed Plan'; title.style.cssText = `font-size:20px;font-weight:800;color:${{col.text}};letter-spacing:-0.4px;`;
        header.appendChild(icon); header.appendChild(title); panel.appendChild(header);

        const scrollContainer = document.createElement('div');
        scrollContainer.style.cssText = 'overflow-y:auto;flex:1;display:flex;flex-direction:column;gap:12px;padding-right:8px;';

        const tasksData = {ts};
        tasksData.forEach((t, i) => {{
            const card = document.createElement('div');
            card.style.cssText = `background:${{col.input}};border:1px solid ${{col.inputBorder}};border-radius:12px;padding:12px 16px;display:flex;gap:12px;align-items:flex-start;`;

            const num = document.createElement('div');
            num.textContent = i + 1;
            num.style.cssText = `width:24px;height:24px;background:${{col.btnPrimary}};color:${{col.btnPrimaryText}};border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:bold;flex-shrink:0;margin-top:2px;`;

            const content = document.createElement('div');
            content.style.cssText = 'display:flex;flex-direction:column;gap:4px;';
            const tid = document.createElement('div'); tid.textContent = t.task_id; tid.style.cssText = `font-size:11px;font-weight:bold;color:${{col.sub}};text-transform:uppercase;`;
            const desc = document.createElement('div'); desc.textContent = t.description; desc.style.cssText = `font-size:14px;color:${{col.text}};line-height:1.4;`;

            content.appendChild(tid); content.appendChild(desc);
            card.appendChild(num); card.appendChild(content);
            scrollContainer.appendChild(card);
        }});
        panel.appendChild(scrollContainer);

        const inputContainer = document.createElement('div');
        inputContainer.style.cssText = 'display:flex;flex-direction:column;gap:10px;flex-shrink:0;';
        const inputLabel = document.createElement('div'); inputLabel.textContent = 'Feedback (optional):'; inputLabel.style.cssText = `font-size:12px;font-weight:700;color:${{col.sub}};text-transform:uppercase;letter-spacing:0.5px;`;
        const feedbackInput = document.createElement('textarea');
        feedbackInput.placeholder = 'e.g., "Add a step to check for X" or "Skip the second task"';
        feedbackInput.style.cssText = `background:${{col.input}};border:1px solid ${{col.inputBorder}};color:${{col.text}};padding:14px;border-radius:14px;font-size:14px;outline:none;min-height:70px;resize:none;transition:border-color 0.2s;`;
        feedbackInput.onfocus = () => feedbackInput.style.borderColor = 'var(--color-blue-500)';
        feedbackInput.onblur = () => feedbackInput.style.borderColor = col.inputBorder;
        inputContainer.appendChild(inputLabel); inputContainer.appendChild(feedbackInput); panel.appendChild(inputContainer);

        const footer = document.createElement('div');
        footer.style.cssText = 'display:flex;gap:12px;flex-shrink:0;';

        const makeBtn = (label, primary) => {{
            const b = document.createElement('button');
            b.textContent = label;
            b.style.cssText = `flex:1;padding:14px 20px;border-radius:9999px;font-size:15px;font-weight:700;cursor:pointer;transition:all 0.2s;border:1px solid ${{primary ? 'transparent' : col.btnBorder}};background:${{primary ? col.btnPrimary : col.btn}};color:${{primary ? col.btnPrimaryText : col.btnText}};`;
            b.onmouseenter = () => {{ b.style.opacity='0.9'; b.style.transform='translateY(-1px)'; }};
            b.onmouseleave = () => {{ b.style.opacity='1'; b.style.transform='translateY(0)'; }};
            return b;
        }};

        const acceptBtn = makeBtn('Accept Plan', true);
        const feedbackBtn = makeBtn('Send Feedback', false);
        const cancelBtn = makeBtn('Cancel', false);
        cancelBtn.style.background = '#7f1d1d';
        cancelBtn.style.color = '#fecaca';
        cancelBtn.style.borderColor = '#991b1b';

        const cleanup = () => {{ overlay.remove(); }};

        acceptBtn.onclick = () => {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'accept'}})); }};
        feedbackBtn.onclick = () => {{
            const val = feedbackInput.value.trim();
            if (val) {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'feedback', value: val}})); }}
            else {{ acceptBtn.onclick(); }}
        }};
        cancelBtn.onclick = () => {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'cancel'}})); }};

        footer.appendChild(acceptBtn); footer.appendChild(feedbackBtn); footer.appendChild(cancelBtn); panel.appendChild(footer);

        const countdown = document.createElement('div');
        countdown.style.cssText = `font-size:11px;color:${{col.sub}};text-align:center;margin-top:-12px;flex-shrink:0;`;
        panel.appendChild(countdown);

        overlay.appendChild(panel); document.body.appendChild(overlay);
        feedbackInput.focus();

        let remaining = {timeout_s};
        const updateCountdown = () => {{
            countdown.textContent = remaining > 0 ? `Auto-accepting in ${{remaining}}s...` : '';
            if (remaining <= 0) {{ cleanup(); resolve(JSON.stringify({{action:'accept'}})); }}
        }};
        updateCountdown();
        _timer = setInterval(() => {{ remaining--; updateCountdown(); }}, 1000);
      }});
    }})();"""

    # ── Iteration Limit UI ──

    def _build_iteration_limit_js(self, current_iter, max_iter, timeout_s: int = 300) -> str:
        """Build a Continue/Cancel modal for iteration limit reached."""
        return f"""
    return (function() {{
      return new Promise((resolve) => {{
    {self._base_theme_js()}
        const overlay = document.createElement('div');
        overlay.style.cssText = `position:fixed;inset:0;z-index:999999;background:${{col.overlay}};display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(4px);`;
        const panel = document.createElement('div');
        panel.style.cssText = `background:${{col.panel}};border:1px solid ${{col.border}};border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);color:${{col.text}};font-family:ui-sans-serif,system-ui,sans-serif;width:100%;max-width:440px;padding:32px;display:flex;flex-direction:column;gap:20px;`;

        const header = document.createElement('div');
        header.style.cssText = 'display:flex;align-items:center;gap:12px;';
        const icon = document.createElement('div'); icon.textContent = '⚠️'; icon.style.cssText = 'font-size:24px;';
        const title = document.createElement('div'); title.textContent = 'Iteration Limit Reached'; title.style.cssText = `font-size:18px;font-weight:800;color:${{col.text}};`;
        header.appendChild(icon); header.appendChild(title); panel.appendChild(header);

        const msg = document.createElement('div');
        msg.style.cssText = `font-size:14px;color:${{col.sub}};line-height:1.5;`;
        msg.textContent = `The agent has used {current_iter} of {max_iter} iterations. Continue for more?`;
        panel.appendChild(msg);

        const countdown = document.createElement('div');
        countdown.style.cssText = `font-size:11px;color:${{col.sub}};text-align:center;margin-top:-8px;`;
        panel.appendChild(countdown);

        const footer = document.createElement('div');
        footer.style.cssText = 'display:flex;gap:10px;';
        const makeBtn = (label, primary) => {{
            const b = document.createElement('button');
            b.textContent = label;
            b.style.cssText = `flex:1;padding:12px 18px;border-radius:9999px;font-size:14px;font-weight:700;cursor:pointer;border:1px solid ${{primary ? 'transparent' : col.btnBorder}};background:${{primary ? col.btnPrimary : col.btn}};color:${{primary ? col.btnPrimaryText : col.btnText}};`;
            b.onmouseenter = () => {{ b.style.opacity='0.9'; }};
            b.onmouseleave = () => {{ b.style.opacity='1'; }};
            return b;
        }};
        const continueBtn = makeBtn('Continue', true);
        const stopBtn = makeBtn('Stop', false);
        const cleanup = () => {{ overlay.remove(); }};
        let _timer;
        continueBtn.onclick = () => {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'continue'}})); }};
        stopBtn.onclick = () => {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'stop'}})); }};
        footer.appendChild(continueBtn); footer.appendChild(stopBtn); panel.appendChild(footer);

        overlay.appendChild(panel); document.body.appendChild(overlay);

        let remaining = {timeout_s};
        const updateCountdown = () => {{
            countdown.textContent = remaining > 0 ? `Auto-stopping in ${{remaining}}s...` : '';
            if (remaining <= 0) {{ cleanup(); resolve(JSON.stringify({{action:'stop'}})); }}
        }};
        updateCountdown();
        _timer = setInterval(() => {{ remaining--; updateCountdown(); }}, 1000);
      }});
    }})();"""

    # ── Task State String ──

    async def emit_task_update(self, finalize_tasks=False):
        """Emit task progress via Open WebUI's native task list UI.

        When finalize_tasks is True, all remaining pending/in_progress tasks
        are marked as completed so the TaskList UI is dismissed (it only
        renders when at least one task is active).
        """
        if not self.task_list:
            return
        first_outstanding = next(
            (i for i, t in enumerate(self.task_list)
             if t not in self.completed_tasks
             and not any(f["task"] == t for f in self.failed_tasks)),
            None,
        )
        tasks = []
        for i, task in enumerate(self.task_list):
            if task in self.completed_tasks:
                status = "completed"
            elif any(f["task"] == task for f in self.failed_tasks):
                status = "cancelled"
            elif finalize_tasks:
                status = "completed"
            elif i == first_outstanding:
                status = "in_progress"
            else:
                status = "pending"
            tasks.append({"id": str(i + 1), "content": task, "status": status})
        if self.event_emitter:
            try:
                await self.event_emitter({
                    "type": "chat:message:tasks",
                    "data": {"tasks": tasks},
                })
            except Exception:
                pass

    def _build_task_state(self):
        lines = []
        lines.append("Current Tasks:")
        for i, task in enumerate(self.task_list):
            if task in self.completed_tasks:
                status = "[done]"
            elif any(f["task"] == task for f in self.failed_tasks):
                reason = next((f["reason"] for f in self.failed_tasks if f["task"] == task), "")
                status = f"[FAIL: {reason}]"
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
            timeout = self.valves.TOOL_TIMEOUT if self.valves.TOOL_TIMEOUT > 0 else None
            if timeout:
                result = await asyncio.wait_for(callable_fn(**filtered_args), timeout=timeout)
            else:
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
        except asyncio.TimeoutError:
            logger.error(f"Tool execution timed out ({tool_name}): {self.valves.TOOL_TIMEOUT}s")
            return f"Error: Tool '{tool_name}' timed out after {self.valves.TOOL_TIMEOUT}s.", []
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return f"Error executing {tool_name}: {e}", []

    # ── Phase System Prompt ──

    def _build_system_prompt(self):
        """Build system prompt based on current phase using Valves overrides."""
        tool_names = ", ".join(sorted(self.phase_tools_dict.keys()))
        task_state = self._build_task_state()

        # Pick base prompt from Valves or fallback to default
        try:
            if self.phase == self.PHASE_PLAN:
                base = self.valves.PLAN_PROMPT or DEFAULT_PLAN_PROMPT
                return base.format(tool_names=tool_names)

            elif self.phase == self.PHASE_EXECUTE:
                base = self.valves.EXECUTE_PROMPT or DEFAULT_EXECUTE_PROMPT
                return base.format(tool_names=tool_names, task_state=task_state)

            elif self.phase == self.PHASE_REVIEW:
                base = self.valves.REVIEW_PROMPT or DEFAULT_REVIEW_PROMPT
                return base.format(goal=self.goal, task_state=task_state, tool_names=tool_names)

            return DEFAULT_PLAN_PROMPT.format(tool_names=tool_names)
        except (KeyError, IndexError, ValueError):
            # User-provided prompt may have stray braces; fall back to default
            if self.phase == self.PHASE_PLAN:
                return DEFAULT_PLAN_PROMPT.format(tool_names=tool_names)
            elif self.phase == self.PHASE_EXECUTE:
                return DEFAULT_EXECUTE_PROMPT.format(tool_names=tool_names, task_state=task_state)
            elif self.phase == self.PHASE_REVIEW:
                return DEFAULT_REVIEW_PROMPT.format(goal=self.goal, task_state=task_state, tool_names=tool_names)
            return DEFAULT_PLAN_PROMPT.format(tool_names=tool_names)

    # ── Phase Transitions ──

    def _transition_to(self, phase):
        """Transition to a new phase: update tools, system prompt, state."""
        self.phase = phase
        self._consecutive_tool_misses.clear()
        # Rebuild filtered tools for new phase
        self._filter_tools_for_phase(phase)

        # Update system prompt in history
        if self.history:
            self.history[0]["content"] = self._build_system_prompt()

    # ── Context Window Management ──

    def _manage_context_window(self, messages):
        """Trim history to MAX_HISTORY_MESSAGES while keeping tool call pairs intact."""
        if len(messages) <= self.MAX_HISTORY_MESSAGES:
            return messages

        to_remove = len(messages) - self.MAX_HISTORY_MESSAGES
        # Build head: system prompt + goal + [AGENT_STATE] messages (always preserved)
        head = messages[:2]
        state_indices = set()
        for i, m in enumerate(messages[2:], start=2):
            if m.get("role") == "system" and m.get("content", "").startswith("[AGENT_STATE]"):
                head.append(m)
                state_indices.add(i)
        # Non-state messages after head, split into removed and tail
        non_state = [m for i, m in enumerate(messages[2:], start=2) if i not in state_indices]
        removed = non_state[:to_remove]
        tail = non_state[to_remove:]

        # Drop dangling tool results whose assistant call was removed
        while tail and tail[0].get("role") == "tool":
            call_id = tail[0].get("tool_call_id")
            assistant_removed = any(
                m.get("role") == "assistant"
                and any(tc.get("id") == call_id for tc in m.get("tool_calls", []))
                for m in removed
            )
            if assistant_removed:
                tail = tail[1:]
            else:
                break

        # If an assistant with tool_calls is at the front of tail,
        # pull missing tool results back from the removed chunk
        if tail and tail[0].get("role") == "assistant" and tail[0].get("tool_calls"):
            missing = []
            for tc in tail[0]["tool_calls"]:
                tc_id = tc.get("id")
                found_in_tail = any(
                    m.get("role") == "tool" and m.get("tool_call_id") == tc_id
                    for m in tail
                )
                if not found_in_tail:
                    for m in removed:
                        if m.get("role") == "tool" and m.get("tool_call_id") == tc_id:
                            missing.append(m)
                            break
            if missing:
                tail = missing + tail

        return head + tail

    def _get_truncation_limit(self):
        return self.valves.MAX_TOOL_RESULT_CHARS

    def _compress_history(self):
        """
        Compress history for soft replan.
        - Keeps system prompt + goal.
        - Summarizes old execution into a summary block.
        - Preserves last ~6 messages, but never splits a tool call pair.
        - Tool results that are dropped from the "middle" are replaced with
          short previews: [Tool: name → first 200 chars of result].
        """
        # 1. System prompt + goal (always preserved)
        preserved = []
        goal_msg = None
        for msg in self.history:
            role = msg.get("role")
            if role == "system":
                preserved.append(msg)
            elif role == "user" and goal_msg is None:
                goal_msg = msg
        if goal_msg:
            preserved.append(goal_msg)

        # 2. Build execution summary
        summary_parts = ["=== Compressed Execution History ==="]
        summary_parts.append(f"Goal: {self.goal}")
        summary_parts.append(f"Completed tasks: {', '.join(self.completed_tasks) if self.completed_tasks else 'None'}")
        if self.failed_tasks:
            summary_parts.append("Failed tasks:")
            for ft in self.failed_tasks:
                summary_parts.append(f"  - {ft.get('task', 'unknown')}: {ft.get('reason', 'no reason')}")
        if self.task_list:
            summary_parts.append(f"Remaining tasks before replan: {', '.join(self.task_list)}")
        # 3. Middle section: tool results become previews, everything else is dropped
        middle_previews = []
        start_idx = len(preserved)

        keep_last = 6
        if len(self.history) > len(preserved) + keep_last:
            middle = self.history[len(preserved):-keep_last]
        else:
            middle = []

        for msg in middle:
            if msg.get("role") == "tool":
                name = msg.get("name", "unknown")
                content = strip_html(msg.get("content", ""))
                preview = content[:200].replace("\n", " ")
                middle_previews.append(f"[Tool: {name} → {preview}...]")
            elif msg.get("role") == "user" and len(msg.get("content", "")) < 500:
                middle_previews.append(f"[User: {strip_html(msg['content'][:200])}]")

        if middle_previews:
            summary_parts.append("Earlier actions (summarized):")
            summary_parts.extend(middle_previews)

        summary_msg = {"role": "system", "content": "\n".join(summary_parts)}

        # 4. Recent tail — safe extraction that preserves tool call pairs
        recent = self.history[-keep_last:] if len(self.history) > keep_last else self.history[len(preserved):]

        # If the cutoff split an assistant+tool_calls from its results, fix it
        if recent and recent[-1].get("role") == "assistant" and recent[-1].get("tool_calls"):
            call_ids = {tc.get("id") for tc in recent[-1]["tool_calls"] if tc.get("id")}
            for msg in self.history:
                if msg.get("role") == "tool" and msg.get("tool_call_id") in call_ids:
                    if msg not in recent:
                        recent.append(msg)

        compressed = preserved.copy()
        compressed.append(summary_msg)
        compressed.extend(recent)
        return compressed

    # ── Main Loop ──

    async def _run_impl(self, user_msg, last_user_msg_raw, model):
        await self.emit_status("Agent starting...")
        await self.resolve_tools()

        self.consecutive_json_errors = 0
        self.loop_count = 0

        if self._state_restored:
            # Preserve restored task state, phase, and history.
            # Append new user message to the goal and history.
            self.goal = f"{self.goal}; Updated: {user_msg}"
            # Remove stale [AGENT_STATE] system messages from history
            self.history = [
                m for m in self.history
                if not (m.get("role") == "system" and m.get("content", "").startswith("[AGENT_STATE]"))
            ]
            # Refresh the system prompt (phase may have changed)
            if self.history and self.history[0].get("role") == "system":
                self.history[0]["content"] = self._build_system_prompt()
            else:
                self.history.insert(0, {"role": "system", "content": self._build_system_prompt()})
            self.history.append({"role": "user", "content": user_msg})
            # Filter tools for the restored phase
            self._filter_tools_for_phase(self.phase)
        else:
            # Fresh session: initialize everything from scratch
            self.goal = user_msg
            self.phase = self.PHASE_PLAN
            self.task_list = []
            self.completed_tasks = []
            self.failed_tasks = []
            self._filter_tools_for_phase(self.PHASE_PLAN)
            system_prompt = self._build_system_prompt()
            self.history = [
                {"role": "system", "content": system_prompt},
                last_user_msg_raw if last_user_msg_raw else {"role": "user", "content": user_msg},
            ]

        recent_calls = []
        self._output_parts = []

        while True:
            if self.loop_count >= self.valves.MAX_ITERATIONS:
                await self.emit_output(f"\n[WARN] Max iterations ({self.valves.MAX_ITERATIONS}) reached.")
                should_continue = False
                if self.event_call and not (self.user_valves and getattr(self.user_valves, "YOLO_MODE", False)):
                    try:
                        js = self._build_iteration_limit_js(self.loop_count, self.valves.MAX_ITERATIONS)
                        raw = await self.event_call({"type": "execute", "data": {"code": js}})
                        raw_str = raw if isinstance(raw, str) else (raw.get("result") or raw.get("value") or "{}") if raw else "{}"
                        try:
                            res = json.loads(raw_str) if isinstance(raw_str, str) and raw_str.startswith("{") else {}
                        except (json.JSONDecodeError, AttributeError):
                            res = {}
                        should_continue = res.get("action") == "continue"
                    except Exception as e:
                        logger.warning(f"Iteration limit dialog failed: {e}")
                if should_continue:
                    self.loop_count = 0
                    await self.emit_status("Continuing after iteration limit...")
                    continue
                await self.emit_task_update(finalize_tasks=True)
                await self.emit_status("Max iterations", done=True)
                return self._format_output()

            self.loop_count += 1
            recent_calls = recent_calls[-30:]

            phase_icons = {
                self.PHASE_PLAN: "[PLAN]",
                self.PHASE_EXECUTE: "[EXEC]",
                self.PHASE_REVIEW: "[REVU]",
            }
            phase_name = {
                self.PHASE_PLAN: "Plan",
                self.PHASE_EXECUTE: "Execute",
                self.PHASE_REVIEW: "Review",
            }
            icon = phase_icons.get(self.phase, "[LOOP]")
            name = phase_name.get(self.phase, "Loop")

            await self.emit_status(f"{icon} {name} -- step {self.loop_count}/{self.valves.MAX_ITERATIONS}")
            if not self.is_silent:
                await self.emit_output(f"\n### {icon} Loop {self.loop_count}\n")

            self.history = self._manage_context_window(self.history)
            # Strip system messages (including [AGENT_STATE] bookkeeping) from LLM context
            # Only the first message (system prompt) is kept
            call_messages = [self.history[0]] + [m for m in self.history[1:] if m.get("role") != "system"]

            completion_body = {
                **self.body,
                "model": model,
                "messages": call_messages,
                "tools": self.phase_tools_specs if self.phase_tools_specs else None,
                "metadata": self.pipe_metadata,
            }

            try:
                completion_body["messages"] = await add_file_context(
                    copy.deepcopy(call_messages), self.chat_id, self.user
                )
            except Exception:
                pass

            # ── Stream LLM response ──
            tc_dict = {}
            content_chunks = []
            reasoning_chunks = []

            async for event in stream_completion(self.request, completion_body, self.user):
                etype = event.get("type")
                if etype == "error":
                    await self.emit_output(f"\n[ERROR] LLM Error: {event.get('text', 'Unknown')}")
                    await self.emit_task_update(finalize_tasks=True)
                    await self.emit_status("Error", done=True)
                    return self._format_output()
                elif etype == "content":
                    content_chunks.append(event.get("text", ""))
                elif etype == "reasoning":
                    reasoning_chunks.append(event.get("text", ""))
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

            if reasoning_chunks:
                reasoning_text = "".join(reasoning_chunks).strip()
                if reasoning_text:
                    await self.emit_output(
                        f'\n\n\u003cdetails type="reasoning"\u003e\u003csummary\u003eThinking\u003c/summary\u003e{reasoning_text}\u003c/details\u003e\n\n'
                    )

            if not tc_dict:
                # Try XML tool call rescue for hallucinated <ToolCall> blocks
                xml_calls = extract_xml_tool_calls(content or "")
                if xml_calls:
                    tc_dict = {tc["index"]: tc for tc in xml_calls}
                else:
                    # No tool calls and no XML rescue — check for unfinished tasks
                    if content and self.task_list and len(self.completed_tasks) < len(self.task_list):
                        # Tasks remain: inject continuation prompt instead of terminating
                        self.history.append({
                            "role": "assistant",
                            "content": content,
                        })
                        self.history.append({
                            "role": "user",
                            "content": "SYSTEM: You produced text but did not call any tools. You have unfinished tasks. Continue working by calling the appropriate tool. Do NOT just describe what to do — call a tool.",
                        })
                        await self.emit_output(f"\n[WARN] No tool call produced. Re-prompting to continue.\n")
                        continue
                    if content:
                        if not self.is_silent:
                            await self.emit_output("\n\n---\n\n**Ergebnis**\n\n")
                        await self.emit_output(content)
                        if not self.is_silent:
                            await self.emit_output("\n\n---\n")
                    await self.emit_task_update(finalize_tasks=True)
                    await self.emit_status("Done", done=True)
                    return self._format_output()

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
                        await self.emit_output("\n[ERROR] JSON parse failed 3 times. Stopping.")
                        await self.emit_task_update(finalize_tasks=True)
                        await self.emit_status("JSON error", done=True)
                        return self._format_output()
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
                    self._save_state_to_history()
                    await self.emit_task_update(finalize_tasks=True)
                    if content:
                        await self.emit_output(content + "\n\n")
                    await self.emit_output(f"{icon} **Finished:** {result}")
                    await self.emit_status("Finished", done=True)
                    return self._format_output()

                # ── Handle replan ──
                if tool_name == "replan":
                    reason = args.get("reason", "Plan needs adjustment")
                    updated = args.get("updated_tasks", "")
                    mode = args.get("mode", "soft")

                    if not updated or not self._extract_task_list(updated):
                        # No new tasks provided -- check if any remaining tasks exist
                        failed_task_names = {f["task"] for f in self.failed_tasks}
                        remaining = [
                            t for t in self.task_list
                            if t not in self.completed_tasks and t not in failed_task_names
                        ]
                        if not remaining:
                            await self.emit_output(f"\n[WARN] Replan requested but no remaining tasks. Terminating.")
                            await self.emit_task_update(finalize_tasks=True)
                            await self.emit_status("No tasks remaining", done=True)
                            return self._format_output()

                    recent_calls = []
                    self._consecutive_tool_misses.clear()
                    result_json = await self._tool_replan(reason=reason, updated_tasks=updated, mode=mode)
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    mode_label = "soft" if mode == "soft" else "hard"
                    await self.emit_output(f"\n[RPLN] **Re-planning ({mode_label}):** {reason}\n")
                    await self.emit_status(f"[RPLN] Re-planning ({mode_label}): {reason}")
                    continue

                # ── Handle complete_task ──
                if tool_name == "complete_task":
                    result_json = await self._tool_complete_task(**args)
                    result_data = json.loads(result_json)
                    status_icon = "[OK]" if result_data.get("completed") else "[WARN]"
                    await self.emit_output(f"\n{status_icon} Task {args.get('index', '?')} marked complete.\n")
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
                    await self.emit_output(f"\n{status_icon} Task {args.get('index', '?')} marked failed: {args.get('reason', '')}\n")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    # Auto-transition to REVIEW if all tasks are done or failed
                    if self.failed_tasks and len(self.completed_tasks) + len(self.failed_tasks) >= len(self.task_list):
                        self._transition_to(self.PHASE_REVIEW)
                    continue

                # ── Handle fix_plan ──
                if tool_name == "fix_plan":
                    result_json = await self._tool_fix_plan(**args)
                    result_data = json.loads(result_json)
                    if result_data.get("fix_plan"):
                        await self.emit_output(f"\n[FIX] **Plan fixed:** {result_data.get('reason', '')}\n")
                        await self.emit_output(f"[FIX] Inserted tasks: {', '.join(result_data.get('inserted_tasks', []))}\n")
                    else:
                        await self.emit_output(f"\n[FIX] **Fix failed:** {result_data.get('error', 'Unknown error')}\n")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    if self.phase == self.PHASE_REVIEW:
                        self._transition_to(self.PHASE_EXECUTE)
                    continue

                # ── Handle confirm_plan ──
                if tool_name == "confirm_plan":
                    plan_text = args.get("plan", content or "")
                    self.task_list = self._extract_task_list(plan_text or content or "")
                    result_json = await self._tool_confirm_plan(**args)
                    result_data = json.loads(result_json)

                    if result_data.get("action") == "feedback":
                        feedback = result_data.get("value", "")
                        self.history.append({
                            "role": "tool",
                            "content": result_json,
                            "tool_call_id": call_id,
                            "name": tool_name,
                        })
                        self.history.append({
                            "role": "user",
                            "content": f"SYSTEM: User provided feedback on the proposed plan: {feedback}. Please revise the plan and call confirm_plan again with the updated plan.",
                        })
                        await self.emit_output(f"\n[PLAN] Plan rejected — user feedback: {feedback}\n")
                        await self.emit_status("[PLAN] Revising plan based on feedback...")
                        continue
                    elif result_data.get("action") == "cancel":
                        await self.emit_output("\n[PLAN] Plan cancelled by user.\n")
                        await self.emit_task_update(finalize_tasks=True)
                        await self.emit_status("Plan cancelled", done=True)
                        return self._format_output()
                    else:
                        self._transition_to(self.PHASE_EXECUTE)
                        self._save_state_to_history()
                        await self.emit_task_update()
                        self.history.append({
                            "role": "tool",
                            "content": result_json,
                            "tool_call_id": call_id,
                            "name": tool_name,
                        })
                        task_summary = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(self.task_list))
                        await self.emit_output(f"\n[PLAN] Plan approved. Moving to execution.\n\n{task_summary}\n")
                        continue

                # ── Duplicate detection ──
                sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                if recent_calls.count(sig) >= 2:
                    tool_result = f"Error: Identical call to `{tool_name}` repeated. Try a different approach."
                else:
                    recent_calls.append(sig)
                    await self.emit_status(f"Running: {tool_name}...")
                    result_str, result_files = await self._execute_tool(tool_name, args, call_id)
                    # Deduplicate files by id/file_id/url
                    for f in result_files:
                        fid = (f.get("id") or f.get("file_id") or f.get("url") or "") if isinstance(f, dict) else ""
                        if fid and fid not in self._seen_file_ids:
                            self._seen_file_ids.add(fid)
                    truncation_limit = self._get_truncation_limit()
                    tool_result = smart_truncate(result_str, truncation_limit)

                    # ── Consecutive tool-not-found tracking ──
                    if "not found in current phase" in tool_result:
                        self._consecutive_tool_misses[tool_name] = self._consecutive_tool_misses.get(tool_name, 0) + 1
                        if self._consecutive_tool_misses[tool_name] >= 3:
                            await self.emit_output(
                                f"<details><summary>⚠️ Tool Loop Break</summary>"
                                f"Tool `{tool_name}` unavailable after 3 consecutive attempts. Forcing replan."
                                f"</details>"
                            )
                            replan_result = await self._tool_replan(
                                reason=f"Tool '{tool_name}' unavailable after 3 consecutive attempts",
                                updated_tasks="",
                                mode="soft",
                            )
                            self.history.append({
                                "role": "tool",
                                "content": replan_result,
                                "tool_call_id": call_id,
                                "name": "replan",
                            })
                            break
                    else:
                        self._consecutive_tool_misses.clear()

                # Render
                args_preview = smart_truncate(json.dumps(args, ensure_ascii=False), 200)
                result_preview = smart_truncate(tool_result, 600)
                phase_tag = phase_icons.get(self.phase, "[LOOP]")
                # Render OpenWebUI-konformer Tool-Call-Block
                args_json = html.escape(json.dumps(args, ensure_ascii=False))
                result_json = html.escape(json.dumps({"result": tool_result}, ensure_ascii=False))
                detail_block = (
                    f'\n\n\u003cdetails type="tool_calls" done="true" '
                    f'id="{call_id}" name="{tool_name}" '
                    f'arguments="{args_json}"\u003e\n'
                    f'\u003csummary\u003e{tool_name}\u003c/summary\u003e\n'
                    f'{result_json}\n'
                    f'\u003c/details\u003e'
                )
                await self.emit_output(detail_block)

                self.history.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": call_id,
                    "name": tool_name,
                })

            # Auto-transition to REVIEW
            if self.phase == self.PHASE_EXECUTE and self.completed_tasks and len(self.completed_tasks) >= len(self.task_list):
                self._transition_to(self.PHASE_REVIEW)

    async def run(self, user_msg, last_user_msg_raw, model):
        try:
            result = await self._run_impl(user_msg, last_user_msg_raw, model)
            return result
        except GeneratorExit:
            # NOTE: GeneratorExit can only be raised if pipe() becomes an async generator.
            # Currently pipe() returns a string, so this catch will not be triggered by
            # OpenWebUI's normal cancellation mechanism. CancelledError handles asyncio cancellation.
            logger.info("Agent loop cancelled by user (GeneratorExit).")
            self._save_state_to_history()
            await self.emit_task_update(finalize_tasks=True)
            await self.emit_status("Cancelled", done=True)
            raise
        except asyncio.CancelledError:
            logger.info("Agent loop cancelled (CancelledError).")
            self._save_state_to_history()
            await self.emit_task_update(finalize_tasks=True)
            await self.emit_status("Cancelled", done=True)
            raise
        except Exception as e:
            logger.error(f"Agent loop error: {e}", exc_info=True)
            await self.emit_task_update(finalize_tasks=True)
            return f"\n[ERROR] Agent loop failed: {e}"
        finally:
            self._seen_file_ids.clear()

    async def run_stream(self, user_msg, last_user_msg_raw, model):
        """Run the Helix Agent loop and yield live text chunks to the caller.

        In non-silent mode every emit_output() call is pushed into the
        internal queue and yielded immediately so the chat sees the agent
        working live.  The final return value of _run_impl is yielded only
        in silent mode; otherwise it is already part of the stream.
        """
        self._stream_queue = asyncio.Queue()
        loop_task = asyncio.create_task(
            self._run_impl(user_msg, last_user_msg_raw, model)
        )
        try:
            while not loop_task.done() or not self._stream_queue.empty():
                try:
                    chunk = await asyncio.wait_for(self._stream_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if chunk:
                    yield chunk
            result = await loop_task
            if self.is_silent and result and result.strip():
                yield result
        except asyncio.CancelledError:
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass
            raise

    def _extract_task_list(self, text):
        if not text:
            return ["Complete the user's request"]

        # Try JSON parsing first: {"tasks": ["task1", "task2"]} or ["task1", "task2"]
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tasks" in data:
                tasks = [t.get("description", t) if isinstance(t, dict) else str(t) for t in data["tasks"]]
                return tasks if tasks else ["Complete the user's request"]
            if isinstance(data, list):
                tasks = [t.get("description", t) if isinstance(t, dict) else str(t) for t in data]
                return tasks if tasks else ["Complete the user's request"]
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        # Try to extract a JSON block from within the text
        json_match = re.search(r'\{[^{}]*"tasks"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict) and "tasks" in data:
                    tasks = [t.get("description", t) if isinstance(t, dict) else str(t) for t in data["tasks"]]
                    return tasks if tasks else ["Complete the user's request"]
                if isinstance(data, list):
                    tasks = [t.get("description", t) if isinstance(t, dict) else str(t) for t in data]
                    return tasks if tasks else ["Complete the user's request"]
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        # Fallback: regex extraction of numbered/bulleted items
        tasks = []
        for line in text.split("\n"):
            line = line.strip()
            m = re.match(r"^\d+[\.\)]\s+(.+)", line)
            if m:
                tasks.append(m.group(1).strip())
            elif re.match(r"^[-*]\s+", line):
                tasks.append(re.sub(r"^[-*]\s+", "", line).strip())
        if not tasks:
            summary = text.strip().split("\n")[0].strip()[:120]
            if summary:
                tasks = [summary]
            else:
                tasks = ["Complete the user's request"]
        return tasks


class Pipe:
    class Valves(BaseModel):
        AGENT_MODEL: str = Field(
            default="",
            description="Model ID for Helix Agent. The model MUST support function calling (tool use). Examples: gpt-4o, claude-3.5-sonnet, gemini-2.0-flash."
        )
        MAX_ITERATIONS: int = Field(
            default=100,
            description="Maximum Helix Agent iterations before stopping."
        )
        MAX_TOOL_RESULT_CHARS: int = Field(
            default=4200,
            description="Max characters for tool results before truncation."
        )
        TOOL_TIMEOUT: int = Field(
            default=90,
            description="Timeout in seconds for individual tool execution. Set to 0 to disable."
        )

        PLAN_TOOLS: str = Field(
            default="",
            description=(
                "Comma-separated tool names allowed in PLAN phase. "
                "Leave EMPTY to allow ALL tools. "
                "Recommended: read-only tools like read_file, search_web, get_github_file_contents."
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
                "Recommended: read-only tools like read_file, search_web, get_github_file_contents."
            )
        )

        PLAN_PROMPT: str = Field(
            default=DEFAULT_PLAN_PROMPT,
            description="System prompt for PLAN phase. Available placeholders: {tool_names}."
        )
        EXECUTE_PROMPT: str = Field(
            default=DEFAULT_EXECUTE_PROMPT,
            description="System prompt for EXECUTE phase. Available placeholders: {tool_names}, {task_state}."
        )
        REVIEW_PROMPT: str = Field(
            default=DEFAULT_REVIEW_PROMPT,
            description="System prompt for REVIEW phase. Available placeholders: {goal}, {task_state}, {tool_names}."
        )

    class UserValves(BaseModel):
        ENABLE_PLAN_APPROVAL: bool = Field(
            default=False,
            description="Enable plan confirmation UI. When off, plans are auto-approved without asking the user.",
        )
        YOLO_MODE: bool = Field(
            default=False,
            description="Skip all user confirmations. Auto-approve plans and ignore iteration limits.",
        )
        SILENT_MODE: bool = Field(
            default=True,
            description="If True, show only plan approvals, final results, and errors. Hide tool call details, reasoning blocks, and intermediate status messages.",
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def pipes(self):
        model_suffix = f" ({self.valves.AGENT_MODEL})" if self.valves.AGENT_MODEL else ""
        return [{"id": "helix-agent", "name": f"Helix Agent{model_suffix}"}]

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
            raise TypeError("Helix Agent requires __request__.")

        __metadata__ = __metadata__ or body.get("metadata", {})

        from open_webui.models.users import Users
        user_id = __user__.get("id") if isinstance(__user__, dict) else ""
        user_obj = await Users.get_user_by_id(user_id) if user_id else None

        user_valves_raw = (
            __user__.get("valves", None) if isinstance(__user__, dict)
            else getattr(__user__, "valves", None)
        )
        if user_valves_raw and isinstance(user_valves_raw, dict):
            user_valves = self.UserValves(**user_valves_raw)
        elif user_valves_raw and isinstance(user_valves_raw, self.UserValves):
            user_valves = user_valves_raw
        elif hasattr(self, "user_valves") and self.user_valves:
            user_valves = self.user_valves
        else:
            user_valves = self.UserValves()

        model = self.valves.AGENT_MODEL or body.get("model", "")

        messages = body.get("messages", [])
        last_user_msg_raw = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            None
        )
        user_msg = last_user_msg_raw.get("content", "Unknown goal") if last_user_msg_raw else "Unknown goal"

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

        engine = HelixAgentEngine(
            request=__request__,
            user=user_obj or __user__,
            body=body,
            event_emitter=__event_emitter__,
            event_call=__event_call__,
            metadata=metadata,
            valves=self.valves,
            user_valves=user_valves,
        )

        # Restore state from previous [AGENT_STATE] messages in chat history
        engine._restore_state_from_messages(messages)

        if engine.is_silent:
            return await engine.run(user_msg, last_user_msg_raw, model)

        async def _generator():
            async for chunk in engine.run_stream(
                user_msg, last_user_msg_raw, model
            ):
                yield chunk

        return _generator()
