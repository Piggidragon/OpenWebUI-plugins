"""
title: Agent Loop
author: Piggidragon
version: 2.0.0
description: >
  Minimal OpenWebUI-native Agent Loop with Plan/Execute/Review phases.

  Architecture:
  - SINGLE model loop (no sub-agents, no parallel LLM calls)
  - Plan phase: analyse context, read files, create task list, ask user for confirmation
  - Execute phase: work through tasks one by one, auto-review each task
  - Review phase: verify all tasks complete, either finish or replan
  - All tools/skills/filters come from OpenWebUI (no custom tool executor)
  - LLM calls via OWUI's generate_chat_completion (streaming SSE)
  - Tool resolution via OWUI (get_tools, get_builtin_tools, get_terminal_tools)
  - Tool execution via OWUI-resolved callables + process_tool_result
  - Renders tool executions as <details type="tool_calls"> to avoid OWUI retry loop
  - terminate and replan as internal tools for loop control
  - ask_user is NOT included (use your own ask_user tool)

  Compared to agent-pipeline-deprecated.py:
  - No OllamaClient, ToolExecutor, WebRAG, JupyterExecutor, OpenWebUIClient
  - No context compression (your filter handles that)
  - No ask_user (use your own tool)
  - No MCP resolution (OWUI handles MCP natively now)
  - Uses OWUI's native tool/skill/filter infrastructure instead of custom implementations
  - 3 phases instead of flat loop
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
5. After creating the plan, call ask_user to present the plan and ask for confirmation.

Rules:
- Be thorough — read files before planning changes.
- Break complex tasks into small, verifiable steps.
- If the request is simple (1-2 tasks), still list them explicitly.
- Call exactly ONE tool per step.
- When done planning, call ask_user with the plan for confirmation.
- NEVER call terminate in PLAN mode — the user must confirm first.
"""

EXECUTE_PROMPT = """\
You are in EXECUTE mode. Work through tasks one at a time.

PHASE: EXECUTE

Available tools: {tool_names}

Current task list:
{task_list}

Completed: {completed_count} | Remaining: {remaining_count}

What to do:
1. Pick the next incomplete task from the list.
2. Execute it using the appropriate tool(s).
3. After executing, briefly assess whether the task succeeded.
4. Move on to the next task.

Rules:
- Call exactly ONE tool per step.
- If a task fails, try an alternative approach before moving on.
- NEVER repeat identical failed tool calls.
- When all tasks are done, call terminate with a summary.
- If you realise the plan was wrong, call replan with what needs to change.
"""

REVIEW_PROMPT = """\
You are in REVIEW mode. Verify that all original requirements are met.

PHASE: REVIEW

Original goal: {goal}
Original plan:
{task_list}

Completed tasks: {completed_list}

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
What went wrong or is missing: {replan_reason}

Previous plan:
{task_list}

Completed tasks: {completed_list}

What to do:
1. Assess what worked and what didn't.
2. Update the task list — add missing tasks, fix broken ones.
3. After creating the updated plan, call ask_user to present the changes for confirmation.

Rules:
- Don't repeat tasks that already succeeded.
- Focus only on what's broken or missing.
- Call ask_user with the updated plan for confirmation.
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
    """Remove <think>...</think> blocks from model output."""
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

        # 4. Internal tools
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
                "description": "Signal that the current plan needs adjustments. Describe what went wrong and what needs to change.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "What went wrong or what is missing"},
                        "updated_tasks": {"type": "string", "description": "Updated task list as numbered steps"},
                    },
                    "required": ["reason"],
                },
            },
            "callable": self._tool_replan,
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

        if self.phase == self.PHASE_PLAN:
            return PLAN_PROMPT.format(tool_names=tool_names)

        elif self.phase == self.PHASE_EXECUTE:
            task_str = "\n".join(
                f"  {'✅' if t in self.completed_tasks else '⬜'} {t}"
                for t in self.task_list
            ) if self.task_list else "No tasks defined yet."
            return EXECUTE_PROMPT.format(
                tool_names=tool_names,
                task_list=task_str,
                completed_count=len(self.completed_tasks),
                remaining_count=len(self.task_list) - len(self.completed_tasks),
            )

        elif self.phase == self.PHASE_REVIEW:
            completed_str = "\n".join(f"  ✅ {t}" for t in self.completed_tasks) if self.completed_tasks else "None"
            return REVIEW_PROMPT.format(
                goal=self.goal,
                task_list="\n".join(f"  {t}" for t in self.task_list),
                completed_list=completed_str,
            )

        elif self.phase == self.PHASE_REPLAN:
            completed_str = "\n".join(f"  ✅ {t}" for t in self.completed_tasks) if self.completed_tasks else "None"
            return REPLAN_PROMPT.format(
                goal=self.goal,
                replan_reason=self._replan_reason or "Previous plan was incomplete",
                task_list="\n".join(f"  {t}" for t in self.task_list),
                completed_list=completed_str,
            )

        return PLAN_PROMPT.format(tool_names=tool_names)

    # ── Phase Transitions ──

    def _transition_to(self, phase, replan_reason=""):
        """Transition to a new phase and update history."""
        self.phase = phase
        self._replan_reason = replan_reason
        if self.history:
            self.history[0]["content"] = self._build_system_prompt()

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
                except json.JSONDecodeError:
                    self.consecutive_json_errors += 1
                    if self.consecutive_json_errors >= 3:
                        output_parts.append("\n❌ JSON parse failed 3 times. Stopping.")
                        await self.emit_status("JSON error", done=True)
                        return "".join(output_parts)
                    args = {}

                self.consecutive_json_errors = 0

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
                    if updated:
                        new_tasks = [t.strip() for t in updated.split("\n") if t.strip()]
                        if new_tasks:
                            self.task_list = self.completed_tasks + new_tasks

                    self._transition_to(self.PHASE_REPLAN, replan_reason=reason)
                    output_parts.append(f"\n🔄 **Re-planning:** {reason}\n")
                    tool_result = json.dumps({"replan": True, "reason": reason})
                    self.history.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    continue

                # ── Phase transition on ask_user ──
                if self.phase == self.PHASE_PLAN and tool_name == "ask_user":
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
                    tool_result = smart_truncate(result_str, self.valves.MAX_TOOL_RESULT_CHARS)

                # Mark tasks as completed in EXECUTE phase
                if self.phase == self.PHASE_EXECUTE and content:
                    self._mark_completed_tasks(content)

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

    def _mark_completed_tasks(self, content):
        """Best-effort: mark tasks as completed based on conversation content."""
        for task in self.task_list:
            if task not in self.completed_tasks:
                task_words = task.lower().split()[:3]
                if any(w in content.lower() for w in task_words):
                    self.completed_tasks.append(task)


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