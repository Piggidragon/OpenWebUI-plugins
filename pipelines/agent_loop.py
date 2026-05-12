"""
title: Agent Loop
author: Piggidragon
version: 2.0.0
description: >
  Minimal OpenWebUI-native Agent Loop pipe.

  What it does:
  - Iterative agent loop with system prompt + tool calling
  - Uses OWUI's generate_chat_completion for LLM calls (streaming SSE)
  - Uses OWUI's tool resolution (get_tools, get_builtin_tools, get_terminal_tools)
  - Executes tools via OWUI-resolved callables
  - Renders tool executions as <details type="tool_calls"> to avoid OWUI's retry loop
  - Adds one custom tool: terminate (to signal loop completion)

  What it does NOT do (because OWUI handles it):
  - No custom OllamaClient — OWUI does LLM calls
  - No custom ToolExecutor — OWUI resolves tools
  - No context compression — your filter handles that
  - No ask_user — you have it as a tool already
  - No MCP resolution — OWUI handles MCP tools natively now
  - No file handling — OWUI handles file context

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
#  SSE STREAM PARSER
# ──────────────────────────────────────────────────────────────────

async def stream_completion(request, body, user):
    """Stream OWUI completion, yielding structured events.
    
    Yields dicts with keys:
      - type: "content" | "tool_calls" | "reasoning" | "error"
      - text/data: the payload
    """
    body["stream"] = True
    try:
        response = await generate_chat_completion(request, body, user=user)
    except Exception as e:
        logger.error(f"generate_chat_completion failed: {e}")
        yield {"type": "error", "text": str(e)}
        return

    # ── Streaming SSE response ──
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
                            # Reasoning/thinking
                            for rk in ("reasoning", "reasoning_content", "thinking"):
                                rv = delta.get(rk)
                                if rv:
                                    yield {"type": "reasoning", "text": rv}
                            # Content
                            cv = delta.get("content")
                            if cv:
                                yield {"type": "content", "text": cv}
                            # Tool calls
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

    # ── Non-streaming dict response ──
    elif isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            if msg.get("tool_calls"):
                yield {"type": "tool_calls", "data": msg["tool_calls"]}
            if msg.get("content"):
                yield {"type": "content", "text": msg["content"]}


async def non_stream_completion(request, body, user):
    """Non-streaming OWUI completion for one-off calls."""
    body["stream"] = False
    try:
        response = await generate_chat_completion(request, body, user=user)
    except Exception as e:
        return {"choices": [{"message": {"content": f"Error: {e}"}}]}

    if isinstance(response, dict):
        return response
    if hasattr(response, "body_iterator"):
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
        full = "".join(chunks)
        try:
            return json.loads(full)
        except json.JSONDecodeError:
            return {"choices": [{"message": {"content": full}}]}
    return response


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
        self.tools_dict = {}
        self.tools_specs = []
        self.consecutive_json_errors = 0
        self.loop_count = 0

    # ── Events ──

    async def emit_status(self, msg, done=False):
        if self.event_emitter:
            try:
                await self.event_emitter({"type": "status", "data": {"description": msg, "done": done}})
            except Exception:
                pass

    # ── Tool Resolution ──

    async def resolve_tools(self):
        """Resolve tools from OWUI's tool infrastructure + our internal terminate tool."""
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

        # 2. Built-in tools (web search, image gen, code interpreter, etc.)
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

        # 4. Our internal terminate tool
        self.tools_dict["terminate"] = {
            "spec": {
                "name": "terminate",
                "description": "Signal that the task is complete. Provide the final result and whether it was successful.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string", "description": "The final answer or result"},
                        "success": {"type": "boolean", "default": True, "description": "Whether the task was completed successfully"},
                    },
                    "required": ["result"],
                },
            },
            "callable": self._tool_terminate,
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

            # Process via OWUI middleware (handles citations, images, etc.)
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

    # ── System Prompt ──

    def _build_system_prompt(self, is_first, goal, tool_names):
        tool_list = ", ".join(sorted(tool_names))

        base = f"""You are a precise AI agent that solves tasks step-by-step using tools.
CRITICAL: You MUST reason and output all tool calls in ENGLISH ONLY.

Available tools: {tool_list}

RULES:
- Call exactly ONE tool per iteration (except terminate which ends the loop).
- After each tool result, assess progress and decide the next step.
- If goal complete → call terminate with the result.
- NEVER repeat identical failed tool calls with the same arguments.
- Stay focused on the goal.
- If a tool fails, try a different approach instead of retrying the same call."""

        if is_first:
            base += f"\n\nOn the FIRST step, break down the goal into a plan, then call your first tool.\n\nGOAL: {goal}"
        else:
            base += f"\n\nContinue from where you left off.\n\nGOAL: {goal}"

        return base

    # ── Main Loop ──

    async def run(self, user_msg, model):
        """Main agent loop. Yields the final response text."""

        await self.emit_status("Agent starting…")
        await self.resolve_tools()

        tool_names = list(self.tools_dict.keys())
        system_prompt = self._build_system_prompt(True, user_msg, tool_names)

        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        recent_calls = []
        output_parts = []

        while self.loop_count < self.valves.MAX_ITERATIONS:
            self.loop_count += 1
            recent_calls = recent_calls[-30:]

            await self.emit_status(f"Thinking… (step {self.loop_count}/{self.valves.MAX_ITERATIONS})")

            # Prepare messages — keep only one system message at index 0
            call_messages = [self.history[0]] + [m for m in self.history[1:] if m.get("role") != "system"]

            # Build completion body
            completion_body = {
                **self.body,
                "model": model,
                "messages": call_messages,
                "tools": self.tools_specs if self.tools_specs else None,
                "metadata": self.pipe_metadata,
            }

            # Apply file context if built-in tools are present (OWUI handles file injection)
            has_builtin = any(t.get("type") == "builtin" for t in self.tools_dict.values())
            if has_builtin and self.tools_specs:
                try:
                    completion_body["messages"] = await add_file_context(
                        copy.deepcopy(call_messages), self.chat_id, self.user
                    )
                except Exception:
                    pass

            # ── Stream LLM response ──
            tc_dict = {}       # index → accumulated tool call
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

                # Handle terminate
                if tool_name == "terminate":
                    result = args.get("result", "Task complete.")
                    success = args.get("success", True)
                    if content:
                        output_parts.append(content + "\n\n")
                    output_parts.append(f"{'✅' if success else '❌'} {result}")
                    await self.emit_status("Finished", done=True)
                    return "".join(output_parts)

                # Duplicate detection
                sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                if recent_calls.count(sig) >= 2:
                    tool_result = f"Error: Identical call to `{tool_name}` repeated. Try a different approach."
                else:
                    recent_calls.append(sig)
                    await self.emit_status(f"Running: {tool_name}…")
                    result_str, result_files = await self._execute_tool(tool_name, args, call_id)
                    tool_result = smart_truncate(result_str, self.valves.MAX_TOOL_RESULT_CHARS)

                # Render tool execution as <details type="tool_calls"> to avoid OWUI retry loop
                args_preview = smart_truncate(json.dumps(args, ensure_ascii=False), 200)
                result_preview = smart_truncate(tool_result, 600)
                detail_block = (
                    f'<details type="tool_calls">\n'
                    f'<summary>{tool_name}</summary>\n'
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

            # Update system prompt for next iteration
            self.history[0]["content"] = self._build_system_prompt(False, user_msg, tool_names)

        # Max iterations
        output_parts.append(f"\n⚠️ Max iterations ({self.valves.MAX_ITERATIONS}) reached.")
        await self.emit_status("Max iterations", done=True)
        return "".join(output_parts)


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
            default=16,
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