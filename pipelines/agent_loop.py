"""
title: Agent Loop
author: Piggidragon
version: 1.0.0
description: >
  OpenWebUI-native Agent Loop pipe.
  Uses OpenWebUI's internal APIs for LLM calls, tool resolution, and streaming
  — no external Ollama client, no custom tool executor, no standalone services.

  Architecture:
  - Single-agent iterative loop (not sub-agent delegation)
  - LLM calls via OWUI's `generate_chat_completion` (streaming SSE)
  - Tool resolution via OWUI's `get_tools`, `get_builtin_tools`, MCP
  - Tool execution via tool callables resolved by OWUI
  - Context compression via auxiliary model when threshold exceeded
  - Per-user workspace isolation
  - OpenWebUI API integration for files, notes, tasks, etc.

  Compared to the deprecated pipeline:
  - ✅ Much simpler (no OllamaClient, ToolExecutor, WebRAG, JupyterExecutor)
  - ✅ Uses OWUI's native tool/tool-call infrastructure
  - ✅ Streaming via SSE events (not batch responses)
  - ✅ MCP support via OWUI's MCP client
  - ✅ Built-in tool parity (web search, image gen, code interpreter, etc.)
  - ✅ Native file upload/download via OWUI's file handlers
  - ✅ Skill support via OWUI's skill resolution

requirements: open-webui>=0.9.1
"""

import asyncio
import json
import logging
import os
import re
import copy
import time
import uuid
from typing import AsyncGenerator, Callable, Awaitable, Any, Optional
from pydantic import BaseModel, Field

from fastapi import Request

from open_webui.utils.chat import generate_chat_completion as generate_raw_chat_completion
from open_webui.utils.tools import get_tools, get_builtin_tools, get_terminal_tools
from open_webui.utils.middleware import process_tool_result, add_file_context, get_system_oauth_token
from open_webui.utils.mcp.client import MCPClient
from open_webui.utils.access_control import has_connection_access
from open_webui.utils.misc import is_string_allowed
from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS
from open_webui.models.chats import Chats
from open_webui.models.users import Users
from open_webui.models.skills import Skills

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def smart_truncate(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text
    for sep in (". ", ".\n", "\n\n", "\n"):
        idx = text[:max_chars].rfind(sep)
        if idx > max_chars // 2:
            return text[:idx + len(sep)].rstrip() + "\n[truncated]"
    return text[:max_chars].rstrip() + "\n[truncated]"


def merge_workspace_model_dict(app_models: dict, model_id: str) -> dict:
    m = copy.deepcopy(app_models.get(model_id) or {"id": model_id, "info": {}})
    if "info" not in m:
        m["info"] = {}
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  SSE STREAM PARSER & COMPLETION
# ─────────────────────────────────────────────────────────────────────────────

async def stream_completion(request: Request, body: dict, user: Any) -> AsyncGenerator:
    """Stream OWUI completion, yielding structured events."""
    body["stream"] = True
    try:
        response = await generate_raw_chat_completion(request, body, user=user)

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
                    if not payload:
                        continue
                    if payload == "[DONE]":
                        return

                    try:
                        parsed = json.loads(payload)
                        if isinstance(parsed, dict):
                            choices = parsed.get("choices", [])
                            if choices:
                                choice = choices[0] if isinstance(choices[0], dict) else {}
                                delta = choice.get("delta", {}) or {}
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
            content = ""
            choices = response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "") or ""
                tc = msg.get("tool_calls")
                if tc:
                    yield {"type": "tool_calls", "data": tc}
            if content:
                yield {"type": "content", "text": content}
            return
    except Exception as e:
        logger.error(f"Stream completion error: {e}")
        yield {"type": "error", "text": str(e)}


async def non_stream_completion(request: Request, body: dict, user: Any) -> dict:
    """Non-streaming OWUI completion for auxiliary calls (summary, compression)."""
    body["stream"] = False
    try:
        response = await generate_raw_chat_completion(request, body, user=user)
        if isinstance(response, dict):
            return response
        if hasattr(response, "body_iterator"):
            chunks = []
            async for chunk in response.body_iterator:
                decoded = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                chunks.append(decoded)
            full = "".join(chunks)
            try:
                return json.loads(full)
            except json.JSONDecodeError:
                return {"choices": [{"message": {"content": full}}]}
        return response
    except Exception as e:
        logger.error(f"Non-stream completion error: {e}")
        return {"choices": [{"message": {"content": f"Error: {e}"}}]}


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT LOOP ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AgentLoopEngine:
    def __init__(
        self,
        request: Request,
        user: Any,
        body: dict,
        event_emitter: Callable,
        event_call: Optional[Callable],
        metadata: dict,
        valves: Any,
    ):
        self.request = request
        self.user = user
        self.body = body
        self.event_emitter = event_emitter
        self.event_call = event_call
        self.metadata = metadata
        self.valves = valves

        self.pipe_metadata = metadata.get("__metadata__", {})
        self.chat_id = metadata.get("__chat_id__", metadata.get("chat_id", ""))
        self.message_id = metadata.get("__message_id__", metadata.get("message_id", ""))

        self.app_models = getattr(request.app.state, "MODELS", {}) if request else {}

        # Agent loop state
        self.history: list[dict] = []
        self.tools_dict: dict[str, Any] = {}
        self.tools_specs: list[dict] = []
        self.consecutive_json_errors = 0
        self.max_iterations = valves.MAX_ITERATIONS
        self.loop_count = 0

    async def emit_status(self, msg: str, done: bool = False):
        if self.event_emitter:
            try:
                await self.event_emitter({
                    "type": "status",
                    "data": {"description": msg, "done": done}
                })
            except Exception:
                pass

    async def emit_message(self, delta: str):
        if self.event_emitter:
            try:
                await self.event_emitter({"type": "message", "data": {"content": delta}})
            except Exception:
                pass

    async def emit_replace(self, content: str):
        if self.event_emitter:
            try:
                await self.event_emitter({"type": "replace", "data": {"content": content}})
            except Exception:
                pass

    # ── Tool Resolution ──

    async def resolve_tools(self) -> dict[str, Any]:
        """Resolve all tools: external (DB + OpenAPI), MCP, built-in, terminal, + internal."""
        tool_ids = (
            self.pipe_metadata.get("toolIds")
            or self.pipe_metadata.get("tool_ids")
            or []
        )
        user_dict = (
            self.user.model_dump()
            if hasattr(self.user, "model_dump")
            else (self.user.__dict__ if isinstance(self.user, dict) else self.user or {})
        )

        extra_params = {
            "chat_id": self.chat_id,
            "tool_ids": tool_ids,
            "__user__": user_dict,
            "__metadata__": self.metadata,
            "__event_emitter__": self.event_emitter,
            "__event_call__": self.event_call,
        }

        tools_dict: dict[str, Any] = {}

        # 1. External tools (DB + OpenAPI)
        seen_ids = set()
        ordered_ids = []
        for tid in tool_ids:
            if tid and tid not in seen_ids:
                seen_ids.add(tid)
                ordered_ids.append(tid)

        if ordered_ids:
            try:
                resolved = await get_tools(self.request, ordered_ids, self.user, extra_params)
                if resolved:
                    tools_dict.update(resolved)
            except Exception as e:
                logger.error(f"Failed to resolve external tools: {e}")

        # 2. MCP tools
        mcp_tools = await self._resolve_mcp_tools(ordered_ids, extra_params)
        if mcp_tools:
            tools_dict.update(mcp_tools)

        # 3. Built-in tools
        model_info = self.app_models.get(self.body.get("model", ""), {})
        features = self._get_model_features(model_info)
        if features:
            try:
                builtin = await get_builtin_tools(self.request, extra_params, features=features, model=model_info)
                if builtin:
                    tools_dict.update(builtin)
            except Exception as e:
                logger.error(f"Failed to resolve builtin tools: {e}")

        # 4. Terminal tools
        terminal_id = self.pipe_metadata.get("terminal_id")
        if terminal_id:
            try:
                raw_term = await get_terminal_tools(self.request, terminal_id, self.user, extra_params)
                if isinstance(raw_term, tuple) and len(raw_term) == 2:
                    t_tools, _ = raw_term
                else:
                    t_tools = raw_term if isinstance(raw_term, dict) else {}
                if t_tools:
                    tools_dict.update(t_tools)
            except Exception as e:
                logger.error(f"Failed to resolve terminal tools: {e}")

        # 5. Agent Loop internal tools
        tools_dict["terminate"] = {
            "spec": {
                "name": "terminate",
                "description": "Finish with a final answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string", "description": "Final answer"},
                        "success": {"type": "boolean", "default": True},
                    },
                    "required": ["result", "success"],
                },
            },
            "callable": self._tool_terminate,
            "type": "function",
        }
        tools_dict["ask_user"] = {
            "spec": {
                "name": "ask_user",
                "description": "Ask user a question or request confirmation. Use BEFORE destructive actions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question to ask"},
                    },
                    "required": ["question"],
                },
            },
            "callable": self._tool_ask_user,
            "type": "function",
        }

        self.tools_dict = tools_dict
        self.tools_specs = [
            {"type": "function", "function": t["spec"]}
            for t in tools_dict.values()
            if isinstance(t, dict) and "spec" in t
        ]
        return tools_dict

    async def _resolve_mcp_tools(self, tool_ids: list, extra_params: dict) -> dict:
        """Resolve MCP tools from tool IDs."""
        result = {}
        if not tool_ids or not self.request:
            return result

        mcp_connections = getattr(
            self.request.app.state.config, "TOOL_SERVER_CONNECTIONS", []
        )

        try:
            oauth_token = await get_system_oauth_token(self.request, self.user)
        except Exception:
            oauth_token = None

        for tid in tool_ids:
            server_id = None
            if tid.startswith("server:mcp:"):
                server_id = tid[len("server:mcp:"):]
            elif tid.startswith("mcp:"):
                server_id = tid[len("mcp:"):]

            if not server_id:
                continue

            for conn in mcp_connections:
                if conn.get("type") == "mcp" and conn.get("info", {}).get("id") == server_id:
                    try:
                        if not await has_connection_access(self.user, conn):
                            continue

                        headers = {}
                        auth_type = conn.get("auth_type", "")
                        if auth_type == "bearer":
                            headers["Authorization"] = f"Bearer {conn.get('key', '')}"
                        elif auth_type == "system_oauth" and oauth_token:
                            headers["Authorization"] = f"Bearer {oauth_token.get('access_token', '')}"

                        conn_headers = conn.get("headers", {})
                        if isinstance(conn_headers, dict):
                            headers.update(conn_headers)

                        client = MCPClient()
                        await client.connect(url=conn.get("url", ""), headers=headers if headers else None)

                        try:
                            specs = await client.list_tool_specs()
                            for spec in specs:
                                fn = spec.get("name", "")
                                full_name = f"{server_id}_{fn}"

                                def make_callable(mcp_client, fn_name, sid):
                                    async def call(**kwargs):
                                        try:
                                            res = await mcp_client.call_tool(fn_name, function_args=kwargs)
                                            if hasattr(res, "content") and res.content:
                                                return "\n".join(
                                                    c.text if hasattr(c, "text") else str(c)
                                                    for c in res.content
                                                )
                                            return str(res)
                                        except Exception as e:
                                            return f"Error calling MCP tool: {e}"
                                    return call

                                result[full_name] = {
                                    "spec": {**spec, "name": full_name},
                                    "callable": make_callable(client, fn, server_id),
                                    "type": "mcp",
                                    "client": client,
                                    "direct": False,
                                }
                        except Exception as e:
                            logger.error(f"MCP list_tool_specs failed for {server_id}: {e}")
                            try:
                                await client.disconnect()
                            except Exception:
                                pass
                    except Exception as e:
                        logger.error(f"MCP resolution failed for {server_id}: {e}")
                    break
        return result

    def _get_model_features(self, model_info: dict) -> dict:
        info = model_info.get("info", {}) or {}
        meta = info.get("meta", {}) or {}
        params = info.get("params", {}) or {}
        features = {}
        for block in (meta, params):
            if isinstance(block.get("features"), dict):
                for fk, fv in block["features"].items():
                    features[fk] = bool(fv)
        return features

    # ── Internal Tool Implementations ──

    async def _tool_terminate(self, **kwargs):
        return json.dumps({"terminated": True, "result": kwargs.get("result", ""), "success": kwargs.get("success", True)})

    async def _tool_ask_user(self, **kwargs):
        question = kwargs.get("question", "I need more information to continue.")
        if self.event_call:
            try:
                js = f'return (function() {{ return new Promise((resolve) => {{ const result = prompt({json.dumps(question)}); resolve(JSON.stringify({{action: result ? "accept" : "skip", value: result || ""}})); }}); }})()'
                raw = await self.event_call({"type": "execute", "data": {"code": js}})
                raw_str = raw if isinstance(raw, str) else ((raw.get("result") or raw.get("value") or "{}") if raw else "{}")
                try:
                    res = json.loads(raw_str) if isinstance(raw_str, str) and raw_str.startswith("{") else {"action": "accept", "value": str(raw_str)}
                    return res.get("value", res.get("action", "skipped"))
                except json.JSONDecodeError:
                    return str(raw_str)
            except Exception as e:
                logger.error(f"ask_user event_call failed: {e}")
        return question

    # ── Context Management ──

    def estimate_tokens(self, messages: list) -> int:
        return sum(len(str(m.get("content", ""))) for m in messages) // 3

    def needs_compression(self, messages: list) -> bool:
        return self.estimate_tokens(messages) > self.valves.CONTEXT_TOKEN_THRESHOLD

    async def compress_context(self, messages: list, model: str) -> list:
        """Compress conversation context using auxiliary model."""
        text = "\n\n".join(
            f"[{m['role'].upper()}]: {m.get('content', '')}"
            for m in messages if m["role"] != "system"
        )
        summary_prompt = (
            "Summarise the following agent conversation concisely. "
            "Keep all facts, results, file paths, error messages, URLs, code snippets, and numeric values. "
            "Omit dead-end intermediate noise. Max 900 words. Reply with summary only."
        )
        summary_body = {
            "model": self.valves.SUMMARY_MODEL or model,
            "messages": [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": text},
            ],
            "stream": False,
        }
        try:
            result = await non_stream_completion(self.request, summary_body, self.user)
            summary = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if summary:
                new_history = [{"role": "user", "content": f"[COMPRESSED CONTEXT]\n{summary}"}]
                last_tool = next((m for m in reversed(messages) if m.get("role") == "tool"), None)
                last_assist = next((m for m in reversed(messages) if m.get("role") == "assistant" and m.get("tool_calls")), None)
                if last_assist and last_tool:
                    new_history += [last_assist, last_tool]
                elif last_tool:
                    new_history.append(last_tool)
                return new_history
        except Exception as e:
            logger.error(f"Context compression failed: {e}")
        return messages

    # ── Build System Prompt ──

    def _build_system_prompt(self, is_first: bool, goal: str, tool_names: list[str]) -> str:
        tool_list = ", ".join(sorted(tool_names))

        base = f"""You are a precise AI agent that solves tasks step-by-step using tools.
CRITICAL: You MUST reason and output all tool calls in ENGLISH ONLY.

Available tools: {tool_list}

RULES:
- Call exactly ONE tool per step (except terminate/ask_user which end the loop).
- After each tool result, assess progress and decide the next step.
- If goal complete → call terminate with the result.
- If you need user confirmation → call ask_user.
- NEVER repeat identical failed tool calls.
- Stay focused on the goal."""

        if is_first:
            base += f"\n\nOn the FIRST step, break down the goal into a plan, then call your first tool.\n\nGOAL: {goal}"
        else:
            base += f"\n\nContinue from where you left off.\n\nGOAL: {goal}"

        return base

    # ── Execute a Tool Call ──

    async def _execute_tool(self, tool_name: str, args: dict, call_id: str) -> tuple[str, list]:
        """Execute a resolved tool and return (result_str, files)."""
        target = self.tools_dict.get(tool_name)
        if not target:
            available = list(self.tools_dict.keys())
            return f"Tool '{tool_name}' not found. Available: {', '.join(available[:20])}", []

        # Filter args to allowed parameters
        spec_params = target.get("spec", {}).get("parameters", {}).get("properties", {})
        allowed_keys = set(spec_params.keys())
        filtered_args = {k: v for k, v in args.items() if k in allowed_keys}

        # Inject special context variables if in signature
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

            # Process result via OWUI's middleware
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

                # Persist files to chat if any
                if files and self.chat_id and self.message_id:
                    try:
                        await Chats.add_message_files_by_id_and_message_id(
                            self.chat_id, self.message_id, files
                        )
                    except Exception:
                        pass
            except Exception as e:
                result_str = str(result) if result is not None else ""
                logger.warning(f"process_tool_result failed for {tool_name}: {e}")

            return result_str, files

        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return f"Error executing {tool_name}: {e}", []

    # ── Main Agent Loop ──

    async def run(self, user_msg: str, model: str) -> AsyncGenerator[str, None]:
        """Main agent loop — iteratively calls the LLM and executes tool calls."""

        await self.emit_status("Agent starting…")
        await self.resolve_tools()

        tool_names = list(self.tools_dict.keys())
        system_prompt = self._build_system_prompt(True, user_msg, tool_names)

        # Build initial history
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        recent_calls = []
        total_output = ""
        files_accumulated = []

        try:
            while self.loop_count < self.max_iterations:
                self.loop_count += 1
                recent_calls = recent_calls[-30:]

                await self.emit_status(f"Thinking… (step {self.loop_count}/{self.max_iterations})")

                # Context compression check
                if self.needs_compression(self.history):
                    await self.emit_status("Compressing context…")
                    self.history = await self.compress_context(self.history, model)
                    goal_prompt = self._build_system_prompt(False, user_msg, tool_names)
                    self.history[0]["content"] = goal_prompt

                # Prepare messages for LLM — strip internal system messages beyond index 0
                call_messages = self.history[:]
                if len(call_messages) > 1:
                    call_messages = [call_messages[0]] + [m for m in call_messages[1:] if m.get("role") != "system"]

                # Build completion body
                completion_body = {
                    **self.body,
                    "model": model,
                    "messages": call_messages,
                    "tools": self.tools_specs if self.tools_specs else None,
                    "metadata": self.pipe_metadata,
                }

                # Apply file context (OWUI native)
                has_builtin = any(t.get("type") == "builtin" for t in self.tools_dict.values())
                if has_builtin and self.tools_specs:
                    try:
                        completion_body["messages"] = await add_file_context(
                            copy.deepcopy(call_messages), self.chat_id, self.user
                        )
                    except Exception:
                        pass

                # Stream the completion
                tc_dict = {}
                content_chunks = []
                error_occurred = False

                async for event in stream_completion(self.request, completion_body, self.user):
                    etype = event.get("type")
                    if etype == "error":
                        error_msg = f"LLM Error: {event.get('text', 'Unknown')}"
                        await self.emit_status(error_msg, done=True)
                        total_output += f"\n\n> ❌ {error_msg}"
                        yield total_output
                        return
                    elif etype == "reasoning":
                        pass  # reasoning handled silently for now
                    elif etype == "content":
                        content_chunks.append(event.get("text", ""))
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

                content = "".join(content_chunks).strip()

                # Clean thinking tags from content
                content = re.sub(r"<(?:think|thinking|reason|reasoning|thought)>.*?</(?:think|thinking|reason|reasoning|thought)>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
                content = re.sub(r"\|begin_of_thought\|.*?\|end_of_thought\|", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

                # No tool calls → final answer
                if not tc_dict:
                    if content:
                        total_output += content
                        await self.emit_replace(total_output)
                    await self.emit_status("Done", done=True)
                    yield total_output
                    return

                # Process tool calls
                tool_calls_list = list(tc_dict.values())

                # Add assistant message with tool calls to history
                self.history.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls_list,
                })

                # Also add content to output if present
                if content:
                    total_output += f"\n**Step {self.loop_count}**\n"
                    await self.emit_replace(total_output)

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
                            total_output += "\n❌ JSON parse failed 3 times. Stopping."
                            await self.emit_replace(total_output)
                            await self.emit_status("JSON error", done=True)
                            yield total_output
                            return
                        args = {}

                    self.consecutive_json_errors = 0

                    # Check for duplicate calls
                    sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                    if recent_calls.count(sig) >= 2:
                        tool_result = f"Error: Identical call to `{tool_name}` repeated."
                    else:
                        recent_calls.append(sig)

                        # Handle terminate
                        if tool_name == "terminate":
                            result = args.get("result", "Task complete.")
                            success = args.get("success", True)
                            icon = "✅" if success else "❌"
                            total_output += f"\n{icon} Task finished."
                            if result:
                                total_output += f"\n\n{result}"
                            await self.emit_replace(total_output)
                            await self.emit_status("Finished", done=True)
                            yield total_output
                            return

                        # Handle ask_user
                        if tool_name == "ask_user":
                            question = args.get("question", "I need more information.")
                            user_response = await self._tool_ask_user(question=question)
                            tool_result = f"User response: {user_response}"
                        else:
                            # Execute tool
                            await self.emit_status(f"Running: {tool_name}…")
                            result_str, result_files = await self._execute_tool(tool_name, args, call_id)
                            tool_result = smart_truncate(result_str, self.valves.MAX_TOOL_RESULT_CHARS)
                            if result_files:
                                files_accumulated.extend(result_files)

                        total_output += f"\n**Step {self.loop_count}** — `{tool_name}`\n```\n{smart_truncate(json.dumps(args, ensure_ascii=False), 200)}\n```\n*Result:* {smart_truncate(tool_result, 300)}\n"
                        await self.emit_replace(total_output)

                    # Add tool result to history
                    self.history.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })

                # Update system prompt for next iteration
                self.history[0]["content"] = self._build_system_prompt(False, user_msg, tool_names)

            # Max iterations reached
            total_output += f"\n⚠ Max iterations ({self.valves.MAX_ITERATIONS}) reached."
            await self.emit_replace(total_output)
            await self.emit_status("Max iterations", done=True)
            yield total_output

        finally:
            # Persist accumulated files
            if files_accumulated and self.chat_id and self.message_id:
                try:
                    await Chats.add_message_files_by_id_and_message_id(
                        self.chat_id, self.message_id, files_accumulated
                    )
                except Exception:
                    pass

            # Emit final files event
            if files_accumulated and self.event_emitter:
                try:
                    await self.event_emitter({"type": "chat:message:files", "data": {"files": files_accumulated}})
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
#  PIPE (Open WebUI Manifold)
# ─────────────────────────────────────────────────────────────────────────────

class Pipe:
    class Valves(BaseModel):
        AGENT_MODEL: str = Field(
            default="",
            description="Model ID for the agent loop (leave blank to use the selected model). Must support native function calling."
        )
        SUMMARY_MODEL: str = Field(
            default="",
            description="Model ID for context compression (leave blank to use AGENT_MODEL)."
        )
        MAX_ITERATIONS: int = Field(
            default=16,
            description="Maximum agent loop iterations before stopping."
        )
        CONTEXT_TOKEN_THRESHOLD: int = Field(
            default=7000,
            description="Token threshold to trigger context compression."
        )
        MAX_TOOL_RESULT_CHARS: int = Field(
            default=4200,
            description="Maximum characters for tool results before truncation."
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": "agent-loop-pipe", "name": "Agent Loop Pipe"}]

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
    ) -> AsyncGenerator[str, None]:
        if __request__ is None:
            raise TypeError("Agent Loop pipe requires __request__ (FastAPI/Starlette Request).")

        __metadata__ = __metadata__ or body.get("metadata", {})
        if __files__ is None:
            __files__ = []

        # Resolve user
        user_id = __user__.get("id") if isinstance(__user__, dict) else ""
        user_obj = await Users.get_user_by_id(user_id) if user_id else None

        # Determine model
        model = self.valves.AGENT_MODEL or body.get("model", "")
        if not model:
            model = body.get("model", "")

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
            "__files__": __files__,
            "__chat_id__": __chat_id__,
            "__message_id__": __message_id__,
        }

        # Pre-fetch skills
        user_skills = {}
        if user_id:
            try:
                skills = await Skills.get_skills_by_user_id(user_id, "read")
                user_skills = {s.id: s for s in skills if s.is_active}
            except Exception:
                pass
        metadata["__user_skills__"] = user_skills

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

        async for chunk in engine.run(user_msg, model):
            yield chunk