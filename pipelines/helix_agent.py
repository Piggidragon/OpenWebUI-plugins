"""
title: Helix Agent
author: Piggidragon
    version: 0.27.1
description: >
  Helix Agent - OpenWebUI-native agent loop with modular per-phase tool control.

  Architecture:
  - SINGLE model loop (Plan -> Execute -> Review -> Replan -> Execute...)
  - Per-phase tool filtering via Valves - only relevant tools exposed to the LLM at each phase
  - Internal control tools (terminate, replan, fix_plan, complete_task, fail_task, confirm_plan) always available
  - Uses OpenWebUI native tool infrastructure (__tools__ param)
  - Context window management with adaptive history truncation and tool-call pair integrity
  - LLM-based conversational context compression: goal and history are compressed independently
    using a configurable compression model (falls back to AGENT_MODEL). Compression happens
    blocking within the loop when token-based compression threshold is exceeded, and asynchronously
    for goal after each agent run. History compression produces an assistant summary message
    that preserves the conversational flow.

  State & Persistence:
  - State persistence via JSON file attachments synced to the OpenWebUI chat DB
  - Deep DB history recovery: recovers state from file attachments across the entire parent message chain
  - Loop count is persisted and restored to track iteration lifetime across sessions
  - Exponential backoff file sync for robust DB persistence under heavy load

  Features:
  - Plan confirmation via custom JS UI (UserValves: ENABLE_PLAN_APPROVAL, YOLO_MODE)
  - Native OpenWebUI task progress UI via chat:message:tasks events, finalized on termination
  - System prompt refresh: task mutations (complete, fail, fix_plan) update the LLM's task state context
  - Iteration limit with Continue/Cancel modal; graceful shutdown on CancelledError/GeneratorExit
  - File handling: add_file_context + chat_completion_files_handler for native multimodal/text file injection
  - Knowledge bases: native OpenWebUI vector search via model metadata
  - MCP support: MCP server tools provided via __tools__ parameter
  - Skills support: resolves user skills from model metadata and injects them into the system prompt
   - Context compression: single CONTEXT_LENGTH valve (tokens) drives adaptive history and goal compression via an estimated character-to-token ratio

  Safety:
  - Session timeout: MAX_SESSION_SECONDS hard-caps overall agent runtime (default 20 min).
  - Per-iteration timeout: MAX_ITERATION_SECONDS hard-caps a single loop iteration (default 15 min).
  - Stream stall timeout: STREAM_CHUNK_TIMEOUT_SECONDS aborts if the LLM produces no content for a configured duration (default 60 s).
  - LLM call budget: MAX_LLM_CALLS stops the agent after N generate_chat_completion calls (default 100, continue dialog).
  - All timeouts are zero-able via Valves; when any budget is exhausted the session is terminated cleanly.
requirements: open-webui>=0.9.1
"""

import asyncio
import hashlib
import inspect
import io
import json
import logging
import os
import re
import copy
import uuid
from typing import Callable, Set, List, Dict, Any
from pydantic import BaseModel, Field

from fastapi import Request

from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.middleware import (
    process_tool_result,
    add_file_context,
    chat_completion_files_handler,
)
from open_webui.models.skills import Skills

logger = logging.getLogger(__name__)

# --- OpenWebUI optional middleware imports (best-effort, tolerant to API changes) ---
try:
    from open_webui.utils.middleware import (
        get_citation_source_from_tool_result,
        terminal_event_handler,
        apply_source_context_to_messages,
    )
    HAS_MIDDLEWARE_CITATIONS = True
except Exception:
    HAS_MIDDLEWARE_CITATIONS = False

try:
    from open_webui.utils.middleware import process_pipeline_inlet_filter
    HAS_INLET_FILTER = True
except Exception:
    HAS_INLET_FILTER = False

try:
    from open_webui.routers.memories import query_memory, QueryMemoryForm
    HAS_MEMORY = True
except Exception:
    HAS_MEMORY = False

try:
    from open_webui.models.chats import Chats
    from open_webui.routers.files import upload_file_handler, Files
    from starlette.datastructures import UploadFile, Headers

    HAS_DB_PERSISTENCE = True
except Exception:
    HAS_DB_PERSISTENCE = False

DEFAULT_PLAN_PROMPT = """\
You are in PLAN mode. Create a concise, actionable task plan.

{loop_info}

Core rules:
1. Only call ask_user (clarification), terminate (impossible/inappropriate), or confirm_plan (submit plan).
2. No plain text, analysis, or chit-chat — call a tool or the request WILL fail.
3. confirm_plan MUST receive a concrete list of 3–7 actionable tasks. Each task is a discrete, measurable step.
4. "Complete the user's request" or "Do the work" are NOT valid tasks.

Actions:
1. Grasp the request. Do NOT write long analysis.
2. Create a numbered task list covering the full goal.
3. Include verification tasks if writing code or files.
4. Call confirm_plan(tasks=[...], task_dependencies=[[], [0], ...]). Minimize dependencies.

File paths: `/home/[USER_HOME]agent/<slug>/`. NEVER use `/root/`.

These tools are available during EXECUTE only:
{tool_info}
"""

DEFAULT_REPLAN_PROMPT = """\
You are in REPLAN mode. A new task plan is needed.

{loop_info}

Core rules:
1. Only call ask_user (clarification), terminate (impossible), or confirm_plan (submit plan).
2. No plain text, analysis, or chit-chat — call a tool or be discarded.
3. confirm_plan MUST receive a concrete list of minimal, actionable tasks (1–3). Each task is a discrete, measurable step.
4. "Complete the user's request" or "Do the work" are NOT valid tasks.

Actions:
1. Review the previous goal and current request.
2. Create a minimal, focused plan (1–3 tasks). Keep dependencies minimal.
3. Call confirm_plan(tasks=[...], task_dependencies=[[], [0], ...]).

File paths: `/home/[USER_HOME]/agent/<slug>/`. NEVER use `/root/`.

Task status markers: [done] = completed, [FAIL: reason] = failed, [blocked] = unmet dependencies, [    ] = not started.

Reason for replan: {reason}

Original goal: {goal}

Past tasks:
{task_state}

These tools are available during EXECUTE only:
{tool_info}
"""

DEFAULT_EXECUTE_PROMPT = """\
You are in EXECUTE mode. Complete exactly ONE task, then mark it done.

{loop_info}

Core rules:
1. Execute ONLY the Current Task. Do NOT call tools for any other task.
2. Use run_tools_parallel for multiple INDEPENDENT calls within the SAME Current Task.
3. If you need more than ONE external tool in this turn, they MUST be inside run_tools_parallel. Do NOT make separate tool calls in the same turn.
4. When the Current Task is done, call complete_task(index) or fail_task(index, reason).
5. Do NOT read or work ahead. Do NOT call multiple separate tools sequentially.
6. Verify code with lint/compile/test before marking complete.
7. If minor issues, prefer fix_plan over replan. Replan only if everything is completely wrong.

File paths: `/home/[USER_HOME]/agent/<slug>/`. NEVER use `/root/`.

Past tasks:
{past_tasks}

Current task:
{current_task}
"""

DEFAULT_REVIEW_PROMPT = """\
You are in REVIEW mode. Inspect completed work BEFORE deciding.

{loop_info}

Task status markers: [done] = completed, [FAIL: reason] = failed, [    ] = not started.

Core rules:
1. Only read what is STRICTLY NECESSARY for verification. Use grep/search, not full file reads. Use run_command for syntax or code run checks.
2. ALWAYS reread the written files. Do not just use what is in history context.
3. Do NOT copy entire file contents into reasoning.
4. Be honest — don't call proceed_to_output if something is missing.
5. If minor issues, prefer fix_plan over replan. Replan only if everything is completely wrong.

Actions:
1. Verify work using read-only tools (read_file, grep_search, list_files, etc.).
2. Optionally run verification commands (linters, tests).
3. Call exactly ONE of: proceed_to_output(), fix_plan(reason, tasks), or replan(reason).

Original goal: {goal}

Tasks:
{task_state}
"""


DEFAULT_OUTPUT_PROMPT = """\
You are in OUTPUT mode - RENDER TURN.

{loop_info}

Your ONLY job here is to call rendering or visualisation tools (e.g. display_file) if any are available and useful to illustrate the results for the user.
Do NOT write any plain text. Only call the appropriate tools.

Original goal: {goal}

Tasks:
{task_state}

Use `run_tools_parallel` if you are calling multiple independent rendering/visualisation tools.
Only rendering/visualisation tools configured for the OUTPUT phase are available.
"""

DEFAULT_OUTPUT_FINAL_PROMPT = """\
You are in OUTPUT mode - SUMMARY TURN.

{loop_info}

Core rules:
1. Return ONLY a JSON object.
2. Concise summary of the entire loop. Only write what has happened and not how. Don't go in detail.
3. Use the provided JSON schema.

Goal: {goal}

Tasks:
{task_state}
"""

OUTPUT_FINAL_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "output_final_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "3-5 sentence summary of what was accomplished, files created/modified, failed tasks (if any), and where to find results"
                },
                "files_created": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relative paths of files created under agent/"
                },
                "files_modified": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relative paths of files modified under agent/"
                },
                "failed_tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Brief descriptions of any failed tasks and why"
                },
                "status": {
                    "type": "string",
                    "enum": ["completed", "partial", "failed"],
                    "description": "Overall session status"
                }
            },
            "required": ["summary", "files_created", "files_modified", "failed_tasks", "status"],
            "additionalProperties": False
        }
    }
}

# Default hints for built-in OpenWebUI tools. Admin may override or extend
# via the Valves.TOOL_STATUS_MAP JSON field. Each hint is {"label": str, "params": [str]}.
# The first existing param from the list is shown as the preview value.
_DEFAULT_TOOL_STATUS_HINTS = {
    "run_command": {"label": "Run", "params": ["command"]},
    "read_file": {"label": "Read", "params": ["file", "path", "file_path"]},
    "write_file": {"label": "Write", "params": ["file", "path", "file_path"]},
    "list_files": {"label": "List", "params": ["path", "dir", "directory"]},
    "search_web": {"label": "Search", "params": ["query"]},
    "replace_file_content": {"label": "Edit", "params": ["file", "path", "file_path"]},
    "fetch_url": {"label": "Fetch", "params": ["url"]},
    "glob_search": {"label": "Glob", "params": ["pattern"]},
    "grep_search": {"label": "Grep", "params": ["pattern"]},
    "display_file": {"label": "Show", "params": ["file", "path", "file_path"]},
}


def _load_tool_status_hints(raw_json: str) -> dict:
    """Parse admin valve JSON and merge with defaults. Returns resolved hints dict."""
    hints = dict(_DEFAULT_TOOL_STATUS_HINTS)
    if not raw_json or not raw_json.strip():
        return hints
    try:
        parsed = json.loads(raw_json)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if isinstance(v, dict) and "label" in v and isinstance(v.get("params"), list):
                    hints[k] = v
    except json.JSONDecodeError:
        logger.warning("TOOL_STATUS_MAP is invalid JSON, using default hints only.")
    return hints


@staticmethod
def _extract_tool_preview(tool_name: str, args: dict, hints: dict) -> str | None:
    """Build a concise status preview for a tool call based on the hint map.

    Returns None if the tool is not in hints or none of the looked-for params exist.
    Example: >>> _extract_tool_preview("read_file", {"file": "/tmp/a.ts"}, hints)
              'Read | /tmp/a.ts'
    """
    if not isinstance(args, dict) or not hints:
        return None
    hint = hints.get(tool_name)
    if not hint:
        return None
    for param in hint["params"]:
        value = args.get(param)
        if value is not None:
            preview = str(value).replace("\n", " / ").strip()
            if len(preview) > 80:
                preview = preview[:77] + "..."
            return f"{hint['label']} | {preview}"
    return None


async def _parse_sse_payload(payload: str):
    """Parse a single SSE data payload and yield structured events."""
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return
    if not isinstance(parsed, dict):
        return
    choices = parsed.get("choices", [])
    if not choices:
        return
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


def smart_truncate(text, max_chars):
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    for sep in (". ", ".\n", "\n\n", "\n"):
        idx = text[:max_chars].rfind(sep)
        if idx > max_chars // 2:
            return text[:idx + len(sep)].rstrip() + "\n[truncated]"
    return text[:max_chars].rstrip() + "\n[truncated]"


def strip_thinking(text):
    """Remove thinking/reasoning blocks from model output.
    Handles: paired tags, unclosed tags, pipe-style blocks, and reasoning prefixes."""
    text = re.sub(
        r"<(?:think|thinking|reason|reasoning|thought)>.*?</(?:think|thinking|reason|reasoning|thought)>",
        "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"\|begin_of_thought\|.*?\|end_of_thought\|", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<(?:think|thinking|reason|reasoning|thought)>[^<]*",
        "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r"^(?:Thinking|Thought|Reasoning|Analysis|Reason)\s*:\s*",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )
    return text.strip()



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


class HelixAgentEngine:
    """Helix Agent - single-model agent loop with per-phase tool filtering."""

    PHASE_PLAN = "plan"
    PHASE_EXECUTE = "execute"
    PHASE_REVIEW = "review"
    PHASE_OUTPUT = "output"
    PHASE_REPLAN = "replan"

    INTERNAL_TOOLS = {"terminate", "replan", "complete_task", "fail_task", "confirm_plan", "fix_plan", "proceed_to_output", "run_tools_parallel", "ask_user"}

    PHASE_INTERNAL_TOOLS = {
        PHASE_PLAN:       {"terminate", "confirm_plan", "ask_user"},
        PHASE_EXECUTE:    {"replan", "complete_task", "fail_task", "fix_plan", "run_tools_parallel"},
        PHASE_REVIEW:     {"proceed_to_output", "replan", "fix_plan", "run_tools_parallel"},
        PHASE_OUTPUT:     set(),
        PHASE_REPLAN:     {"terminate", "confirm_plan", "ask_user"},
    }

    def __init__(self, request, user, body, event_emitter, event_call, metadata, valves, user_valves=None, incoming_tools=None):
        self.request = request
        self.user = user
        self.body = body
        self.event_emitter = event_emitter
        self.event_call = event_call
        self.metadata = metadata
        self.valves = valves
        self.user_valves = user_valves

        # Caches for dynamic UI helpers
        self._tool_status_hints = _load_tool_status_hints(
            getattr(self.valves, "TOOL_STATUS_MAP", "")
        )

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
        self.task_dependencies: Dict[int, List[int]] = {}
        self.produced_files = []
        self._files_lock = asyncio.Lock()
        self._output_turn = 0
        self._output_rendering_skipped = False
        self._seen_file_ids: Set[str] = set()
        self._output_parts = []
        self.loop_count = 0
        self.goal = ""
        self._skill_prompt = ""
        self._replan_reason = ""
        self._plan_questions_asked = 0
        self._plan_reprompt_count = 0
        self._extra_grace = 0
        self._extra_llm_grace = 0
        self._last_compression_loop = 0

        # Token tracking state
        self._total_tool_calls = 0
        self._memory_context = ""
        self._memory_injected = False
        self._rag_sources: list = []

        # Throttling / debounce state
        self._last_state_save_ts: float = 0.0
        self._last_task_state_str: str = ""
        self._last_state_save_hash: str = ""

        self._incoming_tools = dict(incoming_tools) if incoming_tools else {}

        # Session lifetime & API budget guards
        self._session_start_ts = asyncio.get_event_loop().time()
        self._llm_call_count = 0

        # Cache model_knowledge from workspace model metadata for native vector search
        self._model_knowledge = self._resolve_model_knowledge()

    def _resolve_model_knowledge(self):
        """Extract knowledge base config from the workspace model metadata."""
        if not self.request or not self.body:
            return None
        try:
            app_models = getattr(self.request.app.state, "MODELS", {})
            model_info = app_models.get(self.body.get("model", ""), {})
            if not model_info:
                return None
            info_block = model_info.get("info", model_info)
            meta = info_block.get("meta", {}) if isinstance(info_block, dict) else {}
            return meta.get("knowledge") or meta.get("model_knowledge")
        except Exception:
            return None

    def _format_output(self):
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
            if s.startswith("[OUT]"):
                continue
            if s.startswith("[PLAN]"):
                continue
            filtered.append(part)
        return "".join(filtered)

    def _parse_output_json(self, text: str) -> dict | None:
        """Robustly parse JSON from model output, handling markdown blocks and whitespace."""
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _render_output_markdown(data: dict) -> str:
        """Render structured output data as a clean markdown summary."""
        lines = []

        summary = data.get("summary", "")
        if summary:
            lines.append(summary)
            lines.append("")

        status = data.get("status", "")
        if status == "completed":
            lines.append("**Status:** Completed")
        elif status == "partial":
            lines.append("**Status:** Partial")
        elif status == "failed":
            lines.append("**Status:** Failed")
        elif status:
            lines.append(f"**Status:** {status}")

        files_created = data.get("files_created", [])
        if files_created:
            lines.append("")
            lines.append("**Files created:**")
            for f in files_created:
                lines.append(f"- `{f}`")

        files_modified = data.get("files_modified", [])
        if files_modified:
            lines.append("")
            lines.append("**Files modified:**")
            for f in files_modified:
                lines.append(f"- `{f}`")

        failed_tasks = data.get("failed_tasks", [])
        if failed_tasks:
            lines.append("")
            lines.append("**Failed tasks:**")
            for t in failed_tasks:
                lines.append(f"- {t}")

        return "\n".join(lines)

    async def _stream_completion(self, body, max_retries: int = 1, iter_deadline: float = None):
        """Stream OWUI completion, yielding structured events. Retries on transient errors."""

        # --- Iteration deadline guard (fail fast before burning an API call) ---
        if iter_deadline and asyncio.get_event_loop().time() >= iter_deadline:
            yield {"type": "error", "text": "Iteration deadline reached before LLM call"}
            return

        body["stream"] = True
        last_error = None
        self._llm_call_count += 1

        for attempt in range(max_retries + 1):
            try:
                remaining = None
                if iter_deadline:
                    remaining = max(0, iter_deadline - asyncio.get_event_loop().time())
                    if remaining <= 0:
                        yield {"type": "error", "text": "Iteration deadline reached before LLM call"}
                        return

                if remaining:
                    response = await asyncio.wait_for(
                        generate_chat_completion(self.request, body, user=self.user),
                        timeout=remaining
                    )
                else:
                    response = await generate_chat_completion(self.request, body, user=self.user)
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
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"generate_chat_completion failed: {e}")
                yield {"type": "error", "text": str(e)}
                return

        if hasattr(response, "body_iterator"):
            sse_buffer = ""
            chunk_timeout = getattr(self.valves, "STREAM_CHUNK_TIMEOUT_SECONDS", 60)
            iterator = response.body_iterator
            while True:
                try:
                    if self._is_yolo_mode:
                        chunk = await iterator.__anext__()
                    elif chunk_timeout > 0:
                        chunk = await asyncio.wait_for(iterator.__anext__(), timeout=chunk_timeout)
                    else:
                        chunk = await iterator.__anext__()
                except asyncio.TimeoutError:
                    if self._is_yolo_mode:
                        continue
                    logger.warning(f"Stream stalled: no output for {chunk_timeout}s, treating as retryable")
                    yield {"type": "error", "text": f"Stream stalled: no output for {chunk_timeout}s (retryable)", "retryable": True}
                    return
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"SSE stream error: {e}")
                    # SSE stream errors are treated as retryable errors so the agent can recover
                    yield {"type": "error", "text": f"SSE stream error: {e}", "retryable": True}
                    return

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

                    async for event in _parse_sse_payload(payload):
                        yield event

            # Flush remaining buffer
            if sse_buffer.strip():
                for line in sse_buffer.strip().splitlines():
                    stripped = line.strip()
                    if stripped.startswith("data:"):
                        payload = stripped[5:].lstrip()
                        if payload and payload != "[DONE]":
                            async for event in _parse_sse_payload(payload):
                                yield event

        elif isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                if msg.get("tool_calls"):
                    yield {"type": "tool_calls", "data": msg["tool_calls"]}
                if msg.get("content"):
                    yield {"type": "content", "text": msg["content"]}

    async def _save_state_to_file(self, force: bool = False) -> None:
        """Serialize agent state to a JSON file and bind it to the chat DB.

        Writes are throttled: at most one every 2 seconds, and skipped if the
        state payload is unchanged since the last successful save.
        Call with force=True to bypass throttling (e.g. on termination).
        After a successful save, stale helix_state files in the chat are cleaned up.
        """
        if not HAS_DB_PERSISTENCE or not self.chat_id or not self.request or not self.user:
            return

        try:
            state_data = {
                "goal": self.goal,
                "task_list": self.task_list,
                "completed": self.completed_tasks,
                "failed": self.failed_tasks,
                "task_dependencies": {str(k): v for k, v in self.task_dependencies.items()},
                "phase": self.phase,
                "loop_count": self.loop_count,
                "extra_grace": self._extra_grace,
                "extra_llm_grace": self._extra_llm_grace,
                "history": self.history,
            }
            content = json.dumps(state_data, sort_keys=True, ensure_ascii=False).encode("utf-8")
            payload_hash = hashlib.sha256(content).hexdigest()

            now = asyncio.get_event_loop().time()
            if not force:
                if now - self._last_state_save_ts < 2.0:
                    return
                if payload_hash == self._last_state_save_hash:
                    return

            filename = f"helix_state_{self.chat_id}_{self.loop_count}.json"

            file_upload = UploadFile(
                file=io.BytesIO(content),
                filename=filename,
                headers=Headers({"content-type": "application/json"}),
            )

            file_item = await upload_file_handler(
                request=self.request,
                file=file_upload,
                metadata={},
                process=False,
                user=self.user,
            )

            if not file_item:
                return
            file_id = getattr(file_item, "id", None)
            if not file_id:
                return
            file_info = {"file_id": str(file_id), "name": filename}

            # Update throttling bookkeeping
            self._last_state_save_ts = now
            self._last_state_save_hash = payload_hash

            # Update internal metadata
            self._upsert_metadata_files(file_info)

            # Direct DB binding
            if self.chat_id and self.message_id:
                try:
                    await Chats.add_message_files_by_id_and_message_id(
                        self.chat_id,
                        self.message_id,
                        [file_info],
                    )
                    logger.info(
                        f"Persisted state file to chat {self.chat_id} message {self.message_id}"
                    )
                except Exception as db_err:
                    logger.warning(f"DB file binding failed: {db_err}")
                    return  # Abort cleanup if DB binding failed

            # Emit for immediate UI feedback
            if self.event_emitter:
                await self.event_emitter(
                    {"type": "chat:message:files", "data": {"files": [file_info]}}
                )

            # Cleanup stale helix_state attachments across the entire chat
            await self._cleanup_chat_state_files(exclude_file_id=str(file_id))
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")

    async def _cleanup_chat_state_files(self, exclude_file_id: str) -> None:
        """Remove stale helix_state file attachments from all messages in the chat.

        Deletes File DB records and removes the attachment references from
        messages. Skips the file identified by exclude_file_id (the newest).
        """
        if not HAS_DB_PERSISTENCE or not self.chat_id:
            return

        try:
            chat_obj = await Chats.get_chat_by_id(self.chat_id)
            if not chat_obj or not hasattr(chat_obj, "chat"):
                return
            messages_map = chat_obj.chat.get("history", {}).get("messages", {})
            current_id = chat_obj.chat.get("history", {}).get("currentId")

            deleted_count = 0
            visited = set()
            msg_id = current_id
            while msg_id and msg_id not in visited:
                visited.add(msg_id)
                msg = messages_map.get(msg_id)
                if not msg:
                    break
                msg_files = msg.get("files")
                if msg_files and isinstance(msg_files, list):
                    # Identify stale helix_state attachments
                    stale = []
                    keep = []
                    for f in msg_files:
                        if self._is_helix_state_file(f):
                            fid = f.get("id") or f.get("file_id")
                            if fid and str(fid) != exclude_file_id:
                                stale.append(f)
                                continue
                        keep.append(f)

                    if stale:
                        # Delete File DB records (best-effort)
                        for f in stale:
                            fid = f.get("id") or f.get("file_id")
                            if fid:
                                try:
                                    await Files.delete_file_by_id(fid)
                                    logger.info(f"[Helix GC] Deleted stale state file {fid}")
                                except Exception as del_err:
                                    logger.warning(f"[Helix GC] Failed to delete file {fid}: {del_err}")

                        # Update message files array directly
                        msg["files"] = keep
                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                            self.chat_id,
                            msg_id,
                            {"files": keep},
                        )
                        deleted_count += len(stale)
                msg_id = msg.get("parentId")

            if deleted_count:
                logger.info(f"[Helix GC] Pruned {deleted_count} stale helix_state attachments from chat {self.chat_id}")
        except Exception as e:
            logger.warning(f"[Helix GC] Cleanup failed: {e}")

    def _check_session_timeouts(self) -> tuple[bool, str]:
        """Return (should_stop, reason) if session timeout exceeded."""
        if self._is_yolo_mode:
            return False, ""

        max_session = getattr(self.valves, "MAX_SESSION_SECONDS", 1200)
        if max_session > 0:
            elapsed = asyncio.get_event_loop().time() - self._session_start_ts
            if elapsed >= max_session:
                return True, f"Session timeout ({int(elapsed)}s / {max_session}s)"

        return False, ""

    async def _check_timeouts_or_abort(self) -> bool:
        """Return True if the agent should abort now."""
        should_stop, reason = self._check_session_timeouts()
        if should_stop:
            await self.emit_output(f"\n[ERROR] {reason}. Stopping immediately.\n")
            await self.emit_task_update(finalize_tasks=True)
            await self.emit_status(f"Stopped: {reason}", done=True)
            logger.warning(f"Helix aborting: {reason}")
            return True
        return False

    async def _recover_state_from_files(self, body: dict) -> None:
        """Restore agent state from JSON file attachments in the chat.

        Picks the newest helix_state file by the loop number embedded in the filename.
        """
        if not HAS_DB_PERSISTENCE:
            return

        def _extract_loop(name: str) -> int:
            # helix_state_<chat_id>_<loop>.json
            try:
                parts = name.rsplit("_", 1)
                return int(parts[-1].replace(".json", ""))
            except (ValueError, IndexError):
                return 0

        state_candidates = []
        # 1. Look in current message attachments (body files)
        current_files = body.get("files") or body.get("__files__")
        # Also scan self.metadata files since they persist across turns
        metadata_files = self.metadata.get("__files__") or self.metadata.get("files")
        for file_list in (current_files, metadata_files):
            if file_list:
                for f in file_list:
                    name = f.get("name", f.get("filename", ""))
                    if self._is_helix_state_file(f):
                        state_candidates.append((f, _extract_loop(name)))

        # 2. Deep DB history scan
        if not state_candidates and self.chat_id:
            logger.info(f"Deep history scan for chat {self.chat_id}...")
            try:
                chat_obj = await Chats.get_chat_by_id(self.chat_id)
                if chat_obj and hasattr(chat_obj, "chat"):
                    messages_map = chat_obj.chat.get("history", {}).get("messages", {})
                    current_id = chat_obj.chat.get("history", {}).get("currentId")
                    visited = set()
                    while current_id and current_id not in visited:
                        visited.add(current_id)
                        msg = messages_map.get(current_id)
                        if not msg:
                            break
                        msg_files = msg.get("files")
                        if msg_files:
                            for f in msg_files:
                                name = f.get("name", f.get("filename", ""))
                                if self._is_helix_state_file(f):
                                    state_candidates.append((f, _extract_loop(name)))
                        current_id = msg.get("parentId")
            except Exception as e:
                logger.warning(f"DB history scan failed: {e}")

        if not state_candidates:
            logger.info("No Helix state file found in attachments or DB history.")
            return

        # Pick the state file with the highest loop number
        state_file, _ = max(state_candidates, key=lambda item: item[1])

        try:
            file_id = state_file.get("file_id") or state_file.get("id")
            if not file_id or Files is None:
                return
            file_obj = await Files.get_file_by_id(file_id)
            if not file_obj:
                return
            file_path = getattr(file_obj, "path", None)
            if not file_path and hasattr(file_obj, "meta"):
                file_path = file_obj.meta.get("path")
            if not file_path or not os.path.exists(file_path):
                return
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.task_list = data.get("task_list", [])
            self.completed_tasks = data.get("completed", [])
            self.failed_tasks = data.get("failed", [])
            raw_deps = data.get("task_dependencies", {})
            self.task_dependencies = {int(k): v for k, v in raw_deps.items()}
            self.phase = data.get("phase", self.PHASE_PLAN)
            self.goal = data.get("goal") if "goal" in data else self.goal
            self.loop_count = data.get("loop_count", 0)
            self._extra_grace = data.get("extra_grace", 0)
            self._extra_llm_grace = data.get("extra_llm_grace", 0)
            self.history = data.get("history", self.history)
            # Resynchronize compression loop counter to avoid immediate compression after recovery
            self._last_compression_loop = self.loop_count
            logger.info("Helix state recovered from file attachment.")
        except Exception as e:
            logger.warning(f"State recovery from file failed: {e}")

    async def emit_status(self, msg, done=False):
        if self.event_emitter:
            try:
                await self.event_emitter({"type": "status", "data": {"description": msg, "done": done}})
            except Exception:
                pass

    async def emit_output(self, text):
        self._output_parts.append(text)

    @property
    def _is_yolo_mode(self) -> bool:
        """Return True if YOLO mode is enabled (all safety limits ignored)."""
        return self.user_valves is not None and getattr(self.user_valves, "YOLO_MODE", False)

    def _total_history_chars(self) -> int:
        """Return total character count of all messages in self.history."""
        return sum(len(str(m.get("content", ""))) for m in self.history)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count using configured chars-per-token ratio."""
        if not text:
            return 0
        ratio = getattr(self.valves, "CHARS_PER_TOKEN_ESTIMATE", 3.5)
        return max(1, round(len(text) / ratio))

    def _total_history_tokens(self) -> int:
        """Return estimated token count of all messages in self.history."""
        return sum(self._estimate_tokens(str(m.get("content", ""))) for m in self.history)


    def _format_token_status(self) -> str:
        """Return a concise token status string showing current context size."""
        tokens = self._total_history_tokens()
        return f"Tokens: {tokens}"

    def _get_history_compression_threshold(self) -> int:
        """Return token threshold at which history compression should trigger.
        Derived from CONTEXT_LENGTH * 0.70.
        """
        ctx = getattr(self.valves, "CONTEXT_LENGTH", 128000)
        return int(ctx * 0.70)

    def _get_goal_compression_threshold(self) -> int:
        """Return token threshold at which goal compression should trigger.
        Derived from CONTEXT_LENGTH * 0.05.
        """
        ctx = getattr(self.valves, "CONTEXT_LENGTH", 128000)
        return int(ctx * 0.05)

    def _dedupe_files(self, file_list: list) -> list:
        """Remove duplicate files by id/file_id/url, preserving order."""
        seen = set()
        unique = []
        for f in file_list:
            if not isinstance(f, dict):
                continue
            fid = f.get("id") or f.get("file_id") or f.get("url")
            if fid and fid not in seen:
                seen.add(fid)
                unique.append(f)
        return unique

    @staticmethod
    def _is_helix_state_file(f) -> bool:
        return isinstance(f, dict) and f.get("name", "").startswith("helix_state_")

    def _upsert_metadata_files(self, file_info: dict):
        """Replace any existing helix_state files in metadata and append the new one."""
        mfiles = self.metadata.get("__files__")
        if isinstance(mfiles, list):
            mfiles[:] = [f for f in mfiles if not self._is_helix_state_file(f)]
            mfiles.append(file_info)
        else:
            self.metadata["__files__"] = [file_info]

    def get_current_files(self) -> list:
        """Return a deduplicated list of all known files (metadata + produced + DB canonical)."""
        files_map = {}
        # 1. Metadata files
        for f in self.metadata.get("__files__", []):
            if isinstance(f, dict):
                fid = f.get("id") or f.get("file_id") or f.get("url")
                if fid:
                    files_map[fid] = f
        # 2. Produced files from this turn
        for f in self.produced_files:
            if isinstance(f, dict):
                fid = f.get("id") or f.get("file_id") or f.get("url")
                if fid:
                    files_map[fid] = f
        return list(files_map.values()) if files_map else []

    async def resolve_tools(self):
        """Resolve ALL tools from the Pipe __tools__ parameter.

        __tools__ is a dict[str, dict] passed by OpenWebUI middleware containing
        already-resolved tools with callable, spec, and type. We treat it as the
        authoritative source and only add our internal control tools on top.
        """
        self.all_tools_dict = {}

        model_info = self.app_models.get(self.body.get("model", ""), {})
        skill_ids, skill_prompt = await self._resolve_model_skills(model_info)
        self._skill_prompt = skill_prompt

        if self._incoming_tools:
            self.all_tools_dict.update(self._incoming_tools)

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
                "description": "Restart planning. Use when the current approach is not working and you need to create a new plan. Preserves conversation history and context. After calling this tool, the agent will enter Replan mode where you must create a new plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "What went wrong or why the plan needs to change"},
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
                "description": "Present the task plan to the user for approval. Call this after creating the plan in PLAN or REPLAN phase. Provide the tasks as an array of clear, actionable steps. If some tasks depend on others, also provide task_dependencies.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {"type": "string", "description": "A clear, actionable step"},
                            "description": "Array of tasks to accomplish. Each task must be a clear, actionable step.",
                        },
                        "task_dependencies": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "List of task indices that this task depends on"
                            },
                            "description": "Optional. A list aligned with tasks. Index i contains the indices of tasks that task i depends on. If task 2 depends on task 0 being completed first, pass task_dependencies[2]=[0]. Empty arrays mean no dependencies. Avoid cycles and self-referencing.",
                        },
                    },
                    "required": ["tasks"],
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
                        "tasks": {
                            "type": "array",
                            "items": {"type": "string", "description": "A new or corrected task step"},
                            "description": "Array of new/corrected tasks to add. These tasks will be appended or replace failed tasks.",
                        },
                        "task_dependencies": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "List of task indices that this task depends on (relative to the new tasks array)"
                            },
                            "description": "Optional. Dependencies among the new tasks being added. Same format as confirm_plan.",
                        },
                    },
                    "required": ["reason", "tasks"],
                },
            },
            "callable": self._tool_fix_plan,
            "type": "function",
        }
        self.all_tools_dict["proceed_to_output"] = {
            "spec": {
                "name": "proceed_to_output",
                "description": "Move from REVIEW to OUTPUT phase to generate the polished final answer. Call this when everything is done and correct.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notes": {"type": "string", "description": "Optional brief notes about the review decision."},
                    },
                },
            },
            "callable": self._tool_proceed_to_output,
            "type": "function",
        }
        self.all_tools_dict["run_tools_parallel"] = {
            "spec": {
                "name": "run_tools_parallel",
                "description": "Execute multiple independent tool calls in parallel for faster results. Use this when you need to call 2+ tools that do not depend on each other (e.g., two searches, or a file read and a web fetch). Provide each call as {name: str, args: dict}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_calls": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Tool function name to call"},
                                    "args": {"type": "object", "description": "Arguments to pass to the tool"},
                                },
                                "required": ["name", "args"],
                            },
                            "description": "List of tool calls. Each item must have 'name' and 'args'. Example: [{\"name\": \"search_web\", \"args\": {\"query\": \"Python\"}}]",
                        },
                    },
                    "required": ["tool_calls"],
                },
            },
            "callable": self._tool_run_tools_parallel,
            "type": "function",
        }
        self.all_tools_dict["ask_user"] = {
            "spec": {
                "name": "ask_user",
                "description": "Ask the user an interactive question with selectable options and optional custom free-text input. Use this ONLY when you need clarification or a decision from the user before you can continue planning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to display to the user."},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 2-6 options for the user to pick from.",
                        },
                        "allow_custom": {
                            "type": "boolean",
                            "default": True,
                            "description": "If true, the user can type a custom answer instead of picking an option.",
                        },
                    },
                    "required": ["question", "options"],
                },
            },
            "callable": self._tool_ask_user,
            "type": "function",
        }


    async def _resolve_model_skills(self, model_info: dict) -> tuple[list[str], str]:
        """Resolve skillIds from model metadata and fetch user skills from DB."""
        skill_ids: list[str] = []
        skill_prompt = ""
        if not model_info:
            return skill_ids, skill_prompt

        meta = model_info.get("info", {}).get("meta", {})
        if not meta:
            meta = model_info
        model_skill_ids = meta.get("skillIds", [])
        if not model_skill_ids:
            return skill_ids, skill_prompt

        try:
            user_id = None
            if hasattr(self.user, "id"):
                user_id = self.user.id
            elif isinstance(self.user, dict):
                user_id = self.user.get("id")
            if not user_id:
                return skill_ids, skill_prompt

            user_skills = await Skills.get_skills_by_user_id(user_id, "read")
            accessible = {s.id: s for s in user_skills if s.is_active}
            available = []
            for sid in model_skill_ids:
                sk = accessible.get(sid)
                if sk:
                    available.append(sk)
                    skill_ids.append(sid)

            if available:
                descriptions = ""
                for sk in available:
                    descriptions += f"<skill>\n<name>{sk.name}</name>\n<description>{sk.description or ''}</description>\n</skill>\n"
                skill_prompt = (
                    f"<available_skills>\n{descriptions}</available_skills>\n\n"
                    "You have access to the above skills. ONLY use them when they are directly useful for the current task or context. Do not invoke a skill unless it provides clear value for what the user is asking."
                )
        except Exception as e:
            logger.error(f"Error resolving skills: {e}")

        return skill_ids, skill_prompt


    def _filter_tools_for_phase(self, phase: str):
        """Build phase_tools_dict from all_tools_dict based on Valves config."""
        # Determine which tool names are allowed for this phase
        allowlist: Set[str] = set()

        if phase == self.PHASE_EXECUTE:
            allowlist = set(_comma_list(self.valves.EXECUTE_TOOLS))
        elif phase == self.PHASE_REVIEW:
            allowlist = set(_comma_list(self.valves.REVIEW_TOOLS))
        elif phase == self.PHASE_OUTPUT:
            allowlist = set(_comma_list(self.valves.OUTPUT_TOOLS))

        # Phase-specific internal tools (only show relevant ones per phase)
        phase_internal_tools = self.PHASE_INTERNAL_TOOLS.get(phase, self.INTERNAL_TOOLS)

        # PLAN and REPLAN phases: only internal tools (no external tools)
        if phase in (self.PHASE_PLAN, self.PHASE_REPLAN):
            self.phase_tools_dict = {
                name: tool for name, tool in self.all_tools_dict.items()
                if name in phase_internal_tools
            }
            self.phase_tools_specs = [
                {"type": "function", "function": t["spec"]}
                for t in self.phase_tools_dict.values()
                if isinstance(t, dict) and "spec" in t
            ]
            return

        # If allowlist is empty -> allow ALL tools
        # If allowlist has entries -> only those tools (plus phase-internal ones)
        self.phase_tools_dict = {}

        for name, tool in self.all_tools_dict.items():
            # Internal tools are included ONLY if in phase_internal_tools
            if name in self.INTERNAL_TOOLS:
                if name in phase_internal_tools:
                    self.phase_tools_dict[name] = tool
                continue

            # Allowlist filtering for non-internal tools
            if allowlist:
                if name in allowlist:
                    self.phase_tools_dict[name] = tool
            elif phase != self.PHASE_OUTPUT:
                # Non-OUTPUT phases: empty allowlist means ALL non-internal tools
                self.phase_tools_dict[name] = tool
            # OUTPUT phase: empty allowlist means NO non-internal tools (strict rendering control)

        # Build OpenAI-format tool specs
        self.phase_tools_specs = [
            {"type": "function", "function": t["spec"]}
            for t in self.phase_tools_dict.values()
            if isinstance(t, dict) and "spec" in t
        ]


    async def _tool_terminate(self, **kwargs):
        return json.dumps({"terminated": True, "result": kwargs.get("result", ""), "success": kwargs.get("success", True)})

    async def _tool_replan(self, reason: str, **kwargs):
        """Process a replan: transition to REPLAN phase for the LLM to create a new plan."""
        self._replan_reason = reason
        self.completed_tasks = []
        self.failed_tasks = []
        self.task_list = []
        self.task_dependencies = {}

        # Reset counters for the new plan session
        self._total_tool_calls = 0

        # Transition to REPLAN phase
        self._transition_to(self.PHASE_REPLAN)
        self.loop_count = 0
        self._plan_questions_asked = 0
        self._plan_reprompt_count = 0
        self._extra_grace = 0
        await self._save_state_to_file()
        await self.emit_task_update()
        return json.dumps({"replan": True, "reason": reason})

    async def _tool_complete_task(self, **kwargs):
        idx = kwargs.get("index", -1)
        if 0 <= idx < len(self.task_list):
            task = self.task_list[idx]
            if task not in self.completed_tasks:
                self.completed_tasks.append(task)
            await self._save_state_to_file()
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
            await self._save_state_to_file()
            await self.emit_task_update()
            return json.dumps({"failed": True, "task": task, "index": idx, "reason": reason})
        return json.dumps({"failed": False, "error": f"Invalid task index {idx}"})

    # Lightweight correction tool. Use this instead of replan for minor issues.
    async def _tool_fix_plan(self, reason: str, tasks: list, task_dependencies: list = None, **kwargs):
        if not self.task_list:
            return json.dumps({"fix_plan": False, "error": "No task list available"})

        if not isinstance(tasks, list) or not tasks:
            return json.dumps({"fix_plan": False, "error": "No tasks provided"})

        new_tasks = [str(t) for t in tasks]

        # Compute insertion index BEFORE removing failed tasks
        failed_names = {f["task"] for f in self.failed_tasks}
        insert_idx = len(self.task_list)  # default: append
        for i, t in enumerate(self.task_list):
            if t in failed_names:
                insert_idx = i
                break

        # Remove failed tasks from the task list
        self.task_list = [t for t in self.task_list if t not in failed_names]
        if insert_idx > len(self.task_list):
            insert_idx = len(self.task_list)
        self.failed_tasks = []

        self.task_list[insert_idx:insert_idx] = new_tasks

        # Handle task_dependencies for new tasks
        new_deps = self._validate_task_dependencies(task_dependencies, len(new_tasks))
        if new_deps is None:
            return json.dumps({"fix_plan": False, "error": "Invalid task_dependencies: out-of-range or cyclic dependencies detected."})
        # Merge into existing deps: shift indices for inserted tasks
        old_deps = dict(self.task_dependencies)
        shifted_deps = {}
        for old_idx, old_deps_list in old_deps.items():
            if old_idx < insert_idx:
                shifted_deps[old_idx] = old_deps_list
            else:
                shifted_deps[old_idx + len(new_tasks)] = [d + len(new_tasks) for d in old_deps_list]
        # Add new deps offset by insert_idx
        for local_idx, deps_list in new_deps.items():
            shifted_deps[insert_idx + local_idx] = [d + insert_idx for d in deps_list]
        self.task_dependencies = shifted_deps

        await self._save_state_to_file()
        await self.emit_task_update()
        return json.dumps({"fix_plan": True, "inserted_tasks": new_tasks, "reason": reason})

    async def _tool_proceed_to_output(self, **kwargs):
        """Transition from REVIEW to OUTPUT phase."""
        if self.phase == self.PHASE_REVIEW:
            self._transition_to(self.PHASE_OUTPUT)
            await self._save_state_to_file()
            await self.emit_task_update()
            return json.dumps({"proceed_to_output": True})
        return json.dumps({"proceed_to_output": False, "error": f"Cannot proceed to output from {self.phase} phase"})

    def _validate_task_dependencies(self, raw_deps, num_tasks):
        """Validate and normalize task_dependencies. Returns dict[int, List[int]] or None if invalid."""
        if raw_deps is None:
            return {}
        if not isinstance(raw_deps, list):
            return None
        if len(raw_deps) != num_tasks:
            # Allow shorter lists, fill with empty lists
            raw_deps = list(raw_deps) + [[] for _ in range(num_tasks - len(raw_deps))]
        parsed = {}
        for i, dep_entry in enumerate(raw_deps):
            if dep_entry is None:
                dep_entry = []
            if isinstance(dep_entry, int):
                dep_entry = [dep_entry]
            if not isinstance(dep_entry, list):
                return None
            deps = []
            for d in dep_entry:
                try:
                    d_idx = int(d)
                    if d_idx < 0 or d_idx >= num_tasks or d_idx == i:
                        return None
                    deps.append(d_idx)
                except (ValueError, TypeError):
                    return None
            parsed[i] = deps
        # Check for cycles via simple DFS
        visited = {}
        def has_cycle(node, stack):
            visited[node] = True
            stack.add(node)
            for neighbor in parsed.get(node, []):
                if neighbor in stack or (not visited.get(neighbor) and has_cycle(neighbor, stack)):
                    return True
            stack.remove(node)
            return False
        for node in range(num_tasks):
            if not visited.get(node):
                if has_cycle(node, set()):
                    return None
        return parsed

    def _compute_blocked_tasks(self):
        """Return set of task indices that are blocked due to unmet dependencies."""
        blocked = set()
        completed_set = {i for i, task in enumerate(self.task_list) if task in self.completed_tasks}
        for idx, deps in self.task_dependencies.items():
            if idx not in completed_set and not all(d in completed_set for d in deps):
                blocked.add(idx)
        return blocked

    def _get_current_task_index(self):
        """Return the index of the current task: first unblocked, incomplete, non-failed task."""
        failed_set = {next((i for i, task in enumerate(self.task_list) if task == f["task"]), -1) for f in self.failed_tasks}
        blocked = self._compute_blocked_tasks()
        for i, task in enumerate(self.task_list):
            if task in self.completed_tasks or i in failed_set or i in blocked:
                continue
            return i
        return None

    async def _tool_confirm_plan(self, **kwargs):
        tasks = kwargs.get("tasks", [])
        if not isinstance(tasks, list):
            tasks = []
        uv = self.user_valves

        # --- STRICT VALIDATION ---
        # Reject empty or completely missing task lists
        if not tasks:
            return json.dumps({
                "action": "error",
                "error": "CRITICAL: confirm_plan was called with no tasks. You MUST provide a concrete, non-empty list of actionable tasks. Plain text or empty plans are NOT accepted. Call confirm_plan again with a valid plan."
            })

        # Reject overly generic single-task plans
        forbidden_terms = ["complete the user", "do the work", "handle the request", "process the request", "fulfill the request", "address the request", "finish the task", "do everything", "execute all", "perform all"]
        all_text = " ".join(str(t).lower() for t in tasks)
        if len(tasks) == 1 and any(term in all_text for term in forbidden_terms):
            return json.dumps({
                "action": "error",
                "error": f"CRITICAL: Your task plan is too generic: '{tasks[0]}'. Break the work into specific, concrete steps. Call confirm_plan again with a detailed task list."
            })

        # Warn if any individual task is very vague (3 words or less and contains forbidden keywords)
        vague_tasks = []
        for t in tasks:
            words = str(t).lower().split()
            if len(words) <= 3 and any(term in str(t).lower() for term in forbidden_terms):
                vague_tasks.append(str(t))
        if vague_tasks:
            return json.dumps({
                "action": "error",
                "error": f"CRITICAL: Task(s) are too vague: {vague_tasks}. Each task must be a specific, measurable action step. Call confirm_plan again with concrete tasks."
            })

        # Validate and store task dependencies
        raw_deps = kwargs.get("task_dependencies")
        validated_deps = self._validate_task_dependencies(raw_deps, len(tasks))
        if validated_deps is None:
            return json.dumps({"action": "error", "error": "Invalid task_dependencies: out-of-range indices, self-references, or cyclic dependencies detected."})
        self.task_dependencies = validated_deps

        # Proceed with existing approval / auto-accept logic
        if uv and (getattr(uv, "YOLO_MODE", False) or not getattr(uv, "ENABLE_PLAN_APPROVAL", False)):
            return json.dumps({"action": "accept", "tasks": tasks})

        if self.phase == self.PHASE_REPLAN:
            return json.dumps({"action": "accept", "tasks": tasks})

        if not self.event_call:
            return json.dumps({"action": "accept", "tasks": tasks})

        tasks_data = [{"task_id": f"T{i+1}", "description": str(t)} for i, t in enumerate(tasks)]
        if not tasks_data:
            return json.dumps({
                "action": "error",
                "error": "CRITICAL: No valid tasks were generated for plan approval. Call confirm_plan again with a concrete task list."
            })

        timeout_s = getattr(self.valves, "PLAN_APPROVAL_TIMEOUT", 600)
        js = self._build_plan_approval_js(tasks_data, timeout_s=timeout_s)
        try:
            raw = await self.event_call({"type": "execute", "data": {"code": js}})
        except Exception as e:
            logger.error(f"Plan approval event_call failed: {e}")
            return json.dumps({"action": "error", "error": f"Plan confirmation failed: {e}"})

        raw_str = raw if isinstance(raw, str) else (raw.get("result") or raw.get("value") or "{}") if raw else "{}"
        try:
            res = json.loads(raw_str) if isinstance(raw_str, str) and raw_str.startswith("{") else {"action": "error", "error": "Malformed confirmation response."}
        except (json.JSONDecodeError, AttributeError):
            res = {"action": "error", "error": "Malformed confirmation response."}

        res["tasks"] = tasks
        return json.dumps(res)

    async def _tool_ask_user(self, question: str, options: list, allow_custom: bool = True, **kwargs):
        """Interactive user question tool. Only available during PLAN and REPLAN phases."""
        if not self.event_call:
            return json.dumps({"type": "error", "response": "Interactive input not available in this context.", "skipped": True})

        if not options or not isinstance(options, list):
            return json.dumps({"type": "error", "response": "Provide at least one option.", "skipped": True})

        if self.phase in (self.PHASE_PLAN, self.PHASE_REPLAN):
            self._plan_questions_asked += 1
            max_questions = getattr(self.user_valves, "MAX_PLAN_QUESTIONS", 3)
            if self._plan_questions_asked > max_questions:
                return json.dumps({
                    "type": "error",
                    "response": (
                        f"CRITICAL: You have reached the maximum number of clarification questions "
                        f"({max_questions}). You must NOT ask more questions. "
                        "Call confirm_plan with your best plan NOW."
                    ),
                    "skipped": True,
                })

        opts = [str(o) for o in options[:8]]
        if not opts:
            return json.dumps({"type": "error", "response": "Options list is empty.", "skipped": True})

        js = self._build_ask_user_js(question=question, options=opts, allow_custom=allow_custom)
        try:
            raw = await self.event_call({"type": "execute", "data": {"code": js}})
        except Exception as e:
            logger.error(f"ask_user event_call failed: {e}")
            return json.dumps({"type": "error", "response": f"Error getting user input: {e}", "skipped": True})

        raw_str = raw if isinstance(raw, str) else (raw.get("result") or raw.get("value") or "{}") if raw else "{}"
        try:
            parsed = json.loads(raw_str) if isinstance(raw_str, str) and raw_str.startswith("{") else {}
        except (json.JSONDecodeError, AttributeError):
            parsed = {}

        rtype = parsed.get("type")
        if rtype == "select":
            return json.dumps({"type": "select", "response": parsed.get("value", ""), "skipped": False})
        elif rtype == "custom":
            return json.dumps({"type": "custom", "response": parsed.get("value", ""), "skipped": False})
        elif rtype == "skip":
            return json.dumps({"type": "skip", "response": "User skipped the question.", "skipped": True})
        return json.dumps({"type": "unknown", "response": f"User response: {raw_str}", "skipped": False})


    def _normalize_parallel_calls(self, tool_calls: list) -> list:
        """Normalize and validate parallel tool call items."""
        if not isinstance(tool_calls, list):
            raise ValueError("tool_calls must be a list")
        normalized = []
        for i, call in enumerate(tool_calls):
            if isinstance(call, str):
                try:
                    call = json.loads(call)
                except json.JSONDecodeError:
                    raise ValueError(f"tool_calls[{i}] is an unparseable string")
            if not isinstance(call, dict):
                raise ValueError(f"tool_calls[{i}] must be an object")
            name = call.get("name", call.get("tool_name", ""))
            if not name:
                raise ValueError(f"tool_calls[{i}] missing 'name' field")
            if isinstance(name, str) and name.startswith("functions."):
                name = name[len("functions."):]
            args = call.get("args", call.get("arguments", call.get("parameters", {})))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            normalized.append({"name": name, "args": args})
        return normalized

    async def _execute_parallel_single(self, call_item: dict, call_id: str) -> dict:
        """Execute a single tool call for parallel batching."""
        tool_name = call_item["name"]
        args = call_item["args"]
        result_str, result_files = await self._execute_tool(tool_name, args, call_id)
        # Track produced files
        new_files = await self._append_produced_files(result_files)
        if new_files and self.event_emitter:
            await self.event_emitter({
                "type": "chat:message:files",
                "data": {"files": new_files},
            })
        truncation_limit = self._get_truncation_limit()
        was_truncated = False
        truncated_result = result_str
        if truncation_limit and result_str and len(result_str) > truncation_limit:
            was_truncated = True
            await self.emit_status(
                f"Truncated result for {tool_name} ({len(result_str)} -> {truncation_limit} chars)"
            )
            # Prepend a truncation notice so the model knows result is incomplete.
            truncated_result = (
                f"[TRUNCATED] Tool '{tool_name}' result was cut from {len(result_str)} to {truncation_limit} chars. "
                f"If you need the full output, refine your arguments or run a more targeted query.\n\n"
                + smart_truncate(result_str, truncation_limit)
            )
        parsed_result = truncated_result
        if isinstance(truncated_result, str):
            try:
                parsed_result = json.loads(truncated_result)
                if was_truncated and isinstance(parsed_result, dict):
                    parsed_result["__truncated"] = True
                    parsed_result["__truncated_reason"] = (
                        f"Result truncated from {len(result_str)} to {truncation_limit} chars"
                    )
            except json.JSONDecodeError:
                pass
        return {
            "tool_name": tool_name,
            "result": parsed_result,
        }

    async def _tool_run_tools_parallel(self, tool_calls: list = None, **kwargs):
        """Execute multiple independent tool calls in parallel.

        Partial execution: valid calls run in parallel, invalid ones
        return errors without blocking the rest.
        """
        if not tool_calls:
            return json.dumps({"error": "No tool_calls provided"})
        try:
            calls = self._normalize_parallel_calls(tool_calls)
        except ValueError as e:
            return json.dumps({"error": str(e)})

        internal_in_parallel = [c["name"] for c in calls if c["name"] in self.INTERNAL_TOOLS]
        if internal_in_parallel:
            return json.dumps({
                "error": f"run_tools_parallel may NOT contain internal Helix tools: {', '.join(internal_in_parallel)}. These MUST be called individually, not inside a parallel batch.",
            })

        # ── pre-flight validation, preserving original order ──
        valid_calls = []          # list of (original_index, call)
        invalid_results = {}      # original_index -> error dict
        for original_idx, c in enumerate(calls):
            tool_name = c["name"]
            tool_entry = self.phase_tools_dict.get(tool_name)
            if not tool_entry:
                available = ", ".join(sorted(self.phase_tools_dict.keys())[:20])
                invalid_results[original_idx] = {
                    "tool_name": tool_name,
                    "result": f"[ERR] Tool '{tool_name}' is NOT available in phase '{self.phase}'. Available: {available}.",
                }
                continue
            spec = tool_entry.get("spec", {})
            errs = self._validate_tool_args(spec, c.get("args", {}))
            if errs:
                invalid_results[original_idx] = {
                    "tool_name": tool_name,
                    "result": f"[ERR] Validation failed for '{tool_name}': {', '.join(errs)}",
                }
                continue
            valid_calls.append((original_idx, c))

        if not valid_calls:
            # Every call was invalid – return them in original order
            all_invalid = [invalid_results[i] for i in sorted(invalid_results.keys())]
            return json.dumps({"results": all_invalid}, ensure_ascii=False)

        names = ", ".join(c["name"] for _, c in valid_calls)
        await self.emit_status(f"Running parallel: {names}...")

        tasks = [
            self._execute_parallel_single(c, f"{self.message_id or 'parallel'}_{i}")
            for i, c in valid_calls
        ]
        exec_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge valid results back, preserving original order
        final_results = []
        valid_idx = 0
        for original_idx in range(len(calls)):
            if original_idx in invalid_results:
                final_results.append(invalid_results[original_idx])
            else:
                res = exec_results[valid_idx]
                if isinstance(res, Exception):
                    final_results.append({
                        "tool_name": valid_calls[valid_idx][1]["name"],
                        "result": f"[ERR] Execution failed: {res}",
                    })
                else:
                    final_results.append(res)
                valid_idx += 1

        return json.dumps({"results": final_results}, ensure_ascii=False)

    def _base_theme_js(self):
        return """
            const col = {
                overlay: 'rgba(0,0,0,0.62)', panel: '#1e1e2e', border: '#45475a',
                text: '#cdd6f4', sub: '#a6adc8', input: '#313244', inputBorder: '#45475a',
                btn: '#313244', btnText: '#cdd6f4', btnBorder: '#45475a',
                btnPrimary: '#E8713A', btnPrimaryText: '#ffffff',
                muted: '#6c7086',
            };
        """

    def _build_ask_user_js(self, question: str, options: list, allow_custom: bool = True) -> str:
        q_safe = json.dumps(question)
        opts_safe = json.dumps(options)
        custom_js = "true" if allow_custom else "false"
        return f"""
    return (function(){{
      return new Promise((resolve)=>{{
    {self._base_theme_js()}
        const question    = {q_safe};
        const options     = {opts_safe};
        const allowCustom = {custom_js};
        const OVERLAY_ID  = '__owui_helix_ask_user__';

        const existing = document.getElementById(OVERLAY_ID);
        if (existing) existing.remove();

        function finish(payload){{
          panel.style.transform   = 'scale(0.95)';
          panel.style.opacity     = '0';
          overlay.style.opacity   = '0';
          setTimeout(()=>{{
            overlay.remove();
            document.removeEventListener('keydown', onKey);
            resolve(JSON.stringify(payload));
          }},180);
        }}

        const overlay = document.createElement('div');
        overlay.id = OVERLAY_ID;
        Object.assign(overlay.style, {{
          position:'fixed',inset:'0',zIndex:'999999',
          background:col.overlay,
          backdropFilter:'blur(12px)',WebkitBackdropFilter:'blur(12px)',
          display:'flex',alignItems:'center',justifyContent:'center',
          fontFamily:"-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif",
          opacity:'0',transition:'opacity 0.18s ease',
        }});

        const panel = document.createElement('div');
        Object.assign(panel.style, {{
          background:col.panel,border:'1px solid '+col.border,
          borderRadius:'16px',padding:'26px 26px 20px',
          maxWidth:'520px',width:'calc(100vw - 32px)',
          maxHeight:'85vh',overflowY:'auto',
          boxShadow:'0 28px 80px rgba(0,0,0,0.65)',
          display:'flex',flexDirection:'column',gap:'14px',
          transform:'scale(0.92)',opacity:'0',
          transition:'transform 0.22s cubic-bezier(0.34,1.56,0.64,1), opacity 0.18s ease',
        }});

        const header = document.createElement('div');
        Object.assign(header.style, {{ display:'flex',alignItems:'flex-start',gap:'12px' }});
        const questionEl = document.createElement('p');
        Object.assign(questionEl.style, {{
          margin:'0',color:col.text,fontSize:'15px',
          fontWeight:'600',lineHeight:'1.55',flex:'1',wordBreak:'break-word',
        }});
        questionEl.textContent = question;
        const badge = document.createElement('span');
        Object.assign(badge.style, {{
          flexShrink:'0',fontSize:'10px',fontWeight:'700',
          letterSpacing:'0.07em',padding:'3px 9px',borderRadius:'99px',
          background:col.btnPrimary+'26',color:col.btnPrimary,
          marginTop:'2px',whiteSpace:'nowrap',
        }});
        badge.textContent = 'CHOOSE ONE';
        header.appendChild(questionEl); header.appendChild(badge);

        const optContainer = document.createElement('div');
        Object.assign(optContainer.style, {{ display:'flex',flexDirection:'column',gap:'7px' }});

        options.forEach(function(opt, i){{
          const keyLabel = i < 26 ? String.fromCharCode(65 + i) : String(i + 1);

          const btn = document.createElement('button');
          Object.assign(btn.style, {{
            display:'flex',alignItems:'center',gap:'11px',
            background:col.btn,border:'1.5px solid '+col.btnBorder,
            borderRadius:'10px',padding:'11px 13px',
            cursor:'pointer',textAlign:'left',width:'100%',
            minHeight:'48px',outline:'none',fontFamily:'inherit',boxSizing:'border-box',
            transition:'background 0.12s, border-color 0.12s, transform 0.1s',
          }});

          const keyBadge = document.createElement('span');
          Object.assign(keyBadge.style, {{
            flexShrink:'0',width:'26px',height:'26px',
            borderRadius:'6px',background:col.btnBorder,
            display:'flex',alignItems:'center',justifyContent:'center',
            fontSize:'11px',fontWeight:'700',color:col.btnPrimary,
            transition:'background 0.12s, color 0.12s',userSelect:'none',
          }});
          keyBadge.textContent = keyLabel;

          const textBlock = document.createElement('div');
          Object.assign(textBlock.style, {{ flex:'1',minWidth:'0' }});
          const optLabel = document.createElement('span');
          Object.assign(optLabel.style, {{
            display:'block',color:col.text,fontSize:'14px',
            fontWeight:'500',wordBreak:'break-word',
          }});
          optLabel.textContent = opt;
          textBlock.appendChild(optLabel);

            btn.appendChild(keyBadge); btn.appendChild(textBlock);

          function applySelected(){{
            btn.style.background    = col.btnPrimary+'1c';
            btn.style.borderColor   = col.btnPrimary;
            keyBadge.style.background = col.btnPrimary;
            keyBadge.style.color    = '#ffffff';
          }}
          function applyDeselected(){{
            btn.style.background    = col.btn;
            btn.style.borderColor   = col.btnBorder;
            keyBadge.style.background = col.btnBorder;
            keyBadge.style.color    = col.btnPrimary;
          }}

          btn.addEventListener('mouseenter', function(){{
            btn.style.background  = '#3c3c52';
            btn.style.borderColor = col.btnPrimary+'77';
            btn.style.transform   = 'translateY(-1px)';
          }});
          btn.addEventListener('mouseleave', function(){{
            applyDeselected();
            btn.style.transform = '';
          }});
          btn.addEventListener('mouseleave', function(){{
            if (btn.dataset.selected !== '1') applyDeselected();
            btn.style.transform = '';
          }});

          btn.addEventListener('click', function(){{
            finish({{type:'select',index:i,value:opt}});
          }});

          optContainer.appendChild(btn);
        }});

        if (allowCustom){{
          const customRow = document.createElement('div');
          Object.assign(customRow.style, {{ display:'flex',gap:'7px',alignItems:'stretch' }});

          const customInput = document.createElement('input');
          customInput.type = 'text';
          customInput.placeholder = 'Or type a custom answer\u2026';
          Object.assign(customInput.style, {{
            flex:'1',background:col.input,border:'1.5px solid '+col.inputBorder,
            borderRadius:'8px',padding:'10px 12px',color:col.text,
            fontSize:'14px',minHeight:'44px',outline:'none',
            fontFamily:'inherit',transition:'border-color 0.12s',
            boxSizing:'border-box',
          }});
          customInput.addEventListener('focus', ()=>{{ customInput.style.borderColor=col.btnPrimary; }});
          customInput.addEventListener('blur', ()=>{{ customInput.style.borderColor=customInput.value?col.btnPrimary:col.inputBorder; }});
          customInput.addEventListener('keydown', function(e){{
            if (e.key==='Enter' && customInput.value.trim()){{
              e.stopPropagation();
              finish({{type:'custom',value:customInput.value.trim()}});
            }}
          }});

          const sendBtn = document.createElement('button');
          sendBtn.title = 'Submit custom answer (Enter)';
          Object.assign(sendBtn.style, {{
            background:col.btnPrimary,border:'none',borderRadius:'8px',
            padding:'10px 16px',cursor:'pointer',fontSize:'16px',
            color:col.btnPrimaryText,fontWeight:'700',minHeight:'44px',
            flexShrink:'0',transition:'opacity 0.12s, transform 0.1s',
            fontFamily:'inherit',
          }});
          sendBtn.textContent = '\u21b5';
          sendBtn.addEventListener('mouseenter', ()=>{{ sendBtn.style.transform='scale(1.08)'; }});
          sendBtn.addEventListener('mouseleave', ()=>{{ sendBtn.style.transform=''; }});
          sendBtn.addEventListener('click', ()=>{{
            if (customInput.value.trim()){{
              finish({{type:'custom',value:customInput.value.trim()}});
            }}
          }});

          customRow.appendChild(customInput); customRow.appendChild(sendBtn);
          optContainer.appendChild(customRow);
        }}

        const footer = document.createElement('div');
        Object.assign(footer.style, {{ display:'flex',gap:'8px',justifyContent:'flex-end',marginTop:'2px' }});

        const skipBtn = document.createElement('button');
        skipBtn.textContent = 'Skip';
        Object.assign(skipBtn.style, {{
          background:'transparent',border:'1.5px solid '+col.btnBorder,
          borderRadius:'8px',padding:'9px 18px',color:'#6c7086',
          fontSize:'13px',cursor:'pointer',fontFamily:'inherit',
          transition:'border-color 0.12s, color 0.12s',
        }});
        skipBtn.addEventListener('mouseenter', ()=>{{ skipBtn.style.borderColor='#9399b2'; skipBtn.style.color=col.text; }});
        skipBtn.addEventListener('mouseleave', ()=>{{ skipBtn.style.borderColor=col.btnBorder; skipBtn.style.color='#6c7086'; }});
        skipBtn.addEventListener('click', ()=>{{ finish({{type:'skip'}}); }});
        footer.appendChild(skipBtn);

        function onKey(e){{
          if (e.key==='Escape'){{ skipBtn.click(); return; }}
        }}
        document.addEventListener('keydown', onKey);

        panel.appendChild(header); panel.appendChild(optContainer); panel.appendChild(footer);
        overlay.appendChild(panel); document.body.appendChild(overlay);

        requestAnimationFrame(()=>{{
          requestAnimationFrame(()=>{{
            overlay.style.opacity='1';
            panel.style.transform='scale(1)';
            panel.style.opacity='1';
          }});
        }});
      }});
    }})();
"""

    def _build_plan_approval_js(self, tasks: list, timeout_s: int = 600) -> str:
        ts = json.dumps(tasks)
        return f"""
    return (function(){{
      return new Promise((resolve)=>{{
    {self._base_theme_js()}
        const OVERLAY_ID = '__owui_helix_plan__';
        const existing = document.getElementById(OVERLAY_ID);
        if (existing) existing.remove();

        let _timer;

        function cleanup(){{
          panel.style.transform='scale(0.95)';
          panel.style.opacity='0';
          overlay.style.opacity='0';
          setTimeout(()=>{{
            overlay.remove();
            document.removeEventListener('keydown',onKey);
          }},180);
        }}

        const overlay = document.createElement('div');
        overlay.id = OVERLAY_ID;
        Object.assign(overlay.style, {{
          position:'fixed',inset:'0',zIndex:'999999',
          background:col.overlay,
          backdropFilter:'blur(12px)',WebkitBackdropFilter:'blur(12px)',
          display:'flex',alignItems:'center',justifyContent:'center',
          fontFamily:"-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif",
          opacity:'0',transition:'opacity 0.18s ease',
        }});

        const panel = document.createElement('div');
        Object.assign(panel.style, {{
          background:col.panel,border:'1px solid '+col.border,
          borderRadius:'16px',padding:'26px 26px 20px',
          maxWidth:'520px',width:'calc(100vw - 32px)',
          maxHeight:'85vh',overflowY:'auto',
          boxShadow:'0 28px 80px rgba(0,0,0,0.65)',
          display:'flex',flexDirection:'column',gap:'14px',
          transform:'scale(0.92)',opacity:'0',
          transition:'transform 0.22s cubic-bezier(0.34,1.56,0.64,1), opacity 0.18s ease',
        }});

        const header = document.createElement('div');
        Object.assign(header.style, {{ display:'flex',alignItems:'flex-start',gap:'12px' }});
        const titleText = document.createElement('p');
        Object.assign(titleText.style, {{
          margin:'0',color:col.text,fontSize:'15px',
          fontWeight:'600',lineHeight:'1.55',flex:'1',wordBreak:'break-word',
        }});
        titleText.textContent = 'Review Proposed Plan';
        const badge = document.createElement('span');
        Object.assign(badge.style, {{
          flexShrink:'0',fontSize:'10px',fontWeight:'700',
          letterSpacing:'0.07em',padding:'3px 9px',borderRadius:'99px',
          background:col.btnPrimary+'26',color:col.btnPrimary,
          marginTop:'2px',whiteSpace:'nowrap',
        }});
        badge.textContent = 'PLAN REVIEW';
        header.appendChild(titleText); header.appendChild(badge);

        const optContainer = document.createElement('div');
        Object.assign(optContainer.style, {{ display:'flex',flexDirection:'column',gap:'7px' }});

        const tasksData = {ts};
        tasksData.forEach((t,i)=>{{
            const card = document.createElement('div');
            Object.assign(card.style, {{
              display:'flex',alignItems:'flex-start',gap:'12px',
              background:col.input,border:'1.5px solid '+col.inputBorder,
              borderRadius:'10px',padding:'11px 13px',
              minHeight:'48px',transition:'background 0.12s, border-color 0.12s',
              boxSizing:'border-box',
            }});
            const num = document.createElement('span');
            Object.assign(num.style, {{
              flexShrink:'0',width:'26px',height:'26px',
              borderRadius:'6px',background:col.btnPrimary,
              display:'flex',alignItems:'center',justifyContent:'center',
              fontSize:'11px',fontWeight:'700',color:col.btnPrimaryText,
              marginTop:'2px',userSelect:'none',
            }});
            num.textContent = String(i+1);
            const content = document.createElement('div');
            Object.assign(content.style, {{ display:'flex',flexDirection:'column',gap:'2px',flex:'1',minWidth:'0' }});
            const tid = document.createElement('span');
            Object.assign(tid.style, {{
              fontSize:'11px',fontWeight:'700',color:col.sub,
              textTransform:'uppercase',letterSpacing:'0.05em',
            }});
            tid.textContent = t.task_id;
            const desc = document.createElement('span');
            Object.assign(desc.style, {{
              fontSize:'14px',color:col.text,lineHeight:'1.5',wordBreak:'break-word',
            }});
            desc.textContent = t.description;
            content.appendChild(tid); content.appendChild(desc);
            card.appendChild(num); card.appendChild(content);
            optContainer.appendChild(card);
        }});

        const inputContainer = document.createElement('div');
        Object.assign(inputContainer.style, {{ display:'flex',flexDirection:'column',gap:'8px' }});
        const inputLabel = document.createElement('div');
        inputLabel.textContent = 'Feedback (optional):';
        Object.assign(inputLabel.style, {{
          fontSize:'11px',fontWeight:'700',color:col.sub,
          textTransform:'uppercase',letterSpacing:'0.06em',
        }});
        const feedbackInput = document.createElement('textarea');
        feedbackInput.placeholder = 'e.g., \"Add a step to check for X\" or \"Skip the second task\"';
        Object.assign(feedbackInput.style, {{
          background:col.input,border:'1.5px solid '+col.inputBorder,
          color:col.text,padding:'12px 14px',
          borderRadius:'10px',fontSize:'14px',outline:'none',
          minHeight:'64px',resize:'vertical',
          fontFamily:'inherit',boxSizing:'border-box',
          transition:'border-color 0.15s',
        }});
        feedbackInput.addEventListener('focus', ()=>{{ feedbackInput.style.borderColor=col.btnPrimary; }});
        feedbackInput.addEventListener('blur', ()=>{{ feedbackInput.style.borderColor=feedbackInput.value?col.btnPrimary:col.inputBorder; }});
        inputContainer.appendChild(inputLabel); inputContainer.appendChild(feedbackInput);

        const footer = document.createElement('div');
        Object.assign(footer.style, {{ display:'flex',gap:'8px',justifyContent:'flex-end',marginTop:'2px' }});

        function makeBtn(label,primary){{
          const b = document.createElement('button');
          b.textContent = label;
          Object.assign(b.style, {{
            padding:'10px 18px',borderRadius:'8px',
            fontSize:'13px',fontWeight:'700',
            cursor:'pointer',fontFamily:'inherit',boxSizing:'border-box',
            border:'1.5px solid '+(primary?'transparent':col.btnBorder),
            background:primary?col.btnPrimary:col.btn,
            color:primary?col.btnPrimaryText:col.btnText,
            transition:'opacity 0.12s, transform 0.1s, border-color 0.12s',
          }});
          b.addEventListener('mouseenter', ()=>{{ b.style.opacity='0.9'; b.style.transform='translateY(-1px)'; }});
          b.addEventListener('mouseleave', ()=>{{ b.style.opacity='1'; b.style.transform=''; }});
          return b;
        }}

        const acceptBtn = makeBtn('Accept Plan',true);
        const feedbackBtn = makeBtn('Send Feedback',false);
        const cancelBtn = makeBtn('Cancel',false);
        Object.assign(cancelBtn.style, {{
          background:col.btn,color:'#f38ba8',borderColor:'#f38ba8',
        }});
        cancelBtn.addEventListener('mouseenter', ()=>{{ cancelBtn.style.opacity='0.85'; cancelBtn.style.transform='translateY(-1px)'; }});
        cancelBtn.addEventListener('mouseleave', ()=>{{ cancelBtn.style.opacity='1'; cancelBtn.style.transform=''; }});

        acceptBtn.addEventListener('click', ()=>{{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'accept'}})); }});
        feedbackBtn.addEventListener('click', ()=>{{
            const val = feedbackInput.value.trim();
            if (val){{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'feedback',value:val}})); }}
            else {{ acceptBtn.click(); }}
        }});
        cancelBtn.addEventListener('click', ()=>{{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'cancel'}})); }});

        footer.appendChild(cancelBtn); footer.appendChild(feedbackBtn); footer.appendChild(acceptBtn);

        const countdown = document.createElement('div');
        countdown.textContent = '';
        Object.assign(countdown.style, {{
          fontSize:'11px',color:col.sub,textAlign:'center',minHeight:'16px',
        }});

        function onKey(e){{
          if (e.key==='Escape'){{ cancelBtn.click(); return; }}
          if (e.key==='Enter' && e.metaKey){{ acceptBtn.click(); return; }}
        }}
        document.addEventListener('keydown',onKey);

        panel.appendChild(header); panel.appendChild(optContainer); panel.appendChild(inputContainer); panel.appendChild(footer); panel.appendChild(countdown);
        overlay.appendChild(panel); document.body.appendChild(overlay);
        feedbackInput.focus();

        requestAnimationFrame(()=>{{
          requestAnimationFrame(()=>{{
            overlay.style.opacity='1';
            panel.style.transform='scale(1)';
            panel.style.opacity='1';
          }});
        }});

        let remaining = {timeout_s};
        function updateCountdown(){{
            countdown.textContent = remaining>0 ? 'Auto-accepting in '+remaining+'s...' : '';
            if (remaining<=0){{ cleanup(); resolve(JSON.stringify({{action:'accept'}})); }}
        }}
        updateCountdown();
        _timer = setInterval(()=>{{ remaining--; updateCountdown(); }},1000);
      }});
    }})();"""


    def _build_iteration_limit_js(self, current_iter, max_iter, timeout_s: int = 300) -> str:
        """Build a Continue/Cancel modal for iteration limit reached."""
        return f"""
    return (function(){{
      return new Promise((resolve)=>{{
    {self._base_theme_js()}
        const OVERLAY_ID = '__owui_helix_limit__';
        const existing = document.getElementById(OVERLAY_ID);
        if (existing) existing.remove();

        function cleanup(){{
          panel.style.transform='scale(0.95)';
          panel.style.opacity='0';
          overlay.style.opacity='0';
          setTimeout(()=>{{
            overlay.remove();
            document.removeEventListener('keydown',onKey);
          }},180);
        }}

        const overlay = document.createElement('div');
        overlay.id = OVERLAY_ID;
        Object.assign(overlay.style, {{
          position:'fixed',inset:'0',zIndex:'999999',
          background:col.overlay,
          backdropFilter:'blur(12px)',WebkitBackdropFilter:'blur(12px)',
          display:'flex',alignItems:'center',justifyContent:'center',
          fontFamily:"-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif",
          opacity:'0',transition:'opacity 0.18s ease',
        }});

        const panel = document.createElement('div');
        Object.assign(panel.style, {{
          background:col.panel,border:'1px solid '+col.border,
          borderRadius:'16px',padding:'26px 26px 20px',
          maxWidth:'520px',width:'calc(100vw - 32px)',
          maxHeight:'85vh',overflowY:'auto',
          boxShadow:'0 28px 80px rgba(0,0,0,0.65)',
          display:'flex',flexDirection:'column',gap:'14px',
          transform:'scale(0.92)',opacity:'0',
          transition:'transform 0.22s cubic-bezier(0.34,1.56,0.64,1), opacity 0.18s ease',
        }});

        const header = document.createElement('div');
        Object.assign(header.style, {{ display:'flex',alignItems:'flex-start',gap:'12px' }});
        const titleText = document.createElement('p');
        Object.assign(titleText.style, {{
          margin:'0',color:col.text,fontSize:'15px',
          fontWeight:'600',lineHeight:'1.55',flex:'1',wordBreak:'break-word',
        }});
        titleText.textContent = 'Iteration Limit Reached';
        const badge = document.createElement('span');
        Object.assign(badge.style, {{
          flexShrink:'0',fontSize:'10px',fontWeight:'700',
          letterSpacing:'0.07em',padding:'3px 9px',borderRadius:'99px',
          background:col.btnPrimary+'26',color:col.btnPrimary,
          marginTop:'2px',whiteSpace:'nowrap',
        }});
        badge.textContent = 'LIMIT';
        header.appendChild(titleText); header.appendChild(badge);

        const msg = document.createElement('p');
        Object.assign(msg.style, {{
          margin:'0',color:col.sub,fontSize:'14px',lineHeight:'1.55',wordBreak:'break-word',
        }});
        msg.textContent = `The agent has used {current_iter} of {max_iter} iterations. Continue for more?`;

        const footer = document.createElement('div');
        Object.assign(footer.style, {{ display:'flex',gap:'8px',justifyContent:'flex-end',marginTop:'2px' }});

        function makeBtn(label,primary){{
          const b = document.createElement('button');
          b.textContent = label;
          Object.assign(b.style, {{
            padding:'10px 18px',borderRadius:'8px',
            fontSize:'13px',fontWeight:'700',
            cursor:'pointer',fontFamily:'inherit',boxSizing:'border-box',
            border:'1.5px solid '+(primary?'transparent':col.btnBorder),
            background:primary?col.btnPrimary:col.btn,
            color:primary?col.btnPrimaryText:col.btnText,
            transition:'opacity 0.12s, transform 0.1s, border-color 0.12s',
          }});
          b.addEventListener('mouseenter', ()=>{{ b.style.opacity='0.9'; b.style.transform='translateY(-1px)'; }});
          b.addEventListener('mouseleave', ()=>{{ b.style.opacity='1'; b.style.transform=''; }});
          return b;
        }}

        const continueBtn = makeBtn('Continue',true);
        const stopBtn = makeBtn('Stop',false);
        Object.assign(stopBtn.style, {{ background:col.btn,color:'#f38ba8',borderColor:'#f38ba8' }});
        stopBtn.addEventListener('mouseenter', ()=>{{ stopBtn.style.opacity='0.85'; stopBtn.style.transform='translateY(-1px)'; }});
        stopBtn.addEventListener('mouseleave', ()=>{{ stopBtn.style.opacity='1'; stopBtn.style.transform=''; }});

        let _timer;
        continueBtn.addEventListener('click', ()=>{{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'continue'}})); }});
        stopBtn.addEventListener('click', ()=>{{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'stop'}})); }});

        footer.appendChild(stopBtn); footer.appendChild(continueBtn);

        const countdown = document.createElement('div');
        countdown.textContent = '';
        Object.assign(countdown.style, {{
          fontSize:'11px',color:col.sub,textAlign:'center',minHeight:'16px',
        }});

        function onKey(e){{
          if (e.key==='Escape'){{ stopBtn.click(); return; }}
          if (e.key==='Enter' || e.key===' '){{ e.preventDefault(); continueBtn.click(); return; }}
        }}
        document.addEventListener('keydown',onKey);

        panel.appendChild(header); panel.appendChild(msg); panel.appendChild(footer); panel.appendChild(countdown);
        overlay.appendChild(panel); document.body.appendChild(overlay);

        requestAnimationFrame(()=>{{
          requestAnimationFrame(()=>{{
            overlay.style.opacity='1';
            panel.style.transform='scale(1)';
            panel.style.opacity='1';
          }});
        }});

        let remaining = {timeout_s};
        function updateCountdown(){{
            countdown.textContent = remaining>0 ? 'Auto-stopping in '+remaining+'s...' : '';
            if (remaining<=0){{ cleanup(); resolve(JSON.stringify({{action:'stop'}})); }}
        }}
        updateCountdown();
        _timer = setInterval(()=>{{ remaining--; updateCountdown(); }},1000);
      }});
    }})();"""


    def _build_llm_call_limit_js(self, current_calls, max_calls, timeout_s: int = 300) -> str:
        """Build a Continue/Cancel modal for LLM call limit reached."""
        return f"""
    return (function(){{
      return new Promise((resolve)=>{{
    {self._base_theme_js()}
        const OVERLAY_ID = '__owui_helix_llm_limit__';
        const existing = document.getElementById(OVERLAY_ID);
        if (existing) existing.remove();

        function cleanup(){{
          panel.style.transform='scale(0.95)';
          panel.style.opacity='0';
          overlay.style.opacity='0';
          setTimeout(()=>{{
            overlay.remove();
            document.removeEventListener('keydown',onKey);
          }},180);
        }}

        const overlay = document.createElement('div');
        overlay.id = OVERLAY_ID;
        Object.assign(overlay.style, {{
          position:'fixed',inset:'0',zIndex:'999999',
          background:col.overlay,
          backdropFilter:'blur(12px)',WebkitBackdropFilter:'blur(12px)',
          display:'flex',alignItems:'center',justifyContent:'center',
          fontFamily:"-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif",
          opacity:'0',transition:'opacity 0.18s ease',
        }});

        const panel = document.createElement('div');
        Object.assign(panel.style, {{
          background:col.panel,border:'1px solid '+col.border,
          borderRadius:'16px',padding:'26px 26px 20px',
          maxWidth:'520px',width:'calc(100vw - 32px)',
          maxHeight:'85vh',overflowY:'auto',
          boxShadow:'0 28px 80px rgba(0,0,0,0.65)',
          display:'flex',flexDirection:'column',gap:'14px',
          transform:'scale(0.92)',opacity:'0',
          transition:'transform 0.22s cubic-bezier(0.34,1.56,0.64,1), opacity 0.18s ease',
        }});

        const header = document.createElement('div');
        Object.assign(header.style, {{ display:'flex',alignItems:'flex-start',gap:'12px' }});
        const titleText = document.createElement('p');
        Object.assign(titleText.style, {{
          margin:'0',color:col.text,fontSize:'15px',
          fontWeight:'600',lineHeight:'1.55',flex:'1',wordBreak:'break-word',
        }});
        titleText.textContent = 'LLM Call Limit Reached';
        const badge = document.createElement('span');
        Object.assign(badge.style, {{
          flexShrink:'0',fontSize:'10px',fontWeight:'700',
          letterSpacing:'0.07em',padding:'3px 9px',borderRadius:'99px',
          background:col.btnPrimary+'26',color:col.btnPrimary,
          marginTop:'2px',whiteSpace:'nowrap',
        }});
        badge.textContent = 'LIMIT';
        header.appendChild(titleText); header.appendChild(badge);

        const msg = document.createElement('p');
        Object.assign(msg.style, {{
          margin:'0',color:col.sub,fontSize:'14px',lineHeight:'1.55',wordBreak:'break-word',
        }});
        msg.textContent = `The agent has used {current_calls} of {max_calls} LLM calls. Continue for more?`;

        const footer = document.createElement('div');
        Object.assign(footer.style, {{ display:'flex',gap:'8px',justifyContent:'flex-end',marginTop:'2px' }});

        function makeBtn(label,primary){{
          const b = document.createElement('button');
          b.textContent = label;
          Object.assign(b.style, {{
            padding:'10px 18px',borderRadius:'8px',
            fontSize:'13px',fontWeight:'700',
            cursor:'pointer',fontFamily:'inherit',boxSizing:'border-box',
            border:'1.5px solid '+(primary?'transparent':col.btnBorder),
            background:primary?col.btnPrimary:col.btn,
            color:primary?col.btnPrimaryText:col.btnText,
            transition:'opacity 0.12s, transform 0.1s, border-color 0.12s',
          }});
          b.addEventListener('mouseenter', ()=>{{ b.style.opacity='0.9'; b.style.transform='translateY(-1px)'; }});
          b.addEventListener('mouseleave', ()=>{{ b.style.opacity='1'; b.style.transform=''; }});
          return b;
        }}

        const continueBtn = makeBtn('Continue',true);
        const stopBtn = makeBtn('Stop',false);
        Object.assign(stopBtn.style, {{ background:col.btn,color:'#f38ba8',borderColor:'#f38ba8' }});
        stopBtn.addEventListener('mouseenter', ()=>{{ stopBtn.style.opacity='0.85'; stopBtn.style.transform='translateY(-1px)'; }});
        stopBtn.addEventListener('mouseleave', ()=>{{ stopBtn.style.opacity='1'; stopBtn.style.transform=''; }});

        let _timer;
        continueBtn.addEventListener('click', ()=>{{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'continue'}})); }});
        stopBtn.addEventListener('click', ()=>{{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'stop'}})); }});

        footer.appendChild(stopBtn); footer.appendChild(continueBtn);

        const countdown = document.createElement('div');
        countdown.textContent = '';
        Object.assign(countdown.style, {{
          fontSize:'11px',color:col.sub,textAlign:'center',minHeight:'16px',
        }});

        function onKey(e){{
          if (e.key==='Escape'){{ stopBtn.click(); return; }}
          if (e.key==='Enter' || e.key===' '){{ e.preventDefault(); continueBtn.click(); return; }}
        }}
        document.addEventListener('keydown',onKey);

        panel.appendChild(header); panel.appendChild(msg); panel.appendChild(footer); panel.appendChild(countdown);
        overlay.appendChild(panel); document.body.appendChild(overlay);

        requestAnimationFrame(()=>{{
          requestAnimationFrame(()=>{{
            overlay.style.opacity='1';
            panel.style.transform='scale(1)';
            panel.style.opacity='1';
          }});
        }});

        let remaining = {timeout_s};
        function updateCountdown(){{
            countdown.textContent = remaining>0 ? 'Auto-stopping in '+remaining+'s...' : '';
            if (remaining<=0){{ cleanup(); resolve(JSON.stringify({{action:'stop'}})); }}
        }}
        updateCountdown();
        _timer = setInterval(()=>{{ remaining--; updateCountdown(); }},1000);
      }});
    }})();"""

    async def emit_task_update(self, finalize_tasks=False):
        """Emit task progress via Open WebUI's native task list UI, debounced.

        Events are suppressed if the task state is unchanged since the last emit,
        unless finalize_tasks is True, which forces an update.
        """
        if not self.task_list:
            tasks = []
        else:
            blocked = self._compute_blocked_tasks()
            first_outstanding = self._get_current_task_index()
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

        if not finalize_tasks:
            state_key = json.dumps(tasks, sort_keys=True)
            if state_key == self._last_task_state_str:
                return
            self._last_task_state_str = state_key

        if self.event_emitter:
            try:
                await self.event_emitter({
                    "type": "chat:message:tasks",
                    "data": {"tasks": tasks},
                })
            except Exception:
                pass

    def _build_past_tasks(self):
        lines = []
        failed_set = {i for i, task in enumerate(self.task_list) if any(f["task"] == task for f in self.failed_tasks)}
        for i, task in enumerate(self.task_list):
            if task in self.completed_tasks:
                lines.append(f"  [{i}] {task}")
            elif i in failed_set:
                reason = next((f["reason"] for f in self.failed_tasks if f["task"] == task), "")
                lines.append(f"  [{i}] {task} [FAIL: {reason}]")
        return "\n".join(lines) if lines else "(none)"

    def _build_current_task(self):
        current_idx = self._get_current_task_index()
        if current_idx is not None and 0 <= current_idx < len(self.task_list):
            return f"  [{current_idx}] {self.task_list[current_idx]}"
        return "(none)"

    def _build_future_tasks(self):
        lines = []
        completed_set = self.completed_tasks
        failed_names = {f["task"] for f in self.failed_tasks}
        for i, task in enumerate(self.task_list):
            if task not in completed_set and task not in failed_names and i != self._get_current_task_index():
                # Blocked check
                deps = self.task_dependencies.get(i, [])
                blocked = False
                for dep in deps:
                    if dep < len(self.task_list) and self.task_list[dep] not in completed_set:
                        blocked = True
                        break
                if blocked:
                    lines.append(f"  [{i}] {task} [blocked]")
                else:
                    lines.append(f"  [{i}] {task}")
        return "\n".join(lines) if lines else "(none)"

    def _build_task_state(self):
        lines = []
        past = self._build_past_tasks()
        current = self._build_current_task()
        future = self._build_future_tasks()

        lines.append("COMPLETED / DONE:")
        lines.append(past)
        lines.append("")
        lines.append("CURRENT:")
        lines.append(current)
        lines.append("")
        lines.append("FUTURE:")
        lines.append(future)
        lines.append("")
        lines.append(f"Completed: {len(self.completed_tasks)}/{len(self.task_list)}")
        return "\n".join(lines)


    async def _apply_file_prep(self, msgs: list) -> list:
        """Mirror OWUI middleware: add_file_context then chat_completion_files_handler."""
        if not self.request or not self.user or not self.pipe_metadata:
            return msgs

        prep = copy.deepcopy(msgs)
        try:
            prep = await add_file_context(prep, self.chat_id, self.user)
            logger.debug("add_file_context succeeded")
        except Exception as e:
            logger.warning("add_file_context failed: %s", e)

        try:
            udump = self.user.model_dump() if hasattr(self.user, "model_dump") else {}
            extra = {
                "__event_emitter__": self.event_emitter,
                "__metadata__": self.pipe_metadata,
                "__user__": udump,
                "__request__": self.request,
            }
            body, _flags = await chat_completion_files_handler(
                self.request,
                {"messages": prep, "model": self.body.get("model", "")},
                extra,
                self.user,
            )
            prep = body.get("messages", prep)
        except Exception as e:
            logger.warning("chat_completion_files_handler failed: %s", e)

        return prep


    async def _append_produced_files(self, raw_files: list) -> list:
        """Deduplicate and append tool-generated files under lock. Return unique new files."""
        if not raw_files:
            return []
        new_files = []
        async with self._files_lock:
            for f in raw_files:
                if not isinstance(f, dict):
                    continue
                fid = f.get("id") or f.get("file_id") or f.get("url")
                if fid and fid not in self._seen_file_ids:
                    self._seen_file_ids.add(fid)
                    new_files.append(f)
            if new_files:
                self.produced_files.extend(new_files)
                mfiles = self.metadata.get("__files__")
                if isinstance(mfiles, list):
                    mfiles.extend(new_files)
                elif mfiles is None:
                    self.metadata["__files__"] = new_files.copy()
                raw_meta = self.metadata.get("__metadata__")
                if isinstance(raw_meta, dict):
                    meta_files = raw_meta.get("files")
                    if isinstance(meta_files, list):
                        meta_files.extend(new_files)
                    else:
                        raw_meta["files"] = self.produced_files.copy()
        return new_files

    async def _delayed_sync_with_backoff(
        self,
        chat_id: str,
        message_id: str,
        expected_files: list,
        max_attempts: int = 4,
        base_delay: float = 0.5,
    ):
        """Retries file sync with exponential backoff until DB reflects expected files."""
        if not HAS_DB_PERSISTENCE or not chat_id or not message_id:
            return
        expected_ids = {
            f.get("id") or f.get("file_id") or f.get("url")
            for f in expected_files
            if isinstance(f, dict)
        }
        if not expected_ids:
            return
        for attempt in range(max_attempts):
            delay = base_delay * (2 ** attempt)  # 0.5, 1.0, 2.0, 4.0
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            try:
                current_message = await Chats.get_message_by_id_and_message_id(
                    chat_id, message_id
                )
                if current_message:
                    current_ids = {
                        f.get("id") or f.get("file_id") or f.get("url")
                        for f in current_message.get("files", [])
                        if isinstance(f, dict)
                    }
                    missing = expected_ids - current_ids
                    if not missing:
                        logger.debug(
                            f"[Helix] Delayed sync attempt {attempt + 1}: all {len(expected_ids)} files already in DB."
                        )
                        return
                    logger.info(
                        f"[Helix] Delayed sync attempt {attempt + 1}: {len(missing)} files missing. Writing..."
                    )
                await Chats.add_message_files_by_id_and_message_id(
                    chat_id, message_id, expected_files
                )
                logger.info(f"[Helix] Delayed sync succeeded at attempt {attempt + 1}.")
                return
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"[Helix] Delayed sync attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    logger.error(
                        f"[Helix] Delayed sync exhausted all {max_attempts} attempts."
                    )

    async def _sync_produced_files_to_db(self) -> bool:
        """Deduplicate and bind all tool-generated files to the chat DB message.

        Returns True if the sync succeeded (or there was nothing to sync),
        False if a DB error occurred and a retry may be warranted.
        """
        if not HAS_DB_PERSISTENCE or not self.chat_id or not self.message_id:
            return True

        # Deduplicate by id/file_id/url
        async with self._files_lock:
            unique_files = self._dedupe_files(self.produced_files)
            self.produced_files = unique_files.copy()

        if not unique_files:
            return True

        try:
            await Chats.add_message_files_by_id_and_message_id(
                self.chat_id,
                self.message_id,
                unique_files,
            )
            logger.info(f"Synced {len(unique_files)} tool files to DB.")
            return True
        except Exception as e:
            logger.warning(f"Tool file DB sync failed: {e}")
            return False


    def _validate_tool_args(self, spec: dict, args: dict) -> list:
        """Validate args against a tool's JSON schema spec. Returns list of error strings."""
        errors = []
        parameters = spec.get("parameters") or spec.get("inputSchema")
        if not isinstance(parameters, dict):
            return errors

        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            return errors

        required = set(parameters.get("required", []) if isinstance(parameters.get("required"), list) else [])
        for key in required:
            if key not in args:
                errors.append(f"Missing required argument '{key}'")

        for key, value in args.items():
            if key not in properties:
                if parameters.get("additionalProperties") is False:
                    errors.append(f"Unknown argument '{key}'")
                continue

            prop_schema = properties[key]
            if not isinstance(prop_schema, dict):
                continue

            expected_type = prop_schema.get("type")
            if expected_type and not self._check_json_type(value, expected_type):
                errors.append(f"Argument '{key}' must be of type '{expected_type}', got '{type(value).__name__}'")

            enum_values = prop_schema.get("enum")
            if isinstance(enum_values, list) and value not in enum_values:
                errors.append(f"Argument '{key}' must be one of {enum_values}, got '{value}'")

        return errors

    @staticmethod
    def _check_json_type(value, expected_type: str) -> bool:
        """Check if a Python value matches a JSON Schema type string."""
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "array":
            return isinstance(value, list)
        if expected_type == "object":
            return isinstance(value, dict)
        if expected_type == "null":
            return value is None
        return True

    async def _execute_tool(self, tool_name, args, call_id):
        """Execute a single resolved tool from phase_tools_dict."""
        target = self.phase_tools_dict.get(tool_name)
        if not target:
            available = list(self.phase_tools_dict.keys())
            return f"Tool '{tool_name}' not found in current phase ({self.phase}). Available: {', '.join(available[:20])}", []

        def _get_allowed_keys(spec: dict) -> set:
            """Extract allowed arg keys from OpenAI or MCP tool schema."""
            parameters = spec.get("parameters") or spec.get("inputSchema")
            if not isinstance(parameters, dict):
                return set()
            props = parameters.get("properties")
            return set(props.keys()) if isinstance(props, dict) else set()

        allowed_keys = _get_allowed_keys(target.get("spec", {}))
        if allowed_keys:
            filtered_args = {k: v for k, v in args.items() if k in allowed_keys}
        else:
            # If we can't determine schema, pass args through and rely on the tool to error
            filtered_args = dict(args)

        validation_errors = self._validate_tool_args(target.get("spec", {}), filtered_args)
        if validation_errors:
            error_msg = "\n".join(f"- {err}" for err in validation_errors)
            return f"{tool_name} validation failed:\n{error_msg}", []

        callable_fn = target.get("callable")
        if not callable_fn:
            return f"Tool '{tool_name}' has no executable handler.", []

        preview = _extract_tool_preview(tool_name, filtered_args, self._tool_status_hints)
        if preview:
            await self.emit_status(f">> {preview}", done=False)

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


    def _render_prompt(self, template: str, **kwargs) -> str:
        """Safely substitute known placeholders in a prompt template.

        Uses str.replace() so any literal braces in user-provided custom
        prompts (JSON examples, regex, shell variables) are left untouched.
        """
        result = template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    def _build_tool_catalog(self) -> str:
        """Generate a plain list of all non-internal (external) tools.

        Returns a simple bullet list — no headers, no intros.
        """
        external_tools = {
            name: tool for name, tool in self.all_tools_dict.items()
            if name not in self.INTERNAL_TOOLS
        }
        if not external_tools:
            return "(none)"

        lines = []
        for name in sorted(external_tools.keys()):
            tool = external_tools[name]
            spec = tool.get("spec", {})
            desc = spec.get("description", "No description")
            desc = desc.replace("\n", " ").replace("\r", " ")
            desc = " ".join(desc.split())
            if len(desc) > 200:
                desc = desc[:197] + "..."
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    def _build_system_prompt(self):
        """Build system prompt based on current phase using Valves overrides."""
        task_state = self._build_task_state()

        # Build loop counter info for all phases
        phase_name = {
            self.PHASE_PLAN: "PLAN",
            self.PHASE_REPLAN: "REPLAN",
            self.PHASE_EXECUTE: "EXECUTE",
            self.PHASE_REVIEW: "REVIEW",
            self.PHASE_OUTPUT: "OUTPUT",
        }.get(self.phase, "LOOP")

        max_loops = 5
        loop_info = f"[Agent State] Phase: {phase_name} | Loop: {self.loop_count}/{max_loops}"

        base = ""
        if self.phase == self.PHASE_PLAN:
            tool_catalog = self._build_tool_catalog()
            base = self._render_prompt(
                self.valves.PLAN_PROMPT or DEFAULT_PLAN_PROMPT,
                loop_info=loop_info,
            )
            if tool_catalog:
                base = f"{tool_catalog}\n\n{base}"
        elif self.phase == self.PHASE_REPLAN:
            tool_catalog = self._build_tool_catalog()
            base = self._render_prompt(
                DEFAULT_REPLAN_PROMPT,
                goal=self.goal,
                reason=self._replan_reason,
                task_state=task_state,
                loop_info=loop_info,
            )
            if tool_catalog:
                base = f"{tool_catalog}\n\n{base}"
        elif self.phase == self.PHASE_EXECUTE:
            past_tasks = self._build_past_tasks()
            current_task = self._build_current_task()
            future_tasks = self._build_future_tasks()
            base = self._render_prompt(
                self.valves.EXECUTE_PROMPT or DEFAULT_EXECUTE_PROMPT,
                past_tasks=past_tasks,
                current_task=current_task,
                future_tasks=future_tasks,
                loop_info=loop_info,
            )
        elif self.phase == self.PHASE_REVIEW:
            base = self._render_prompt(
                self.valves.REVIEW_PROMPT or DEFAULT_REVIEW_PROMPT,
                goal=self.goal,
                task_state=task_state,
                loop_info=loop_info,
            )
        elif self.phase == self.PHASE_OUTPUT:
            if self._output_turn >= 2:
                base = self._render_prompt(
                    DEFAULT_OUTPUT_FINAL_PROMPT,
                    goal=self.goal,
                    task_state=task_state,
                    loop_info=loop_info,
                )
            else:
                base = self._render_prompt(
                    self.valves.OUTPUT_PROMPT or DEFAULT_OUTPUT_PROMPT,
                    goal=self.goal,
                    task_state=task_state,
                    loop_info=loop_info,
                )
        else:
            base = self._render_prompt(DEFAULT_PLAN_PROMPT, loop_info=loop_info)

        base = base.replace("[USER_HOME]", self._get_user_home())

        if self._memory_context:
            base = f"{self._memory_context}\n\n{base}"

        if self._skill_prompt:
            base = f"{self._skill_prompt}\n\n{base}"
        return base


    def _get_user_home(self):
        """Resolve the actual Linux user home directory for the OpenWebUI user.

        OpenWebUI may run as root or in a container, so ``os.path.expanduser('~')``
        often returns ``/root``.  We look at the real user record from the
        OpenWebUI ``Users`` table first.  If that doesn't yield a usable home
        we fall back to ``getent passwd`` followed by any directory under
        ``/home/`` (excluding ``/home/root``).
        """
        # 1. Try the OpenWebUI user object first.
        user_name = None
        if self.user:
            if hasattr(self.user, "name"):
                user_name = self.user.name
            elif isinstance(self.user, dict):
                user_name = self.user.get("name")

            if user_name:
                # getent passwd is the safest way to resolve a real home dir.
                try:
                    import subprocess
                    result = subprocess.run(
                        ["getent", "passwd", user_name],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        parts = result.stdout.strip().split(":")
                        if len(parts) >= 6:
                            home = parts[5]
                            if home and os.path.isdir(home):
                                return home
                except Exception:
                    pass

        # 2. Fallback: look for any directory under /home/ that isn't /home/root.
        try:
            home_dirs = sorted(
                [d for d in os.listdir("/home")
                 if os.path.isdir(os.path.join("/home", d)) and d != "root"],
                key=lambda x: os.stat(os.path.join("/home", x)).st_mtime,
                reverse=True,
            )
            for d in home_dirs:
                candidate = os.path.join("/home", d)
                if os.path.isdir(candidate):
                    return candidate
        except Exception:
            pass

        # 3. Last resort – whatever expanduser gives us.
        return os.path.expanduser("~")

    # -----------------------------------------------------------------
    # OpenWebUI middleware integration helpers (citations, terminal,
    # memory, filter pipeline, RAG template) - best-effort, fails safe.
    # -----------------------------------------------------------------

    async def _emit_terminal_event(self, tool_name: str, args: dict, result_str: str):
        """Best-effort terminal event emission after file/command tools."""
        if not self.event_emitter or not HAS_MIDDLEWARE_CITATIONS:
            return
        try:
            await terminal_event_handler(tool_name, args, result_str, self.event_emitter)
        except Exception as e:
            logger.debug(f"terminal_event_handler failed for {tool_name}: {e}")

    async def _emit_citation_source(self, tool_name: str, args: dict, result_str: str):
        """Best-effort emission of source/citation events for search/fetch/kb tools."""
        if not self.event_emitter or not HAS_MIDDLEWARE_CITATIONS:
            return
        try:
            sources = get_citation_source_from_tool_result(tool_name, args, result_str)
            if sources:
                for source in sources:
                    await self.event_emitter({"type": "source", "data": source})
        except Exception as e:
            logger.debug(f"get_citation_source_from_tool_result failed for {tool_name}: {e}")

    async def _inject_memory_context(self, user_msg: str) -> str:
        """Query user memories and return a short system-style injection string.

        Returns an empty string if memories are unavailable or the feature is off.
        """
        if not HAS_MEMORY or not self.request or not self.user:
            return ""
        try:
            k = getattr(self.valves, "MEMORY_QUERY_K", 3)
            form = QueryMemoryForm(content=user_msg, k=k)
            mem_task = asyncio.create_task(
                query_memory(self.request, form, self.user)
            )
            mem_result = await mem_task
            if mem_result and isinstance(mem_result, dict):
                memories = mem_result.get("memories", [])
                if memories:
                    lines = ["[User Memories - the following facts have been saved from previous conversations:]"]
                    for m in memories:
                        content = m.get("content", "") if isinstance(m, dict) else str(m)
                        if content:
                            lines.append(f"- {content}")
                    return "\n".join(lines)
            return ""
        except Exception as e:
            logger.debug(f"Memory injection failed: {e}")
            return ""

    async def _apply_filter_pipeline(self, messages: list, user_msg: str) -> list:
        """Run OpenWebUI inlet filter on the message payload before LLM call.

        Returns the potentially modified messages. Falls back to the originals on error.
        """
        if not HAS_INLET_FILTER or not self.request or not self.user:
            return messages
        try:
            payload = {
                **self.body,
                "messages": messages,
            }
            filtered_payload = await process_pipeline_inlet_filter(
                self.request, payload, self.user
            )
            if isinstance(filtered_payload, dict) and "messages" in filtered_payload:
                return filtered_payload["messages"]
        except Exception as e:
            logger.debug(f"process_pipeline_inlet_filter failed: {e}")
        return messages

    async def _apply_rag_template(self, messages: list, sources: list, user_msg: str) -> list:
        """Best-effort RAG template application with source context.

        Only active when the middleware module exposes apply_source_context_to_messages.
        Falls back to the original messages when unavailable.
        """
        if not HAS_MIDDLEWARE_CITATIONS or not self.request or not self.user:
            return messages
        try:
            updated = await apply_source_context_to_messages(
                self.request, messages, sources, user_msg
            )
            if isinstance(updated, list):
                return updated
        except Exception as e:
            logger.debug(f"apply_source_context_to_messages failed: {e}")
        return messages

    def _transition_to(self, phase):
        """Transition to a new phase: update tools, system prompt, state."""
        self.phase = phase
        if phase == self.PHASE_OUTPUT:
            self._output_turn = 0
            self._output_rendering_skipped = False
            skip_rendering = getattr(self.user_valves, "SKIP_OUTPUT_RENDERING", True)
            if skip_rendering and "display_file" not in self._incoming_tools:
                self._output_rendering_skipped = True
                self._output_turn = 1  # Bypass RENDER so first loop is treated as SUMMARY
                asyncio.create_task(self.emit_status("Skipping RENDER (display_file not available)"))
        # Rebuild filtered tools for new phase
        self._filter_tools_for_phase(phase)


    def _manage_context_window(self, messages):
        """Trim history to MAX_HISTORY_MESSAGES while keeping tool call pairs intact."""
        max_history = getattr(self.valves, "MAX_HISTORY_MESSAGES", 100)
        if len(messages) <= max_history:
            return messages

        to_remove = len(messages) - max_history
        head = messages[:1]
        # Non-state messages after head, split into removed and tail
        non_state = messages[1:]
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
        if not getattr(self.valves, "ENABLE_TOOL_TRUNCATION", True):
            return 0
        user_limit = getattr(self.user_valves, "MAX_TOOL_RESULT_CHARS", -1) if self.user_valves else -1
        if user_limit == -1:
            return self.valves.MAX_TOOL_RESULT_CHARS
        return user_limit

    async def _call_compression_model(self, messages: list) -> str:
        """Call LLM for context compression. Uses CONTEXT_COMPRESSION_MODEL or falls back to AGENT_MODEL."""
        model = self.valves.CONTEXT_COMPRESSION_MODEL or self.valves.AGENT_MODEL
        if not model:
            logger.warning("No compression model configured; skipping compression.")
            return ""

        # --- API budget + session timeout guards ---
        if not self._is_yolo_mode:
            max_llm = getattr(self.valves, "MAX_LLM_CALLS", 100)
            if max_llm > 0 and self._llm_call_count >= max_llm + self._extra_llm_grace:
                logger.warning("Skipping compression: LLM call budget exhausted.")
                return ""
        if await self._check_timeouts_or_abort():
            return ""

        try:
            body = {
                **self.body,
                "model": model,
                "messages": messages,
                "stream": False,
            }
            self._llm_call_count += 1
            response = await generate_chat_completion(self.request, body, self.user)
            if isinstance(response, dict) and response.get("choices"):
                raw = response["choices"][0].get("message", {}).get("content", "")
                return raw.strip()
            return ""
        except Exception as e:
            logger.warning(f"Compression model call failed: {e}")
            return ""

    def _format_messages_for_compression(self, messages: list) -> str:
        """Format a list of conversation messages into a single text for LLM compression."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            if role == "tool":
                name = msg.get("name", "unknown")
                content = strip_html(msg.get("content", "")[:1000])
                lines.append(f"[Tool: {name}]\n{content}")
            elif role == "assistant" and msg.get("tool_calls"):
                content = msg.get("content", "")
                tool_names = ", ".join(tc.get("function", {}).get("name", "?") for tc in msg.get("tool_calls", []))
                lines.append(f"[Assistant: calling tools -> {tool_names}]\n{content}")
            else:
                content = strip_html(msg.get("content", ""))
                lines.append(f"[{role.capitalize()}]\n{content}")
        return "\n\n".join(lines)

    @staticmethod
    def _find_safe_compression_split(messages: list, keep_recent: int) -> int:
        """Find an index to split messages so that tool-call pairs are never broken.

        The split must never land inside a tool-call sequence.  Specifically:
        - If a message with role 'tool' would be cut off, its preceding assistant
          message (the one containing the matching tool_calls) must also be kept.
        - If an assistant message with tool_calls is at the split boundary,
          all of its corresponding 'tool' result messages must also be kept.

        Returns the split index (number of messages to include in the OLD part).
        """
        if len(messages) <= keep_recent + 1:
            return 0

        # Start from a naive split point
        split_idx = len(messages) - keep_recent

        # Gather all tool_call IDs from assistant messages at/before the boundary
        assistant_call_ids = set()
        for i in range(split_idx):
            msg = messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id")
                    if tc_id:
                        assistant_call_ids.add(tc_id)

        # Find tool result messages that fall AFTER the split but whose
        # assistant call was BEFORE the split -> move split forward
        while split_idx < len(messages):
            msg = messages[split_idx]
            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id and tc_id in assistant_call_ids:
                    # This tool result belongs to an assistant BEFORE the split,
                    # so we must keep it -> move split forward
                    split_idx += 1
                    continue
            # If the message at split_idx is an assistant with tool_calls,
            # ensure all its tool results are after the split too.
            # If some tool results are BEFORE the split, we can't leave a
            # dangling assistant call either -> move split back
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                missing_tool = False
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id")
                    if tc_id:
                        # Look for the matching tool result in the old part
                        found_in_old = any(
                            m.get("role") == "tool" and m.get("tool_call_id") == tc_id
                            for m in messages[:split_idx]
                        )
                        if found_in_old:
                            missing_tool = True
                            break
                if missing_tool:
                    # Can't split here because the old part already contains
                    # a tool result for this assistant. Move split forward to
                    # include this assistant in the new part.
                    split_idx += 1
                    continue
            break

        return split_idx

    async def _compress_history_llm(self) -> bool:
        """
        Blocking LLM-based history compression.
        Interrupts the loop, compresses old messages into a single assistant summary message.
        Returns True if compression happened, False otherwise.
        """
        keep_recent = getattr(self.valves, "KEEP_RECENT_MESSAGES", 6)
        threshold = self._get_history_compression_threshold()

        total_tokens = self._total_history_tokens()
        if total_tokens <= threshold:
            return False

        if len(self.history) <= keep_recent + 1:
            return False

        split_idx = self._find_safe_compression_split(self.history, keep_recent)
        old_messages = self.history[:split_idx]
        recent_messages = self.history[split_idx:]

        if not old_messages:
            return False

        old_tokens = sum(self._estimate_tokens(str(m.get("content", ""))) for m in old_messages)
        await self.emit_status(f"Compressing history ({old_tokens} tokens)...", done=False)

        # Format old messages for compression prompt
        conversation_text = self._format_messages_for_compression(old_messages)
        compression_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a conversation compressor for an AI coding agent. "
                    "Summarize the following conversation part concisely while preserving ALL important context: "
                    "file paths, key decisions, errors, tool names and their decisive results, user preferences, "
                    "and any unfinished work. Output as a dense narrative. Keep it under 2000 words."
                ),
            },
            {"role": "user", "content": conversation_text},
        ]

        compressed = await self._call_compression_model(compression_prompt)
        if not compressed:
            await self.emit_status("Compression failed, continuing...", done=True)
            return False

        compressed_tokens = self._estimate_tokens(compressed)
        recent_tokens = sum(self._estimate_tokens(str(m.get("content", ""))) for m in recent_messages)
        saved = old_tokens - compressed_tokens
        saved_pct = round((saved / total_tokens) * 100) if total_tokens else 0

        # Build the compressed context message as a system message
        # so it does not interfere with assistant/tool conversation flow.
        compressed_msg = {
            "role": "assistant",
            "content": f"[Compressed Context Summary]\n{compressed}",
        }

        self.history = [compressed_msg] + recent_messages

        await self.emit_status(f"History compressed: saved {saved_pct}% ({saved} tokens)", done=True)
        return True

    async def _compress_goal_if_needed(self):
        """Compress the goal after the agent run completes if it exceeds the threshold."""
        threshold = self._get_goal_compression_threshold()
        goal_tokens = self._estimate_tokens(self.goal)
        if goal_tokens <= threshold:
            return

        prompt = [
            {
                "role": "system",
                "content": (
                    "Compress the following user goal into a concise instruction. "
                    "Preserve all technical requirements, file paths, constraints, and the latest request."
                ),
            },
            {"role": "user", "content": self.goal},
        ]

        compressed = await self._call_compression_model(prompt)
        if compressed and len(compressed) < len(self.goal):
            original_tokens = self._estimate_tokens(self.goal)
            self.goal = compressed
            new_tokens = self._estimate_tokens(self.goal)
            saved = original_tokens - new_tokens
            saved_pct = round((saved / original_tokens) * 100)
            logger.info(f"Goal compressed: saved {saved_pct}% ({saved} tokens)")




    async def _run_impl(self, user_msg, last_user_msg_raw, model):
        await self.resolve_tools()

        if getattr(self.valves, "DEBUG_MODE", False):
            msg_lower = (user_msg or "").strip().lower()
            if msg_lower in ("show tools", "debug tools", "list tools"):
                builtin_tools = []
                external_tools = []
                mcp_tools = []
                terminal_tools = []
                helix_tools = []

                for name, info in self.all_tools_dict.items():
                    if name in self.INTERNAL_TOOLS:
                        helix_tools.append(name)
                        continue
                    tool_type = info.get("type", "") if isinstance(info, dict) else ""
                    if tool_type == "builtin":
                        builtin_tools.append(name)
                    elif tool_type == "mcp":
                        mcp_tools.append(name)
                    elif tool_type == "terminal":
                        terminal_tools.append(name)
                    else:
                        external_tools.append(name)

                builtin_tools.sort()
                external_tools.sort()
                mcp_tools.sort()
                terminal_tools.sort()
                helix_tools.sort()

                def _fmt(name):
                    base = f"- {name}"
                    return base

                sections = []
                sections.append(f"**Tools from Pipe (__tools__):** {len(self._incoming_tools)}")

                if builtin_tools:
                    sections.append(
                        f"**Built-in Tools ({len(builtin_tools)}):**\n"
                        + "\n".join(_fmt(t) for t in builtin_tools)
                    )
                else:
                    sections.append("**Built-in Tools:** none")

                if external_tools:
                    sections.append(
                        f"**External / Workspace Tools ({len(external_tools)}):**\n"
                        + "\n".join(_fmt(t) for t in external_tools)
                    )
                else:
                    sections.append("**External / Workspace Tools:** none")

                if mcp_tools:
                    sections.append(
                        f"**MCP Tools ({len(mcp_tools)}):**\n"
                        + "\n".join(_fmt(t) for t in mcp_tools)
                    )
                else:
                    sections.append("**MCP Tools:** none")

                if terminal_tools:
                    sections.append(
                        f"**Terminal Tools ({len(terminal_tools)}):**\n"
                        + "\n".join(_fmt(t) for t in terminal_tools)
                    )
                else:
                    sections.append("**Terminal Tools:** none")

                if helix_tools:
                    sections.append(
                        f"**Helix Tools ({len(helix_tools)}):**\n"
                        + "\n".join(_fmt(t) for t in helix_tools)
                    )
                else:
                    sections.append("**Helix Tools:** none")

                return "\n\n".join(sections)

        await self.emit_status("Agent starting...")

        await self._recover_state_from_files(self.body if isinstance(self.body, dict) else {})

        # --- Memory injection (once, before first plan LLM call) ---
        if getattr(self.valves, "ENABLE_MEMORY_INJECTION", True) and not self._memory_injected:
            self._memory_context = await self._inject_memory_context(user_msg)
            if self._memory_context:
                self._memory_injected = True
                logger.info("Injected user memory context into system prompt.")

        max_size_mb = getattr(self.valves, "MAX_ATTACHMENT_SIZE_MB", 5)
        if max_size_mb > 0 and self.request:
            max_bytes = max_size_mb * 1024 * 1024
            incoming_files = list(self.metadata.get("__files__") or [])
            if self.body and isinstance(self.body, dict):
                body_files = self.body.get("files") or self.body.get("__files__")
                if body_files:
                    seen = {f.get("id") or f.get("file_id") for f in incoming_files if isinstance(f, dict)}
                    for f in body_files:
                        if isinstance(f, dict):
                            fid = f.get("id") or f.get("file_id")
                            if fid and fid not in seen:
                                incoming_files.append(f)
                                seen.add(fid)
            oversized = []
            for file_info in incoming_files:
                if not isinstance(file_info, dict):
                    continue
                file_size = file_info.get("size")
                if file_size is None and HAS_DB_PERSISTENCE:
                    fid = file_info.get("file_id") or file_info.get("id")
                    if fid:
                        try:
                            file_obj = await Files.get_file_by_id(fid)
                            if file_obj:
                                fpath = getattr(file_obj, "path", None)
                                if fpath and os.path.exists(fpath):
                                    file_size = os.path.getsize(fpath)
                        except Exception:
                            pass
                if file_size and file_size > max_bytes:
                    oversized.append((file_info.get("name", "unknown"), file_size))
            if oversized:
                items = "\n".join(f"- `{name}` ({size / (1024*1024):.1f} MB)" for name, size in oversized)
                err = (
                    f"**Error: File(s) too large ({max_size_mb} MB max)**\n\n"
                    f"The following attached file(s) exceed the maximum allowed size ({max_size_mb} MB):\n"
                    f"{items}\n\n"
                    f"Please upload large documents to a Knowledge Base instead."
                )
                await self.emit_status("File too large", done=True)
                return err

        # If every task is completed or failed, treat this as a brand-new request.
        has_remaining_tasks = False
        if self.task_list:
            failed_names = {f["task"] for f in self.failed_tasks}
            remaining = [
                t for t in self.task_list
                if t not in self.completed_tasks and t not in failed_names
            ]
            has_remaining_tasks = bool(remaining)

        if self.goal and self.task_list and has_remaining_tasks:
            max_allowed = max(0, self.valves.MAX_ITERATIONS - 5)
            if self.loop_count > max_allowed:
                logger.info(f"Clamped loop_count from {self.loop_count} to {max_allowed} after state restore.")
                self.loop_count = max_allowed
            self.goal = f"{self.goal}; Updated: {user_msg}"
            self.history.append({"role": "user", "content": user_msg})
            self._filter_tools_for_phase(self.phase)
            await self.emit_task_update()
        elif self.goal and not self.task_list and self.phase == self.PHASE_REPLAN:
            logger.info("Resuming interrupted Replan phase.")
            self.loop_count = 0
            self._plan_questions_asked = 0
            if user_msg not in self.goal:
                self.goal = self.goal + "\n\nNEW REQUEST:\n" + user_msg
            self.history.append({"role": "user", "content": user_msg})
            self._filter_tools_for_phase(self.PHASE_REPLAN)
            await self.emit_task_update()
        elif self.goal and self.task_list and not has_remaining_tasks and self.user_valves and getattr(self.user_valves, "SKIP_PLAN_ON_RESUME", True):
            logger.info("Previous session finished; entering Replan phase.")
            self.loop_count = 0
            self._plan_questions_asked = 0
            self.task_list = []
            self.completed_tasks = []
            self.failed_tasks = []
            self.goal = self.goal + "\n\nNEW REQUEST:\n" + user_msg
            self.phase = self.PHASE_REPLAN
            self.history.append({"role": "user", "content": user_msg})
            self._filter_tools_for_phase(self.PHASE_REPLAN)
            await self.emit_task_update()
        else:
            if self.goal and self.task_list and not has_remaining_tasks:
                logger.info("All tasks completed or failed; starting fresh session.")
            self.goal = user_msg
            self.phase = self.PHASE_PLAN
            self.task_list = []
            self.completed_tasks = []
            self.failed_tasks = []
            self.loop_count = 0
            self._plan_questions_asked = 0
            self._total_tool_calls = 0
            self._filter_tools_for_phase(self.PHASE_PLAN)
            self.history = [last_user_msg_raw if last_user_msg_raw else {"role": "user", "content": user_msg}]

        recent_calls = []
        self._output_parts = []

        while True:
            # --- Global session guards: time budget + LLM call budget ---
            if await self._check_timeouts_or_abort():
                return self._format_output()

            effective_max = self.valves.MAX_ITERATIONS + self._extra_grace

            if self.phase == self.PHASE_OUTPUT:
                self._output_turn += 1
                if self._output_turn == 1:
                    has_rendering_tools = any(
                        name not in self.INTERNAL_TOOLS
                        for name in self.phase_tools_dict.keys()
                    )
                    if not has_rendering_tools:
                        self._output_turn = 2
                        await self.emit_status("No rendering tools configured - skipping RENDER")
                if self._output_turn > 2 and not self._is_yolo_mode:
                    await self.emit_task_update(finalize_tasks=True)
                    await self.emit_status("Output exceeded max turns", done=True)
                    return self._format_output()

            if self.loop_count >= effective_max and not self._is_yolo_mode:
                await self.emit_output(f"\n[WARN] Max iterations ({effective_max}) reached.")
                should_continue = False
                if self.event_call and not (self.user_valves and getattr(self.user_valves, "YOLO_MODE", False)):
                    try:
                        timeout_s = getattr(self.valves, "ITERATION_LIMIT_TIMEOUT", 300)
                        js = self._build_iteration_limit_js(self.loop_count, effective_max, timeout_s=timeout_s)
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
                    self._extra_grace += self.valves.MAX_ITERATIONS
                    await self.emit_status("Continuing after iteration limit...")
                    continue
                await self.emit_task_update(finalize_tasks=True)
                await self.emit_status("Max iterations", done=True)
                return self._format_output()

            # --- LLM call budget guard with continue dialog ---
            max_llm = getattr(self.valves, "MAX_LLM_CALLS", 100)
            if max_llm > 0 and self._llm_call_count >= max_llm + self._extra_llm_grace and not self._is_yolo_mode:
                await self.emit_output(f"\n[WARN] Max LLM calls ({max_llm + self._extra_llm_grace}) reached.")
                should_continue = False
                if self.event_call and not (self.user_valves and getattr(self.user_valves, "YOLO_MODE", False)):
                    try:
                        timeout_s = getattr(self.valves, "ITERATION_LIMIT_TIMEOUT", 300)
                        js = self._build_llm_call_limit_js(self._llm_call_count, max_llm + self._extra_llm_grace, timeout_s=timeout_s)
                        raw = await self.event_call({"type": "execute", "data": {"code": js}})
                        raw_str = raw if isinstance(raw, str) else (raw.get("result") or raw.get("value") or "{}") if raw else "{}"
                        try:
                            res = json.loads(raw_str) if isinstance(raw_str, str) and raw_str.startswith("{") else {}
                        except (json.JSONDecodeError, AttributeError):
                            res = {}
                        should_continue = res.get("action") == "continue"
                    except Exception as e:
                        logger.warning(f"LLM call limit dialog failed: {e}")
                if should_continue:
                    self._extra_llm_grace += max_llm
                    await self.emit_status("Continuing after LLM call limit...")
                    continue
                await self.emit_task_update(finalize_tasks=True)
                await self.emit_status("Max LLM calls", done=True)
                return self._format_output()

            self.loop_count += 1
            recent_calls = recent_calls[-30:]

            max_replan_loops = getattr(self.valves, "MAX_REPLAN_LOOPS", 3)
            if self.phase == self.PHASE_REPLAN and self.loop_count >= max_replan_loops and not self._is_yolo_mode:
                logger.warning("REPLAN reached %d loops; aborting with error.", max_replan_loops)
                await self.emit_output(f"\n[ERROR] Replanning failed after {max_replan_loops} attempts. The agent could not generate a valid task plan. Please rephrase your request or check your instructions.\n")
                await self.emit_task_update(finalize_tasks=True)
                await self.emit_status("Planning failed", done=True)
                return self._format_output()

            phase_name = {
                self.PHASE_PLAN: "Plan",
                self.PHASE_REPLAN: "Replan",
                self.PHASE_EXECUTE: "Execute",
                self.PHASE_REVIEW: "Review",
                self.PHASE_OUTPUT: "Output",
            }
            name = phase_name.get(self.phase, "Loop")
            if self.phase == self.PHASE_OUTPUT:
                sub = "RENDER" if self._output_turn == 1 else ("SUMMARY" if self._output_turn == 2 else "Output")
                name = f"{phase_name[self.PHASE_OUTPUT]} ({sub})"

            effective_max = self.valves.MAX_ITERATIONS + self._extra_grace
            token_status = f", {self._format_token_status()}"
            await self.emit_status(f"Mode: {name}, Loop: {self.loop_count}/{effective_max}{token_status}")


            self.history = self._manage_context_window(self.history)

            total_tokens = self._total_history_tokens()
            if total_tokens > self._get_history_compression_threshold():
                if self.loop_count - getattr(self, "_last_compression_loop", 0) >= self.valves.COMPRESSION_INTERVAL:
                    compressed = await self._compress_history_llm()
                    if compressed:
                        self._last_compression_loop = self.loop_count

            system_prompt = self._build_system_prompt()
            call_messages = [{"role": "system", "content": system_prompt}] + [m for m in self.history if m.get("role") != "system"]

            # ── Filter pipeline (inlet) before LLM call ──
            if getattr(self.valves, "ENABLE_FILTER_PIPELINE", True):
                call_messages = await self._apply_filter_pipeline(call_messages, user_msg)

            completion_body = {
                **self.body,
                "model": model,
                "messages": call_messages,
                "tools": self.phase_tools_specs if self.phase_tools_specs else None,
                "metadata": self.pipe_metadata,
            }

            if self.phase == self.PHASE_OUTPUT and self._output_turn >= 2:
                completion_body["tools"] = None
                completion_body["response_format"] = OUTPUT_FINAL_JSON_SCHEMA

            mk = getattr(self, "_model_knowledge", None)
            if mk:
                completion_body.setdefault("metadata", {})
                if isinstance(completion_body.get("metadata"), dict):
                    completion_body["metadata"]["knowledge"] = mk
                    completion_body["metadata"]["__model_knowledge__"] = mk

            try:
                completion_body["messages"] = await self._apply_file_prep(copy.deepcopy(call_messages))
            except Exception as e:
                logger.warning("_apply_file_prep failed: %s", e)

            tc_dict = {}
            content_chunks = []

            max_llm_retries = getattr(self.valves, "LLM_RETRY_COUNT", 1)
            max_iter_seconds = getattr(self.valves, "MAX_ITERATION_SECONDS", 300)
            iter_deadline = None
            if max_iter_seconds > 0:
                iter_deadline = asyncio.get_event_loop().time() + max_iter_seconds

            retryable_error_occurred = False
            try:
                async for event in self._stream_completion(
                    completion_body, max_retries=max_llm_retries, iter_deadline=iter_deadline
                ):
                    etype = event.get("type")
                    if etype == "error":
                        error_text = event.get("text", "Unknown")
                        if event.get("retryable"):
                            # Retryable stream error: feed back to the model and try again
                            await self.emit_status(f"LLM stream recovered: {error_text[:100]}...")
                            self.history.append({
                                "role": "user",
                                "content": f"SYSTEM: LLM call was interrupted by a transient error ({error_text}). The stream timed out or disconnected. Please retry your last action.",
                            })
                            retryable_error_occurred = True
                            break
                        else:
                            # Non-retryable (budget/session exhausted) – fatal
                            await self.emit_output(f"\n[ERROR] LLM Error: {error_text}")
                            await self.emit_task_update(finalize_tasks=True)
                            await self.emit_status("Error", done=True)
                            return self._format_output()
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
            except asyncio.CancelledError:
                raise

            # Skip the rest of this iteration if a retryable error was handled above
            if retryable_error_occurred:
                continue

            content = strip_thinking("".join(content_chunks).strip())

            if not tc_dict:
                if self.phase in (self.PHASE_PLAN, self.PHASE_REPLAN):
                    self.history.append({
                        "role": "assistant",
                        "content": content or "",
                    })
                    max_plan_loops = 5
                    # Increment reprompt counter so we don't loop forever on stubborn models
                    self._plan_reprompt_count = getattr(self, '_plan_reprompt_count', 0) + 1
                    max_reprompts = getattr(self.valves, "MAX_PLAN_REPROMPTS", 2)
                    if self._plan_reprompt_count > max_reprompts:
                        await self.emit_output(f"\n[ERROR] The agent failed to produce a valid plan after multiple attempts.\n")
                        await self.emit_task_update(finalize_tasks=True)
                        await self.emit_status("Planning failed", done=True)
                        return self._format_output()
                    urgency = ""
                    if self.loop_count >= max_plan_loops - 1:
                        urgency = f" URGENT: You are at loop {self.loop_count} of {max_plan_loops}. You MUST call confirm_plan NOW with your best plan. Do NOT ask more questions or produce more text."
                    self.history.append({
                        "role": "system",
                        "content": f"CRITICAL REMINDER: You are in {self.phase.upper()} mode. You MUST call one of these tools ONLY: ask_user, terminate, or confirm_plan. Any other output is rejected.{urgency}",
                    })
                    await self.emit_output(f"\n[WARN] No tool call produced in {self.phase} phase. Re-prompting to enforce confirm_plan.\n")
                    continue
                if self.phase == self.PHASE_OUTPUT:
                    if self._output_turn == 1:
                        # RENDER with no tools: skip text and proceed to SUMMARY
                        self.history.append({
                            "role": "assistant",
                            "content": content or "",
                        })
                        continue
                    if content:
                        parsed = self._parse_output_json(content)
                        if parsed:
                            rendered = self._render_output_markdown(parsed)
                            if rendered:
                                await self.emit_output(rendered)
                            else:
                                summary = parsed.get("summary", "")
                                if summary:
                                    await self.emit_output(summary)
                                else:
                                    await self.emit_output(content)
                        else:
                            await self.emit_output(content)
                    await self.emit_task_update(finalize_tasks=True)
                    await self.emit_status("Done", done=True)
                    return self._format_output()
                if self.phase == self.PHASE_REVIEW:
                    self.history.append({
                        "role": "assistant",
                        "content": content or "",
                    })
                    self.history.append({
                        "role": "user",
                        "content": "SYSTEM: You produced text but did not call any tools. In REVIEW phase you MUST call exactly one of: proceed_to_output(), fix_plan(reason, updated_tasks), or replan(reason). Do NOT output text-call the appropriate tool.",
                    })
                    await self.emit_output(f"\n[WARN] No tool call produced in REVIEW phase. Re-prompting to enforce transition tool.\n")
                    continue
                if content and self.task_list and len(self.completed_tasks) < len(self.task_list):
                    # Tasks remain: inject continuation prompt instead of terminating
                    self.history.append({
                        "role": "assistant",
                        "content": content,
                    })
                    self.history.append({
                        "role": "user",
                        "content": "SYSTEM: You produced text but did not call any tools. You have unfinished tasks. Continue working by calling the appropriate tool. Do NOT just describe what to do - call a tool.",
                    })
                    await self.emit_output(f"\n[WARN] No tool call produced. Re-prompting to continue.\n")
                    continue
                if content:
                    await self.emit_output(content)
                await self.emit_status("Done", done=True)
                return self._format_output()

            tool_calls_list = list(tc_dict.values())
            self.history.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls_list,
            })

            # ── EXECUTE phase: enforce one action per turn ──
            # Allow run_tools_parallel (it counts as one tool call that executes multiple sub-calls internally).
            # allow_multiple_tools is True only for that case or for internal state-transition tools (complete_task,
            # fail_task, terminate, replan) which naturally should not coexist with external tools anyway.
            if self.phase == self.PHASE_EXECUTE:
                # collect names
                names = [tc.get("function", {}).get("name", "") for tc in tool_calls_list]
                # run_tools_parallel is allowed because it is a single orchestrated call
                is_parallel = len(names) == 1 and names[0] == "run_tools_parallel"
                if not is_parallel and len(tool_calls_list) > 1:
                    # pick the first non-internal tool (or the first overall as fallback)
                    chosen_idx = 0
                    for i, n in enumerate(names):
                        if n not in self.INTERNAL_TOOLS:
                            chosen_idx = i
                            break
                    chosen_tc = tool_calls_list[chosen_idx]
                    rejected_names = [n for j, n in enumerate(names) if j != chosen_idx]
                    # Inject a system reminder so next turn the LLM knows why extra calls were dropped
                    rejected_tool_list = ", ".join(rejected_names)
                    reminder = (
                        f"SYSTEM: You attempted to call multiple independent tools in the same turn. "
                        f"Only '{chosen_tc.get('function',{}).get('name')}' was executed. "
                        f"You dropped: {rejected_tool_list}. "
                        f"If you need {len(rejected_names) + 1} or more independent tools, use run_tools_parallel. "
                        f"Do not make separate tool calls in a single turn."
                    )
                    self.history.append({"role": "system", "content": reminder})
                    tool_calls_list = [chosen_tc]
                    await self.emit_output(f"\n[WARN] EXECUTE: Only one tool per turn. Executed '{names[chosen_idx]}', dropped: {rejected_names}.\n")

            _pending_phase_transition = None
            for tc in tool_calls_list:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                call_id = tc.get("id", str(uuid.uuid4()))

                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    error_detail = (
                        f"Error: Invalid JSON in tool arguments for '{tool_name}'. "
                        f"The arguments provided were: {raw_args}. "
                        f"Please ensure they are a valid JSON object with exactly the keys expected by this tool."
                    )
                    await self.emit_status(f"JSON parse error: {tool_name}")
                    args = {}
                    self.history.append({
                        "role": "tool",
                        "content": error_detail,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    await self._save_state_to_file()
                    continue

                if tool_name == "terminate":
                    result = args.get("result", "Task complete.")
                    success = args.get("success", True)
                    icon = "[OK]" if success else "[FAIL]"
                    await self._save_state_to_file(force=True)
                    await self.emit_task_update(finalize_tasks=True)
                    if content:
                        await self.emit_output(content + "\n\n")
                    await self.emit_output(f"{icon} **Finished:** {result}")
                    await self.emit_status("Finished", done=True)
                    return self._format_output()

                if tool_name == "replan":
                    reason = args.get("reason", "Plan needs adjustment")

                    recent_calls = []
                    result_json = await self._tool_replan(reason=reason)
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    await self.emit_output(f"\n[RPLN] **Re-planning:** {reason}\n")
                    await self.emit_status(f"[RPLN] Re-planning: {reason}")
                    continue

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
                    if (len(self.completed_tasks) + len(self.failed_tasks)) >= len(self.task_list):
                        _pending_phase_transition = self.PHASE_REVIEW
                        await self.emit_status("Moving to review phase...")
                    continue

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
                    if self.failed_tasks and len(self.completed_tasks) + len(self.failed_tasks) >= len(self.task_list):
                        _pending_phase_transition = self.PHASE_REVIEW
                        await self.emit_status("Moving to review phase...")
                    continue

                if tool_name == "proceed_to_output":
                    result_json = await self._tool_proceed_to_output(**args)
                    result_data = json.loads(result_json)
                    if result_data.get("proceed_to_output"):
                        _pending_phase_transition = self.PHASE_OUTPUT
                        await self.emit_status("Moving to output phase...")
                    else:
                        await self.emit_status(f"Output transition failed: {result_data.get('error', 'Unknown error')[:120]}")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    continue

                if tool_name == "fix_plan":
                    result_json = await self._tool_fix_plan(**args)
                    result_data = json.loads(result_json)
                    if result_data.get("fix_plan"):
                        await self.emit_output(f"\n[FIX] **Plan fixed:** {result_data.get('reason', '')}\n")
                        await self.emit_output(f"[FIX] Inserted tasks: {', '.join(result_data.get('inserted_tasks', []))}\n")
                    else:
                        await self.emit_status(f"[FIX] Fix failed: {result_data.get('error', 'Unknown error')[:120]}")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    if self.phase in (self.PHASE_REVIEW, self.PHASE_OUTPUT):
                        _pending_phase_transition = self.PHASE_EXECUTE
                    continue

                if tool_name == "confirm_plan":
                    result_json = await self._tool_confirm_plan(**args)
                    result_data = json.loads(result_json)
                    action = result_data.get("action", "")

                    if action in ("error", "timeout"):
                        error_msg = result_data.get("error", "Plan confirmation failed.")
                        # Feed error back to the model so it can self-correct instead of aborting.
                        self.history.append({
                            "role": "tool",
                            "content": result_json,
                            "tool_call_id": call_id,
                            "name": tool_name,
                        })
                        self.history.append({
                            "role": "user",
                            "content": f"SYSTEM: Plan confirmation encountered an error ({error_msg}). Please call confirm_plan(tasks=[...]) again with a proper task list.",
                        })
                        await self.emit_status(f"Plan confirmation error: {error_msg[:120]}")
                        continue

                    if action == "cancel":
                        await self.emit_output("\nThe plan was cancelled by the user. The agent will not proceed.\n")
                        await self.emit_task_update(finalize_tasks=True)
                        await self.emit_status("Plan cancelled", done=True)
                        return self._format_output()

                    if action == "feedback":
                        feedback = result_data.get("value", "")
                        self.history.append({
                            "role": "tool",
                            "content": result_json,
                            "tool_call_id": call_id,
                            "name": tool_name,
                        })
                        self.history.append({
                            "role": "user",
                            "content": f"SYSTEM: User provided feedback on the proposed plan: {feedback}. Please revise the plan and call confirm_plan(tasks=[...]) again with the updated task list.",
                        })
                        await self.emit_output(f"\n[PLAN] Plan rejected - user feedback: {feedback}\n")
                        await self.emit_status("[PLAN] Revising plan based on feedback...")
                        continue

                    tasks = args.get("tasks", [])
                    if not isinstance(tasks, list):
                        tasks = []
                    # Defensive fallback: if task_list ends up empty, abort planning
                    if not tasks:
                        self.history.append({
                            "role": "user",
                            "content": "SYSTEM: confirm_plan was called but produced an empty task list. You MUST provide a concrete, actionable list of tasks. Call confirm_plan again.",
                        })
                        await self.emit_status("Plan confirmation returned empty task list")
                        continue
                    self.task_list = [str(t) for t in tasks]
                    _pending_phase_transition = self.PHASE_EXECUTE
                    await self._save_state_to_file()
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

                if tool_name == "run_tools_parallel":
                    result_json = await self._tool_run_tools_parallel(**args)
                    result_data = json.loads(result_json)
                    if result_data.get("error"):
                        await self.emit_status(f"Parallel execution: {result_data['error'][:200]}")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    continue

                if tool_name == "ask_user":
                    result_json = await self._tool_ask_user(**args)
                    result_data = json.loads(result_json)
                    user_response = result_data.get("response", "")
                    skipped = result_data.get("skipped", False)
                    if skipped:
                        await self.emit_output(f"\n[ASK] **User question skipped:** {user_response}\n")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    continue

                sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                if recent_calls.count(sig) >= 2:
                    tool_result = f"Error: Identical call to `{tool_name}` repeated. Try a different approach."
                else:
                    recent_calls.append(sig)
                    await self.emit_status(f"Running: {tool_name}...")

                    validation_failed = False
                    target_tool = self.phase_tools_dict.get(tool_name)
                    if target_tool:
                        v_errs = self._validate_tool_args(target_tool.get("spec", {}), args)
                        if v_errs:
                            tool_result = (
                                f"Validation error for `{tool_name}`:\n"
                                + "\n".join(f"- {e}" for e in v_errs)
                            )
                            validation_failed = True

                    if not validation_failed:
                        try:
                            result_str, result_files = await self._execute_tool(tool_name, args, call_id)
                        except Exception as exec_err:
                            # Unhandled exception from tool – recoverable loop-level error
                            await self.emit_status(f"Tool error: {tool_name} threw an exception. Recovering...")
                            self.history.append({
                                "role": "tool",
                                "content": f"[ERR] Tool '{tool_name}' encountered an unexpected error: {exec_err}. Please adjust your arguments and try again.",
                                "tool_call_id": call_id,
                                "name": tool_name,
                            })
                            continue

                        new_files = await self._append_produced_files(result_files)
                        if new_files and self.event_emitter:
                            await self.event_emitter({
                                "type": "chat:message:files",
                                "data": {"files": new_files},
                            })
                        truncation_limit = self._get_truncation_limit()
                        was_truncated = False
                        original_len = len(result_str) if result_str else 0
                        if truncation_limit and result_str and len(result_str) > truncation_limit:
                            was_truncated = True
                            await self.emit_status(
                                f"Truncated result for {tool_name} ({original_len} -> {truncation_limit} chars)"
                            )
                            tool_result = smart_truncate(result_str, truncation_limit)
                        else:
                            tool_result = result_str

                        if was_truncated:
                            tool_result = (
                                f"[TRUNCATED] Tool '{tool_name}' result was cut from {original_len} to {truncation_limit} chars. "
                                f"If you need the full output, refine your arguments or run a more targeted query.\n\n"
                                f"{tool_result}"
                            )
                        self._total_tool_calls += 1

                        # ── Terminal sync for file/command tools ──
                        if getattr(self.valves, "ENABLE_TERMINAL_SYNC", True) and tool_name in (
                            "display_file", "write_file", "replace_file_content", "run_command", "replace_note_content", "write_note"
                        ) and result_str:
                            await self._emit_terminal_event(tool_name, args, result_str)

                        # ── Citation / source events for search/fetch/kb tools ──
                        if getattr(self.valves, "ENABLE_CITATIONS", True) and tool_name in (
                            "search_web", "fetch_url", "query_knowledge_files", "view_knowledge_file", "query_knowledge_bases", "view_knowledge_bases"
                        ) and result_str:
                            await self._emit_citation_source(tool_name, args, result_str)
                            if getattr(self.valves, "ENABLE_RAG_TEMPLATE", False):
                                try:
                                    from open_webui.utils.middleware import get_source_context
                                    src = get_source_context(tool_name, args, result_str)
                                    if src:
                                        self._rag_sources.append(src)
                                except Exception:
                                    pass

                        if "not found in current phase" in tool_result:
                            available_tools = ", ".join(sorted(self.phase_tools_dict.keys())[:30])
                            tool_result = (
                                f"Tool '{tool_name}' is not available in the current phase ({self.phase}). "
                                f"Available tools for this phase: {available_tools}. "
                                f"Please check the tool name and use only tools listed for this phase."
                            )
                            await self.emit_status(f"Tool unavailable: {tool_name}")

                self.history.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": call_id,
                    "name": tool_name,
                })

            if _pending_phase_transition:
                self._transition_to(_pending_phase_transition)
            else:
                if self.phase == self.PHASE_EXECUTE and (len(self.completed_tasks) + len(self.failed_tasks)) >= len(self.task_list):
                    self._transition_to(self.PHASE_REVIEW)

            if self.phase == self.PHASE_OUTPUT and self._output_turn == 1:
                continue

    async def run(self, user_msg, last_user_msg_raw, model):
        try:
            result = await self._run_impl(user_msg, last_user_msg_raw, model)
            await self._compress_goal_if_needed()
            return result
        except GeneratorExit:
            logger.info("Agent loop cancelled by user (GeneratorExit).")
            await self._save_state_to_file(force=True)
            await self.emit_task_update(finalize_tasks=True)
            await self.emit_status("Cancelled", done=True)
            raise
        except asyncio.CancelledError:
            logger.info("Agent loop cancelled (CancelledError).")
            await self._save_state_to_file(force=True)
            await self.emit_task_update(finalize_tasks=True)
            await self.emit_status("Cancelled", done=True)
            raise
        except Exception as e:
            logger.error(f"Agent loop error: {e}", exc_info=True)
            await self.emit_task_update(finalize_tasks=True)
            return f"\n[ERROR] Agent loop failed: {e}"
        finally:
            sync_success = await self._sync_produced_files_to_db()
            if not sync_success and HAS_DB_PERSISTENCE and self.chat_id and self.message_id:
                # Only schedule backup retry when the immediate sync actually failed
                snapshot = list(self.produced_files)
                asyncio.get_running_loop().create_task(
                    self._delayed_sync_with_backoff(
                        self.chat_id, self.message_id, snapshot
                    )
                )
            if self._total_tool_calls > 0:
                summary = f"\n[Session Stats] Tools: {self._total_tool_calls} | Loops: {self.loop_count}"
                await self.emit_output(summary)
                await self.emit_status(f"Session: {self._format_token_status()}", done=True)
            self._seen_file_ids.clear()
            self.produced_files.clear()

class Pipe:
    class Valves(BaseModel):
        AGENT_MODEL: str = Field(
            default="",
            description="Model ID for Helix Agent. The model MUST support native tool calling."
        )
        CONTEXT_COMPRESSION_MODEL: str = Field(
            default="",
            description="Model ID for context compression. If empty, falls back to AGENT_MODEL.",
        )

        MAX_ITERATIONS: int = Field(
            default=100,
            description="Maximum Helix Agent iterations before stopping."
        )
        MAX_REPLAN_LOOPS: int = Field(
            default=3,
            ge=0,
            description="Safety cap: after this many REPLAN loops the agent falls back to single-task EXECUTE.",
        )
        LLM_RETRY_COUNT: int = Field(
            default=1,
            ge=0,
            description="Number of retries for transient LLM API errors (ConnectionError, TimeoutError). Set to 0 to disable retries.",
        )
        TOOL_TIMEOUT: int = Field(
            default=90,
            description="Timeout in seconds for individual tool execution. Set to 0 to disable."
        )

        MAX_SESSION_SECONDS: int = Field(
            default=1200,
            ge=0,
            description="Hard session lifetime limit in seconds. If exceeded, the agent shuts down immediately regardless of state. Default 1200 (20 min). Set to 0 to disable."
        )
        STREAM_CHUNK_TIMEOUT_SECONDS: int = Field(
            default=60,
            ge=0,
            description="Max seconds to wait for the next chunk from the LLM stream. If the model produces no output for this duration, the stream is aborted as retryable. Default 60. Set to 0 to disable.",
        )
        MAX_ITERATION_SECONDS: int = Field(
            default=900,
            ge=0,
            description="Hard per-iteration timeout in seconds (one trip through the main loop). If exceeded, the agent shuts down immediately. Default 900 (15 min). Set to 0 to disable."
        )
        MAX_LLM_CALLS: int = Field(
            default=100,
            ge=0,
            description="Absolute maximum number of LLM API calls (generate_chat_completion) allowed per session. When reached, a continue dialog appears. Default 100. Set to 0 to disable."
        )

        CONTEXT_LENGTH: int = Field(
            default=128000,
            ge=1000,
            description="Context window length in tokens. A single valve that drives all adaptive compression thresholds.",
        )
        CHARS_PER_TOKEN_ESTIMATE: float = Field(
            default=3.5,
            ge=1.0,
            description="Estimated characters per token for the active model. Used to convert token-based context limits (CONTEXT_LENGTH) into character-based internal thresholds.",
        )
        COMPRESSION_INTERVAL: int = Field(
            default=5,
            ge=1,
            description="Minimum loop iterations between consecutive history compressions to avoid token thrashing.",
        )
        KEEP_RECENT_MESSAGES: int = Field(
            default=6,
            ge=2,
            description="Number of recent messages to always keep uncompressed in history. Older messages are candidates for compression.",
        )
        MAX_HISTORY_MESSAGES: int = Field(
            default=100,
            ge=10,
            description="Maximum total conversation messages retained in context. Older messages are dropped while keeping tool-call pairs intact.",
        )

        EXECUTE_TOOLS: str = Field(
            default=(
                "calculate_timestamp, create_calendar_event, delete_calendar_event, "
                "fetch_url, get_current_timestamp, get_process_status, "
                "glob_search, grep_search, kill_process, list_files, list_knowledge_bases, "
                "list_memories, list_processes, query_knowledge_bases, query_knowledge_files, "
                "read_file, replace_file_content, replace_note_content, run_command, "
                "search_calendar_events, search_channel_messages, search_channels, search_chats, "
                "search_knowledge_bases, search_knowledge_files, search_memories, "
                "search_notes, search_web, send_process_input, update_calendar_event, "
                "view_channel_message, view_channel_thread, view_chat, "
                "view_knowledge_file, view_note, view_skill, write_file, write_note"
            ),
            description=(
                "Comma-separated tool names allowed in EXECUTE phase. "
                "Leave EMPTY to allow ALL tools. "
                "Default excludes write-memory tools (add_memory, delete_memory, replace_memory_content)."
            )
        )
        REVIEW_TOOLS: str = Field(
            default=(
                "calculate_timestamp, get_current_timestamp, get_process_status, "
                "glob_search, grep_search, list_files, list_knowledge_bases, list_memories, "
                "list_processes, query_knowledge_bases, query_knowledge_files, read_file, "
                "run_command, search_calendar_events, search_channel_messages, search_channels, "
                "search_chats, search_knowledge_bases, search_knowledge_files, "
                "search_memories, search_notes, "
                "view_channel_message, view_channel_thread, view_chat, "
                "view_knowledge_file, view_note, view_skill"
            ),
            description=(
                "Comma-separated tool names allowed in REVIEW phase. "
                "Leave EMPTY to allow ALL tools. "
                "Default is a read-only / review-safe set + command execution."
            )
        )
        OUTPUT_TOOLS: str = Field(
            default="display_file",
            description=(
                "Comma-separated rendering/visualization tool names allowed in OUTPUT phase RENDER turn. "
                "Default is display_file for rendering produced files."
            )
        )

        PLAN_PROMPT: str = Field(
            default=DEFAULT_PLAN_PROMPT,
            description="System prompt for PLAN phase. Available placeholders: {tool_info}, {loop_info}."
        )
        EXECUTE_PROMPT: str = Field(
            default=DEFAULT_EXECUTE_PROMPT,
            description="System prompt for EXECUTE phase. Available placeholders: {tool_info}, {past_tasks}, {current_task}, {future_tasks}, {loop_info}."
        )
        REVIEW_PROMPT: str = Field(
            default=DEFAULT_REVIEW_PROMPT,
            description="System prompt for REVIEW phase. Available placeholders: {goal}, {task_state}, {loop_info}."
        )
        OUTPUT_PROMPT: str = Field(
            default=DEFAULT_OUTPUT_PROMPT,
            description="System prompt for OUTPUT phase - RENDER turn. Only rendering/visualisation tools are called here. Available placeholders: {goal}, {task_state}, {loop_info}.",
        )

        ENABLE_TOOL_TRUNCATION: bool = Field(
            default=True,
            description="If True, tool results are truncated to MAX_TOOL_RESULT_CHARS. If False, truncation is completely disabled and full tool results are passed to the LLM regardless of size.",
        )
        MAX_TOOL_RESULT_CHARS: int = Field(
            default=12000,
            description="Max characters for tool results before truncation. This valve is character-based (not token-based) so that tool output limits remain directly inspectable and predictable.",
        )
        MAX_ATTACHMENT_SIZE_MB: int = Field(
            default=5,
            description="Maximum allowed size of individual attached files in megabytes. Files larger than this will trigger an error at the start of the conversation, suggesting the user upload them to a Knowledge Base instead. Set to 0 to disable the size check."
        )
        PLAN_APPROVAL_TIMEOUT: int = Field(
            default=600,
            ge=0,
            description="Timeout in seconds for the plan approval modal. After this time the plan is auto-approved.",
        )
        ITERATION_LIMIT_TIMEOUT: int = Field(
            default=300,
            ge=0,
            description="Timeout in seconds for the iteration limit Continue/Cancel modal. After this time the agent auto-stops.",
        )

        DEBUG_MODE: bool = Field(
            default=False,
            description="Enable debug mode. When on, messages 'show tools' return the available items directly without any LLM call.",
        )
        TOOL_STATUS_MAP: str = Field(
            default='{"run_command": {"label": "Run", "params": ["command"]}, "read_file": {"label": "Read", "params": ["file", "path", "file_path"]}, "write_file": {"label": "Write", "params": ["file", "path", "file_path"]}, "list_files": {"label": "List", "params": ["path", "dir", "directory"]}, "search_web": {"label": "Search", "params": ["query"]}, "replace_file_content": {"label": "Edit", "params": ["file", "path", "file_path"]}, "fetch_url": {"label": "Fetch", "params": ["url"]}, "glob_search": {"label": "Glob", "params": ["pattern"]}, "grep_search": {"label": "Grep", "params": ["pattern"]}, "display_file": {"label": "Show", "params": ["file", "path", "file_path"]}}',
            description=(
                "JSON object that maps tool names to status preview hints."
                " This setting is only effective when set to a custom value."
                " Format: {\"tool_name\": {\"label\": \"Run\", \"params\": [\"command\"]}}."
                " 'params' is a priority list; the first found arg is shown."
                " Non-listed tools produce no preview. Overrides/extends defaults."
            ),
        )

        ENABLE_CITATIONS: bool = Field(
            default=True,
            description="If True, emits source/citation events after web_search, fetch_url, query_knowledge_files, and view_knowledge_file tool calls so users see clickable sources."
        )
        ENABLE_TERMINAL_SYNC: bool = Field(
            default=True,
            description="If True, emits terminal events after write_file, replace_file_content, display_file, and run_command so the OpenWebUI Terminal tab refreshes automatically."
        )
        ENABLE_MEMORY_INJECTION: bool = Field(
            default=True,
            description="If True and the OpenWebUI memory module is available, injects relevant user memories into the system prompt before the first plan."
        )
        ENABLE_FILTER_PIPELINE: bool = Field(
            default=True,
            description="If True and the OpenWebUI inlet filter is available, runs process_pipeline_inlet_filter on the chat payload before each LLM call so admin-configured security/compliance filters are honoured."
        )
        ENABLE_RAG_TEMPLATE: bool = Field(
            default=False,
            description="If True and the OpenWebUI RAG middleware is available, formats source context from tool results via apply_source_context_to_messages instead of plain-text injection."
        )
        MEMORY_QUERY_K: int = Field(
            default=3,
            description="Number of top user memories to retrieve when ENABLE_MEMORY_INJECTION is active."
        )

    class UserValves(BaseModel):
        YOLO_MODE: bool = Field(
            default=False,
            description="Skip all user confirmations and safety limits. Auto-approve plans, ignore session timeout, iteration limit, LLM call budget, plan reprompt and ask_user restrictions.",
        )
        ENABLE_PLAN_APPROVAL: bool = Field(
            default=True,
            description="Enable plan confirmation UI. When off, plans are auto-approved without asking the user.",
        )
        SKIP_PLAN_ON_RESUME: bool = Field(
            default=True,
            description="When the previous session is finished, skip the full PLAN phase for a new user request and jump straight to a Replan. Set to False to always start fresh with full PLAN phase.",
        )
        SKIP_OUTPUT_RENDERING: bool = Field(
            default=True,
            description="If True, skip the OUTPUT phase RENDER turn (rendering/visualization collection) and go straight to the SUMMARY turn. Useful when no rendering/visualization tools are configured.",
        )

        MAX_PLAN_QUESTIONS: int = Field(
            default=3,
            description="Maximum number of clarification questions (ask_user) the agent may ask per planning phase before it is forced to finalise the plan.",
        )
        MAX_TOOL_RESULT_CHARS: int = Field(
            default=12000,
            ge=-1,
            description="Max characters for individual tool results before truncation. Admin default is 12000; users may override to a personal preference. Set to -1 to use admin default, 0 to disable truncation entirely.",
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
        __tools__: dict = None,
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
            incoming_tools=__tools__,
        )

        return await engine.run(user_msg, last_user_msg_raw, model)