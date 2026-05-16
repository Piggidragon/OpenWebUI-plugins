"""
title: Helix Agent
author: Piggidragon
version: 0.21.0
description: >
  Helix Agent — OpenWebUI-native agent loop with modular per-phase tool control.

  Architecture:
  - SINGLE model loop (Plan -> Execute -> Review -> Replan -> Execute...)
  - Per-phase tool filtering via Valves — only relevant tools exposed to the LLM at each phase
  - Internal control tools (terminate, replan, fix_plan, complete_task, fail_task, confirm_plan, rag_search) always available
  - Uses OpenWebUI native tool infrastructure (get_tools, get_builtin_tools, get_terminal_tools)
  - Context window management with adaptive history truncation and tool-call pair integrity

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
  - RAG search built-in: agent can query attached large files via vector search
  - MCP support: resolves and calls MCP server tools via OpenWebUI's MCPClient
  - Skills support: resolves user skills from model metadata and injects them into the system prompt
requirements: open-webui>=0.9.1, chromadb, sentence-transformers, langchain-text-splitters
"""

import asyncio
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
from open_webui.utils.tools import get_tools, get_builtin_tools, get_terminal_tools
from open_webui.utils.middleware import (
    process_tool_result,
    add_file_context,
    chat_completion_files_handler,
    get_system_oauth_token,
)
from open_webui.utils.mcp.client import MCPClient
from open_webui.utils.access_control import has_connection_access
from open_webui.utils.misc import is_string_allowed
from open_webui.utils.headers import include_user_info_headers
from open_webui.env import (
    ENABLE_FORWARD_USER_INFO_HEADERS,
    FORWARD_SESSION_INFO_HEADER_CHAT_ID,
    FORWARD_SESSION_INFO_HEADER_MESSAGE_ID,
)
from open_webui.models.skills import Skills

logger = logging.getLogger(__name__)

# v4.6+: OpenWebUI DB-backed state persistence
try:
    from open_webui.models.chats import Chats
    from open_webui.routers.files import upload_file_handler, Files
    from starlette.datastructures import UploadFile, Headers

    HAS_DB_PERSISTENCE = True
except Exception:
    HAS_DB_PERSISTENCE = False

# ──────────────────────────────────────────────────────────────────
#  DEFAULT PROMPTS (overridable via Valves)
# ──────────────────────────────────────────────────────────────────

DEFAULT_PLAN_PROMPT = """\
You are in PLAN mode. Your job is to understand the user's request and create a clear, actionable task plan.

PHASE: PLAN

Available tools: {tool_names}

What to do:
1. Analyse the user's request thoroughly.
2. Use your available tools to understand the scope (read files, search knowledge, etc.), but do NOT perform the actual task yet (e.g., do not write files, push code, or execute actions).
3. Create a numbered task list that covers the entire goal.
4. Each task should be a clear, actionable step.
5. After creating the plan, call confirm_plan with the plan text to present it for review.

File paths: If the plan involves creating files, decide on a project folder name (short slug based on the goal) under `/home/userxy/agent/`. All files for this task must be written within that project folder.

Plan format for confirm_plan:
When calling confirm_plan, provide the plan parameter as a numbered list with one task per line:
1. First task description
2. Second task description
3. Third task description

Alternatively, you may provide the plan as JSON: {{"tasks": ["task 1", "task 2", "task 3"]}}

Rules:
- Focus purely on planning — do NOT attempt to perform the task or execute actions.
- You may use available tools to gather context and inspect files or code relevant to the goal.
- Break complex tasks into small, verifiable steps.
- If the request is simple (1-2 tasks), still list them explicitly.
- Call exactly ONE tool per step.
- When done planning, call confirm_plan with the plan for confirmation.
- If the request is inappropriate or impossible, call terminate with a brief explanation.
- NEVER call replan in PLAN mode.
- If a tool returns an error during planning (e.g., file not found), note the limitation in your plan. Do not call fix_plan for planning-stage errors.
- If the user rejects your plan with feedback, revise the plan based on their feedback and call confirm_plan again with the updated plan. Do NOT repeat the same plan unchanged.
- If the user cancels the plan, acknowledge it and stop.
- Use the ask_user tool if you need clarification from the user (e.g., ambiguous request, missing details, unclear scope).
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

File paths: Create a project folder under `/home/userxy/agent/` named after the current task/goal (use a short slug). All files for this project must be written within that folder. Do not scatter files across unrelated directories.

Rules:
- Call exactly ONE tool per step.
- NEVER repeat identical failed tool calls (duplicate detection is active).
- When all tasks are done, the system will move to review automatically. You may also call early_finish(reason) if you believe you're done early.
- If a tool returns an error, analyze it and retry with corrected parameters. You do NOT need to call fix_plan for trivial errors.
- Only call fix_plan if the same task fails repeatedly (3+ attempts) or if the task design was wrong.
- Only call replan(mode='soft') if the entire approach is wrong. Use replan(mode='hard') only for complete strategy replacement.
- If you need to think step-by-step before acting, do so — reasoning will be captured in a collapsible block.
- You MUST call complete_task(index) or fail_task(index, reason) after working on a task.
- Use `run_tools_parallel` to call multiple independent tools at once for efficiency.
- You may only use the tools listed above. Do NOT ask the user questions.
"""

DEFAULT_REVIEW_PROMPT = """\
You are in REVIEW mode. Your ONLY job is to pick one of these actions.

PHASE: REVIEW
Original goal: {goal}
Available tools: {tool_names}

{task_state}

Task status markers: [done] = completed, [FAIL: reason] = failed with reason, [    ] = not started.

You MUST call exactly ONE of these tools:

1. `proceed_to_output()` — Everything is done and correct. Move to the OUTPUT phase to generate the polished final answer.
2. `fix_plan(reason, updated_tasks)` — Only minor fixes are needed (a task failed or needs a small correction). List just the new/corrected tasks.
3. `replan(reason, updated_tasks, mode="soft")` — The overall strategy is broken and tasks need to be replaced entirely.

Rules:
- If there are only minor issues with individual tasks, ALWAYS prefer `fix_plan` over `replan`. Only use `replan` if the overall strategy is broken.
- Be honest — don't call `proceed_to_output` if something is missing or wrong.
- If the result is good enough, call `proceed_to_output`. Don't gold-plate.
- Provide a brief reasoning for your assessment before calling the final tool.
- You may only use the tools listed above.
"""

DEFAULT_OUTPUT_PROMPT = """\
You are in OUTPUT mode — COLLECTION. Gather missing context and render visualisations if needed. Do not produce any answer text yet.

Goal: {goal}

{task_state}
"""

DEFAULT_OUTPUT_FINAL_PROMPT = """\
Write the final answer.

Goal: {goal}
"""

DEFAULT_REPLAN_SKIP_PROMPT = """\
You are in QUICK REPLAN mode. A previous session completed, and the user has a new request.

PHASE: QUICK REPLAN

Available tools: {tool_names}

Recent context (previous goal):
{goal}

What to do:
1. Review the previous goal and the user's new request.
2. Create a minimal, focused task plan (1-3 tasks) that addresses the new request in the context of what was already done.
3. Call confirm_plan with the updated plan.

Plan format for confirm_plan:
When calling confirm_plan, provide the plan parameter as a numbered list with one task per line:
1. First task description
2. Second task description

Alternatively, you may provide the plan as JSON: {{"tasks": ["task 1", "task 2"]}}

Rules:
- Keep the plan short and actionable. 1-3 tasks maximum.
- Re-use previous context where relevant.
- Call exactly ONE tool per step.
- When done, call confirm_plan to move to execution.
- NEVER call terminate in QUICK REPLAN mode.
- You may use the ask_user tool to ask the user a multiple-choice question if you need clarification.
- You may only use the tools listed above.
- ABSOLUTELY CRITICAL: You MUST call confirm_plan to finish QUICK REPLAN mode. Do NOT answer the user directly. Do NOT generate story text, code, or any other output. Only call tools.
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
    PHASE_OUTPUT = "output"
    PHASE_REPLAN_SKIP = "replan_skip"  # Quick replan when previous session finished

    MAX_HISTORY_MESSAGES = 50
    MAX_REPLAN_SKIP_LOOPS = 3  # Safety net: max loops in quick replan phase

    # Internal tools that are ALWAYS available regardless of phase filters
    INTERNAL_TOOLS = {"terminate", "replan", "complete_task", "fail_task", "confirm_plan", "fix_plan", "rag_search", "proceed_to_output", "early_finish", "run_tools_parallel"}

    PHASE_INTERNAL_TOOLS = {
        PHASE_PLAN:       {"terminate", "replan", "complete_task", "fail_task", "confirm_plan", "fix_plan", "rag_search"},
        PHASE_EXECUTE:    {"replan", "complete_task", "fail_task", "fix_plan", "rag_search", "early_finish", "run_tools_parallel"},
        PHASE_REVIEW:     {"proceed_to_output", "replan", "fix_plan", "rag_search"},
        PHASE_OUTPUT:     {"rag_search"},
        PHASE_REPLAN_SKIP:{"confirm_plan", "rag_search", "ask_user"},  # Quick replan has minimal internal tools
    }

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
        self.produced_files = []
        self._files_lock = asyncio.Lock()
        self._consecutive_tool_misses: Dict[str, int] = {}
        self._output_turn = 0
        self._seen_file_ids: Set[str] = set()
        self._output_parts = []
        self.loop_count = 0
        self.goal = ""
        self._skill_prompt = ""
        self._mcp_clients: Dict[str, Any] = {}
        self._extra_grace = 0
        self._plan_questions_asked = 0  # Counter for ask_user calls in PLAN/REPLAN_SKIP

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
            if s.startswith("[PLAN]"):
                continue
            filtered.append(part)
        return "".join(filtered)

    # ── State Persistence (DB + File Attachments) ──

    async def _save_state_to_file(self) -> None:
        """Serialize agent state to a JSON file and bind it to the chat DB."""
        if not HAS_DB_PERSISTENCE or not self.chat_id or not self.request or not self.user:
            return
        try:
            state_data = {
                "goal": self.goal,
                "task_list": self.task_list,
                "completed": self.completed_tasks,
                "failed": self.failed_tasks,
                "phase": self.phase,
                "loop_count": self.loop_count,
                "extra_grace": self._extra_grace,
            }
            filename = f"helix_state_{self.chat_id}.json"
            content = json.dumps(state_data, ensure_ascii=False).encode("utf-8")

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

            # Update internal metadata (prune old state files first)
            internal_files = self.metadata.get("__files__")
            if isinstance(internal_files, list):
                internal_files[:] = [
                    f for f in internal_files
                    if not (isinstance(f, dict) and f.get("name", "").startswith("helix_state_"))
                ]
                internal_files.append(file_info)
            else:
                self.metadata["__files__"] = [file_info]

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

            # Emit for immediate UI feedback
            if self.event_emitter:
                await self.event_emitter(
                    {"type": "chat:message:files", "data": {"files": [file_info]}}
                )
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")

    async def _recover_state_from_files(self, body: dict) -> None:
        """Restore agent state from JSON file attachments in the chat."""
        if not HAS_DB_PERSISTENCE:
            return

        state_file = None
        # 1. Look in current message attachments (body files)
        current_files = body.get("files") or body.get("__files__")
        # Also scan self.metadata files since they persist across turns
        metadata_files = self.metadata.get("__files__") or self.metadata.get("files")
        for file_list in (current_files, metadata_files):
            if file_list:
                for f in reversed(file_list):
                    name = f.get("name", f.get("filename", ""))
                    if "helix_state" in name and name.endswith(".json"):
                        state_file = f
                        break
                if state_file:
                    break

        # 2. Deep DB history scan
        if not state_file and self.chat_id:
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
                            for f in reversed(msg_files):
                                name = f.get("name", f.get("filename", ""))
                                if "helix_state" in name and name.endswith(".json"):
                                    state_file = f
                                    logger.info(f"Recovered state from DB history: {name}")
                                    break
                        if state_file:
                            break
                        current_id = msg.get("parentId")
            except Exception as e:
                logger.warning(f"DB history scan failed: {e}")

        if not state_file:
            logger.info("No Helix state file found in attachments or DB history.")
            return

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
            self.phase = data.get("phase", self.PHASE_PLAN)
            self.goal = data.get("goal", self.goal)
            self.loop_count = data.get("loop_count", 0)
            self._extra_grace = data.get("extra_grace", 0)
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

        # 0. Resolve skills from model metadata (injected into system prompt later)
        model_info = self.app_models.get(self.body.get("model", ""), {})
        skill_ids, skill_prompt = await self._resolve_model_skills(model_info)
        self._skill_prompt = skill_prompt

        # 1. External tools (DB + OpenAPI)
        unique_ids = list(dict.fromkeys(tid for tid in tool_ids if tid))
        if unique_ids:
            try:
                resolved = await get_tools(self.request, unique_ids, self.user, extra_params)
                if resolved:
                    self.all_tools_dict.update(resolved)
            except Exception as e:
                logger.error(f"get_tools failed: {e}")

        # 2. MCP tools (from server:mcp: / mcp: prefixed IDs)
        try:
            mcp_tools = await self._resolve_mcp_tools(unique_ids, extra_params)
            if mcp_tools:
                self.all_tools_dict.update(mcp_tools)
                # Track clients for later cleanup
                for name, tool in mcp_tools.items():
                    client = tool.get("client")
                    if client and hasattr(client, "disconnect"):
                        self._mcp_clients[name] = client
        except Exception as e:
            logger.error(f"MCP tool resolution failed: {e}")

        # 3. Built-in tools (pass skill_ids so view_skill is available)
        if skill_ids:
            extra_params["__skill_ids__"] = skill_ids
        features = self._get_model_features(model_info)
        if features or extra_params.get("__skill_ids__"):
            try:
                builtin = await get_builtin_tools(self.request, extra_params, features=features, model=model_info)
                if builtin:
                    self.all_tools_dict.update(builtin)
            except Exception as e:
                logger.error(f"get_builtin_tools failed: {e}")

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
                    self.all_tools_dict.update(t_tools)
            except Exception as e:
                logger.error(f"get_terminal_tools failed: {e}")

        # 5. Add internal control tools (always available, stored in all_tools_dict too)
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
        self.all_tools_dict["early_finish"] = {
            "spec": {
                "name": "early_finish",
                "description": "Signal that the current work is complete enough to move to the next phase. Only available in EXECUTE phase.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Brief reason why you're finishing early"},
                    },
                    "required": ["reason"],
                },
            },
            "callable": self._tool_early_finish,
            "type": "function",
        }
        self.all_tools_dict["rag_search"] = {
            "spec": {
                "name": "rag_search",
                "description": "Semantic search inside an attached file using RAG (Retrieval-Augmented Generation). Use this when the attached file is too large to inline. Returns the most relevant chunks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string", "description": "The file ID or name of the attached document to search."},
                        "query": {"type": "string", "description": "The search query describing what you are looking for in the document."},
                        "top_k": {"type": "integer", "default": 5, "description": "Number of top relevant chunks to return."},
                    },
                    "required": ["file_id", "query"],
                },
            },
            "callable": self._tool_rag_search,
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
                "description": "Ask the user an interactive question with selectable options and optional custom free-text input. Use this ONLY when you need clarification or a decision from the user before you can continue planning. NOT available in EXECUTE, REVIEW or OUTPUT phases.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to display to the user."},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 2–6 options for the user to pick from.",
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

    # ── Skills & MCP Helpers ──

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

    async def _resolve_mcp_tools(self, tool_ids: list[str], extra_params: dict) -> dict[str, Any]:
        """Resolve MCP server tools from tool_ids, mirroring planner_v3 logic."""
        out: dict[str, Any] = {}
        if not tool_ids or not self.request or not self.user:
            return out

        oauth_token = None
        try:
            oauth_token = await get_system_oauth_token(self.request, self.user)
        except Exception:
            pass

        resolved_servers = set()
        for tool_id in tool_ids:
            if not isinstance(tool_id, str):
                continue

            server_id = None
            if tool_id.startswith("server:mcp:"):
                server_id = tool_id[len("server:mcp:"):]
            elif tool_id.startswith("mcp:"):
                server_id = tool_id[len("mcp:"):]
                mcp_connections = getattr(
                    self.request.app.state.config, "TOOL_SERVER_CONNECTIONS", []
                )
                for server_connection in mcp_connections:
                    if server_connection.get("type", "") == "mcp":
                        sid = server_connection.get("info", {}).get("id")
                        if sid and (tool_id == sid or tool_id.startswith(f"{sid}_")):
                            server_id = sid
                            break

            if not server_id:
                continue
            if server_id in resolved_servers:
                continue
            resolved_servers.add(server_id)

            try:
                mcp_server_connection = None
                for conn in self.request.app.state.config.TOOL_SERVER_CONNECTIONS:
                    if (
                        conn.get("type", "") == "mcp"
                        and conn.get("info", {}).get("id") == server_id
                    ):
                        mcp_server_connection = conn
                        break

                if not mcp_server_connection:
                    logger.error("MCP server %s not found in connections", server_id)
                    continue

                if not await has_connection_access(self.user, mcp_server_connection):
                    logger.warning("Access denied to MCP server %s", server_id)
                    continue

                auth_type = mcp_server_connection.get("auth_type", "")
                headers: dict[str, str] = {}
                if auth_type == "bearer":
                    headers["Authorization"] = f"Bearer {mcp_server_connection.get('key', '')}"
                elif auth_type == "none":
                    pass
                elif auth_type == "session":
                    tok = getattr(getattr(self.request, "state", None), "token", None)
                    creds = getattr(tok, "credentials", None) if tok else None
                    if creds:
                        headers["Authorization"] = f"Bearer {creds}"
                elif auth_type == "system_oauth":
                    if oauth_token:
                        headers["Authorization"] = f"Bearer {oauth_token.get('access_token', '')}"
                elif auth_type == "oauth_2.1":
                    try:
                        splits = server_id.split(":")
                        sid = splits[-1] if len(splits) > 1 else server_id
                        mgr = getattr(self.request.app.state, "oauth_client_manager", None)
                        if mgr:
                            ot = await mgr.get_oauth_token(
                                getattr(self.user, "id", ""), f"mcp:{sid}"
                            )
                            if ot:
                                headers["Authorization"] = f"Bearer {ot.get('access_token', '')}"
                    except Exception as e:
                        logger.error("OAuth token for MCP: %s", e)

                connection_headers = mcp_server_connection.get("headers")
                if connection_headers and isinstance(connection_headers, dict):
                    headers.update(connection_headers)

                if ENABLE_FORWARD_USER_INFO_HEADERS and self.user:
                    headers = include_user_info_headers(headers, self.user)
                    cid = self.chat_id
                    if cid:
                        headers[FORWARD_SESSION_INFO_HEADER_CHAT_ID] = cid
                    mid = self.message_id
                    if mid:
                        headers[FORWARD_SESSION_INFO_HEADER_MESSAGE_ID] = mid

                client = MCPClient()
                await client.connect(
                    url=mcp_server_connection.get("url", ""),
                    headers=headers if headers else None,
                )

                if not hasattr(client, "_call_lock"):
                    client._call_lock = asyncio.Lock()

                function_name_filter_list = mcp_server_connection.get("config", {}).get(
                    "function_name_filter_list", ""
                )
                if isinstance(function_name_filter_list, str):
                    function_name_filter_list = function_name_filter_list.split(",")

                async with client._call_lock:
                    tool_specs = await client.list_tool_specs()

                def make_tool_function(mcp_client, function_name, sid):
                    async def tool_function(**kwargs):
                        try:
                            logger.debug(
                                "[MCP] Calling '%s' on server '%s' with args: %s",
                                function_name, sid, kwargs,
                            )
                            async with mcp_client._call_lock:
                                result = await mcp_client.call_tool(
                                    function_name, function_args=kwargs
                                )
                            if hasattr(result, "content") and result.content:
                                texts = []
                                for c in result.content:
                                    if hasattr(c, "text") and c.text:
                                        texts.append(c.text)
                                    elif hasattr(c, "image"):
                                        texts.append(
                                            f"[Image Content: {c.image[:50]}...]"
                                            if isinstance(c.image, str)
                                            else "[Image Content]"
                                        )
                                    else:
                                        texts.append(str(c))
                                return "\n".join(texts)
                            if hasattr(result, "isError") and result.isError:
                                return f"MCP Error from {sid}: {result}"
                            return str(result)
                        except Exception as e:
                            logger.error(
                                "Failed to call MCP tool '%s' on '%s': %s",
                                function_name, sid, e, exc_info=True,
                            )
                            return f"Error calling MCP tool: {e}"

                    return tool_function

                for tool_spec in tool_specs:
                    if function_name_filter_list:
                        if not is_string_allowed(tool_spec["name"], function_name_filter_list):
                            continue
                    tool_function = make_tool_function(client, tool_spec["name"], server_id)
                    prefixed_name = f'{server_id}_{tool_spec["name"]}'
                    out[prefixed_name] = {
                        "spec": {**tool_spec, "name": prefixed_name},
                        "callable": tool_function,
                        "type": "mcp",
                        "client": client,
                        "direct": False,
                    }
            except Exception as e:
                logger.debug("MCP tool load failed for %s: %s", tool_id, e)
                if self.event_emitter:
                    try:
                        await self.event_emitter({
                            "type": "chat:message:error",
                            "data": {"error": {"content": f"Failed to connect to MCP server '{server_id}'"}},
                        })
                    except Exception:
                        pass
                continue

        return out

    # ── Phase-aware Tool Filtering ──

    def _filter_tools_for_phase(self, phase: str):
        """Build phase_tools_dict from all_tools_dict based on Valves config."""
        # Determine which tool names are allowed for this phase
        allowlist: Set[str] = set()

        if phase == self.PHASE_PLAN:
            allowlist = set(_comma_list(self.valves.PLAN_TOOLS))
        elif phase == self.PHASE_REPLAN_SKIP:
            allowlist = set(_comma_list(self.valves.PLAN_TOOLS))  # Same tools as PLAN for quick replan
        elif phase == self.PHASE_EXECUTE:
            allowlist = set(_comma_list(self.valves.EXECUTE_TOOLS))
        elif phase == self.PHASE_REVIEW:
            allowlist = set(_comma_list(self.valves.REVIEW_TOOLS))
        elif phase == self.PHASE_OUTPUT:
            allowlist = set(_comma_list(self.valves.OUTPUT_TOOLS))

        # Phase-specific internal tools (only show relevant ones per phase)
        phase_internal_tools = self.PHASE_INTERNAL_TOOLS.get(phase, self.INTERNAL_TOOLS)

        # If allowlist is empty -> allow ALL tools
        # If allowlist has entries -> only those tools (plus phase-internal ones)
        self.phase_tools_dict = {}

        for name, tool in self.all_tools_dict.items():
            # Internal tools are included ONLY if in phase_internal_tools
            if name in self.INTERNAL_TOOLS:
                if name in phase_internal_tools:
                    self.phase_tools_dict[name] = tool
                continue

            # ask_user is shown only in PLAN and REPLAN_SKIP phases (planning / clarification only)
            if name == "ask_user":
                if phase in (self.PHASE_PLAN, self.PHASE_REPLAN_SKIP):
                    self.phase_tools_dict[name] = tool
                continue

            # Allowlist filtering for non-internal tools
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
            # Grant 3 extra iterations without rewinding the persistent counter
            self._extra_grace += 3
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

        if self.phase in (self.PHASE_REVIEW, self.PHASE_OUTPUT):
            self._transition_to(self.PHASE_EXECUTE)

        await self._save_state_to_file()
        await self.emit_task_update()
        return json.dumps({"replan": True, "reason": reason, "updated_tasks": updated_tasks, "mode": mode})

    async def _tool_complete_task(self, **kwargs):
        idx = kwargs.get("index", -1)
        if 0 <= idx < len(self.task_list):
            task = self.task_list[idx]
            if task not in self.completed_tasks:
                self.completed_tasks.append(task)
            await self._save_state_to_file()
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
            await self._save_state_to_file()
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

        await self._save_state_to_file()
        self.history[0]["content"] = self._build_system_prompt()
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

    async def _tool_confirm_plan(self, **kwargs):
        plan_text = kwargs.get("plan", "")
        uv = self.user_valves

        if uv and (getattr(uv, "YOLO_MODE", False) or not getattr(uv, "ENABLE_PLAN_APPROVAL", False)):
            return json.dumps({"action": "accept"})

        # Auto-accept when in QUICK REPLAN phase to keep the skip flow fast
        if self.phase == self.PHASE_REPLAN_SKIP:
            return json.dumps({"action": "accept"})

        if not self.event_call:
            return json.dumps({"action": "accept"})

        # Task list extraction is deferred to the accept branch in _run_impl
        tasks = self._extract_task_list(plan_text)
        tasks_data = [{"task_id": f"T{i+1}", "description": t} for i, t in enumerate(tasks)]
        if not tasks_data:
            tasks_data = [{"task_id": "T1", "description": plan_text}]

        js = self._build_plan_approval_js(tasks_data)
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

        return json.dumps(res)

    async def _tool_early_finish(self, reason: str, **kwargs):
        """Signal early completion and move to REVIEW from EXECUTE."""
        if self.phase == self.PHASE_EXECUTE:
            self._transition_to(self.PHASE_REVIEW)
            await self._save_state_to_file()
            await self.emit_task_update()
            return json.dumps({"early_finish": True, "phase": "review", "reason": reason})
        return json.dumps({"early_finish": False, "error": f"Cannot early_finish from {self.phase}"})

    async def _tool_ask_user(self, question: str, options: list, allow_custom: bool = True, **kwargs):
        """Interactive user question tool. Only available during PLAN and QUICK REPLAN phases."""
        if self.phase in (self.PHASE_PLAN, self.PHASE_REPLAN_SKIP):
            self._plan_questions_asked += 1
            max_questions = getattr(self.valves, "MAX_PLAN_QUESTIONS", 3)
            if self._plan_questions_asked >= max_questions:
                return json.dumps({
                    "type": "error",
                    "response": (
                        f"CRITICAL: You have reached the maximum number of clarification questions "
                        f"({max_questions}). You must NOT ask more questions. "
                        "Call confirm_plan with your best plan NOW."
                    ),
                    "skipped": True,
                })

        if not self.event_call:
            return json.dumps({"type": "error", "response": "Interactive input not available in this context.", "skipped": True})

        if not options or not isinstance(options, list):
            return json.dumps({"type": "error", "response": "Provide at least one option.", "skipped": True})

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

    async def _tool_rag_search(self, file_id: str, query: str, top_k: int = 5, **kwargs):
        """RAG search: chunk, embed (if needed) and retrieve top_k chunks."""
        try:
            from chromadb import PersistentClient
            from sentence_transformers import SentenceTransformer
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as imp_err:
            return json.dumps({"error": f"RAG dependencies missing: {imp_err}"})

        if not file_id or not query:
            return json.dumps({"error": "file_id and query are required"})

        # Resolve upload dir & read file bytes
        upload_dirs = [
            getattr(self.request.app.state, "UPLOAD_DIR", None),
            "/app/backend/data/uploads",
            "/app/data/uploads",
            "/data/uploads",
            "./data/uploads",
            "data/uploads",
        ]
        file_path = None
        for d in upload_dirs:
            if d:
                p = os.path.join(d, file_id)
                if os.path.isfile(p):
                    file_path = p
                    break

        if not file_path:
            return json.dumps({"error": f"File {file_id} not found on disk"})

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except Exception as read_err:
            return json.dumps({"error": f"Could not read {file_id}: {read_err}"})

        # Persistent vector DB path
        db_path = os.environ.get("HELIX_RAG_DB", "/app/backend/data/helix_rag_db")
        os.makedirs(db_path, exist_ok=True)

        try:
            client = PersistentClient(path=db_path)
            collection = client.get_or_create_collection(file_id)
            embedder = SentenceTransformer("all-MiniLM-L6-v2")

            # Check if already embedded
            existing = collection.count()
            if existing == 0:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                chunks = splitter.split_text(text)
                if not chunks:
                    return json.dumps({"error": "No extractable text in file"})

                embeddings = embedder.encode(chunks).tolist()
                collection.add(
                    ids=[f"{file_id}-{i}" for i in range(len(chunks))],
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=[{"source": file_id}] * len(chunks),
                )

            query_embedding = embedder.encode([query]).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
            )

            chunks = results.get("documents", [[]])[0]
            if not chunks:
                return json.dumps({"result": "No relevant chunks found."})

            context = "\n\n".join(
                f'<source id="{i+1}">{chunk}</source>' for i, chunk in enumerate(chunks)
            )
            return json.dumps({
                "result": f"Use the following context:\n{context}\n\nQuestion: {query}",
                "chunks": chunks,
            })
        except Exception as rag_err:
            logger.error("RAG search failed: %s", rag_err)
            return json.dumps({"error": f"RAG search failed: {rag_err}"})

    # ── Parallel Tool Execution ──

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
            # Strip functions. prefix if present
            if isinstance(name, str) and name.startswith("functions."):
                name = name[len("functions."):]
            # Resolve args / arguments / parameters
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
        # Try to parse JSON result back to native type for cleaner aggregation
        parsed_result = result_str
        if isinstance(result_str, str):
            try:
                parsed_result = json.loads(result_str)
            except json.JSONDecodeError:
                pass
        return {
            "tool_name": tool_name,
            "result": parsed_result,
        }

    async def _tool_run_tools_parallel(self, tool_calls: list = None, **kwargs):
        """Execute multiple independent tool calls in parallel."""
        if not tool_calls:
            return json.dumps({"error": "No tool_calls provided"})
        try:
            calls = self._normalize_parallel_calls(tool_calls)
        except ValueError as e:
            return json.dumps({"error": str(e)})

        # Validate each tool exists in current phase
        missing = []
        for c in calls:
            if c["name"] not in self.phase_tools_dict:
                missing.append(c["name"])
        if missing:
            available = ", ".join(sorted(self.phase_tools_dict.keys())[:20])
            return json.dumps({
                "error": f"Tools not available in current phase: {', '.join(missing)}. Available: {available}",
            })

        # Emit status for each parallel sub-call
        names = ", ".join(c["name"] for c in calls)
        await self.emit_status(f"Running parallel: {names}...")

        # Execute all in parallel
        tasks = [
            self._execute_parallel_single(c, f"{self.message_id or 'parallel'}_{i}")
            for i, c in enumerate(calls)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                processed.append({
                    "tool_name": calls[i]["name"],
                    "result": f"Error: {res}",
                })
            else:
                processed.append(res)

        return json.dumps({"results": processed}, ensure_ascii=False)

    def _base_theme_js(self):
        return """
            const col = {
                overlay: 'rgba(0,0,0,0.62)', panel: '#1e1e2e', border: '#45475a',
                text: '#cdd6f4', sub: '#a6adc8', input: '#313244', inputBorder: '#45475a',
                btn: '#313244', btnText: '#cdd6f4', btnBorder: '#45475a',
                btnPrimary: '#E8713A', btnPrimaryText: '#ffffff',
            };
            try { const s = getComputedStyle(document.documentElement);
              col.panel = s.getPropertyValue('--color-gray-900').trim() || col.panel;
              col.text = s.getPropertyValue('--color-gray-50').trim() || col.text;
              col.btnPrimary = s.getPropertyValue('--color-blue-500').trim() || col.btnPrimary;
            } catch(e) {}
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

          const check = document.createElement('span');
          Object.assign(check.style, {{
            flexShrink:'0',fontSize:'15px',color:col.btnPrimary,
            opacity:'0',transition:'opacity 0.12s',display:'none',
          }});
          check.textContent = '\u2713';

          btn.appendChild(keyBadge); btn.appendChild(textBlock); btn.appendChild(check);

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
            if (btn.dataset.selected !== '1'){{
              btn.style.background  = '#3c3c52';
              btn.style.borderColor = col.btnPrimary+'77';
              btn.style.transform   = 'translateY(-1px)';
            }}
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
            if (e.key==='Enter' \u0026\u0026 customInput.value.trim()){{
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
        let _timer;
        const OVERLAY_ID = '__owui_helix_plan__';
        const existing = document.getElementById(OVERLAY_ID);
        if (existing) existing.remove();

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
        const iconBlock = document.createElement('div');
        iconBlock.textContent = '\uD83D\uDCCB';
        Object.assign(iconBlock.style, {{ fontSize:'22px',flexShrink:'0',marginTop:'2px' }});
        const titleText = document.createElement('p');
        Object.assign(titleText.style, {{
          margin:'0',color:col.text,fontSize:'16px',
          fontWeight:'700',lineHeight:'1.4',flex:'1',wordBreak:'break-word',
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
        header.appendChild(iconBlock); header.appendChild(titleText); header.appendChild(badge);

        const scrollContainer = document.createElement('div');
        Object.assign(scrollContainer.style, {{
          overflowY:'auto',flex:'1',display:'flex',flexDirection:'column',gap:'7px',
          paddingRight:'4px',
        }});

        const tasksData = {ts};
        tasksData.forEach((t,i)=>{{
            const card = document.createElement('div');
            Object.assign(card.style, {{
              display:'flex',alignItems:'flex-start',gap:'12px',
              background:col.input,border:'1.5px solid '+col.inputBorder,
              borderRadius:'10px',padding:'11px 13px',
              minHeight:'48px',transition:'background 0.12s, border-color 0.12s',
            }});
            const num = document.createElement('span');
            Object.assign(num.style, {{
              flexShrink:'0',width:'26px',height:'26px',
              borderRadius:'6px',background:col.btnPrimary,
              display:'flex',alignItems:'center',justifyContent:'center',
              fontSize:'11px',fontWeight:'700',color:col.btnPrimaryText,
              marginTop:'2px',
            }});
            num.textContent = String(i+1);
            const content = document.createElement('div');
            Object.assign(content.style, {{ display:'flex',flexDirection:'column',gap:'4px',flex:'1',minWidth:'0' }});
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
            scrollContainer.appendChild(card);
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
          minHeight:'64px',resize:'none',
          fontFamily:'inherit',boxSizing:'border-box',
          transition:'border-color 0.15s',
        }});
        feedbackInput.addEventListener('focus', ()=>{{ feedbackInput.style.borderColor=col.btnPrimary; }});
        feedbackInput.addEventListener('blur', ()=>{{ feedbackInput.style.borderColor=col.inputBorder; }});
        inputContainer.appendChild(inputLabel); inputContainer.appendChild(feedbackInput);

        const footer = document.createElement('div');
        Object.assign(footer.style, {{ display:'flex',gap:'8px',justifyContent:'flex-end',marginTop:'2px' }});

        function makeBtn(label,primary){{
          const b = document.createElement('button');
          b.textContent = label;
          Object.assign(b.style, {{
            padding:'10px 18px',borderRadius:'8px',
            fontSize:'13px',fontWeight:'700',
            cursor:'pointer',fontFamily:'inherit',
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
          background:'#313244',color:'#f38ba8',borderColor:'#f38ba8',
        }});
        cancelBtn.addEventListener('mouseenter', ()=>{{ cancelBtn.style.opacity='0.85'; cancelBtn.style.transform='translateY(-1px)'; }});
        cancelBtn.addEventListener('mouseleave', ()=>{{ cancelBtn.style.opacity='1'; cancelBtn.style.transform=''; }});

        const cleanup = ()=>{{
          panel.style.transform='scale(0.95)';
          panel.style.opacity='0';
          overlay.style.opacity='0';
          setTimeout(()=>{{
            overlay.remove();
            document.removeEventListener('keydown',onKey);
          }},180);
        }};

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

        panel.appendChild(header); panel.appendChild(scrollContainer); panel.appendChild(inputContainer); panel.appendChild(footer); panel.appendChild(countdown);
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

    # ── Iteration Limit UI ──

    def _build_iteration_limit_js(self, current_iter, max_iter, timeout_s: int = 300) -> str:
        """Build a Continue/Cancel modal for iteration limit reached."""
        return f"""
    return (function(){{
      return new Promise((resolve)=>{{
    {self._base_theme_js()}
        const OVERLAY_ID = '__owui_helix_limit__';
        const existing = document.getElementById(OVERLAY_ID);
        if (existing) existing.remove();

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
          maxWidth:'440px',width:'calc(100vw - 32px)',
          boxShadow:'0 28px 80px rgba(0,0,0,0.65)',
          display:'flex',flexDirection:'column',gap:'14px',
          transform:'scale(0.92)',opacity:'0',
          transition:'transform 0.22s cubic-bezier(0.34,1.56,0.64,1), opacity 0.18s ease',
        }});

        const header = document.createElement('div');
        Object.assign(header.style, {{ display:'flex',alignItems:'flex-start',gap:'12px' }});
        const iconBlock = document.createElement('div');
        iconBlock.textContent = '\u26A0\uFE0F';
        Object.assign(iconBlock.style, {{ fontSize:'22px',flexShrink:'0',marginTop:'2px' }});
        const titleText = document.createElement('p');
        Object.assign(titleText.style, {{
          margin:'0',color:col.text,fontSize:'16px',
          fontWeight:'700',lineHeight:'1.4',flex:'1',wordBreak:'break-word',
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
        header.appendChild(iconBlock); header.appendChild(titleText); header.appendChild(badge);

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
            cursor:'pointer',fontFamily:'inherit',
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
        Object.assign(stopBtn.style, {{ background:'#313244',color:'#f38ba8',borderColor:'#f38ba8' }});
        stopBtn.addEventListener('mouseenter', ()=>{{ stopBtn.style.opacity='0.85'; stopBtn.style.transform='translateY(-1px)'; }});
        stopBtn.addEventListener('mouseleave', ()=>{{ stopBtn.style.opacity='1'; stopBtn.style.transform=''; }});

        const cleanup = ()=>{{
          panel.style.transform='scale(0.95)';
          panel.style.opacity='0';
          overlay.style.opacity='0';
          setTimeout(()=>{{
            overlay.remove();
            document.removeEventListener('keydown',onKey);
          }},180);
        }};

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

    # ── File Context Preparation (planner_v3 parity) ──

    async def _apply_file_prep(self, msgs: list) -> list:
        """Mirror OWUI middleware: add_file_context then chat_completion_files_handler."""
        if not self.request or not self.user or not self.pipe_metadata:
            return msgs

        # 1. add_file_context — injects text-file content into messages
        prep = copy.deepcopy(msgs)
        try:
            prep = await add_file_context(prep, self.chat_id, self.user)
            logger.debug("add_file_context succeeded")
        except Exception as e:
            logger.warning("add_file_context failed: %s", e)

        # 2. chat_completion_files_handler — converts remaining attachments to multimodal blocks
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

    # ── File Persistence (tools → DB sync) ──

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
                # Mutate metadata so OWUI core persists files at stream cleanup
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

    async def _sync_produced_files_to_db(self):
        """Deduplicate and bind all tool-generated files to the chat DB message."""
        if not HAS_DB_PERSISTENCE or not self.chat_id or not self.message_id:
            return

        # Deduplicate by id/file_id/url
        seen = set()
        unique_files = []
        async with self._files_lock:
            for f in self.produced_files:
                if not isinstance(f, dict):
                    continue
                fid = f.get("id") or f.get("file_id") or f.get("url")
                if fid and fid not in seen:
                    seen.add(fid)
                    unique_files.append(f)
            self.produced_files = unique_files.copy()

        if not unique_files:
            return

        try:
            await Chats.add_message_files_by_id_and_message_id(
                self.chat_id,
                self.message_id,
                unique_files,
            )
            logger.info(f"Synced {len(unique_files)} tool files to DB.")
        except Exception as e:
            logger.warning(f"Tool file DB sync failed: {e}")

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

        base = ""
        try:
            if self.phase == self.PHASE_PLAN:
                base = (self.valves.PLAN_PROMPT or DEFAULT_PLAN_PROMPT).format(tool_names=tool_names)
            elif self.phase == self.PHASE_REPLAN_SKIP:
                base = DEFAULT_REPLAN_SKIP_PROMPT.format(tool_names=tool_names, goal=self.goal)
            elif self.phase == self.PHASE_EXECUTE:
                base = (self.valves.EXECUTE_PROMPT or DEFAULT_EXECUTE_PROMPT).format(tool_names=tool_names, task_state=task_state)
            elif self.phase == self.PHASE_REVIEW:
                base = (self.valves.REVIEW_PROMPT or DEFAULT_REVIEW_PROMPT).format(goal=self.goal, task_state=task_state, tool_names=tool_names)
            elif self.phase == self.PHASE_OUTPUT:
                if self._output_turn >= 2:
                    base = DEFAULT_OUTPUT_FINAL_PROMPT.format(goal=self.goal, task_state=task_state)
                else:
                    base = (self.valves.OUTPUT_PROMPT or DEFAULT_OUTPUT_PROMPT).format(goal=self.goal, task_state=task_state)
            else:
                base = DEFAULT_PLAN_PROMPT.format(tool_names=tool_names)
        except (KeyError, IndexError, ValueError):
            # User-provided prompt may have stray braces; fall back to default
            if self.phase == self.PHASE_PLAN:
                base = DEFAULT_PLAN_PROMPT.format(tool_names=tool_names)
            elif self.phase == self.PHASE_REPLAN_SKIP:
                base = DEFAULT_REPLAN_SKIP_PROMPT.format(tool_names=tool_names, goal=self.goal)
            elif self.phase == self.PHASE_EXECUTE:
                base = DEFAULT_EXECUTE_PROMPT.format(tool_names=tool_names, task_state=task_state)
            elif self.phase == self.PHASE_REVIEW:
                base = DEFAULT_REVIEW_PROMPT.format(goal=self.goal, task_state=task_state, tool_names=tool_names)
            elif self.phase == self.PHASE_OUTPUT:
                if self._output_turn >= 2:
                    base = DEFAULT_OUTPUT_FINAL_PROMPT.format(goal=self.goal, task_state=task_state)
                else:
                    base = DEFAULT_OUTPUT_PROMPT.format(goal=self.goal, task_state=task_state)
            else:
                base = DEFAULT_PLAN_PROMPT.format(tool_names=tool_names)

        # Prepend skill prompt if resolved (available in all phases)
        if self._skill_prompt:
            base = f"{self._skill_prompt}\n\n{base}"
        return base

    # ── Phase Transitions ──

    def _transition_to(self, phase):
        """Transition to a new phase: update tools, system prompt, state."""
        self.phase = phase
        self._consecutive_tool_misses.clear()
        # Reset output turn counter when entering OUTPUT phase
        if phase == self.PHASE_OUTPUT:
            self._output_turn = 0
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
        # Build head: system prompt + goal (always preserved)
        head = messages[:2]
        # Non-state messages after head, split into removed and tail
        non_state = messages[2:]
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
        await self.resolve_tools()

        # --- Debug mode intercept ---
        if getattr(self.valves, "DEBUG_MODE", False):
            msg_lower = (user_msg or "").strip().lower()
            if msg_lower in ("show tools", "debug tools", "list tools"):
                # Only external tools (exclude Helix internal control tools)
                external_tools = sorted(
                    name for name in self.all_tools_dict.keys()
                    if name not in self.INTERNAL_TOOLS
                )
                tool_list = "\n".join(f"- {t}" for t in external_tools)
                tool_csv = ", ".join(external_tools)
                return f"**Available tools ({len(external_tools)}):**\n\n{tool_list}\n\n**Comma-separated:** {tool_csv}"
            if msg_lower in ("show mcp", "debug mcp", "list mcp"):
                mcp_tools = sorted(
                    name for name, info in self.all_tools_dict.items()
                    if info.get("type") == "mcp" and name not in self.INTERNAL_TOOLS
                )
                tool_list = "\n".join(f"- {t}" for t in mcp_tools)
                tool_csv = ", ".join(mcp_tools)
                return f"**Available MCP tools ({len(mcp_tools)}):**\n\n{tool_list}\n\n**Comma-separated:** {tool_csv}"
            if msg_lower in ("show skills", "debug skills", "list skills"):
                if self._skill_prompt:
                    return f"**Injected skills prompt:**\n\n{self._skill_prompt}"
                return "**No skills injected.**"

        await self.emit_status("Agent starting...")

        # Attempt DB-backed state recovery at start of turn
        await self._recover_state_from_files(self.body if isinstance(self.body, dict) else {})

        # After recovery, decide whether to continue the old session or start fresh.
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
            # State was restored and there are still open tasks - continue execution
            max_allowed = max(0, self.valves.MAX_ITERATIONS - 5)
            if self.loop_count > max_allowed:
                logger.info(f"Clamped loop_count from {self.loop_count} to {max_allowed} after state restore.")
                self.loop_count = max_allowed
            self.goal = f"{self.goal}; Updated: {user_msg}"
            if self.history and self.history[0].get("role") == "system":
                self.history[0]["content"] = self._build_system_prompt()
            else:
                self.history.insert(0, {"role": "system", "content": self._build_system_prompt()})
            self.history.append({"role": "user", "content": user_msg})
            self._filter_tools_for_phase(self.phase)
            await self.emit_task_update()
        elif self.goal and self.task_list and not has_remaining_tasks and self.user_valves and getattr(self.user_valves, "SKIP_PLAN_ON_RESUME", True):
            # Previous session finished. Use Quick Replan phase to let the agent plan the new request.
            logger.info("Previous session finished; entering Quick Replan phase.")
            self.loop_count = 0
            self._plan_questions_asked = 0
            self.task_list = []
            self.completed_tasks = []
            self.failed_tasks = []
            self.goal = self.goal + "\n\nNEW REQUEST:\n" + user_msg
            self.phase = self.PHASE_REPLAN_SKIP
            if self.history and self.history[0].get("role") == "system":
                self.history[0]["content"] = self._build_system_prompt()
            else:
                self.history.insert(0, {"role": "system", "content": self._build_system_prompt()})
            self.history.append({"role": "user", "content": user_msg})
            self._filter_tools_for_phase(self.PHASE_REPLAN_SKIP)
            await self.emit_task_update()
        else:
            # Fresh session: reset state and start from PLAN
            if self.goal and self.task_list and not has_remaining_tasks:
                logger.info("All tasks completed or failed; starting fresh session.")
            self.goal = user_msg
            self.phase = self.PHASE_PLAN
            self.task_list = []
            self.completed_tasks = []
            self.failed_tasks = []
            self.loop_count = 0
            self._plan_questions_asked = 0
            self._filter_tools_for_phase(self.PHASE_PLAN)
            system_prompt = self._build_system_prompt()
            self.history = [
                {"role": "system", "content": system_prompt},
                last_user_msg_raw if last_user_msg_raw else {"role": "user", "content": user_msg},
            ]

        recent_calls = []
        self._output_parts = []

        while True:
            # ── OUTPUT phase hard limit: max 2 turns (tool prep + final text) ──
            if self.phase == self.PHASE_OUTPUT:
                self._output_turn += 1
                if self._output_turn > 2:
                    # Safety net: exceeded max output turns, return immediately
                    await self.emit_task_update(finalize_tasks=True)
                    await self.emit_status("Output phase exceeded max turns", done=True)
                    return self._format_output()
                # Refresh system prompt for the current turn
                if self.history and self.history[0].get("role") == "system":
                    self.history[0]["content"] = self._build_system_prompt()

            effective_max = self.valves.MAX_ITERATIONS + self._extra_grace
            if self.loop_count >= effective_max:
                await self.emit_output(f"\n[WARN] Max iterations ({effective_max}) reached.")
                should_continue = False
                if self.event_call and not (self.user_valves and getattr(self.user_valves, "YOLO_MODE", False)):
                    try:
                        js = self._build_iteration_limit_js(self.loop_count, effective_max)
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

            self.loop_count += 1
            recent_calls = recent_calls[-30:]

            # Safety-net for REPLAN_SKIP: max 3 loops, then fallback to single-task EXECUTE
            if self.phase == self.PHASE_REPLAN_SKIP and self.loop_count > self.MAX_REPLAN_SKIP_LOOPS:
                logger.warning("REPLAN_SKIP exceeded %d loops; falling back to single-task EXECUTE.", self.MAX_REPLAN_SKIP_LOOPS)
                self.task_list = [user_msg]
                if self.history and len(self.history) > 0:
                    self.history[0]["content"] = self._build_system_prompt()
                self._transition_to(self.PHASE_EXECUTE)
                await self.emit_task_update()

            phase_icons = {
                self.PHASE_PLAN: "[PLAN]",
                self.PHASE_REPLAN_SKIP: "[QPLN]",
                self.PHASE_EXECUTE: "[EXEC]",
                self.PHASE_REVIEW: "[REVU]",
                self.PHASE_OUTPUT: "[OUT]",
            }
            phase_name = {
                self.PHASE_PLAN: "Plan",
                self.PHASE_REPLAN_SKIP: "Quick Replan",
                self.PHASE_EXECUTE: "Execute",
                self.PHASE_REVIEW: "Review",
                self.PHASE_OUTPUT: "Output",
            }
            icon = phase_icons.get(self.phase, "[LOOP]")
            name = phase_name.get(self.phase, "Loop")

            effective_max = self.valves.MAX_ITERATIONS + self._extra_grace
            await self.emit_status(f"Mode: {name}, Loop: {self.loop_count}/{effective_max}")


            self.history = self._manage_context_window(self.history)
            # Strip system messages from LLM context
            # Only the first message (system prompt) is kept
            call_messages = [self.history[0]] + [m for m in self.history[1:] if m.get("role") != "system"]

            completion_body = {
                **self.body,
                "model": model,
                "messages": call_messages,
                "tools": self.phase_tools_specs if self.phase_tools_specs else None,
                "metadata": self.pipe_metadata,
            }

            # In OUTPUT phase turn 2, remove tools to force pure text output
            if self.phase == self.PHASE_OUTPUT and self._output_turn >= 2:
                completion_body["tools"] = None

            # Apply OpenWebUI file context prep (add_file_context + chat_completion_files_handler)
            try:
                completion_body["messages"] = await self._apply_file_prep(copy.deepcopy(call_messages))
            except Exception as e:
                logger.warning("_apply_file_prep failed: %s", e)

            # ── Stream LLM response ──
            tc_dict = {}
            content_chunks = []

            async for event in stream_completion(self.request, completion_body, self.user):
                etype = event.get("type")
                if etype == "error":
                    await self.emit_output(f"\n[ERROR] LLM Error: {event.get('text', 'Unknown')}")
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

            content = strip_thinking("".join(content_chunks).strip())

            if not tc_dict:
                # Try XML tool call rescue for hallucinated <ToolCall> blocks
                xml_calls = extract_xml_tool_calls(content or "")
                if xml_calls:
                    tc_dict = {tc["index"]: tc for tc in xml_calls}
                else:
                    # No tool calls and no XML rescue
                    # PLAN phase: enforce that the model must call confirm_plan.
                    if self.phase in (self.PHASE_PLAN, self.PHASE_REPLAN_SKIP):
                        self.history.append({
                            "role": "assistant",
                            "content": content or "",
                        })
                        self.history.append({
                            "role": "user",
                            "content": "SYSTEM: You produced text but did not call any tools. You MUST call the confirm_plan tool with the plan to proceed. Do NOT output the plan as text—call the tool.",
                        })
                        await self.emit_output(f"\n[WARN] No tool call produced in {self.phase} phase. Re-prompting to enforce confirm_plan.\n")
                        continue
                    # OUTPUT phase: never emit text in Turn 1, always proceed to Turn 2
                    if self.phase == self.PHASE_OUTPUT:
                        if self._output_turn == 1:
                            # Turn 1 with no tools: skip text and proceed to Turn 2
                            self.history.append({
                                "role": "assistant",
                                "content": content or "",
                            })
                            # Refresh final prompt for Turn 2
                            if self.history and self.history[0].get("role") == "system":
                                self.history[0]["content"] = self._build_system_prompt()
                            continue
                        # Turn 2: emit final text and return
                        if content:
                            await self.emit_output(content)
                        await self.emit_task_update(finalize_tasks=True)
                        await self.emit_status("Done", done=True)
                        return self._format_output()
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
                        await self.emit_output(content)
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

                # ── Handle early_finish ──
                if tool_name == "early_finish":
                    if self.phase != self.PHASE_EXECUTE:
                        result_json = json.dumps({"early_finish": False, "error": f"early_finish is only available in EXECUTE phase, not {self.phase}"})
                        self.history.append({
                            "role": "tool",
                            "content": result_json,
                            "tool_call_id": call_id,
                            "name": tool_name,
                        })
                        continue
                    result_json = await self._tool_early_finish(**args)
                    result_data = json.loads(result_json)
                    if result_data.get("early_finish"):
                        await self.emit_output(f"\n[FIN] Finishing early: {result_data.get('reason', '')}\n")
                        await self.emit_status("Finishing early...")
                    else:
                        await self.emit_output(f"\n[FIN] Early finish failed: {result_data.get('error', 'Unknown error')}\n")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    continue

                # ── Handle terminate ──
                if tool_name == "terminate":
                    result = args.get("result", "Task complete.")
                    success = args.get("success", True)
                    icon = "[OK]" if success else "[FAIL]"
                    await self._save_state_to_file()
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

                # ── Handle proceed_to_output ──
                if tool_name == "proceed_to_output":
                    result_json = await self._tool_proceed_to_output(**args)
                    result_data = json.loads(result_json)
                    if result_data.get("proceed_to_output"):
                        await self.emit_output(f"\n[OUT] **Proceeding to output generation...**\n")
                        await self.emit_status("[OUT] Moving to output phase...")
                    else:
                        await self.emit_output(f"\n[OUT] **Output transition failed:** {result_data.get('error', 'Unknown error')}\n")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
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
                    if self.phase in (self.PHASE_REVIEW, self.PHASE_OUTPUT):
                        self._transition_to(self.PHASE_EXECUTE)
                    continue

                # ── Handle confirm_plan ──
                if tool_name == "confirm_plan":
                    result_json = await self._tool_confirm_plan(**args)
                    result_data = json.loads(result_json)
                    action = result_data.get("action", "")

                    # Error / timeout → stop immediately
                    if action in ("error", "timeout"):
                        error_msg = result_data.get("error", "Plan confirmation failed.")
                        await self.emit_output(f"\n[PLAN] **Plan confirmation error:** {error_msg}\n")
                        await self.emit_task_update(finalize_tasks=True)
                        await self.emit_status("Plan confirmation error", done=True)
                        return self._format_output()

                    # Cancel → stop immediately
                    if action == "cancel":
                        await self.emit_output("\nThe plan was cancelled by the user. The agent will not proceed.\n")
                        await self.emit_task_update(finalize_tasks=True)
                        await self.emit_status("Plan cancelled", done=True)
                        return self._format_output()

                    # Feedback → stay in PLAN, revise
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
                            "content": f"SYSTEM: User provided feedback on the proposed plan: {feedback}. Please revise the plan and call confirm_plan again with the updated plan.",
                        })
                        await self.emit_output(f"\n[PLAN] Plan rejected — user feedback: {feedback}\n")
                        await self.emit_status("[PLAN] Revising plan based on feedback...")
                        continue

                    # Accept (or unknown safe fallback) → extract tasks, transition
                    plan_text = args.get("plan", content or "")
                    self.task_list = self._extract_task_list(plan_text or content or "")
                    self._transition_to(self.PHASE_EXECUTE)
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

                # ── Handle run_tools_parallel ──
                if tool_name == "run_tools_parallel":
                    result_json = await self._tool_run_tools_parallel(**args)
                    result_data = json.loads(result_json)
                    if result_data.get("error"):
                        await self.emit_output(f"\n[ERR] **Parallel execution failed:** {result_data['error']}\n")
                    else:
                        executed_names = [r.get("tool_name", "?") for r in result_data.get("results", [])]
                        await self.emit_output(f"\n[PAR] **Parallel done:** {', '.join(executed_names)}\n")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    continue

                # ── Handle ask_user ──
                if tool_name == "ask_user":
                    result_json = await self._tool_ask_user(**args)
                    result_data = json.loads(result_json)
                    user_response = result_data.get("response", "")
                    skipped = result_data.get("skipped", False)
                    if skipped:
                        await self.emit_output(f"\n[ASK] **User question skipped:** {user_response}\n")
                    else:
                        await self.emit_output(f"\n[ASK] **User answered:** {user_response}\n")
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    continue

                # ── Duplicate detection ──
                sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                if recent_calls.count(sig) >= 2:
                    tool_result = f"Error: Identical call to `{tool_name}` repeated. Try a different approach."
                else:
                    recent_calls.append(sig)
                    await self.emit_status(f"Running: {tool_name}...")
                    result_str, result_files = await self._execute_tool(tool_name, args, call_id)
                    # Track and persist tool-generated files safely
                    new_files = await self._append_produced_files(result_files)
                    if new_files and self.event_emitter:
                        await self.event_emitter({
                            "type": "chat:message:files",
                            "data": {"files": new_files},
                        })
                    truncation_limit = self._get_truncation_limit()
                    tool_result = smart_truncate(result_str, truncation_limit)

                    # ── Consecutive tool-not-found tracking ──
                    if "not found in current phase" in tool_result:
                        self._consecutive_tool_misses[tool_name] = self._consecutive_tool_misses.get(tool_name, 0) + 1
                        if self._consecutive_tool_misses[tool_name] >= 3:
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

                self.history.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": call_id,
                    "name": tool_name,
                })

            # Auto-transition to REVIEW
            if self.phase == self.PHASE_EXECUTE and self.completed_tasks and len(self.completed_tasks) >= len(self.task_list):
                self._transition_to(self.PHASE_REVIEW)

            # OUTPUT phase: after tool calls in turn 1, continue to turn 2 (no tools)
            if self.phase == self.PHASE_OUTPUT and self._output_turn == 1:
                continue

    async def _disconnect_mcp_clients(self):
        """Gracefully disconnect any MCP clients opened during tool resolution."""
        if not self._mcp_clients:
            return
        for name, client in list(self._mcp_clients.items()):
            try:
                if hasattr(client, "disconnect"):
                    await asyncio.wait_for(client.disconnect(), timeout=3.0)
            except Exception:
                pass
        self._mcp_clients.clear()

    async def run(self, user_msg, last_user_msg_raw, model):
        try:
            result = await self._run_impl(user_msg, last_user_msg_raw, model)
            return result
        except GeneratorExit:
            logger.info("Agent loop cancelled by user (GeneratorExit).")
            await self._save_state_to_file()
            await self.emit_task_update(finalize_tasks=True)
            await self.emit_status("Cancelled", done=True)
            raise
        except asyncio.CancelledError:
            logger.info("Agent loop cancelled (CancelledError).")
            await self._save_state_to_file()
            await self.emit_task_update(finalize_tasks=True)
            await self.emit_status("Cancelled", done=True)
            raise
        except Exception as e:
            logger.error(f"Agent loop error: {e}", exc_info=True)
            await self.emit_task_update(finalize_tasks=True)
            return f"\n[ERROR] Agent loop failed: {e}"
        finally:
            await self._disconnect_mcp_clients()
            await self._sync_produced_files_to_db()
            if HAS_DB_PERSISTENCE and self.chat_id and self.message_id:
                snapshot = list(self.produced_files)
                asyncio.get_running_loop().create_task(
                    self._delayed_sync_with_backoff(
                        self.chat_id, self.message_id, snapshot
                    )
                )
            self._seen_file_ids.clear()

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
        DEBUG_MODE: bool = Field(
            default=False,
            description="Enable debug mode. When on, messages 'show tools', 'show mcp', or 'show skills' return the available items directly without any LLM call.",
        )
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
            default=(
                "read_file, calculate_timestamp, fetch_url, get_current_timestamp, "
                "glob_search, grep_search, list_files, list_knowledge_bases, list_memories, "
                "query_knowledge_bases, query_knowledge_files, search_knowledge_bases, search_knowledge_files, search_memories, search_notes, "
                "search_papers, search_web, view_chat, view_knowledge_file, view_note, view_skill"
            ),
            description=(
                "Comma-separated tool names allowed in PLAN phase. "
                "Leave EMPTY to allow ALL tools. "
                "Default is a read-only / research-safe set."
            )
        )
        EXECUTE_TOOLS: str = Field(
            default=(
                "add_memory, delete_memory, list_memories, replace_memory_content, search_memories, "
                "bash_command, run_command, get_process_status, list_processes, "
                "read_file, write_file, copy_file, move_file, delete_file, make_directory, compress_files, edit_file, "
                "glob_search, grep_search, list_files, "
                "fetch_url, get_current_timestamp, calculate_timestamp, "
                "search_web, search_papers, search_notes, search_knowledge_bases, search_knowledge_files, search_memories, "
                "search_chats, view_chat, view_note, view_knowledge_file, view_skill, "
                "query_knowledge_bases, query_knowledge_files, list_knowledge_bases, list_memories, "
                "calculate, render_visualization, display_file, show_map, get_weather_forecast, "
                "github_create_branch, github_create_or_update_file, github_create_pull_request"
            ),
            description=(
                "Comma-separated tool names allowed in EXECUTE phase. "
                "Leave EMPTY to allow ALL tools."
            )
        ),
        REVIEW_TOOLS: str = Field(
            default=(
                "read_file, calculate_timestamp, fetch_url, get_current_timestamp, get_process_status, run_command, "
                "glob_search, grep_search, list_files, list_knowledge_bases, list_memories, list_processes, "
                "query_knowledge_bases, query_knowledge_files, "
                "search_chats, search_knowledge_bases, search_knowledge_files, search_memories, search_notes, search_web, "
                "view_chat, view_knowledge_file, view_note, view_skill"
            ),
            description=(
                "Comma-separated tool names allowed in REVIEW phase. "
                "Leave EMPTY to allow ALL tools. "
                "Default is a read-only / review-safe set + command execution."
            )
        )
        OUTPUT_TOOLS: str = Field(
            default=(
                "show_map, get_weather_forecast, render_visualization, "
                "list_files, read_file, display_file, grep_search, glob_search, "
                "list_processes, get_process_status, get_current_timestamp, calculate_timestamp, "
                "list_knowledge_bases, list_memories, search_knowledge_bases, query_knowledge_bases, "
                "search_knowledge_files, query_knowledge_files, view_knowledge_file, "
                "search_chats, search_memories, view_chat, search_notes, view_note, view_skill"
            ),
            description=(
                "Comma-separated tool names allowed in OUTPUT phase. "
                "Leave EMPTY to allow ALL tools. "
                "Default includes rendering/output tools and read-only helpers."
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
        OUTPUT_PROMPT: str = Field(
            default=DEFAULT_OUTPUT_PROMPT,
            description="System prompt for OUTPUT phase — Turn 1 (collection / rendering allowed). Available placeholders: {goal}, {task_state}."
        )

    class UserValves(BaseModel):
        ENABLE_PLAN_APPROVAL: bool = Field(
            default=True,
            description="Enable plan confirmation UI. When off, plans are auto-approved without asking the user.",
        )
        YOLO_MODE: bool = Field(
            default=False,
            description="Skip all user confirmations. Auto-approve plans and ignore iteration limits.",
        )
        SKIP_PLAN_ON_RESUME: bool = Field(
            default=True,
            description="When the previous session is finished, skip the full PLAN phase for a new user request and jump straight to a Quick Replan. Set to False to always start fresh with full PLAN phase.",
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

        return await engine.run(user_msg, last_user_msg_raw, model)