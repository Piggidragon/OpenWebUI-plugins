"""
title: Helix Agent
author: Piggidragon
version: 0.23.1
description: >
  Helix Agent — OpenWebUI-native agent loop with modular per-phase tool control.

  Architecture:
  - SINGLE model loop (Plan -> Execute -> Review -> Replan -> Execute...)
  - Per-phase tool filtering via Valves — only relevant tools exposed to the LLM at each phase
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
5. If the goal involves writing code or creating files, include explicit verification tasks (e.g., syntax check, lint, run a quick test).
6. After creating the plan, call confirm_plan with the plan text to present it for review.

File paths: If the plan involves creating files, decide on a ONE project folder name (short slug based on the goal) under `[USER_HOME]/agent/`. ALL files for this task and any follow-up turns on the SAME topic must be written within that project folder. Do NOT write into the bare `[USER_HOME]/agent/` root.

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

File paths: All files MUST be written into the SAME project subfolder under `[USER_HOME]/agent/` (e.g. `[USER_HOME]/agent/website-redesign/`). Use a short slug based on the current task/goal. EVERY file created in this or follow-up turns for the SAME topic must go into that SAME project folder. NEVER scatter files across unrelated directories and NEVER write into the bare `[USER_HOME]/agent/` root.

Code verification: If a task involves writing code, you MUST verify syntax before calling complete_task. Use appropriate validation tools (e.g. `python -m py_compile`, `bash -n`, `node --check`, a linter, or any available syntax-check tool). Only mark the task complete once the code passes validation or the validation failure has been documented.

Rules:
- Call exactly ONE tool per step OR use run_tools_parallel for multiple independent calls.
- NEVER repeat identical failed tool calls (duplicate detection is active).
- When all tasks are done, the system will move to review automatically. You may also call early_finish(reason) if you believe you're done early.
- If a tool returns an error, analyze it and retry with corrected parameters. You do NOT need to call fix_plan for trivial errors.
- Only call fix_plan if the same task fails repeatedly (3+ attempts) or if the task design was wrong.
- Only call replan(reason, updated_tasks) if the entire approach is wrong and a new plan is needed. It will enter Replan mode where you must then call confirm_plan with the new tasks.
- If you need to think step-by-step before acting, do so — reasoning will be captured in a collapsible block.
- You MUST call complete_task(index) or fail_task(index, reason) after working on a task.
- Use `run_tools_parallel` to call multiple independent tools at once for efficiency.
- You may only use the tools listed above. Do NOT ask the user questions.
"""

DEFAULT_REVIEW_PROMPT = """\
You are in REVIEW mode. Inspect the completed work using the available tools BEFORE deciding on an action.

PHASE: REVIEW
Original goal: {goal}
Available tools: {tool_names}

{task_state}

Task status markers: [done] = completed, [FAIL: reason] = failed with reason, [    ] = not started.

What to do:
1. Use the available read-only tools (e.g. read_file, grep_search, list_files, view_knowledge_file) to verify the work. Check file contents, code correctness, and whether files exist in `[USER_HOME]/agent/<project>/`.
2. Once you have inspected the work, call exactly ONE of these tools:

- `proceed_to_output()` — Everything is done and correct. Move to the OUTPUT phase to generate the polished final answer.
- `fix_plan(reason, updated_tasks)` — Only minor fixes are needed (a task failed or needs a small correction). List just the new/corrected tasks.
- `replan(reason, updated_tasks)` — The overall strategy is broken and tasks need to be replaced entirely. The agent will enter Replan mode and you must then call confirm_plan with the updated plan.

Rules:
- ALWAYS verify before deciding. Don't guess — read files, run checks, inspect outputs.
- If there are only minor issues with individual tasks, ALWAYS prefer `fix_plan` over `replan`. Only use `replan` if the overall strategy is broken.
- Be honest — don't call `proceed_to_output` if something is missing or wrong.
- If the result is good enough, call `proceed_to_output`. Don't gold-plate.
- Provide a brief reasoning for your assessment before calling the final tool.
- You may only use the tools listed above.
"""

DEFAULT_OUTPUT_PROMPT = """\
You are in OUTPUT mode — RENDERING / VISUALISATION TURN.
This is turn 1 of 2 in the output phase.

Your ONLY job here is to call rendering or visualisation tools (e.g. render_visualization, show_map, display_file) if any are available and useful to illustrate the results for the user.
Do NOT write summary text or answer the user yet. That happens in turn 2.

Goal: {goal}

{task_state}

Available tools: {tool_names}
"""

DEFAULT_OUTPUT_FINAL_PROMPT = """\
You are in OUTPUT mode — FINAL SUMMARY PHASE.
This is turn 2 of 2 in the output phase. You may NOT use any tools.

Task:
Write a concise 3-5 sentence summary for the user that covers:
1. What was accomplished during this session (completed tasks).
2. Which files were created or modified.
3. Any failed tasks and why (if applicable).
4. Where the user can find the results (project folder under `[USER_HOME]/agent/`).

Do NOT reproduce file contents, code, or large text blocks here. All results are already persisted in files. Keep the reply short and focused.

Goal: {goal}
"""

DEFAULT_REPLAN_PROMPT = """\
You are in REPLAN mode. A previous session completed or the approach needs to change, and a new task plan is required.

PHASE: REPLAN

Available tools: {tool_names}

Recent context (previous goal):
{goal}

What to do:
1. Review the previous goal and the current request.
2. Create a minimal, focused task plan (1-3 tasks) that addresses the new request in the context of what was already done.
3. If the goal involves writing code or creating files, include explicit verification tasks (e.g., syntax check, lint, run a quick test).
4. Call confirm_plan with the updated plan.

File paths: If the plan involves creating files, decide on a ONE project folder name (short slug based on the goal) under `[USER_HOME]/agent/`. ALL files for this task and any follow-up turns on the SAME topic must be written within that project folder. Do NOT write into the bare `[USER_HOME]/agent/` root.

Plan format for confirm_plan:
- Numbered list (1., 2., 3.) with short, specific action items.

Rules:
- You may call ask_user ONCE if the new request is ambiguous.
- ABSOLUTELY CRITICAL: You MUST call confirm_plan to finish REPLAN mode. Do NOT answer the user directly. Do NOT generate story text, code, or any other output. Only call tools.
"""


# ──────────────────────────────────────────────────────────────────
#  SSE STREAM PARSER
# ──────────────────────────────────────────────────────────────────

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


async def stream_completion(request, body, user, max_retries: int = 1):
    """Stream OWUI completion, yielding structured events. Retries on transient errors."""
    body["stream"] = True
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
    PHASE_REPLAN = "replan"  # Replan when previous session finished

    # Internal tools that are ALWAYS available regardless of phase filters
    INTERNAL_TOOLS = {"terminate", "replan", "complete_task", "fail_task", "confirm_plan", "fix_plan", "proceed_to_output", "early_finish", "run_tools_parallel"}

    PHASE_INTERNAL_TOOLS = {
        PHASE_PLAN:       {"terminate", "complete_task", "fail_task", "confirm_plan", "fix_plan"},
        PHASE_EXECUTE:    {"replan", "complete_task", "fail_task", "fix_plan", "early_finish", "run_tools_parallel"},
        PHASE_REVIEW:     {"proceed_to_output", "replan", "fix_plan"},
        PHASE_OUTPUT:     set(),
        PHASE_REPLAN:    {"confirm_plan", "ask_user"},  # Replan phase has minimal internal tools
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
        self.consecutive_json_errors = 0
        self._plan_questions_asked = 0

        # Throttling / debounce state
        self._last_state_save_ts: float = 0.0
        self._last_task_state_str: str = ""
        self._last_state_save_hash: str = ""

        self._incoming_tools = dict(incoming_tools) if incoming_tools else {}

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
            if s.startswith("[PLAN]"):
                continue
            filtered.append(part)
        return "".join(filtered)

    # ── State Persistence (DB + File Attachments) ──

    async def _save_state_to_file(self, force: bool = False) -> None:
        """Serialize agent state to a JSON file and bind it to the chat DB.

        Writes are throttled: at most one every 2 seconds, and skipped if the
        state payload is unchanged since the last successful save.
        Call with force=True to bypass throttling (e.g. on termination).
        """
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
            content = json.dumps(state_data, sort_keys=True, ensure_ascii=False).encode("utf-8")
            payload_hash = hashlib.sha256(content).hexdigest()

            now = asyncio.get_event_loop().time()
            if not force:
                if now - self._last_state_save_ts < 2.0:
                    return
                if payload_hash == self._last_state_save_hash:
                    return

            filename = f"helix_state_{self.chat_id}.json"

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

    def _get_history_compression_threshold(self) -> int:
        """Return **token** threshold at which history compression should trigger.
        Derived from CONTEXT_LENGTH (tokens) * 0.70.
        """
        ctx = getattr(self.valves, "CONTEXT_LENGTH", 128000)
        return int(ctx * 0.70)

    def _get_goal_compression_threshold(self) -> int:
        """Return **token** threshold at which goal compression should trigger.
        Derived from CONTEXT_LENGTH (tokens) * 0.05.
        """
        ctx = getattr(self.valves, "CONTEXT_LENGTH", 128000)
        return int(ctx * 0.05)

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

        # 0. Resolve skills from model metadata (injected into system prompt later)
        model_info = self.app_models.get(self.body.get("model", ""), {})
        skill_ids, skill_prompt = await self._resolve_model_skills(model_info)
        self._skill_prompt = skill_prompt

        # 1. Tools from the Pipe __tools__ parameter (authoritative source)
        if self._incoming_tools:
            self.all_tools_dict.update(self._incoming_tools)

        # 2. Add internal control tools (always available)
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
                "description": "Restart planning. Use when the current approach is not working and you need to create a new plan. Preserves conversation history and context. After calling this tool, the agent will enter Replan mode where you must call confirm_plan with the updated task list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "What went wrong or why the plan needs to change"},
                        "updated_tasks": {"type": "string", "description": "Updated task list as numbered steps (only what is still needed). If empty, existing non-completed tasks are kept."},
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

    # ── Phase-aware Tool Filtering ──

    def _filter_tools_for_phase(self, phase: str):
        """Build phase_tools_dict from all_tools_dict based on Valves config."""
        # Determine which tool names are allowed for this phase
        allowlist: Set[str] = set()

        if phase == self.PHASE_PLAN:
            allowlist = set(_comma_list(self.valves.PLAN_TOOLS))
        elif phase == self.PHASE_REPLAN:
            allowlist = set(_comma_list(self.valves.PLAN_TOOLS))  # Same tools as PLAN for replan
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

            # ask_user is shown only in PLAN and REPLAN phases (planning / clarification only)
            if name == "ask_user":
                if phase in (self.PHASE_PLAN, self.PHASE_REPLAN):
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

    # ── Internal Tools ──

    async def _tool_terminate(self, **kwargs):
        return json.dumps({"terminated": True, "result": kwargs.get("result", ""), "success": kwargs.get("success", True)})

    # Replan: transition to REPLAN phase to create a new plan while preserving context.
    async def _tool_replan(self, reason: str, updated_tasks: str = "", **kwargs):
        """Process a replan: transition to REPLAN phase for the LLM to create a new plan."""
        new_tasks = self._extract_task_list(updated_tasks) if updated_tasks else []

        # Update task list if new tasks were provided
        if new_tasks:
            self.task_list = new_tasks
        else:
            # Keep only tasks that are not completed or failed
            failed_task_names = {f["task"] for f in self.failed_tasks}
            self.task_list = [
                t for t in self.task_list
                if t not in self.completed_tasks and t not in failed_task_names
            ]

        self.completed_tasks = []
        self.failed_tasks = []
        self.consecutive_json_errors = 0

        # Transition to REPLAN phase
        self._transition_to(self.PHASE_REPLAN)
        self.loop_count = 0
        self._plan_questions_asked = 0
        await self._save_state_to_file()
        await self.emit_task_update()
        return json.dumps({"replan": True, "reason": reason, "tasks": self.task_list})

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
            if t in failed_names:
                insert_idx = i
                break

        # Remove failed tasks from the task list
        self.task_list = [t for t in self.task_list if t not in failed_names]
        if insert_idx > len(self.task_list):
            insert_idx = len(self.task_list)
        self.failed_tasks = []

        self.task_list[insert_idx:insert_idx] = new_tasks

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

    async def _tool_confirm_plan(self, **kwargs):
        plan_text = kwargs.get("plan", "")
        uv = self.user_valves

        if uv and (getattr(uv, "YOLO_MODE", False) or not getattr(uv, "ENABLE_PLAN_APPROVAL", False)):
            return json.dumps({"action": "accept"})

        # Auto-accept when in REPLAN phase to keep the skip flow fast
        if self.phase == self.PHASE_REPLAN:
            return json.dumps({"action": "accept"})

        if not self.event_call:
            return json.dumps({"action": "accept"})

        # Task list extraction is deferred to the accept branch in _run_impl
        tasks = self._extract_task_list(plan_text)
        tasks_data = [{"task_id": f"T{i+1}", "description": t} for i, t in enumerate(tasks)]
        if not tasks_data:
            tasks_data = [{"task_id": "T1", "description": plan_text}]

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
        """Interactive user question tool. Only available during PLAN and REPLAN phases."""
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

        # Validate each tool exists in current phase AND validate args against schema
        missing = []
        validation_errors = []
        for i, c in enumerate(calls):
            tool_name = c["name"]
            tool_entry = self.phase_tools_dict.get(tool_name)
            if not tool_entry:
                missing.append(tool_name)
                continue
            spec = tool_entry.get("spec", {})
            errs = self._validate_tool_args(spec, c.get("args", {}))
            if errs:
                validation_errors.append(
                    {"index": i, "tool": tool_name, "errors": errs}
                )
        if missing:
            available = ", ".join(sorted(self.phase_tools_dict.keys())[:20])
            phase_name = self.phase
            return json.dumps({
                "error": f"[{phase_name}] The following tools are NOT available in this phase: {', '.join(missing)}. Available tools in this phase: {available}. Please only use tools listed for the current phase.",
            })
        if validation_errors:
            error_lines = []
            for err in validation_errors:
                error_lines.append(f"- {err['tool']}: {', '.join(err['errors'])}")
            return json.dumps({
                "error": "Validation failed for one or more parallel tool calls:\n" + "\n".join(error_lines),
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
        """Emit task progress via Open WebUI's native task list UI, debounced.

        Events are suppressed if the task state is unchanged since the last emit,
        unless finalize_tasks is True, which forces an update.
        """
        if not self.task_list:
            tasks = []
        else:
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

    async def _sync_produced_files_to_db(self) -> bool:
        """Deduplicate and bind all tool-generated files to the chat DB message.

        Returns True if the sync succeeded (or there was nothing to sync),
        False if a DB error occurred and a retry may be warranted.
        """
        if not HAS_DB_PERSISTENCE or not self.chat_id or not self.message_id:
            return True

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

    # ── Execute Tool ──

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
            """Extract allowed arg keys from various tool schema shapes."""
            parameters = spec.get("parameters") or spec.get("inputSchema")
            if isinstance(parameters, dict):
                props = parameters.get("properties")
                if isinstance(props, dict):
                    return set(props.keys())
                # Handle JSON Schema top-level properties directly
                if "properties" in parameters:
                    top_props = parameters.get("properties")
                    if isinstance(top_props, dict):
                        return set(top_props.keys())
                # Fallback: keys that aren't JSON Schema metadata keys
                meta_keys = {"type", "required", "properties", "description", "title", "$defs", "additionalProperties"}
                return {k for k in parameters.keys() if k not in meta_keys}
            # Some schemas specify args as a dict under a key named "args" or "arguments"
            for fallback_key in ("args", "arguments", "params", "input"):
                fb = spec.get(fallback_key)
                if isinstance(fb, dict):
                    nested = fb.get("properties") or {}
                    return set(nested.keys()) if isinstance(nested, dict) else set(fb.keys())
            return set()

        allowed_keys = _get_allowed_keys(target.get("spec", {}))
        if allowed_keys:
            filtered_args = {k: v for k, v in args.items() if k in allowed_keys}
        else:
            # If we can't determine schema, pass args through and rely on the tool to error
            filtered_args = dict(args)

        # ── Schema validation: check args against tool spec BEFORE execution ──
        validation_errors = self._validate_tool_args(target.get("spec", {}), filtered_args)
        if validation_errors:
            error_msg = "\n".join(f"- {err}" for err in validation_errors)
            return f"{tool_name} validation failed:\n{error_msg}", []

        callable_fn = target.get("callable")
        if not callable_fn:
            return f"Tool '{tool_name}' has no executable handler.", []

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

    def _build_system_prompt(self):
        """Build system prompt based on current phase using Valves overrides."""
        tool_names = ", ".join(sorted(self.phase_tools_dict.keys()))
        task_state = self._build_task_state()

        base = ""
        try:
            if self.phase == self.PHASE_PLAN:
                base = self._render_prompt(
                    self.valves.PLAN_PROMPT or DEFAULT_PLAN_PROMPT,
                    tool_names=tool_names,
                )
            elif self.phase == self.PHASE_REPLAN:
                base = self._render_prompt(
                    DEFAULT_REPLAN_PROMPT,
                    tool_names=tool_names,
                    goal=self.goal,
                )
            elif self.phase == self.PHASE_EXECUTE:
                base = self._render_prompt(
                    self.valves.EXECUTE_PROMPT or DEFAULT_EXECUTE_PROMPT,
                    tool_names=tool_names,
                    task_state=task_state,
                )
            elif self.phase == self.PHASE_REVIEW:
                base = self._render_prompt(
                    self.valves.REVIEW_PROMPT or DEFAULT_REVIEW_PROMPT,
                    goal=self.goal,
                    task_state=task_state,
                    tool_names=tool_names,
                )
            elif self.phase == self.PHASE_OUTPUT:
                if self._output_turn >= 2:
                    base = self._render_prompt(
                        DEFAULT_OUTPUT_FINAL_PROMPT,
                        goal=self.goal,
                        task_state=task_state,
                    )
                else:
                    base = self._render_prompt(
                        self.valves.OUTPUT_PROMPT or DEFAULT_OUTPUT_PROMPT,
                        goal=self.goal,
                        task_state=task_state,
                        tool_names=tool_names,
                    )
            else:
                base = self._render_prompt(DEFAULT_PLAN_PROMPT, tool_names=tool_names)
        except Exception:
            # Last-resort fallback if rendering itself somehow fails
            if self.phase == self.PHASE_PLAN:
                base = self._render_prompt(DEFAULT_PLAN_PROMPT, tool_names=tool_names)
            elif self.phase == self.PHASE_REPLAN:
                base = self._render_prompt(DEFAULT_REPLAN_PROMPT, tool_names=tool_names, goal=self.goal)
            elif self.phase == self.PHASE_EXECUTE:
                base = self._render_prompt(DEFAULT_EXECUTE_PROMPT, tool_names=tool_names, task_state=task_state)
            elif self.phase == self.PHASE_REVIEW:
                base = self._render_prompt(DEFAULT_REVIEW_PROMPT, goal=self.goal, task_state=task_state, tool_names=tool_names)
            elif self.phase == self.PHASE_OUTPUT:
                if self._output_turn >= 2:
                    base = self._render_prompt(DEFAULT_OUTPUT_FINAL_PROMPT, goal=self.goal, task_state=task_state)
                else:
                    base = self._render_prompt(DEFAULT_OUTPUT_PROMPT, goal=self.goal, task_state=task_state, tool_names=tool_names)
            else:
                base = self._render_prompt(DEFAULT_PLAN_PROMPT, tool_names=tool_names)

        base = base.replace("[USER_HOME]", os.path.expanduser("~"))

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
            self._output_rendering_skipped = False
            # Skip rendering turn if user enabled and display_file is not available in __tools__
            skip_rendering = getattr(self.user_valves, "SKIP_OUTPUT_RENDERING", True)
            if skip_rendering and "display_file" not in self._incoming_tools:
                self._output_rendering_skipped = True
                self._output_turn = 1  # Start at 1 so the first loop iteration is treated as turn 2 (final)
                asyncio.create_task(self.emit_status("Skipping OUTPUT rendering (display_file not available)"))
        # Rebuild filtered tools for new phase
        self._filter_tools_for_phase(phase)

    # ── Context Window Management ──

    def _manage_context_window(self, messages):
        """Trim history to MAX_HISTORY_MESSAGES while keeping tool call pairs intact."""
        max_history = getattr(self.valves, "MAX_HISTORY_MESSAGES", 100)
        if len(messages) <= max_history:
            return messages

        to_remove = len(messages) - max_history
        # Build head: goal (always preserved) – system prompt is injected separately
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
        # User override wins over admin default; -1 = use admin default, 0 = disabled
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

        try:
            body = {
                **self.body,
                "model": model,
                "messages": messages,
                "stream": False,
            }
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
                lines.append(f"[Assistant: calling tools → {tool_names}]\n{content}")
            else:
                content = strip_html(msg.get("content", ""))
                lines.append(f"[{role.capitalize()}]\n{content}")
        return "\n\n".join(lines)

    async def _compress_history_llm(self) -> bool:
        """
        Blocking LLM-based history compression.
        Interrupts the loop, compresses old messages into a single assistant summary message.
        Returns True if compression happened, False otherwise.
        """
        keep_recent = getattr(self.valves, "KEEP_RECENT_MESSAGES", 6)
        threshold = self._get_history_compression_threshold()

        # Calculate total tokens in history
        total_tokens = self._total_history_tokens()
        if total_tokens <= threshold:
            return False

        if len(self.history) <= keep_recent + 1:
            return False

        # Split: old messages (to compress) + recent messages (keep intact)
        old_messages = self.history[:-keep_recent]
        recent_messages = self.history[-keep_recent:]

        # Emit status: compression started
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

        # Build the compressed assistant message
        compressed_msg = {
            "role": "assistant",
            "content": f"=== Compressed Context ===\n{compressed}",
        }

        # Replace history: compressed msg + recent messages
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



    # ── Main Loop ──

    async def _run_impl(self, user_msg, last_user_msg_raw, model):
        await self.resolve_tools()

        # --- Debug mode intercept ---
        if getattr(self.valves, "DEBUG_MODE", False):
            msg_lower = (user_msg or "").strip().lower()
            if msg_lower in ("show tools", "debug tools", "list tools"):
                builtin_tools = []
                external_tools = []
                mcp_tools = []
                terminal_tools = []
                helix_tools = []

                for name, info in self.all_tools_dict.items():
                    if name in self.INTERNAL_TOOLS or name == "ask_user":
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

        # Attempt DB-backed state recovery at start of turn
        await self._recover_state_from_files(self.body if isinstance(self.body, dict) else {})

        # ── Attachment size guard ──
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
                file_size = None
                if HAS_DB_PERSISTENCE:
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
                if file_size is None:
                    upload_dir = getattr(self.request.app.state, "UPLOAD_DIR", None)
                    fname = file_info.get("name") or file_info.get("filename") or file_info.get("file_id") or file_info.get("id")
                    if upload_dir and fname:
                        fpath = os.path.join(upload_dir, fname)
                        if os.path.exists(fpath):
                            file_size = os.path.getsize(fpath)
                if file_size and file_size > max_bytes:
                    oversized.append((file_info.get("name", "unknown"), file_size))
            if oversized:
                items = "\n".join(f"- `{name}` ({size / (1024*1024):.1f} MB)" for name, size in oversized)
                err = (
                    f"**Error: File(s) too large ({max_size_mb} MB max)**\n\n"
                    f"The following attached file(s) exceed the maximum allowed size ({max_size_mb} MB):\n"
                    f"{items}\n\n"
                    f"**Please upload large documents to a Knowledge Base instead.**\n"
                    f"1. Go to your Workspace/Knowledge settings\n"
                    f"2. Create or select a Knowledge Base\n"
                    f"3. Upload the file there\n"
                    f"4. Link the Knowledge Base to this model\n"
                    f"5. Try again"
                )
                await self.emit_status("File too large", done=True)
                return err

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
            self.history.append({"role": "user", "content": user_msg})
            self._filter_tools_for_phase(self.phase)
            await self.emit_task_update()
        elif self.goal and not self.task_list and self.phase == self.PHASE_REPLAN:
            # Interrupted Replan (task_list empty because confirm_plan not yet called)
            logger.info("Resuming interrupted Replan phase.")
            self.loop_count = 0
            self._plan_questions_asked = 0
            if user_msg not in self.goal:
                self.goal = self.goal + "\n\nNEW REQUEST:\n" + user_msg
            self.history.append({"role": "user", "content": user_msg})
            self._filter_tools_for_phase(self.PHASE_REPLAN)
            await self.emit_task_update()
        elif self.goal and self.task_list and not has_remaining_tasks and self.user_valves and getattr(self.user_valves, "SKIP_PLAN_ON_RESUME", True):
            # Previous session finished. Use Replan phase to let the agent plan the new request.
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
            self.history = [last_user_msg_raw if last_user_msg_raw else {"role": "user", "content": user_msg}]

        recent_calls = []
        self._output_parts = []

        while True:
            effective_max = self.valves.MAX_ITERATIONS + self._extra_grace

            # ── OUTPUT phase hard limit: max 2 turns (tool prep + final text) ──
            if self.phase == self.PHASE_OUTPUT:
                self._output_turn += 1
                if self._output_turn == 1:
                    # Check if Turn 1 has any actual rendering tools configured
                    has_rendering_tools = any(
                        name not in self.INTERNAL_TOOLS
                        for name in self.phase_tools_dict.keys()
                    )
                    if not has_rendering_tools:
                        # No rendering tools available — skip Turn 1 and go straight to Final Summary
                        self._output_turn = 2
                        await self.emit_status("No rendering tools configured — skipping OUTPUT turn 1")
                if self._output_turn > 2 and not self._is_yolo_mode:
                    # Safety net: exceeded max output turns, return immediately
                    await self.emit_task_update(finalize_tasks=True)
                    await self.emit_status("Output phase exceeded max turns", done=True)
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

            self.loop_count += 1
            recent_calls = recent_calls[-30:]

            # Safety-net for REPLAN: max loops, then fallback to single-task EXECUTE
            max_replan_loops = getattr(self.valves, "MAX_REPLAN_LOOPS", 3)
            if self.phase == self.PHASE_REPLAN and self.loop_count >= max_replan_loops and not self._is_yolo_mode:
                logger.warning("REPLAN reached %d loops; falling back to single-task EXECUTE.", max_replan_loops)
                self.task_list = [user_msg]
                self._transition_to(self.PHASE_EXECUTE)
                await self.emit_task_update()

            phase_name = {
                self.PHASE_PLAN: "Plan",
                self.PHASE_REPLAN: "Replan",
                self.PHASE_EXECUTE: "Execute",
                self.PHASE_REVIEW: "Review",
                self.PHASE_OUTPUT: "Output",
            }
            name = phase_name.get(self.phase, "Loop")

            effective_max = self.valves.MAX_ITERATIONS + self._extra_grace
            await self.emit_status(f"Mode: {name}, Loop: {self.loop_count}/{effective_max}")


            self.history = self._manage_context_window(self.history)

            # ── Context Compression: check threshold and compress if needed ──
            total_tokens = self._total_history_tokens()
            if total_tokens > self._get_history_compression_threshold():
                if self.loop_count - getattr(self, "_last_compression_loop", 0) >= self.valves.COMPRESSION_INTERVAL:
                    compressed = await self._compress_history_llm()
                    if compressed:
                        self._last_compression_loop = self.loop_count

            # Build fresh system prompt and prepend it to history
            system_prompt = self._build_system_prompt()
            call_messages = [{"role": "system", "content": system_prompt}] + [m for m in self.history if m.get("role") != "system"]

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

            # Inject model knowledge for native OpenWebUI vector search
            mk = getattr(self, "_model_knowledge", None)
            if mk:
                completion_body.setdefault("metadata", {})
                if isinstance(completion_body.get("metadata"), dict):
                    completion_body["metadata"]["knowledge"] = mk
                    completion_body["metadata"]["__model_knowledge__"] = mk

            # Apply OpenWebUI file context prep (add_file_context + chat_completion_files_handler)
            try:
                completion_body["messages"] = await self._apply_file_prep(copy.deepcopy(call_messages))
            except Exception as e:
                logger.warning("_apply_file_prep failed: %s", e)

            # ── Stream LLM response ──
            tc_dict = {}
            content_chunks = []

            max_llm_retries = getattr(self.valves, "LLM_RETRY_COUNT", 1)
            async for event in stream_completion(self.request, completion_body, self.user, max_retries=max_llm_retries):
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
                    if self.phase in (self.PHASE_PLAN, self.PHASE_REPLAN):
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
                    max_errors = self.valves.MAX_CONSECUTIVE_ERRORS
                    should_stop = getattr(self.valves, "ENABLE_HARD_STOP_ON_ERRORS", False) and self.consecutive_json_errors >= max_errors and not self._is_yolo_mode
                    error_detail = f"Error: Invalid JSON in tool arguments for '{tool_name}'. The arguments provided were: {raw_args}. Please ensure they are a valid JSON object with exactly the keys expected by this tool."
                    if should_stop:
                        await self.emit_output(f"\n[ERROR] JSON parse failed {max_errors} times. Stopping.\n")
                        await self.emit_task_update(finalize_tasks=True)
                        await self.emit_status("JSON error", done=True)
                        return self._format_output()
                    args = {}
                    self.history.append({
                        "role": "tool",
                        "content": error_detail,
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
                    await self._save_state_to_file(force=True)
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
                    result_json = await self._tool_replan(reason=reason, updated_tasks=updated)
                    self.history.append({
                        "role": "tool",
                        "content": result_json,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })
                    await self.emit_output(f"\n[RPLN] **Re-planning:** {reason}\n")
                    await self.emit_status(f"[RPLN] Re-planning: {reason}")
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

                    # ── Schema validation for single tool calls ──
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
                        result_str, result_files = await self._execute_tool(tool_name, args, call_id)
                        # Track and persist tool-generated files safely
                        new_files = await self._append_produced_files(result_files)
                        if new_files and self.event_emitter:
                            await self.event_emitter({
                                "type": "chat:message:files",
                                "data": {"files": new_files},
                            })
                        truncation_limit = self._get_truncation_limit()
                        if truncation_limit and result_str and len(result_str) > truncation_limit:
                            await self.emit_status(
                                f"Truncated call {tool_name} because result was {len(result_str)}/{truncation_limit} chars"
                            )
                        tool_result = smart_truncate(result_str, truncation_limit)

                        # ── Consecutive tool-not-found tracking ──
                        if "not found in current phase" in tool_result:
                            self._consecutive_tool_misses[tool_name] = self._consecutive_tool_misses.get(tool_name, 0) + 1
                            miss_count = self._consecutive_tool_misses[tool_name]
                            max_errors = self.valves.MAX_CONSECUTIVE_ERRORS
                            available_tools = ", ".join(sorted(self.phase_tools_dict.keys())[:30])
                            warning_msg = (
                                f"[WARN] Tool '{tool_name}' is not available in the current phase ({self.phase}). "
                                f"Attempt {miss_count}. Available tools: {available_tools}. "
                                f"Please check the tool name and use only tools listed for this phase."
                            )
                            should_stop = getattr(self.valves, "ENABLE_HARD_STOP_ON_ERRORS", False) and miss_count >= max_errors
                            if should_stop:
                                await self.emit_output(f"\n[ERROR] Tool '{tool_name}' unavailable {max_errors} times. Stopping.\n")
                                await self.emit_task_update(finalize_tasks=True)
                                await self.emit_status("Tool unavailable", done=True)
                                return self._format_output()
                            # Append a strong warning that the model will see in its next turn,
                            # replacing the generic error with a more informative one.
                            tool_result = warning_msg
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

    async def run(self, user_msg, last_user_msg_raw, model):
        try:
            result = await self._run_impl(user_msg, last_user_msg_raw, model)
            # Compress goal after the agent run completes
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
            self._seen_file_ids.clear()
            self.produced_files.clear()

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
        # ── 1. Model ──
        AGENT_MODEL: str = Field(
            default="",
            description="Model ID for Helix Agent. The model MUST support native tool calling."
        )
        CONTEXT_COMPRESSION_MODEL: str = Field(
            default="",
            description="Model ID for context compression. If empty, falls back to AGENT_MODEL.",
        )

        # ── 2. Iteration & Loop Safety ──
        MAX_ITERATIONS: int = Field(
            default=100,
            description="Maximum Helix Agent iterations before stopping."
        )
        MAX_REPLAN_LOOPS: int = Field(
            default=3,
            ge=0,
            description="Safety cap: after this many REPLAN loops the agent falls back to single-task EXECUTE.",
        )
        ENABLE_HARD_STOP_ON_ERRORS: bool = Field(
            default=False,
            description="If True, the agent will hard-stop (return final output) after MAX_CONSECUTIVE_ERRORS consecutive tool call failures (e.g., JSON parse errors, unavailable tools). If False, the model receives the error in its history and is free to self-correct."
        )
        MAX_CONSECUTIVE_ERRORS: int = Field(
            default=3,
            description="Number of consecutive errors that triggers a hard stop when ENABLE_HARD_STOP_ON_ERRORS is True."
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

        # ── 3. Context & Memory ──
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

        # ── 4. Phase Tools & Prompts ──
        PLAN_TOOLS: str = Field(
            default=(
                "calculate_timestamp, fetch_url, get_current_timestamp, get_process_status, "
                "glob_search, grep_search, list_files, list_knowledge_bases, list_memories, "
                "list_processes, query_knowledge_bases, query_knowledge_files, read_file, "
                "search_calendar_events, search_channel_messages, search_channels, search_chats, "
                "search_knowledge_bases, search_knowledge_files, search_memories, search_notes, "
                "search_web, view_channel_message, view_channel_thread, view_chat, "
                "view_knowledge_file, view_note, view_skill"
            ),
            description=(
                "Comma-separated tool names allowed in PLAN phase. "
                "Leave EMPTY to allow ALL tools. "
                "Default is a read-only / research-safe set."
            )
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
                "Leave EMPTY to allow ALL tools."
            )
        ),
        REVIEW_TOOLS: str = Field(
            default=(
                "calculate_timestamp, fetch_url, get_current_timestamp, get_process_status, "
                "glob_search, grep_search, list_files, list_knowledge_bases, list_memories, "
                "list_processes, query_knowledge_bases, query_knowledge_files, read_file, "
                "run_command, search_calendar_events, search_channel_messages, search_channels, "
                "search_chats, search_knowledge_bases, search_knowledge_files, "
                "search_memories, search_notes, search_web, "
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
                "Comma-separated rendering/visualization tool names allowed in OUTPUT phase turn 1. "
                "Default is display_file for rendering produced files."
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
            description="System prompt for OUTPUT phase - Turn 1 (rendering / visualisation). Only rendering/visualisation tools are called here. Available placeholders: {goal}, {task_state}, {tool_names}.",
        )

        # ── 5. Safety & Limits ──
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

        # ── 6. Debug ──
        DEBUG_MODE: bool = Field(
            default=False,
            description="Enable debug mode. When on, messages 'show tools' return the available items directly without any LLM call.",
        )

    class UserValves(BaseModel):
        # ── 1. Behaviour ──
        YOLO_MODE: bool = Field(
            default=False,
            description="Skip all user confirmations. Auto-approve plans and ignore iteration limits.",
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
            description="If True, skip the OUTPUT phase turn 1 (rendering/visualization collection) and go straight to the final summary turn 2. Useful when no rendering/visualization tools are configured.",
        )

        # ── 2. Limits ──
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
        __tools__: list = None,
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