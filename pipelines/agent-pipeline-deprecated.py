"""
title: Agent Pipeline
author: custom
version: 0.24.0
description: >
  OpenWebUI-native integration:
  - per-user API key mapping via valves (JSON)
  - OpenWebUI context references (#knowledge/#memory/#prompt/#chat/#note/#website/#file)
  - memory/prompt/chat/document lookup via OpenWebUI API (memories: read-only)
  - Notes full CRUD with Lessons-Learned pattern (destructive ops require confirmation)
  - Knowledge Base creation and file ingestion with status polling
  - Two-tier file system: local workspace (scratch) + OWUI files (persistent/RAG)
  - OWUI file download to local workspace for editing
  - OpenWebUI file upload + direct download links + RAG processing
  - OpenWebUI knowledge retrieval & querying
  - Chat history read-only access
  - optional discovery of OpenWebUI tools/functions endpoints
  - Path traversal protection via realpath-based sandbox
  - event_call confirmation for all destructive operations
  Migration note (v21): Notes CRUD, KB management, file download, chat read,
  event_call confirmations, and path-safety fixes added.
requirements: httpx, beautifulsoup4, websockets
"""

import asyncio
import httpx
import json
import os
import re
from pydantic import BaseModel, Field, field_validator
from typing import AsyncGenerator, Optional, Any

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

PLANNING_PROMPT = """You are a precise AI agent that solves tasks step-by-step using tools.
CRITICAL: You MUST reason and output all tool calls in ENGLISH ONLY. Never use the user's language for internal steps.

Available tools: {tool_names}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE SYSTEM — TWO TIERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER 1 — Local Workspace (Hidden Scratchpad):
  - Tools: read_file, write_file, edit_file, search_in_file, list_files, delete_file, run_python
  - Purpose: Temporary working environment, intermediate data, code execution.
  - User cannot see these files directly.

TIER 2 — OpenWebUI Files (Final & Persistent):
  - Tools: openwebui_upload_file, openwebui_download_file_to_workspace
  - Purpose: Final results the user requested.
  - Rule: If you generated a file the user cares about, YOU MUST `openwebui_upload_file` it at the end.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEMORY & TODO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. MEMORIES (Read-Only):
   - On first step, call `openwebui_get_memories` to load user preferences. NEVER write or modify memories.
2. TASK TRACKING (Dual-Track):
   - Local todo.md: internal scratchpad, updated after every step.
   - OpenWebUI Tasks (openwebui_create_task / openwebui_update_task): create one OWUI task per major milestone so the user sees live progress. Update status to 'done' when the milestone is complete.

{lessons_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESTRUCTIVE ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALWAYS call `ask_user` BEFORE deleting non-temporary files or completely overwriting major notes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL GUIDANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{openwebui_tool_guidance}
- List files before reading if unsure what exists locally.
- Call `terminate` as soon as the goal is gracefully fully achieved and important results are uploaded.

On the FIRST step:
1. Break down the goal into a concise plan.
2. Initialize `todo.md` locally if the request requires tracking multiple complex steps.
3. Call exactly ONE tool to start (or terminate if done).

GOAL: {goal}"""

BASE_AGENT_PROMPT = """\
You are a precise AI agent working step-by-step.
CRITICAL: You MUST reason and output all tool calls in ENGLISH ONLY.

Available tools: {tool_names}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATE & WORKSPACE REMINDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Task Tracking (Dual-Track): keep local `todo.md` as internal scratchpad and update it after every step.
- OpenWebUI Tasks: use task create/update tools for major milestones so user-visible progress stays live; set status to `done` when a milestone is complete.
- OpenWebUI Files: persistent uploads. Upload final deliverables here.
- Memories: READ-ONLY global preferences.
{lessons_section}
- Destructive Actions: ALWAYS call `ask_user` for confirmation BEFORE deleting or destroying data.

Each step:
1. Assess the latest tool result.
2. If goal complete → ensure final results are uploaded, update tracker/lessons if necessary, then call `terminate`.
3. If blocked or need destructive confirmation → call `ask_user`.
4. Else → call exactly ONE best next tool. Do not repeat identical failed tool calls.

{openwebui_tool_guidance}

GOAL: {goal}"""

SUMMARY_PROMPT = """\
Summarise the following agent conversation concisely in ENGLISH.
Keep all facts and results. State what is done vs remaining. Max 900 words.
CRITICAL: Preserve exactly — variable names, file paths, error messages, URLs, code snippets, and numeric values. Do not paraphrase technical details.
Omit dead-end intermediate noise.
Reply with summary only."""

RESUME_PROMPT = """\
You are a precise AI agent working step-by-step. You are resuming a task after context compression.
CRITICAL: You MUST reason and output all tool calls in ENGLISH ONLY.

The progress summary is in the user message above. Continue seamlessly from where you left off.

Rules:
- You have the same tools available: {tool_names}
- Check your local `todo.md` if you lost track of exact multi-step progression.
- ALWAYS ask for confirmation (`ask_user`) before destructive actions.
- Call exact ONE tool per step. If the goal is fully achieved based on the summary, call `terminate`.

GOAL: {goal}"""

TRANSLATE_PROMPT = """\
Translate text into {target_language}. Keep natural style.
Preserve markdown, URLs, file paths, code blocks, emojis exactly.
Only translate human prose.
Reply with translation only.

TEXT:
{text}"""

LANG_DETECT_PROMPT = """\
Detect the language of this message.
Return exactly one lowercase language name (e.g. english, german, french).
No explanation.

TEXT:
{text}"""

RAG_SCORE_PROMPT = (
    "Rate relevance of this text chunk to query from 1-5.\n"
    "Query: {query}\nChunk: {chunk}\n"
    "Respond with one digit only."
)

_TRANSLATION_CACHE_MAX = 256
_TRANSLATION_CACHE_EVICT = 64
_translation_cache: dict[tuple[str, str], str] = {}

# ─────────────────────────────────────────────────────────────────────────────
#  TOOL SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOL_SCHEMAS = [
    # ── Web
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search current web info. Keep query short.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch URL and extract relevant text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
    # ── Local workspace files
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read local workspace file. Use offset/limit for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "default": 0},
                    "limit": {"type": "integer", "default": 0},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "Search text/regex in local workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "context_lines": {"type": "integer", "default": 2},
                    "max_matches": {"type": "integer", "default": 20},
                },
                "required": ["path", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write/append a text file in local workspace. Set user_visible=true to also upload to OpenWebUI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "string", "enum": ["write", "append"], "default": "write"},
                    "user_visible": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace specific text in local workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in local workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "recursive": {"type": "boolean", "default": False},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a local workspace file. Always ask_user for confirmation before calling this.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python in persistent Jupyter kernel.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },

    # ── OpenWebUI: References & context
    {
        "type": "function",
        "function": {
            "name": "openwebui_resolve_references",
            "description": "Resolve #references in user text (#knowledge #memory #prompt #chat #note #website #file).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "limit_per_type": {"type": "integer", "default": 5},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_get_memories",
            "description": "Get user memories from OpenWebUI (READ ONLY — never write memories). Endpoint: GET /api/v1/memories",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_get_prompt",
            "description": "Get a saved prompt by id/name/title from OpenWebUI prompts. Endpoint: GET /api/v1/prompts",
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string"},
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_search_entity",
            "description": "Search OpenWebUI entities: chats, documents, files, prompts, memories, tools, functions, models, notes, tasks, websites, knowledge, automations, calendars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["entity_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_upload_file",
            "description": "Upload a local workspace file to OpenWebUI (tier-2) and return download URL. Endpoint: POST /api/v1/files/",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_download_file_to_workspace",
            "description": "Download an OpenWebUI file (by file_id) into the local workspace for editing. Endpoint: GET /api/v1/files/{id}/content",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string",
                                "description": "OpenWebUI file ID from openwebui_search_entity or upload response."},
                    "local_path": {"type": "string",
                                   "description": "Relative path in local workspace to save the file."},
                },
                "required": ["file_id", "local_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_get_defaults",
            "description": "Best-effort read of OpenWebUI defaults/configs (models, rag/ranker hints, etc).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },

    # ── OpenWebUI: Retrieval / RAG
    {
        "type": "function",
        "function": {
            "name": "openwebui_retrieval_query",
            "description": "Query OpenWebUI knowledge/RAG collections for relevant chunks. Endpoint: POST /api/v1/retrieval/query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_process_file_for_rag",
            "description": "Process an uploaded file into an OpenWebUI RAG collection. Endpoint: POST /api/v1/retrieval/process/file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string"},
                    "collection_name": {"type": "string", "default": ""},
                },
                "required": ["file_id"],
            },
        },
    },

    # ── OpenWebUI: Knowledge Base management
    {
        "type": "function",
        "function": {
            "name": "openwebui_create_knowledge",
            "description": "Create a new OpenWebUI Knowledge Base. Endpoint: POST /api/v1/knowledge/create",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the new knowledge base."},
                    "description": {"type": "string", "default": ""},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_add_file_to_knowledge",
            "description": "Add an uploaded file to an existing Knowledge Base. Polls file processing status first. Endpoint: POST /api/v1/knowledge/{id}/file/add",
            "parameters": {
                "type": "object",
                "properties": {
                    "knowledge_id": {"type": "string"},
                    "file_id": {"type": "string"},
                },
                "required": ["knowledge_id", "file_id"],
            },
        },
    },

    # ── OpenWebUI: Notes CRUD
    {
        "type": "function",
        "function": {
            "name": "openwebui_get_notes",
            "description": "List and search user Notes from OpenWebUI. Endpoint: GET /api/v1/notes",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "default": ""},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_create_note",
            "description": "Create a new Note in OpenWebUI. Use for Lessons-Learned and persistent session summaries. Endpoint: POST /api/v1/notes",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string", "description": "Markdown content of the note."},
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_update_note",
            "description": "Update an existing Note. For significant content changes, ask_user for confirmation first. Endpoint: PUT /api/v1/notes/{id}",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                    "title": {"type": "string", "default": ""},
                    "content": {"type": "string", "default": ""},
                },
                "required": ["note_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_create_task",
            "description": "Create a visible task in OpenWebUI UI to show the user a step is in progress. Use alongside local todo.md for internal tracking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done"],
                        "default": "pending"
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_update_task",
            "description": "Update the status or title of an existing OpenWebUI task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_list_tasks",
            "description": "List current OpenWebUI tasks for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status_filter": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_delete_note",
            "description": "Delete a Note permanently. ALWAYS call ask_user for confirmation before this tool. Endpoint: DELETE /api/v1/notes/{id}",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                    "note_title": {"type": "string", "description": "Title for logging/confirmation display."},
                },
                "required": ["note_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_list_calendar_events",
            "description": "List calendar events from OpenWebUI. Optionally filter by date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_date": {"type": "string", "description": "Optional start of range in ISO 8601 format."},
                    "to_date": {"type": "string", "description": "Optional end of range in ISO 8601 format."},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_create_calendar_event",
            "description": "Create a calendar event in OpenWebUI. Always confirm details with user before creating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string", "description": "Start datetime in ISO 8601 format."},
                    "end": {"type": "string", "description": "Optional end datetime in ISO 8601 format."},
                    "description": {"type": "string"},
                    "reminder_minutes": {"type": "integer", "default": None},
                },
                "required": ["title", "start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_delete_calendar_event",
            "description": "Delete a calendar event by ID. ALWAYS call ask_user for confirmation first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                },
                "required": ["event_id"],
            },
        },
    },

    # ── OpenWebUI: Chats (read only)
    {
        "type": "function",
        "function": {
            "name": "openwebui_list_chats",
            "description": "List user chat sessions from OpenWebUI (read-only). Endpoint: GET /api/v1/chats",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "query": {"type": "string", "default": ""},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openwebui_get_chat",
            "description": "Get full message history of a chat session (read-only). Endpoint: GET /api/v1/chats/{chat_id}",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string"},
                },
                "required": ["chat_id"],
            },
        },
    },

    # ── OpenWebUI: Discovery
    {
        "type": "function",
        "function": {
            "name": "openwebui_list_tools_functions",
            "description": "List all OpenWebUI tools and functions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },

    # ── Conversation flow
    {
        "type": "function",
        "function": {
            "name": "openwebui_list_automations",
            "description": "List scheduled automations for the current user from OpenWebUI. Read-only — creating/deleting automations is done by the user in the UI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional search string to filter automations by name or title (case-insensitive)."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of automations to return (default 10).",
                        "default": 10
                    }
                }
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask user a question or request confirmation. Use BEFORE any destructive action (delete note, overwrite, etc).",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminate",
            "description": "Finish with final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "success": {"type": "boolean"},
                },
                "required": ["result", "success"],
            },
        },
    },
]

OPENWEBUI_TOOL_NAMES = {
    "openwebui_resolve_references",
    "openwebui_get_memories",
    "openwebui_get_prompt",
    "openwebui_search_entity",
    "openwebui_upload_file",
    "openwebui_download_file_to_workspace",
    "openwebui_get_defaults",
    "openwebui_retrieval_query",
    "openwebui_process_file_for_rag",
    "openwebui_create_knowledge",
    "openwebui_add_file_to_knowledge",
    "openwebui_get_notes",
    "openwebui_create_note",
    "openwebui_update_note",
    "openwebui_delete_note",
    "openwebui_create_task",
    "openwebui_update_task",
    "openwebui_list_tasks",
    "openwebui_list_calendar_events",
    "openwebui_create_calendar_event",
    "openwebui_delete_calendar_event",
    "openwebui_list_chats",
    "openwebui_get_chat",
    "openwebui_list_tools_functions",
    "openwebui_list_automations",
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def smart_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    w = text[:max_chars]
    for sep in (". ", ".\n", "\n\n", "\n"):
        idx = w.rfind(sep)
        if idx > max_chars // 2:
            return w[: idx + len(sep)].rstrip() + "\n[truncated]"
    return w.rstrip() + "\n[truncated]"


def smart_trim_jupyter(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    is_error = bool(re.search(r"(Error|Traceback|Exception)", output))
    if is_error:
        return "[output trimmed]\n...\n" + output[-max_chars:]
    half = max_chars // 2
    return output[:half] + "\n...[trimmed]...\n" + output[-half:]


# ─────────────────────────────────────────────────────────────────────────────
#  CONTEXT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ContextManager:
    @staticmethod
    def estimate_tokens(messages: list) -> int:
        return sum(len(str(m.get("content", ""))) for m in messages) // 3

    @staticmethod
    def needs_compression(messages: list, threshold: int) -> bool:
        return ContextManager.estimate_tokens(messages) > threshold

    @staticmethod
    def build_summary_messages(messages: list) -> list:
        text = "\n\n".join(
            f"[{m['role'].upper()}]: {m.get('content', '')}"
            for m in messages if m["role"] != "system"
        )
        return [{"role": "user", "content": f"{SUMMARY_PROMPT}\n\n---\n\n{text}"}]

    @staticmethod
    def compact_history(messages: list, max_tool_msgs: int = 6) -> list:
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
        tool_indices = [i for i, m in enumerate(messages) if m["role"] == "tool"]
        keep = set(user_indices[-6:])
        if user_indices:
            keep.add(user_indices[0])
        for i in tool_indices[-max_tool_msgs:]:
            keep.add(i)
            if i - 1 >= 0 and messages[i - 1]["role"] == "assistant":
                keep.add(i - 1)
        seen, out = set(), []
        for i, m in enumerate(messages):
            if i not in keep:
                continue
            sig = (m.get("role"), m.get("name"), str(m.get("content", ""))[:300])
            if sig not in seen:
                seen.add(sig)
                out.append(m)
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  OLLAMA CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: int, num_ctx: int,
                 max_retries: int = 2, temperature: float = 0.1):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(connect=10, read=timeout_seconds, write=30, pool=10)
        self.num_ctx = num_ctx
        self.max_retries = max_retries
        self.temperature = temperature
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=30),
            )
        return self._client

    async def aclose(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(self, model: str, messages: list, tools: list | None = None,
                   num_ctx_override: int | None = None, num_predict: int = 768,
                   temperature_override: float | None = None) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": self.temperature if temperature_override is None else temperature_override,
                "num_predict": num_predict,
                "num_ctx": num_ctx_override if num_ctx_override else self.num_ctx,
            },
        }
        if tools:
            payload["tools"] = tools
        for attempt in range(self.max_retries + 1):
            try:
                c = await self._get_client()
                r = await c.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
                return r.json()
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.TimeoutException) as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Ollama transient transport error after retries: {e}")
            except Exception:
                raise

    async def chat_text(self, model: str, messages: list, num_ctx_override: int | None = None,
                        num_predict: int = 512, temperature_override: float | None = None) -> str:
        d = await self.chat(model, messages, None, num_ctx_override, num_predict, temperature_override)
        return d.get("message", {}).get("content", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
#  OPENWEBUI CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class OpenWebUIClient:
    """
    OpenWebUI API helper covering:
    - Management API: /api/*
    - File & RAG: /api/v1/files, /api/v1/knowledge, /api/v1/retrieval
    - Notes CRUD: /api/v1/notes
    - Chats (read): /api/v1/chats
    - Proxied Ollama: /ollama/*
    """

    def __init__(self, base_url: str, timeout_seconds: int = 20):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(connect=8, read=timeout_seconds, write=20, pool=8)
        self._client: Optional[httpx.AsyncClient] = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)
        return self._client

    async def aclose(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _headers(self, api_key: str | None = None) -> dict:
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    async def get(self, path: str, api_key: str | None = None, params: dict | None = None) -> Any:
        c = await self._http()
        r = await c.get(f"{self.base_url}{path}", headers=self._headers(api_key), params=params or {})
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            return r.json()
        return {"text": r.text}

    async def get_bytes(self, path: str, api_key: str | None = None) -> bytes:
        c = await self._http()
        h = {}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        r = await c.get(f"{self.base_url}{path}", headers=h)
        r.raise_for_status()
        return r.content

    async def post(self, path: str, api_key: str | None = None, data: dict | None = None,
                   files: Any = None) -> Any:
        c = await self._http()
        if files is not None:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            r = await c.post(f"{self.base_url}{path}", headers=headers, data=data or {}, files=files)
        else:
            r = await c.post(f"{self.base_url}{path}", headers=self._headers(api_key), json=data or {})
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            return r.json()
        return {"text": r.text}

    async def put(self, path: str, api_key: str | None = None, data: dict | None = None) -> Any:
        c = await self._http()
        r = await c.put(f"{self.base_url}{path}", headers=self._headers(api_key), json=data or {})
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            return r.json()
        return {"text": r.text}

    async def delete(self, path: str, api_key: str | None = None) -> Any:
        c = await self._http()
        r = await c.delete(f"{self.base_url}{path}", headers=self._headers(api_key))
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            return r.json()
        return {"text": r.text, "status_code": r.status_code}

    # ── Best-effort helpers (existing) ──

    async def best_effort_defaults(self, api_key: str | None = None) -> dict:
        out = {"configs": None, "models": None, "ollama_tags": None, "errors": []}
        candidates = [
            ("/api/v1/configs", "configs"),
            ("/api/configs", "configs"),
            ("/api/v1/models", "models"),
            ("/api/models", "models"),
            ("/ollama/api/tags", "ollama_tags"),
        ]
        for path, key in candidates:
            try:
                out[key] = await self.get(path, api_key=api_key)
            except Exception as e:
                out["errors"].append(f"{path}: {e}")
        return out

    async def best_effort_list(self, entity_type: str, api_key: str | None = None,
                               query: str = "", limit: int = 10) -> dict:
        et = (entity_type or "").lower().strip()
        endpoints = {
            "memories": ["/api/v1/memories"],
            "prompts": ["/api/v1/prompts"],
            "chats": ["/api/v1/chats"],
            "documents": ["/api/v1/documents"],
            "files": ["/api/v1/files"],
            "tools": ["/api/v1/tools"],
            "functions": ["/api/v1/functions"],
            "models": ["/api/v1/models"],
            "notes": ["/api/v1/notes"],
            "tasks": ["/api/v1/tasks"],
            "websites": ["/api/v1/websites"],
            "knowledge": ["/api/v1/knowledge", "/api/knowledge", "/api/v1/documents", "/api/documents"],
            "retrieval": ["/api/v1/retrieval", "/api/retrieval"],
            "automations": ["/api/v1/automations"],
            "calendars": ["/api/v1/calendars"],
        }
        if et not in endpoints:
            supported = sorted(endpoints.keys())
            return {
                "error": f"Unsupported entity_type '{entity_type}'. Supported: {', '.join(supported)}",
                "supported": supported,
            }
        errs = []
        for p in endpoints[et]:
            try:
                data = await self.get(p, api_key=api_key)
                items = data if isinstance(data, list) else data.get("data", data.get("items", []))
                if not isinstance(items, list):
                    items = [items]
                if query:
                    q = query.lower()
                    items = [x for x in items if q in json.dumps(x, ensure_ascii=False).lower()]
                return {"entity_type": et, "endpoint": p, "count": len(items[:limit]), "items": items[:limit]}
            except Exception as e:
                errs.append(f"{p}: {e}")
        return {"entity_type": et, "error": "No endpoint succeeded", "errors": errs}

    async def best_effort_get_prompt(self, identifier: str, api_key: str | None = None) -> dict:
        listing = await self.best_effort_list("prompts", api_key=api_key, query=identifier, limit=20)
        if listing.get("items"):
            items = listing["items"]
            for p in items:
                fields = [str(p.get("id", "")), str(p.get("name", "")), str(p.get("title", ""))]
                if identifier in fields:
                    return {"matched": p, "via": "exact-id/name/title"}
            return {"matched": items[0], "via": "best-text-match"}
        return listing

    async def upload_file(self, file_path: str, api_key: str | None = None) -> dict:
        errs = []
        name = os.path.basename(file_path)
        for path in ("/api/v1/files",):
            try:
                with open(file_path, "rb") as f:
                    files = {"file": (name, f)}
                    resp = await self.post(path, api_key=api_key, files=files)
                file_id = None
                if isinstance(resp, dict):
                    file_id = resp.get("id") or resp.get("file_id") or (resp.get("data", {}) or {}).get("id")
                if file_id:
                    content_url = f"{self.base_url}/api/v1/files/{file_id}/content"
                    return {"uploaded": True, "endpoint": path, "id": file_id, "download_url": content_url, "raw": resp}
                return {"uploaded": True, "endpoint": path, "raw": resp}
            except Exception as e:
                errs.append(f"{path}: {e}")
        return {"uploaded": False, "errors": errs}

    async def file_status_poll(self, file_id: str, api_key: str | None = None,
                               timeout_seconds: int = 60, interval_seconds: float = 2.0) -> dict:
        """Poll GET /api/v1/files/{id}/process/status until status is 'completed' or 'failed'."""
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            try:
                resp = await self.get(f"/api/v1/files/{file_id}/process/status", api_key=api_key)
                status = resp.get("status", "")
                if status == "completed":
                    return {"ready": True, "status": status, "raw": resp}
                if status == "failed":
                    return {"ready": False, "status": status, "error": resp.get("error", ""), "raw": resp}
            except Exception as e:
                pass  # endpoint may not exist on older OWUI — treat as ready
            await asyncio.sleep(interval_seconds)
        return {"ready": True, "status": "unknown", "note": "Status endpoint timed out or unavailable; proceeding."}

    async def best_effort_retrieval_search(self, query: str, top_k: int = 5, api_key: str | None = None) -> dict:
        errs = []
        endpoints = [
            "/api/v1/retrieval/query",
            "/api/retrieval/query",
            "/api/v1/retrieval/search",
            "/api/retrieval/search",
        ]
        payloads = [
            {"query": query, "top_k": top_k},
            {"queries": [query], "k": top_k},
            {"text": query, "limit": top_k},
        ]
        for path in endpoints:
            for payload in payloads:
                try:
                    resp = await self.post(path, api_key=api_key, data=payload)
                    return {"success": True, "endpoint": path, "payload": payload, "results": resp}
                except Exception as e:
                    errs.append(f"{path} ({payload}): {e}")
        return {"success": False, "errors": errs}

    async def best_effort_process_file(self, file_id: str, collection_name: str | None = None,
                                       api_key: str | None = None) -> dict:
        errs = []
        endpoints = ["/api/v1/retrieval/process/file", "/api/retrieval/process/file"]
        payload = {"file_id": file_id}
        if collection_name:
            payload["collection_name"] = collection_name
        for path in endpoints:
            try:
                resp = await self.post(path, api_key=api_key, data=payload)
                return {"success": True, "endpoint": path, "result": resp}
            except Exception as e:
                errs.append(f"{path}: {e}")
        return {"success": False, "errors": errs}

    # ── Notes CRUD ──

    async def notes_list(self, api_key: str | None = None) -> list:
        """GET /api/v1/notes"""
        try:
            resp = await self.get("/api/v1/notes", api_key=api_key)
            if isinstance(resp, list):
                return resp
            return resp.get("data", resp.get("notes", []))
        except Exception:
            return []

    async def notes_create(self, title: str, content: str, api_key: str | None = None) -> dict:
        """POST /api/v1/notes"""
        return await self.post("/api/v1/notes", api_key=api_key, data={"title": title, "content": content})

    async def notes_update(self, note_id: str, title: str | None = None,
                           content: str | None = None, api_key: str | None = None) -> dict:
        """PUT /api/v1/notes/{id}"""
        payload = {}
        if title is not None:
            payload["title"] = title
        if content is not None:
            payload["content"] = content
        return await self.put(f"/api/v1/notes/{note_id}", api_key=api_key, data=payload)

    async def notes_delete(self, note_id: str, api_key: str | None = None) -> dict:
        """DELETE /api/v1/notes/{id}"""
        return await self.delete(f"/api/v1/notes/{note_id}", api_key=api_key)

    # ── Tasks CRUD ──

    async def tasks_list(self, api_key: str | None = None, status_filter: str = "", limit: int = 20) -> dict:
        """GET /api/v1/tasks"""
        params = {}
        if status_filter:
            params["status"] = status_filter
        resp = await self.get("/api/v1/tasks", api_key=api_key, params=params)
        items = resp if isinstance(resp, list) else resp.get("data", resp.get("items", resp.get("tasks", [])))
        if not isinstance(items, list):
            items = []
        if status_filter:
            sf = status_filter.lower()
            items = [x for x in items if str(x.get("status", "")).lower() == sf]
        return {"count": len(items[:limit]), "tasks": items[:limit]}

    async def tasks_create(self, title: str, description: str = "", status: str = "pending",
                           api_key: str | None = None) -> dict:
        """POST /api/v1/tasks"""
        payload = {"title": title, "status": status}
        if description:
            payload["description"] = description
        return await self.post("/api/v1/tasks", api_key=api_key, data=payload)

    async def tasks_update(self, task_id: str, status: str | None = None, title: str | None = None,
                           description: str | None = None, api_key: str | None = None) -> dict:
        """PUT /api/v1/tasks/{id}"""
        payload = {}
        if status is not None:
            payload["status"] = status
        if title is not None:
            payload["title"] = title
        if description is not None:
            payload["description"] = description
        return await self.put(f"/api/v1/tasks/{task_id}", api_key=api_key, data=payload)

    # ── Knowledge Base management ──

    async def knowledge_create(self, name: str, description: str = "", api_key: str | None = None) -> dict:
        """POST /api/v1/knowledge/create"""
        return await self.post("/api/v1/knowledge/create", api_key=api_key,
                               data={"name": name, "description": description})

    async def knowledge_add_file(self, knowledge_id: str, file_id: str, api_key: str | None = None) -> dict:
        """POST /api/v1/knowledge/{id}/file/add"""
        return await self.post(f"/api/v1/knowledge/{knowledge_id}/file/add",
                               api_key=api_key, data={"file_id": file_id})

    # ── Chats (read) ──

    async def chats_list(self, api_key: str | None = None, limit: int = 20, query: str = "") -> dict:
        """GET /api/v1/chats"""
        try:
            resp = await self.get("/api/v1/chats", api_key=api_key)
            items = resp if isinstance(resp, list) else resp.get("data", resp.get("chats", []))
            if not isinstance(items, list):
                items = []
            if query:
                q = query.lower()
                items = [x for x in items if q in json.dumps(x, ensure_ascii=False).lower()]
            return {"count": len(items[:limit]), "chats": items[:limit]}
        except Exception as e:
            return {"error": str(e)}

    async def chats_get(self, chat_id: str, api_key: str | None = None) -> dict:
        """GET /api/v1/chats/{chat_id}"""
        try:
            return await self.get(f"/api/v1/chats/{chat_id}", api_key=api_key)
        except Exception as e:
            return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
#  WEB RAG
# ─────────────────────────────────────────────────────────────────────────────

class WebRAG:
    def __init__(
            self,
            aux_client: OllamaClient,
            aux_model: str,
            tika_url: str,
            enabled: bool = True,
            rag_chunk_tokens: int = 180,
            rag_chunk_overlap_tokens: int = 60,
            rag_candidates: int = 8,
            rag_top_k: int = 3,
            rag_min_score: int = 3,
            rag_extract_max_chars: int = 16000,
            rag_html_max_chars: int = 300000,
            rag_pdf_max_bytes: int = 8 * 1024 * 1024,
            rag_score_chunk_preview_chars: int = 800,
            rag_score_num_predict: int = 16,
    ):
        self.aux = aux_client
        self.aux_model = aux_model
        self.tika_url = tika_url.rstrip("/") if tika_url else ""
        self.enabled = enabled
        self.rag_chunk_tokens = rag_chunk_tokens
        self.rag_chunk_overlap_tokens = rag_chunk_overlap_tokens
        self.rag_candidates = rag_candidates
        self.rag_top_k = rag_top_k
        self.rag_min_score = rag_min_score
        self.rag_extract_max_chars = rag_extract_max_chars
        self.rag_html_max_chars = rag_html_max_chars
        self.rag_pdf_max_bytes = rag_pdf_max_bytes
        self.rag_score_chunk_preview_chars = rag_score_chunk_preview_chars
        self.rag_score_num_predict = rag_score_num_predict
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_http(self):
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=8, read=25, write=20, pool=8),
                follow_redirects=True,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._http

    async def aclose(self):
        if self._http:
            await self._http.aclose()
            self._http = None

    def _chunk_text(self, text: str) -> list[str]:
        chunk_chars = self.rag_chunk_tokens * 4
        step = max(80, (self.rag_chunk_tokens - self.rag_chunk_overlap_tokens) * 4)
        chunks, i = [], 0
        while i < len(text):
            chunks.append(text[i:i + chunk_chars])
            i += step
        return chunks

    async def _score_chunk(self, chunk: str, query: str) -> int:
        try:
            prompt = RAG_SCORE_PROMPT.format(query=query, chunk=chunk[: self.rag_score_chunk_preview_chars])
            out = await self.aux.chat_text(
                self.aux_model,
                [{"role": "user", "content": prompt}],
                num_ctx_override=2048,
                num_predict=self.rag_score_num_predict
            )
            m = re.search(r"[1-5]", out)
            return int(m.group()) if m else 1
        except Exception:
            return 1

    async def extract_text(self, url: str) -> str:
        c = await self._get_http()
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgentRAG/9.1)"}
        for attempt in range(3):
            try:
                if url.lower().endswith(".pdf") and self.tika_url:
                    resp = await c.get(url, headers=headers)
                    resp.raise_for_status()
                    pdf_data = resp.content[: self.rag_pdf_max_bytes]
                    tika_resp = await c.put(
                        f"{self.tika_url}/tika",
                        content=pdf_data,
                        headers={"Accept": "text/plain", "Content-Type": "application/pdf"},
                    )
                    return tika_resp.text[: self.rag_extract_max_chars]
                resp = await c.get(url, headers=headers)
                resp.raise_for_status()
                html = resp.text[: self.rag_html_max_chars]
                html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
                html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
                html = re.sub(r"<[^>]+>", " ", html)
                return re.sub(r"\s+", " ", html).strip()[: self.rag_extract_max_chars]
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.TimeoutException):
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise

    async def fetch_relevant(self, url: str, query: str | None, max_output_tokens: int = 650) -> str:
        try:
            text = await self.extract_text(url)
        except Exception as e:
            return f"Error fetching {url}: {e}"
        if not self.enabled or not query or len(text) < 2000:
            return smart_truncate(text, max_output_tokens * 4)
        chunks = self._chunk_text(text)
        candidates = chunks[: self.rag_candidates]
        scores = await asyncio.gather(*[self._score_chunk(c, query) for c in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top = [(s, c) for s, c in ranked if s >= self.rag_min_score][: self.rag_top_k] or \
              ranked[:max(1, min(2, len(ranked)))]
        max_chars_each = (max_output_tokens * 4) // max(1, len(top))
        parts = [f"[Relevance {s}/5]\n{smart_truncate(c, max_chars_each)}" for s, c in top]
        return f"Relevant excerpts from {url}:\n\n" + "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  JUPYTER EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

class JupyterExecutor:
    def __init__(self, jupyter_url: str, token: str, persistent: bool,
                 execution_timeout_seconds: int = 30):
        self.base = jupyter_url.rstrip("/")
        self.token = "" if token in (None, "none", "None") else token
        self.persistent = persistent
        self.execution_timeout_seconds = execution_timeout_seconds
        self._kernel_id: str | None = None
        self._http: Optional[httpx.AsyncClient] = None

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"token {self.token}"
        return h

    async def _http_client(self):
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(connect=8, read=20, write=20, pool=8))
        return self._http

    async def aclose(self):
        if self._http:
            await self._http.aclose()
            self._http = None

    async def _get_or_create_kernel(self):
        if self.persistent and self._kernel_id:
            return self._kernel_id
        c = await self._http_client()
        r = await c.post(f"{self.base}/api/kernels", headers=self._headers(), json={"name": "python3"})
        r.raise_for_status()
        kid = r.json()["id"]
        if self.persistent:
            self._kernel_id = kid
        return kid

    async def run(self, code: str) -> str:
        try:
            import websockets
        except ImportError:
            return "Error: websockets package not installed."
        try:
            kid = await self._get_or_create_kernel()
        except Exception as e:
            return f"Jupyter kernel error: {e}"
        ws_scheme = "wss" if self.base.startswith("https") else "ws"
        ws_base = self.base.replace("https://", "").replace("http://", "")
        ws_url = f"{ws_scheme}://{ws_base}/api/kernels/{kid}/channels"
        if self.token:
            ws_url += f"?token={self.token}"
        execute_msg = {
            "header": {"msg_id": f"msg_{id(code)}", "msg_type": "execute_request", "version": "5.3"},
            "parent_header": {},
            "metadata": {},
            "content": {"code": code, "silent": False, "store_history": True},
        }
        outputs = []
        connect_timeout = self.execution_timeout_seconds + 10
        try:
            async with asyncio.timeout(connect_timeout):
                async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
                    await ws.send(json.dumps(execute_msg))
                    async with asyncio.timeout(self.execution_timeout_seconds):
                        while True:
                            raw = await ws.recv()
                            msg = json.loads(raw)
                            mtype = msg.get("header", {}).get("msg_type", "")
                            content = msg.get("content", {})
                            if mtype == "stream":
                                outputs.append(content.get("text", ""))
                            elif mtype == "execute_result":
                                outputs.append(content.get("data", {}).get("text/plain", ""))
                            elif mtype == "error":
                                tb = "\n".join(content.get("traceback", []))
                                tb = re.sub(r"\x1b\[[0-9;]*m", "", tb)
                                outputs.append(f"Error: {content.get('ename')}: {content.get('evalue')}\n{tb}")
                            elif mtype == "status" and content.get("execution_state") == "idle":
                                break
        except asyncio.TimeoutError:
            outputs.append(f"[Execution timed out after {self.execution_timeout_seconds}s]")
        except Exception as e:
            return f"Jupyter WebSocket error: {e}"
        result = smart_trim_jupyter("".join(outputs).strip(), 2500)
        return result if result else "(no output)"


# ─────────────────────────────────────────────────────────────────────────────
#  TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

async def _cached_translate(client: OllamaClient, text: str, lang: str,
                            model: str, num_predict: int) -> str:
    if not text.strip() or lang == "english":
        return text
    key = (text[:300], lang)
    if key in _translation_cache:
        return _translation_cache[key]
    try:
        result = await client.chat_text(
            model,
            [{"role": "user", "content": TRANSLATE_PROMPT.format(target_language=lang, text=text)}],
            num_ctx_override=4096,
            num_predict=num_predict,
        )
    except Exception:
        return text
    _translation_cache[key] = result
    if len(_translation_cache) > _TRANSLATION_CACHE_MAX:
        for k in list(_translation_cache.keys())[:_TRANSLATION_CACHE_EVICT]:
            del _translation_cache[k]
    return result


class ToolExecutor:
    def __init__(
            self,
            searxng_url: str,
            workspace_dir: str,
            web_rag: WebRAG,
            jupyter: JupyterExecutor,
            aux_client: OllamaClient,
            translation_model: str,
            user_language: str,
            max_tool_result_chars: int,
            enabled_tools: set[str],
            translate_num_predict: int,
            fetch_url_max_output_tokens: int,
            owui_client: OpenWebUIClient,
            owui_base_url: str,
            owui_api_key: str,
            current_user: dict,
            openwebui_runtime: dict,
            openwebui_allowed_entities: set[str],
            openwebui_allowed_reference_types: set[str],
            event_call=None,
            file_status_poll_timeout: int = 60,
            file_status_poll_interval: float = 2.0,
    ):
        self.searxng_url = searxng_url.rstrip("/")
        self.workspace_dir = workspace_dir
        self.web_rag = web_rag
        self.jupyter = jupyter
        self.aux_client = aux_client
        self.translation_model = translation_model
        self.user_language = user_language
        self.max_chars = max_tool_result_chars
        self.enabled_tools = enabled_tools
        self.translate_num_predict = translate_num_predict
        self.fetch_url_max_output_tokens = fetch_url_max_output_tokens
        self.owui = owui_client
        self.owui_base_url = owui_base_url.rstrip("/")
        self.owui_api_key = owui_api_key or ""
        self.current_user = current_user or {}
        self.openwebui_runtime = openwebui_runtime or {}
        self.openwebui_allowed_entities = {
            x.strip().lower() for x in (openwebui_allowed_entities or set()) if x.strip()
        }
        self.openwebui_allowed_reference_types = {
            x.strip().lower() for x in (openwebui_allowed_reference_types or set()) if x.strip()
        }
        self._event_call = event_call
        self.file_status_poll_timeout = file_status_poll_timeout
        self.file_status_poll_interval = file_status_poll_interval
        self._http: Optional[httpx.AsyncClient] = None
        self._search_cache: dict[str, str] = {}

    async def _http_client(self):
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(connect=8, read=20, write=20, pool=8))
        return self._http

    async def aclose(self):
        if self._http:
            await self._http.aclose()
            self._http = None

    async def _translate_user_visible(self, text: str) -> str:
        return await _cached_translate(
            self.aux_client,
            text,
            self.user_language,
            self.translation_model,
            self.translate_num_predict,
        )

    def _full_path(self, path: str) -> str:
        """
        Resolve path safely within workspace using realpath to prevent:
        - Directory traversal: ../../etc/passwd
        - Absolute path injection: /etc/passwd
        - Symlink-based escapes
        """
        clean = path.lstrip("/\\")
        # Ensure workspace exists so realpath can resolve it
        os.makedirs(self.workspace_dir, exist_ok=True)
        full = os.path.realpath(os.path.join(self.workspace_dir, clean))
        root = os.path.realpath(self.workspace_dir)
        if not (full == root or full.startswith(root + os.sep)):
            raise ValueError(
                f"Security error: path '{path}' escapes workspace. "
                f"Resolved: '{full}', allowed root: '{root}'"
            )
        return full

    # ── OpenWebUI reference helper ──

    def _extract_refs(self, text: str) -> list[tuple[str, str]]:
        refs = []
        for m in re.finditer(r"#([a-zA-Z_][\w\-]*)(?::([^\n#]+))?", text):
            kind = m.group(1).strip().lower()
            val = (m.group(2) or "").strip()
            refs.append((kind, val))
        return refs

    # ── Dispatch ──

    async def execute(self, name: str, args: dict) -> str:
        if name not in self.enabled_tools:
            if name in OPENWEBUI_TOOL_NAMES:
                return (
                    f"Tool '{name}' is unavailable: OpenWebUI API integration is disabled "
                    "or no valid API key is available."
                )
            return f"Tool '{name}' is disabled by configuration."
        try:
            # ── Web
            if name == "web_search":         return await self._web_search(**args)
            if name == "fetch_url":          return await self._fetch_url(**args)
            # ── Local workspace
            if name == "read_file":          return await self._read_file(**args)
            if name == "search_in_file":     return await self._search_in_file(**args)
            if name == "write_file":         return await self._write_file(**args)
            if name == "edit_file":          return await self._edit_file(**args)
            if name == "list_files":         return await self._list_files(**args)
            if name == "delete_file":        return await self._delete_file(**args)
            if name == "run_python":         return await self._run_python(**args)
            # ── OWUI context
            if name == "openwebui_resolve_references":   return await self._openwebui_resolve_references(**args)
            if name == "openwebui_get_memories":         return await self._openwebui_get_memories(**args)
            if name == "openwebui_get_prompt":           return await self._openwebui_get_prompt(**args)
            if name == "openwebui_search_entity":        return await self._openwebui_search_entity(**args)
            if name == "openwebui_upload_file":          return await self._openwebui_upload_file(**args)
            if name == "openwebui_download_file_to_workspace": return await self._openwebui_download_file_to_workspace(
                **args)
            if name == "openwebui_get_defaults":         return await self._openwebui_get_defaults(**args)
            # ── OWUI retrieval
            if name == "openwebui_retrieval_query":      return await self._openwebui_retrieval_query(**args)
            if name == "openwebui_process_file_for_rag": return await self._openwebui_process_file_for_rag(**args)
            # ── OWUI Knowledge Base management
            if name == "openwebui_create_knowledge":     return await self._openwebui_create_knowledge(**args)
            if name == "openwebui_add_file_to_knowledge": return await self._openwebui_add_file_to_knowledge(**args)
            # ── OWUI Notes
            if name == "openwebui_get_notes":            return await self._openwebui_get_notes(**args)
            if name == "openwebui_create_note":          return await self._openwebui_create_note(**args)
            if name == "openwebui_update_note":          return await self._openwebui_update_note(**args)
            if name == "openwebui_delete_note":          return await self._openwebui_delete_note(**args)
            if name == "openwebui_create_task":          return await self._openwebui_create_task(**args)
            if name == "openwebui_update_task":          return await self._openwebui_update_task(**args)
            if name == "openwebui_list_tasks":           return await self._openwebui_list_tasks(**args)
            # ── OWUI Calendar
            if name == "openwebui_list_calendar_events": return await self._openwebui_list_calendar_events(**args)
            if name == "openwebui_create_calendar_event": return await self._openwebui_create_calendar_event(**args)
            if name == "openwebui_delete_calendar_event": return await self._openwebui_delete_calendar_event(**args)
            # ── OWUI Chats (read)
            if name == "openwebui_list_chats":           return await self._openwebui_list_chats(**args)
            if name == "openwebui_get_chat":             return await self._openwebui_get_chat(**args)
            # ── OWUI Automations (read)
            if name == "openwebui_list_automations":     return await self._openwebui_list_automations(**args)
            # ── OWUI discovery
            if name == "openwebui_list_tools_functions": return await self._openwebui_list_tools_functions(**args)
            return f"Unknown tool: {name}"
        except TypeError as e:
            return f"Tool argument error ({name}): {e}"
        except ValueError as e:
            # Security errors from _full_path bubble up here
            return f"Security error ({name}): {e}"
        except Exception as e:
            return f"Tool error ({name}): {e}"

    # ── Web tools ──

    async def _web_search(self, query: str, num_results: int = 5) -> str:
        key = f"{query.lower().strip()}:{num_results}"
        if key in self._search_cache:
            return self._search_cache[key]
        c = await self._http_client()
        try:
            num_results = max(1, min(10, int(num_results)))
            r = await c.get(
                f"{self.searxng_url}/search",
                params={"q": query, "format": "json", "safesearch": "0"}
            )
            r.raise_for_status()
            results = r.json().get("results", [])[:num_results]
            if not results:
                return f"No results for: {query}"
            lines = [f"Search results for '{query}':\n"]
            for i, item in enumerate(results, 1):
                lines.append(
                    f"{i}. {smart_truncate(item.get('title', ''), 120)}\n"
                    f"   {item.get('url', '')}\n"
                    f"   {smart_truncate(item.get('content', ''), 180)}\n"
                )
            out = smart_truncate("\n".join(lines), self.max_chars)
            self._search_cache[key] = out
            return out
        except Exception as e:
            return f"Search error: {e}"

    async def _fetch_url(self, url: str, query: str | None = None) -> str:
        return await self.web_rag.fetch_relevant(url, query, max_output_tokens=self.fetch_url_max_output_tokens)

    # ── Local file tools ──

    async def _read_file(self, path: str, offset: int = 0, limit: int = 0) -> str:
        fp = self._full_path(path)
        if not os.path.exists(fp):
            return f"File not found: {path}"
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        offset = max(0, offset)
        selected = lines[offset: offset + limit] if limit and limit > 0 else lines[offset:]
        content = "".join(selected)
        header = f"Contents of {path} (lines {offset + 1}–{offset + len(selected)} of {total}):\n\n"
        return header + smart_truncate(content, self.max_chars)

    async def _write_file(self, path: str, content: str, mode: str = "write",
                          user_visible: bool = False) -> str:
        fp = self._full_path(path)
        os.makedirs(os.path.dirname(fp) or self.workspace_dir, exist_ok=True)
        final_content = content
        if user_visible:
            ext = os.path.splitext(path)[1].lower()
            non_translatable = {
                ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".cpp",
                ".c", ".h", ".hpp", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
                ".xml", ".sql", ".sh", ".bat", ".ps1"
            }
            if ext not in non_translatable:
                final_content = await self._translate_user_visible(content)
        with open(fp, "w" if mode == "write" else "a", encoding="utf-8") as f:
            f.write(final_content)
        filename = os.path.basename(fp)
        size_kb = round(os.path.getsize(fp) / 1024, 1)
        msg = f"✅ {'Written' if mode == 'write' else 'Appended'}: {path} ({size_kb} KB)\n📁 Local: {fp}"
        if user_visible and self.openwebui_runtime.get("openwebui_tools_enabled", False):
            up = await self.owui.upload_file(fp, api_key=self.owui_api_key)
            if up.get("uploaded"):
                dl = up.get("download_url", "")
                if dl:
                    msg += f"\n📥 OpenWebUI download: {dl}"
                file_id = up.get("id")
                if file_id:
                    rag_resp = await self.owui.best_effort_process_file(file_id, api_key=self.owui_api_key)
                    if rag_resp.get("success"):
                        msg += f"\n📄 Processed for RAG successfully."
                    else:
                        msg += f"\n⚠ RAG processing failed: {rag_resp.get('errors')}"
            else:
                msg += f"\n⚠ OpenWebUI upload failed: {up.get('errors')}"
        return msg

    async def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        fp = self._full_path(path)
        if not os.path.exists(fp):
            return f"File not found: {path}"
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if old_text in content:
            content = content.replace(old_text, new_text, 1)
            with open(fp, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✅ Edit applied to {path}."
        return "Error: 'old_text' not found in file."

    async def _search_in_file(self, path: str, pattern: str, context_lines: int = 2,
                              max_matches: int = 20) -> str:
        fp = self._full_path(path)
        if not os.path.exists(fp):
            return f"File not found: {path}"
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
        match_indices = [i for i, line in enumerate(lines) if regex.search(line)]
        if not match_indices:
            return f"No matches for '{pattern}' in {path}."
        total = len(match_indices)
        match_indices = match_indices[:max_matches]
        context_lines = max(0, min(context_lines, 10))
        parts = []
        for idx in match_indices:
            lo = max(0, idx - context_lines)
            hi = min(len(lines) - 1, idx + context_lines)
            block = []
            for i in range(lo, hi + 1):
                marker = ">>>" if regex.search(lines[i]) else "   "
                block.append(f"{marker} {i + 1:4d} | {lines[i].rstrip()}")
            parts.append("\n".join(block))
        header = f"Found {total} match(es) for '{pattern}' in {path}:\n"
        return smart_truncate(header + "\n---\n".join(parts), self.max_chars)

    async def _list_files(self, path: str = ".", recursive: bool = False) -> str:
        fp = self._full_path(path)
        if not os.path.exists(fp):
            return f"Path not found: {path}"
        if not os.path.isdir(fp):
            return f"Not a directory: {path}"
        lines = [f"Contents of {path}:\n"]
        if recursive:
            for root, dirs, files in os.walk(fp):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                rel_root = os.path.relpath(root, fp)
                prefix = "" if rel_root == "." else f"{rel_root}/"
                for d in sorted(dirs):
                    lines.append(f"  📁 {prefix}{d}/")
                for fn in sorted(files):
                    f_path = os.path.join(root, fn)
                    size_kb = round(os.path.getsize(f_path) / 1024, 1)
                    lines.append(f"  📄 {prefix}{fn} ({size_kb} KB)")
        else:
            for entry in sorted(os.listdir(fp)):
                if entry.startswith("."):
                    continue
                full = os.path.join(fp, entry)
                if os.path.isdir(full):
                    lines.append(f"  📁 {entry}/")
                else:
                    size_kb = round(os.path.getsize(full) / 1024, 1)
                    lines.append(f"  📄 {entry} ({size_kb} KB)")
        if len(lines) == 1:
            lines.append("  (empty)")
        return smart_truncate("\n".join(lines), self.max_chars)

    async def _delete_file(self, path: str) -> str:
        fp = self._full_path(path)
        if not os.path.exists(fp):
            return f"File not found: {path}"
        if os.path.isdir(fp):
            return f"Cannot delete directory with this tool: {path}"
        # Reject symlinks to prevent escaping via symlink delete
        if os.path.islink(fp):
            return f"Cannot delete symlinks: {path}"
        os.remove(fp)
        return f"✅ Deleted: {path}"

    async def _run_python(self, code: str) -> str:
        return await self.jupyter.run(code)

    # ── OpenWebUI: Defaults ──

    async def _openwebui_get_defaults(self) -> str:
        d = await self.owui.best_effort_defaults(api_key=self.owui_api_key)
        return smart_truncate(json.dumps(d, ensure_ascii=False, indent=2), self.max_chars)

    # ── OpenWebUI: Memories (read only) ──

    async def _openwebui_get_memories(self, query: str = "", limit: int = 20) -> str:
        d = await self.owui.best_effort_list(
            "memories", api_key=self.owui_api_key, query=query or "", limit=limit
        )
        return smart_truncate(json.dumps(d, ensure_ascii=False, indent=2), self.max_chars)

    # ── OpenWebUI: Prompts ──

    async def _openwebui_get_prompt(self, identifier: str) -> str:
        d = await self.owui.best_effort_get_prompt(identifier, api_key=self.owui_api_key)
        return smart_truncate(json.dumps(d, ensure_ascii=False, indent=2), self.max_chars)

    # ── OpenWebUI: Entity search ──

    async def _openwebui_search_entity(self, entity_type: str, query: str = "", limit: int = 10) -> str:
        entity_alias = {
            "memory": "memories", "prompt": "prompts", "chat": "chats",
            "document": "documents", "file": "files", "knowledge": "knowledge",
            "tool": "tools", "function": "functions", "model": "models",
            "note": "notes", "website": "websites",
        }
        et = entity_alias.get(entity_type.lower().strip(), entity_type.lower().strip())
        if self.openwebui_allowed_entities and et not in self.openwebui_allowed_entities:
            return f"Entity type '{entity_type}' is not allowed by configuration."
        d = await self.owui.best_effort_list(et, api_key=self.owui_api_key, query=query, limit=limit)
        return smart_truncate(json.dumps(d, ensure_ascii=False, indent=2), self.max_chars)

    # ── OpenWebUI: File upload ──

    async def _openwebui_upload_file(self, path: str) -> str:
        fp = self._full_path(path)
        if not os.path.exists(fp):
            return f"File not found in workspace: {path}"
        up = await self.owui.upload_file(fp, api_key=self.owui_api_key)
        if up.get("uploaded"):
            return smart_truncate(
                "✅ Uploaded to OpenWebUI\n" + json.dumps(up, ensure_ascii=False, indent=2),
                self.max_chars
            )
        return f"Upload failed: {up.get('errors')}"

    # ── OpenWebUI: File download to workspace ──

    async def _openwebui_download_file_to_workspace(self, file_id: str, local_path: str) -> str:
        """
        Download an OWUI file (GET /api/v1/files/{id}/content) into the local workspace.
        """
        fp = self._full_path(local_path)
        os.makedirs(os.path.dirname(fp) or self.workspace_dir, exist_ok=True)
        try:
            content_bytes = await self.owui.get_bytes(
                f"/api/v1/files/{file_id}/content", api_key=self.owui_api_key
            )
            with open(fp, "wb") as f:
                f.write(content_bytes)
            size_kb = round(len(content_bytes) / 1024, 1)
            return (
                f"✅ Downloaded OWUI file '{file_id}' to workspace path '{local_path}' ({size_kb} KB).\n"
                f"You can now use read_file / edit_file / run_python on it."
            )
        except Exception as e:
            return f"Download failed for file_id '{file_id}': {e}"

    # ── OpenWebUI: Retrieval ──

    async def _openwebui_retrieval_query(self, query: str, top_k: int = 5) -> str:
        d = await self.owui.best_effort_retrieval_search(query, top_k=top_k, api_key=self.owui_api_key)
        return smart_truncate(json.dumps(d, ensure_ascii=False, indent=2), self.max_chars)

    async def _openwebui_process_file_for_rag(self, file_id: str, collection_name: str = "") -> str:
        d = await self.owui.best_effort_process_file(
            file_id, collection_name=collection_name if collection_name else None,
            api_key=self.owui_api_key
        )
        return smart_truncate(json.dumps(d, ensure_ascii=False, indent=2), self.max_chars)

    # ── OpenWebUI: Knowledge Base management ──

    async def _openwebui_create_knowledge(self, name: str, description: str = "") -> str:
        try:
            resp = await self.owui.knowledge_create(name, description, api_key=self.owui_api_key)
            kb_id = resp.get("id", "")
            return (
                f"✅ Knowledge Base created.\n"
                f"  Name: {name}\n"
                f"  ID: {kb_id}\n"
                f"  Use openwebui_add_file_to_knowledge(knowledge_id='{kb_id}', file_id='...') to add files."
            )
        except Exception as e:
            return f"Failed to create Knowledge Base '{name}': {e}"

    async def _openwebui_add_file_to_knowledge(self, knowledge_id: str, file_id: str) -> str:
        """
        Polls GET /api/v1/files/{id}/process/status first, then
        calls POST /api/v1/knowledge/{id}/file/add.
        """
        # Step 1: Poll file processing status
        status = await self.owui.file_status_poll(
            file_id, api_key=self.owui_api_key,
            timeout_seconds=self.file_status_poll_timeout,
            interval_seconds=self.file_status_poll_interval,
        )
        if not status.get("ready"):
            return (
                f"File '{file_id}' processing failed or timed out. "
                f"Status: {status.get('status')}. Error: {status.get('error', '')}. "
                f"Cannot add to Knowledge Base."
            )
        note = f" (Note: {status['note']})" if status.get("note") else ""
        # Step 2: Add to KB
        try:
            resp = await self.owui.knowledge_add_file(
                knowledge_id, file_id, api_key=self.owui_api_key
            )
            return (
                    f"✅ File '{file_id}' added to Knowledge Base '{knowledge_id}'.{note}\n"
                    + json.dumps(resp, ensure_ascii=False, indent=2)
            )
        except Exception as e:
            return f"Failed to add file '{file_id}' to KB '{knowledge_id}': {e}{note}"

    # ── OpenWebUI: Notes CRUD ──

    async def _openwebui_get_notes(self, query: str = "", limit: int = 20) -> str:
        notes = await self.owui.notes_list(api_key=self.owui_api_key)
        if query:
            q = query.lower()
            notes = [n for n in notes if q in json.dumps(n, ensure_ascii=False).lower()]
        notes = notes[:limit]
        return smart_truncate(
            json.dumps({"count": len(notes), "notes": notes}, ensure_ascii=False, indent=2),
            self.max_chars
        )

    async def _openwebui_create_note(self, title: str, content: str) -> str:
        try:
            resp = await self.owui.notes_create(title, content, api_key=self.owui_api_key)
            note_id = resp.get("id", "")
            return (
                f"✅ Note created.\n"
                f"  Title: {title}\n"
                f"  ID: {note_id}\n"
                f"  The note is now accessible in OpenWebUI and can be referenced as #note:{title}"
            )
        except Exception as e:
            return f"Failed to create note '{title}': {e}"

    async def _openwebui_update_note(self, note_id: str, title: str = "",
                                     content: str = "") -> str:
        try:
            resp = await self.owui.notes_update(
                note_id,
                title=title if title else None,
                content=content if content else None,
                api_key=self.owui_api_key,
            )
            return f"✅ Note '{note_id}' updated.\n" + json.dumps(resp, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Failed to update note '{note_id}': {e}"

    async def _openwebui_delete_note(self, note_id: str, note_title: str = "") -> str:
        """
        Deletes a note. The agent MUST call ask_user for confirmation BEFORE calling this tool.
        This is enforced by instruction — the tool itself does not re-confirm to avoid double-prompts.
        """
        try:
            resp = await self.owui.notes_delete(note_id, api_key=self.owui_api_key)
            label = f"'{note_title}' ({note_id})" if note_title else f"'{note_id}'"
            return f"✅ Note {label} permanently deleted."
        except Exception as e:
            return f"Failed to delete note '{note_id}': {e}"

    async def _openwebui_create_task(self, title: str, description: str = "", status: str = "pending") -> str:
        try:
            allowed_status = {"pending", "in_progress", "done"}
            if status not in allowed_status:
                return f"Invalid status '{status}'. Allowed: pending, in_progress, done."
            resp = await self.owui.tasks_create(
                title,
                description=description,
                status=status,
                api_key=self.owui_api_key,
            )
            task_id = ""
            if isinstance(resp, dict):
                task_id = str(resp.get("id") or (resp.get("data", {}) or {}).get("id") or "")
            return (
                f"✅ Task created.\n"
                f"  Title: {title}\n"
                f"  Status: {status}\n"
                f"  ID: {task_id or '(unknown)'}"
            )
        except Exception as e:
            return f"Failed to create task '{title}': {e}"

    async def _openwebui_update_task(self, task_id: str, status: str | None = None,
                                     title: str | None = None, description: str | None = None) -> str:
        try:
            if status is not None and status not in {"pending", "in_progress", "done"}:
                return f"Invalid status '{status}'. Allowed: pending, in_progress, done."
            if status is None and title is None and description is None:
                return "No update fields provided. Set at least one of: status, title, description."
            resp = await self.owui.tasks_update(
                task_id,
                status=status,
                title=title,
                description=description,
                api_key=self.owui_api_key,
            )
            return f"✅ Task '{task_id}' updated.\n" + json.dumps(resp, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Failed to update task '{task_id}': {e}"

    async def _openwebui_list_tasks(self, status_filter: str = "", limit: int = 20) -> str:
        try:
            if status_filter and status_filter not in {"pending", "in_progress", "done"}:
                return f"Invalid status_filter '{status_filter}'. Allowed: pending, in_progress, done."
            d = await self.owui.tasks_list(
                api_key=self.owui_api_key,
                status_filter=status_filter,
                limit=limit,
            )
            return smart_truncate(json.dumps(d, ensure_ascii=False, indent=2), self.max_chars)
        except Exception as e:
            return f"Failed to list tasks: {e}"

    async def _openwebui_list_calendar_events(self, from_date: str = "", to_date: str = "",
                                              limit: int = 20) -> str:
        try:
            params = {}
            if from_date:
                params["from_date"] = from_date
            if to_date:
                params["to_date"] = to_date
            resp = await self.owui.get("/api/v1/calendars", api_key=self.owui_api_key, params=params)
            events = resp if isinstance(resp, list) else resp.get("data", resp.get("items", resp.get("events", [])))
            if not isinstance(events, list):
                events = []
            events = events[:limit]
            return smart_truncate(
                json.dumps({"count": len(events), "events": events}, ensure_ascii=False, indent=2),
                self.max_chars
            )
        except Exception as e:
            return f"Failed to list calendar events: {e}"

    async def _openwebui_create_calendar_event(self, title: str, start: str, end: str = "",
                                               description: str = "",
                                               reminder_minutes: int | None = None) -> str:
        try:
            payload = {"title": title, "start": start}
            if end:
                payload["end"] = end
            if description:
                payload["description"] = description
            if reminder_minutes is not None:
                payload["reminder_minutes"] = reminder_minutes
            resp = await self.owui.post("/api/v1/calendars", api_key=self.owui_api_key, data=payload)
            event_id = ""
            if isinstance(resp, dict):
                event_id = str(resp.get("id") or (resp.get("data", {}) or {}).get("id") or "")
            return (
                f"✅ Calendar event created.\n"
                f"  Title: {title}\n"
                f"  ID: {event_id or '(unknown)'}"
            )
        except Exception as e:
            return f"Failed to create calendar event '{title}': {e}"

    async def _openwebui_delete_calendar_event(self, event_id: str) -> str:
        try:
            await self.owui.delete(f"/api/v1/calendars/{event_id}", api_key=self.owui_api_key)
            return f"✅ Calendar event '{event_id}' deleted."
        except Exception as e:
            return f"Failed to delete calendar event '{event_id}': {e}"

    # ── OpenWebUI: Chats (read only) ──

    async def _openwebui_list_chats(self, limit: int = 20, query: str = "") -> str:
        d = await self.owui.chats_list(api_key=self.owui_api_key, limit=limit, query=query)
        return smart_truncate(json.dumps(d, ensure_ascii=False, indent=2), self.max_chars)

    async def _openwebui_get_chat(self, chat_id: str) -> str:
        d = await self.owui.chats_get(chat_id, api_key=self.owui_api_key)
        return smart_truncate(json.dumps(d, ensure_ascii=False, indent=2), self.max_chars)

    async def _openwebui_list_automations(self, query: str = "", limit: int = 10) -> str:
        try:
            d = await self.owui.get("/api/v1/automations", api_key=self.owui_api_key)
            # Handle if response is {"data": [...]} or just a list
            items = d if isinstance(d, list) else d.get("data", [])
            if query:
                q = query.lower()
                items = [item for item in items if
                         q in item.get("name", "").lower() or q in item.get("title", "").lower()]

            items = items[:limit]
            res = {"count": len(items), "items": items}
            return json.dumps(res, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Error listing automations: {str(e)}"

    # ── OpenWebUI: Discovery ──

    async def _openwebui_list_tools_functions(self) -> str:
        tools = await self.owui.best_effort_list("tools", api_key=self.owui_api_key, limit=20)
        functions = await self.owui.best_effort_list("functions", api_key=self.owui_api_key, limit=20)
        return smart_truncate(
            json.dumps({"tools": tools, "functions": functions}, ensure_ascii=False, indent=2),
            self.max_chars
        )

    # ── OpenWebUI: References resolver ──

    async def _openwebui_resolve_references(self, text: str, limit_per_type: int = 5) -> str:
        refs = self._extract_refs(text)
        if not refs:
            return "No #references found."
        out = {"references": [], "notes": []}
        kind_alias = {
            "knowledge": "knowledge", "kb": "knowledge",
            "memory": "memories", "memories": "memories",
            "prompt": "prompts", "prompts": "prompts",
            "chat": "chats", "chats": "chats",
            "note": "notes", "notes": "notes",
            "website": "websites", "web": "websites",
            "file": "files", "files": "files",
            "document": "documents", "documents": "documents",
            "tool": "tools", "function": "functions", "model": "models",
        }
        for kind, val in refs:
            canonical_kind = kind_alias.get(kind, kind)
            if (self.openwebui_allowed_reference_types and
                    kind not in self.openwebui_allowed_reference_types and
                    canonical_kind not in self.openwebui_allowed_reference_types):
                out["notes"].append(f"Reference type '#{kind}' is not allowed by configuration.")
                continue
            et = kind_alias.get(kind, "")
            if not et:
                out["notes"].append(f"Unknown ref type '#{kind}'.")
                continue
            q = val or ""
            if et == "knowledge" and q:
                retrieval = await self.owui.best_effort_retrieval_search(
                    q, top_k=limit_per_type, api_key=self.owui_api_key
                )
                if retrieval.get("success"):
                    out["references"].append(
                        {"ref": f"#{kind}:{val}", "resolved": retrieval, "method": "retrieval_search"}
                    )
                    continue
                else:
                    out["notes"].append(
                        f"Retrieval query for '#{kind}:{val}' failed. Falling back to entity search."
                    )
            d = await self.owui.best_effort_list(
                et, api_key=self.owui_api_key, query=q, limit=limit_per_type
            )
            out["references"].append(
                {"ref": f"#{kind}:{val}" if val else f"#{kind}", "resolved": d, "method": "entity_search"}
            )
        return smart_truncate(json.dumps(out, ensure_ascii=False, indent=2), self.max_chars)


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:
    class Valves(BaseModel):
        # ── Models
        main_agent_model: str = Field(default=os.getenv("PIPE_MAIN_AGENT_MODEL", "gemma4:26b"))
        summary_model: str = Field(default=os.getenv("PIPE_SUMMARY_MODEL", "gemma4:e2b"))
        translation_model: str = Field(default=os.getenv("PIPE_TRANSLATION_MODEL", "gemma4:e2b"))
        language_detection_model: str = Field(default=os.getenv("PIPE_LANGUAGE_DETECTION_MODEL", "gemma4:e2b"))
        ranker_model: str = Field(default=os.getenv("PIPE_RANKER_MODEL", "gemma4:e2b"))

        # ── Servers
        main_ollama_url: str = Field(
            default=os.getenv("PIPE_MAIN_OLLAMA_URL", os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")))
        aux_ollama_url: Optional[str] = Field(default=os.getenv("PIPE_AUX_OLLAMA_URL", ""))
        use_aux_services: bool = Field(default=os.getenv("PIPE_USE_AUX_SERVICES", "false").lower() == "true")
        searxng_url: str = Field(default=os.getenv("PIPE_SEARXNG_URL", "http://searxng:8080"))
        tika_url: str = Field(default=os.getenv("PIPE_TIKA_URL", "http://tika:9998"))
        jupyter_url: str = Field(default=os.getenv("PIPE_JUPYTER_URL", "http://jupyter:8888"))

        # ── OpenWebUI base
        openwebui_base_url: str = Field(
            default=os.getenv("PIPE_OPENWEBUI_BASE_URL", "http://open-webui:8080"),
            description="Base URL of OpenWebUI for management API (/api/*) and file download links."
        )
        enable_openwebui_api: bool = Field(
            default=os.getenv("PIPE_ENABLE_OPENWEBUI_API", "false").lower() == "true",
            description="Master switch for all OpenWebUI API integration."
        )
        enable_user_api_key_mapping: bool = Field(
            default=os.getenv("PIPE_ENABLE_USER_API_KEY_MAPPING", "false").lower() == "true",
            description="Enable per-user API key mapping from openwebui_user_api_keys_json."
        )
        allow_openwebui_default_api_key_fallback: bool = Field(
            default=os.getenv("PIPE_ALLOW_OPENWEBUI_DEFAULT_API_KEY_FALLBACK", "false").lower() == "true",
            description="Allow fallback to openwebui_default_api_key when user mapping does not match."
        )
        allow_mapping_default_key: bool = Field(
            default=os.getenv("PIPE_ALLOW_MAPPING_DEFAULT_KEY", "false").lower() == "true",
            description="Allow use of 'default' key from openwebui_user_api_keys_json."
        )
        openwebui_key_resolution_order: str = Field(
            default=os.getenv("PIPE_OPENWEBUI_KEY_RESOLUTION_ORDER", "id,username,email"),
            description="CSV lookup order for mapping keys (id,username,email)."
        )
        openwebui_disable_tools_without_api_key: bool = Field(
            default=os.getenv("PIPE_OPENWEBUI_DISABLE_TOOLS_WITHOUT_API_KEY", "true").lower() == "true",
            description="Disable OpenWebUI tools when no API key is available."
        )
        openwebui_expose_diagnostics: bool = Field(
            default=os.getenv("PIPE_OPENWEBUI_EXPOSE_DIAGNOSTICS", "false").lower() == "true",
            description="Emit safe OpenWebUI diagnostics on startup (no secrets)."
        )
        openwebui_allowed_entities_csv: str = Field(
            default=os.getenv("PIPE_OPENWEBUI_ALLOWED_ENTITIES_CSV", ""),
            description="CSV allowlist for openwebui_search_entity entity types. Empty = allow all."
        )
        openwebui_allowed_reference_types_csv: str = Field(
            default=os.getenv("PIPE_OPENWEBUI_ALLOWED_REFERENCE_TYPES_CSV", ""),
            description="CSV allowlist for openwebui_resolve_references #types. Empty = allow all."
        )

        # ── API key mapping
        openwebui_user_api_keys_json: str = Field(
            default=os.getenv("PIPE_OPENWEBUI_USER_API_KEYS_JSON", "{}"),
            description=(
                'JSON: {"usernames":{"alice":"sk-..."},"emails":{"a@b.com":"sk-..."},'
                '"ids":{"123":"sk-..."},"default":"sk-..."}'
            )
        )
        openwebui_default_api_key: str = Field(
            default=os.getenv("PIPE_OPENWEBUI_DEFAULT_API_KEY", ""),
            description="Fallback API key if no user mapping matched."
        )

        # ── Jupyter
        jupyter_token: Optional[str] = Field(default=os.getenv("PIPE_JUPYTER_TOKEN", ""))
        jupyter_persistent_kernel: bool = Field(
            default=os.getenv("PIPE_JUPYTER_PERSISTENT", "true").lower() == "true")
        jupyter_execution_timeout_seconds: int = Field(
            default=int(os.getenv("PIPE_JUPYTER_EXEC_TIMEOUT", "45")))

        # ── Workspace
        workspace_dir: str = Field(default=os.getenv("PIPE_WORKSPACE_DIR", "/workspace"))
        user_scoped_workspace: bool = Field(
            default=os.getenv("PIPE_USER_SCOPED_WORKSPACE", "true").lower() == "true",
            description="If true, files are isolated under /workspace/<user-id>/"
        )

        # ── Runtime limits
        agent_num_ctx: int = Field(default=int(os.getenv("PIPE_NUM_CTX", "32768")))
        request_timeout_seconds: int = Field(default=int(os.getenv("PIPE_REQUEST_TIMEOUT", "180")))
        max_iterations: int = Field(default=int(os.getenv("PIPE_MAX_ITERATIONS", "16")))
        context_token_threshold: int = Field(default=int(os.getenv("PIPE_CONTEXT_THRESHOLD", "7000")))
        max_tool_result_chars: int = Field(default=int(os.getenv("PIPE_MAX_TOOL_RESULT_CHARS", "4200")))
        max_recent_tool_messages: int = Field(default=int(os.getenv("PIPE_MAX_RECENT_TOOL_MESSAGES", "8")))

        # ── Generation
        temperature_main: float = Field(default=float(os.getenv("PIPE_TEMPERATURE_MAIN", "0.05")))
        temperature_aux: float = Field(default=float(os.getenv("PIPE_TEMPERATURE_AUX", "0.05")))
        main_num_predict_step: int = Field(default=int(os.getenv("PIPE_MAIN_NUM_PREDICT_STEP", "1100")))
        summary_num_predict: int = Field(default=int(os.getenv("PIPE_SUMMARY_NUM_PREDICT", "800")))
        translate_num_predict: int = Field(default=int(os.getenv("PIPE_TRANSLATE_NUM_PREDICT", "1200")))
        lang_detect_num_predict: int = Field(default=int(os.getenv("PIPE_LANG_DETECT_NUM_PREDICT", "24")))
        rag_score_num_predict: int = Field(default=int(os.getenv("PIPE_RAG_SCORE_NUM_PREDICT", "24")))

        # ── Web-RAG
        web_rag_enabled: bool = Field(
            default=os.getenv("PIPE_WEB_RAG_ENABLED", "true").lower() == "true")
        rag_chunk_tokens: int = Field(default=int(os.getenv("PIPE_RAG_CHUNK_TOKENS", "260")))
        rag_chunk_overlap_tokens: int = Field(default=int(os.getenv("PIPE_RAG_CHUNK_OVERLAP", "90")))
        rag_candidates: int = Field(default=int(os.getenv("PIPE_RAG_CANDIDATES", "12")))
        rag_top_k: int = Field(default=int(os.getenv("PIPE_RAG_TOP_K", "4")))
        rag_min_score: int = Field(default=int(os.getenv("PIPE_RAG_MIN_SCORE", "3")))
        rag_extract_max_chars: int = Field(default=int(os.getenv("PIPE_RAG_EXTRACT_MAX_CHARS", "24000")))
        rag_html_max_chars: int = Field(default=int(os.getenv("PIPE_RAG_HTML_MAX_CHARS", "450000")))
        rag_pdf_max_bytes: int = Field(
            default=int(os.getenv("PIPE_RAG_PDF_MAX_BYTES", str(12 * 1024 * 1024))))
        rag_score_chunk_preview_chars: int = Field(
            default=int(os.getenv("PIPE_RAG_SCORE_PREVIEW_CHARS", "1200")))
        fetch_url_max_output_tokens: int = Field(
            default=int(os.getenv("PIPE_FETCH_URL_MAX_OUTPUT_TOKENS", "950")))

        # ── Retries
        http_max_retries: int = Field(default=int(os.getenv("PIPE_HTTP_MAX_RETRIES", "3")))
        aux_timeout_seconds: int = Field(default=int(os.getenv("PIPE_AUX_TIMEOUT", "90")))

        # ── File status polling (for Knowledge Base file ingestion)
        owui_file_status_poll_timeout_seconds: int = Field(
            default=int(os.getenv("PIPE_OWUI_FILE_POLL_TIMEOUT", "60")),
            description="Max seconds to poll file processing status before proceeding."
        )
        owui_file_status_poll_interval_seconds: float = Field(
            default=float(os.getenv("PIPE_OWUI_FILE_POLL_INTERVAL", "2.0")),
            description="Seconds between file processing status polls."
        )

        # ── Notes / Lessons Learned
        notes_lesson_learned_enabled: bool = Field(
            default=os.getenv("PIPE_NOTES_LESSON_LEARNED_ENABLED", "true").lower() == "true",
            description="Allow agent to create Lessons-Learned notes at end of sessions."
        )
        notes_lesson_learned_title_prefix: str = Field(
            default=os.getenv("PIPE_NOTES_LESSON_LEARNED_PREFIX", "Agent Lessons"),
            description="Title prefix for Lessons-Learned notes, e.g. 'Agent Lessons: topic'."
        )

        # ── Tool toggles: Web
        enable_tool_web: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_WEB", "true").lower() == "true")
        # ── Tool toggles: Local workspace
        enable_tool_read_file: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_READ_FILE", "true").lower() == "true")
        enable_tool_search_in_file: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_SEARCH_IN_FILE", "true").lower() == "true")
        enable_tool_write_file: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_WRITE_FILE", "true").lower() == "true")
        enable_tool_edit_file: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_EDIT_FILE", "true").lower() == "true")
        enable_tool_run_python: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_RUN_PYTHON", "true").lower() == "true")
        enable_tool_list_files: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_LIST_FILES", "true").lower() == "true")
        enable_tool_delete_file: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_DELETE_FILE", "true").lower() == "true")
        enable_tool_ask_user: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_ASK_USER", "true").lower() == "true")
        # ── Tool toggles: OWUI context & references
        enable_tool_openwebui_refs: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_OPENWEBUI_REFS", "true").lower() == "true")
        enable_tool_openwebui_memories: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_OPENWEBUI_MEMORIES", "true").lower() == "true",
            description="Allow agent to READ user memories (read-only).")
        enable_tool_openwebui_prompts: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_OPENWEBUI_PROMPTS", "true").lower() == "true")
        enable_tool_openwebui_search: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_OPENWEBUI_SEARCH", "true").lower() == "true")
        enable_tool_openwebui_upload: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_OPENWEBUI_UPLOAD", "true").lower() == "true")
        enable_tool_openwebui_download: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_OPENWEBUI_DOWNLOAD", "true").lower() == "true",
            description="Allow agent to download OWUI files into local workspace.")
        enable_tool_openwebui_defaults: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_OPENWEBUI_DEFAULTS", "true").lower() == "true")
        # ── Tool toggles: Notes
        enable_tool_notes_read: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_NOTES_READ", "true").lower() == "true",
            description="Allow agent to list and read Notes.")
        enable_tool_notes_write: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_NOTES_WRITE", "true").lower() == "true",
            description="Allow agent to create and update Notes.")
        enable_tool_notes_delete: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_NOTES_DELETE", "true").lower() == "true",
            description="Allow agent to delete Notes (always requires confirmation).")
        # ── Tool toggles: Tasks
        enable_tool_tasks_read: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_TASKS_READ", "true").lower() == "true",
            description="Allow agent to list OpenWebUI tasks.")
        enable_tool_tasks_write: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_TASKS_WRITE", "true").lower() == "true",
            description="Allow agent to create and update OpenWebUI tasks.")
        # ── Tool toggles: Calendar
        enable_tool_calendar_read: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_CALENDAR_READ", "true").lower() == "true",
            description="Allow agent to list calendar events.")
        enable_tool_calendar_write: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_CALENDAR_WRITE", "true").lower() == "true",
            description="Allow agent to create calendar events.")
        enable_tool_calendar_delete: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_CALENDAR_DELETE", "true").lower() == "true",
            description="Allow agent to delete calendar events (always requires confirmation).")
        # ── Tool toggles: Knowledge Base
        enable_tool_knowledge_management: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_KNOWLEDGE_MGMT", "true").lower() == "true",
            description="Allow agent to create Knowledge Bases and add files to them.")
        # ── Tool toggles: Chats (read)
        enable_tool_chats_readonly: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_CHATS_READONLY", "true").lower() == "true",
            description="Allow agent to read chat history (list + get).")
        # ── Tool toggles: Automations (read)
        enable_tool_automations_readonly: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_AUTOMATIONS_READONLY", "false").lower() == "true",
            description="Allow agent to list scheduled automations (read-only).")
        # ── Tool toggles: Retrieval
        enable_tool_openwebui_retrieval: bool = Field(
            default=os.getenv("PIPE_ENABLE_TOOL_OPENWEBUI_RETRIEVAL", "true").lower() == "true")

        @field_validator("jupyter_token", mode="before")
        @classmethod
        def normalize_jupyter_token(cls, v):
            if v is None:
                return ""
            sv = str(v).strip()
            if sv.lower() in ("none", "null"):
                return ""
            return sv

        @field_validator("rag_min_score")
        @classmethod
        def validate_rag_min_score(cls, v):
            return max(1, min(5, int(v)))

    def __init__(self):
        self.name = "Agent Pipeline"
        self.valves = self.Valves()
        self._jupyter_executor: Optional[JupyterExecutor] = None
        self._jupyter_lock: asyncio.Lock = asyncio.Lock()

    def _resolve_aux_url(self) -> str:
        if self.valves.use_aux_services and self.valves.aux_ollama_url.strip():
            return self.valves.aux_ollama_url
        return self.valves.main_ollama_url

    async def _detect_user_language(self, aux_client: OllamaClient, messages: list) -> str:
        msg = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        if not msg.strip():
            return "english"
        try:
            out = await aux_client.chat_text(
                self.valves.language_detection_model,
                [{"role": "user", "content": LANG_DETECT_PROMPT.format(text=msg[:1500])}],
                num_ctx_override=2048,
                num_predict=self.valves.lang_detect_num_predict,
            )
            lang = re.sub(r"[^a-z]", "", out.lower())[:32]
            return lang or "english"
        except Exception:
            return "english"

    async def _translate(self, aux_client: OllamaClient, text: str, lang: str) -> str:
        return await _cached_translate(
            aux_client,
            text,
            lang,
            self.valves.translation_model,
            self.valves.translate_num_predict,
        )

    def _parse_user_key_map(self) -> dict:
        raw = (self.valves.openwebui_user_api_keys_json or "").strip()
        if not raw:
            return {"ids": {}, "usernames": {}, "emails": {}, "default": ""}
        try:
            d = json.loads(raw)
        except Exception:
            return {"ids": {}, "usernames": {}, "emails": {}, "default": ""}
        if not isinstance(d, dict):
            return {"ids": {}, "usernames": {}, "emails": {}, "default": ""}

        def parse_section(name: str, normalizer) -> dict[str, str]:
            section = d.get(name, {})
            if not isinstance(section, dict):
                return {}
            out = {}
            for k, v in section.items():
                nk = normalizer(k)
                if not nk:
                    continue
                if not isinstance(v, str) or not v.strip():
                    continue
                out[nk] = v.strip()
            return out

        return {
            "ids": parse_section("ids", lambda x: str(x).strip()),
            "usernames": parse_section("usernames", lambda x: str(x).strip().lower()),
            "emails": parse_section("emails", lambda x: str(x).strip().lower()),
            "default": d.get("default", "").strip() if isinstance(d.get("default"), str) else "",
        }

    @staticmethod
    def _parse_csv_values(csv_text: str) -> set[str]:
        return {x.strip().lower() for x in (csv_text or "").split(",") if x.strip()}

    def _parse_openwebui_key_resolution_order(self) -> list[str]:
        allowed = {"id", "username", "email"}
        order = []
        for item in [
            x.strip().lower()
            for x in (self.valves.openwebui_key_resolution_order or "").split(",")
            if x.strip()
        ]:
            if item in allowed and item not in order:
                order.append(item)
        if not order:
            return ["id", "username", "email"]
        return order

    def _resolve_user_api_key(self, user: dict) -> dict:
        mapping_enabled = bool(self.valves.enable_user_api_key_mapping)
        m = self._parse_user_key_map() if mapping_enabled else {
            "ids": {}, "usernames": {}, "emails": {}, "default": ""
        }
        user_id = str(user.get("id", "")).strip()
        username = str(user.get("username") or user.get("name") or user.get("login") or "").strip().lower()
        email = str(user.get("email", "")).strip().lower()
        candidates = {
            "id": (user_id, m["ids"]),
            "username": (username, m["usernames"]),
            "email": (email, m["emails"]),
        }
        for source in self._parse_openwebui_key_resolution_order():
            identifier, section = candidates[source]
            if identifier and identifier in section:
                return {
                    "key": section[identifier], "source": source,
                    "matched_identifier": identifier, "mapping_enabled": mapping_enabled,
                    "fallback_used": False,
                }
        if mapping_enabled and self.valves.allow_mapping_default_key and m.get("default"):
            return {
                "key": m["default"], "source": "default",
                "matched_identifier": "default", "mapping_enabled": mapping_enabled,
                "fallback_used": True,
            }
        if self.valves.allow_openwebui_default_api_key_fallback and (
                self.valves.openwebui_default_api_key or ""
        ).strip():
            return {
                "key": (self.valves.openwebui_default_api_key or "").strip(), "source": "default",
                "matched_identifier": "openwebui_default_api_key", "mapping_enabled": mapping_enabled,
                "fallback_used": True,
            }
        return {
            "key": "", "source": "none", "matched_identifier": "",
            "mapping_enabled": mapping_enabled, "fallback_used": False,
        }

    def _resolve_openwebui_runtime_capabilities(self, user: dict) -> dict:
        key_meta = self._resolve_user_api_key(user)
        api_enabled = bool(self.valves.enable_openwebui_api)
        api_key_present = bool(key_meta.get("key"))
        tools_enabled = api_enabled
        reason = ""
        if not api_enabled:
            tools_enabled = False
            reason = "OpenWebUI API integration is disabled."
        elif (
                self.valves.openwebui_disable_tools_without_api_key and
                not api_key_present
        ):
            tools_enabled = False
            reason = "OpenWebUI tools require a valid API key by configuration."
        return {
            "openwebui_api_enabled": api_enabled,
            "openwebui_tools_enabled": tools_enabled,
            "disable_reason": reason,
            "mapping_enabled": bool(key_meta.get("mapping_enabled")),
            "api_key_present": api_key_present,
            "key_source": key_meta.get("source", "none"),
            "key_meta": key_meta,
        }

    def _get_enabled_tool_names(self, runtime_capabilities: dict | None = None) -> set[str]:
        runtime = runtime_capabilities or self._resolve_openwebui_runtime_capabilities({})
        enabled = {"terminate"}
        # Web
        if self.valves.enable_tool_web:
            enabled.update({"web_search", "fetch_url"})
        # Local workspace
        if self.valves.enable_tool_read_file:
            enabled.add("read_file")
        if self.valves.enable_tool_search_in_file:
            enabled.add("search_in_file")
        if self.valves.enable_tool_write_file:
            enabled.add("write_file")
        if self.valves.enable_tool_edit_file:
            enabled.add("edit_file")
        if self.valves.enable_tool_run_python:
            enabled.add("run_python")
        if self.valves.enable_tool_list_files:
            enabled.add("list_files")
        if self.valves.enable_tool_delete_file:
            enabled.add("delete_file")
        if self.valves.enable_tool_ask_user:
            enabled.add("ask_user")

        if runtime.get("openwebui_tools_enabled"):
            # Always-on OWUI tools (when API is enabled)
            if self.valves.enable_tool_openwebui_retrieval:
                enabled.update({"openwebui_retrieval_query", "openwebui_process_file_for_rag"})
            enabled.add("openwebui_list_tools_functions")
            # Context & references
            if self.valves.enable_tool_openwebui_refs:
                enabled.add("openwebui_resolve_references")
            if self.valves.enable_tool_openwebui_memories:
                enabled.add("openwebui_get_memories")
            if self.valves.enable_tool_openwebui_prompts:
                enabled.add("openwebui_get_prompt")
            if self.valves.enable_tool_openwebui_search:
                enabled.add("openwebui_search_entity")
            if self.valves.enable_tool_openwebui_upload:
                enabled.add("openwebui_upload_file")
            if self.valves.enable_tool_openwebui_download:
                enabled.add("openwebui_download_file_to_workspace")
            if self.valves.enable_tool_openwebui_defaults:
                enabled.add("openwebui_get_defaults")
            # Notes
            if self.valves.enable_tool_notes_read:
                enabled.add("openwebui_get_notes")
            if self.valves.enable_tool_notes_write:
                enabled.update({"openwebui_create_note", "openwebui_update_note"})
            if self.valves.enable_tool_notes_delete:
                enabled.add("openwebui_delete_note")
            # Tasks
            if self.valves.enable_tool_tasks_read:
                enabled.add("openwebui_list_tasks")
            if self.valves.enable_tool_tasks_write:
                enabled.update({"openwebui_create_task", "openwebui_update_task"})
            # Calendar
            if self.valves.enable_tool_calendar_read:
                enabled.add("openwebui_list_calendar_events")
            if self.valves.enable_tool_calendar_write:
                enabled.add("openwebui_create_calendar_event")
            if self.valves.enable_tool_calendar_delete:
                enabled.add("openwebui_delete_calendar_event")
            # Knowledge Base management
            if self.valves.enable_tool_knowledge_management:
                enabled.update({"openwebui_create_knowledge", "openwebui_add_file_to_knowledge"})
            # Chats (read)
            if self.valves.enable_tool_chats_readonly:
                enabled.update({"openwebui_list_chats", "openwebui_get_chat"})
            # Automations (read)
            if self.valves.enable_tool_automations_readonly:
                enabled.add("openwebui_list_automations")
        return enabled

    def _build_openwebui_tool_guidance(self, enabled_tools: set[str]) -> str:
        has_owui = any(t in OPENWEBUI_TOOL_NAMES for t in enabled_tools)
        if not has_owui:
            return "- OpenWebUI API tools are unavailable in this configuration."
        lines = []
        if "openwebui_resolve_references" in enabled_tools:
            lines.append("- Resolve #knowledge/#note/#memory/#file/#chat refs with openwebui_resolve_references.")
        if "openwebui_retrieval_query" in enabled_tools:
            lines.append("- For large document search, prefer openwebui_retrieval_query over reading full files.")
        if "openwebui_get_memories" in enabled_tools:
            lines.append("- Read user memories with openwebui_get_memories (READ ONLY — never write memories).")
        if "openwebui_get_notes" in enabled_tools:
            lines.append(
                "- Read and manage notes with openwebui_get_notes / openwebui_create_note / openwebui_update_note.")
        if "openwebui_delete_note" in enabled_tools:
            lines.append("- ALWAYS ask_user for confirmation before openwebui_delete_note.")
        if "openwebui_create_task" in enabled_tools or "openwebui_list_tasks" in enabled_tools:
            lines.append(
                "- Mirror major milestones to OpenWebUI Tasks via openwebui_create_task / openwebui_update_task; keep local todo.md as internal scratchpad.")
        if "openwebui_list_calendar_events" in enabled_tools:
            lines.append("- List calendar events with openwebui_list_calendar_events and optional date range filters.")
        if "openwebui_create_calendar_event" in enabled_tools:
            lines.append("- Confirm event details with ask_user, then create via openwebui_create_calendar_event.")
        if "openwebui_delete_calendar_event" in enabled_tools:
            lines.append("- ALWAYS ask_user for confirmation before openwebui_delete_calendar_event.")
        if "openwebui_upload_file" in enabled_tools:
            lines.append("- Use openwebui_upload_file to make workspace files user-visible and persistent.")
        if "openwebui_download_file_to_workspace" in enabled_tools:
            lines.append("- Use openwebui_download_file_to_workspace to fetch OWUI files for local editing.")
        if "openwebui_create_knowledge" in enabled_tools:
            lines.append(
                "- Create KBs with openwebui_create_knowledge; add files with openwebui_add_file_to_knowledge.")
        if "openwebui_list_chats" in enabled_tools:
            lines.append("- Read chat history with openwebui_list_chats / openwebui_get_chat (READ ONLY).")
        return "\n".join(lines) if lines else "- OpenWebUI tools available."

    def _get_tool_schemas(self, enabled_tools: set[str]) -> list[dict]:
        return [t for t in ALL_TOOL_SCHEMAS if t["function"]["name"] in enabled_tools]

    def _build_prompt(self, is_first: bool, goal: str, enabled_tool_names_str: str,
                      openwebui_tool_guidance: str) -> str:
        if self.valves.notes_lesson_learned_enabled:
            prefix = self.valves.notes_lesson_learned_title_prefix
            if is_first:
                lessons_section = (
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "LESSONS LEARNED\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"- On first step: call `openwebui_get_notes` and search for a note whose title starts with \"{prefix}\". Read it to load existing technical insights before starting.\n"
                    f"- On task completion: if you discovered new non-obvious technical insights, APPEND them to this exact note via `openwebui_update_note`. Never create a separate note per task.\n"
                    f"- If no \"{prefix}\" note exists yet, create it with `openwebui_create_note` using title \"{prefix}\"."
                )
            else:
                lessons_section = (
                    f"- Lessons Learned: if the task is now complete and you have new insights, "
                    f"append them to the \"{prefix}\" note before calling terminate."
                )
        else:
            lessons_section = ""

        template = PLANNING_PROMPT if is_first else BASE_AGENT_PROMPT
        return template.format(
            goal=goal,
            tool_names=enabled_tool_names_str,
            openwebui_tool_guidance=openwebui_tool_guidance,
            lessons_section=lessons_section,
        )

    async def pipe(
            self,
            body: dict,
            __event_emitter__=None,
            __event_call__=None,
            __user__=None,
    ) -> AsyncGenerator[str, None]:

        async def status(msg: str, done: bool = False):
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": msg, "done": done}})

        messages = body.get("messages", [])
        if not messages:
            yield "No messages received."
            return

        user = __user__ if isinstance(__user__, dict) else {}
        user_id = str(
            user.get("id") or user.get("username") or user.get("name") or "anonymous"
        ).strip()
        runtime = self._resolve_openwebui_runtime_capabilities(user)
        key_meta = runtime.get("key_meta", {})
        api_key = key_meta.get("key", "")

        workspace_dir = self.valves.workspace_dir
        if self.valves.user_scoped_workspace:
            safe_user = re.sub(r"[^a-zA-Z0-9._-]+", "_", user_id)[:80]
            workspace_dir = os.path.join(self.valves.workspace_dir, safe_user)
            os.makedirs(workspace_dir, exist_ok=True)

        goal = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "Unknown goal"
        )
        history = [m for m in messages if m.get("role") != "system"]

        enabled_tools = self._get_enabled_tool_names(runtime)
        tool_schemas = self._get_tool_schemas(enabled_tools)
        enabled_tool_names_str = ", ".join(sorted(enabled_tools))
        openwebui_tool_guidance = self._build_openwebui_tool_guidance(enabled_tools)

        main_client = OllamaClient(
            self.valves.main_ollama_url,
            self.valves.request_timeout_seconds,
            self.valves.agent_num_ctx,
            max_retries=self.valves.http_max_retries,
            temperature=self.valves.temperature_main,
        )
        aux_client = OllamaClient(
            self._resolve_aux_url(),
            self.valves.aux_timeout_seconds,
            4096,
            max_retries=self.valves.http_max_retries,
            temperature=self.valves.temperature_aux,
        )

        user_language = await self._detect_user_language(aux_client, messages)

        web_rag = WebRAG(
            aux_client, self.valves.ranker_model, self.valves.tika_url,
            enabled=self.valves.web_rag_enabled,
            rag_chunk_tokens=self.valves.rag_chunk_tokens,
            rag_chunk_overlap_tokens=self.valves.rag_chunk_overlap_tokens,
            rag_candidates=self.valves.rag_candidates,
            rag_top_k=self.valves.rag_top_k,
            rag_min_score=self.valves.rag_min_score,
            rag_extract_max_chars=self.valves.rag_extract_max_chars,
            rag_html_max_chars=self.valves.rag_html_max_chars,
            rag_pdf_max_bytes=self.valves.rag_pdf_max_bytes,
            rag_score_chunk_preview_chars=self.valves.rag_score_chunk_preview_chars,
            rag_score_num_predict=self.valves.rag_score_num_predict,
        )

        owui = OpenWebUIClient(self.valves.openwebui_base_url, timeout_seconds=20)
        async with self._jupyter_lock:
            if self._jupyter_executor is None:
                self._jupyter_executor = JupyterExecutor(
                    self.valves.jupyter_url,
                    self.valves.jupyter_token,
                    self.valves.jupyter_persistent_kernel,
                    self.valves.jupyter_execution_timeout_seconds,
                )
        executor = ToolExecutor(
            self.valves.searxng_url,
            workspace_dir,
            web_rag,
            self._jupyter_executor,
            aux_client,
            self.valves.translation_model,
            user_language,
            self.valves.max_tool_result_chars,
            enabled_tools,
            self.valves.translate_num_predict,
            self.valves.fetch_url_max_output_tokens,
            owui_client=owui,
            owui_base_url=self.valves.openwebui_base_url,
            owui_api_key=api_key,
            current_user=user,
            openwebui_runtime=runtime,
            openwebui_allowed_entities=self._parse_csv_values(self.valves.openwebui_allowed_entities_csv),
            openwebui_allowed_reference_types=self._parse_csv_values(
                self.valves.openwebui_allowed_reference_types_csv
            ),
            event_call=__event_call__,
            file_status_poll_timeout=self.valves.owui_file_status_poll_timeout_seconds,
            file_status_poll_interval=self.valves.owui_file_status_poll_interval_seconds,
        )

        ctx = ContextManager()
        loop_count = 0
        is_first = True
        recent_calls = []
        consecutive_json_errors = 0

        await status("Agent starting…")
        if self.valves.openwebui_expose_diagnostics:
            await status(
                "OpenWebUI diagnostics: "
                f"api_enabled={runtime.get('openwebui_api_enabled', False)}, "
                f"mapping_enabled={runtime.get('mapping_enabled', False)}, "
                f"api_key_present={runtime.get('api_key_present', False)}, "
                f"key_source={runtime.get('key_source', 'none')}"
            )
        yield "<think>\n"

        try:
            while loop_count < self.valves.max_iterations:
                recent_calls = recent_calls[-30:]
                system = self._build_prompt(is_first, goal, enabled_tool_names_str, openwebui_tool_guidance)
                history = ctx.compact_history(history, self.valves.max_recent_tool_messages)

                if ctx.needs_compression(history, self.valves.context_token_threshold):
                    await status("Compressing context…")
                    yield "> 🗜 Context compressed — summary injected\n\n"
                    summary = await aux_client.chat_text(
                        self.valves.summary_model,
                        ctx.build_summary_messages(history),
                        num_ctx_override=4096,
                        num_predict=self.valves.summary_num_predict,
                    )
                    last_tool = next(
                        (m for m in reversed(history) if m.get("role") == "tool"), None
                    )
                    last_assist = next(
                        (m for m in reversed(history)
                         if m.get("role") == "assistant" and m.get("tool_calls")),
                        None
                    )
                    history = [{"role": "user", "content": f"[COMPRESSED CONTEXT]\n{summary}"}]
                    if last_assist and last_tool:
                        history += [last_assist, last_tool]
                    elif last_tool:
                        history.append(last_tool)
                    system = RESUME_PROMPT.format(goal=goal, tool_names=enabled_tool_names_str)

                call_messages = [{"role": "system", "content": system}] + history
                await status(f"Thinking… (step {loop_count + 1}/{self.valves.max_iterations})")

                try:
                    response = await main_client.chat(
                        self.valves.main_agent_model,
                        call_messages,
                        tools=tool_schemas,
                        num_predict=self.valves.main_num_predict_step,
                    )
                except Exception as e:
                    translated = await self._translate(aux_client, f"Ollama error: {e}", user_language)
                    yield f"\n</think>\n\n❌ {translated}"
                    await status("Ollama error", done=True)
                    return

                message = response.get("message", {})
                tool_calls = message.get("tool_calls") or []
                assistant_text = (message.get("content") or "").strip()

                if not tool_calls:
                    translated = await self._translate(aux_client, assistant_text, user_language)
                    yield f"\n</think>\n\n{translated}"
                    await status("Done", done=True)
                    return

                tc = tool_calls[0]
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                raw_args = fn.get("arguments", {})

                args = None
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = None
                elif isinstance(raw_args, dict):
                    args = raw_args

                if tool_name == "terminate":
                    result = (args or {}).get("result", "Task complete.")
                    success = (args or {}).get("success", True)
                    icon = "✅" if success else "❌"
                    translated = await self._translate(aux_client, result, user_language)
                    yield f"\n{icon} Terminate called.\n</think>\n\n{translated}"
                    await status("Finished", done=True)
                    return

                if tool_name == "ask_user":
                    q = (args or {}).get("question", "I need more information to continue.")
                    translated_q = await self._translate(aux_client, q, user_language)
                    yield f"\n⏸ Agent paused.\n</think>\n\n**Agent asks:** {translated_q}"
                    await status("Waiting for user input", done=True)
                    return

                if args is None:
                    consecutive_json_errors += 1
                    if consecutive_json_errors >= 3:
                        translated = await self._translate(
                            aux_client,
                            "Agent produced invalid JSON 3 times in a row. Stopping.",
                            user_language
                        )
                        yield f"\n❌ JSON parse failed repeatedly.\n</think>\n\n{translated}"
                        await status("JSON error", done=True)
                        return
                    tool_result = f"Error: Could not parse tool arguments as JSON ({consecutive_json_errors}/3)."
                    history.append({"role": "assistant", "content": assistant_text, "tool_calls": tool_calls})
                    history.append({"role": "tool", "content": tool_result, "name": tool_name})
                    loop_count += 1
                    continue

                consecutive_json_errors = 0
                sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                if recent_calls.count(sig) >= 2:
                    tool_result = f"Error: Identical call to `{tool_name}` repeated."
                    history.append({"role": "assistant", "content": assistant_text, "tool_calls": tool_calls})
                    history.append({"role": "tool", "content": tool_result, "name": tool_name})
                    loop_count += 1
                    continue
                recent_calls.append(sig)

                yield (
                    f"\n**Step {loop_count + 1}** — `{tool_name}`\n"
                    f"```\n{smart_truncate(json.dumps(args, ensure_ascii=False), 220)}\n```\n"
                )
                await status(f"Running: {tool_name}…")
                tool_result = smart_truncate(
                    await executor.execute(tool_name, args),
                    self.valves.max_tool_result_chars
                )
                yield f"*Result:* {smart_truncate(tool_result, 380)}\n"

                history.append({"role": "assistant", "content": assistant_text, "tool_calls": tool_calls})
                history.append({"role": "tool", "content": tool_result, "name": tool_name})

                is_first = False
                loop_count += 1

            translated = await self._translate(
                aux_client,
                f"Reached maximum of {self.valves.max_iterations} steps without finishing.",
                user_language
            )
            yield f"\n⚠ Max iterations reached.\n</think>\n\n{translated}"
            await status("Max iterations", done=True)

        finally:
            await executor.aclose()
            await web_rag.aclose()
            await owui.aclose()
            await main_client.aclose()
            await aux_client.aclose()
