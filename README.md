# OpenWebUI Plugins

A collection of tools, pipelines and prompts for [Open WebUI](https://github.com/open-webui/open-webui) — extending the platform with GitHub integration, workflow helpers and more.

## Repository Structure

```
OpenWebUI-plugins/
├── tools/           ← Tool plugins (Python classes exposed as functions to the model)
├── pipelines/       ← Pipeline plugins (custom processing pipelines)
├── prompts/         ← Prompt templates
├── .gitignore
└── LICENSE
```

## Tools

Tools are Python plugins that extend the assistant with new capabilities. Each tool is a single file in `tools/` containing a `Tools` class. The model can call any public `async` method as a function.

---

### `github_access.py` — GitHub Repository Management

Full read/write access to GitHub repositories, issues, pull requests and workflows.

**Capabilities:**
- Browse repositories, files and directories
- Read commits, branches and diffs
- Create branches, create/write/rename/delete files and directories
- Create, read, update, close issues
- Add, list, update, delete comments on issues and PRs
- List, create, update pull requests, request reviewers, get diffs
- Rename files and directories
- Trigger and monitor GitHub Actions workflows

**User Valves:**

| Valve | Default | Controls |
|-------|---------|----------|
| `ENABLE_CONTENT` | on | Read access to repos, files, branches, commits |
| `ENABLE_CONTENT_WRITE` | on | Create branches, create/write/rename/delete files and directories |
| `ENABLE_ISSUES` | on | List, search, get issues |
| `ENABLE_ISSUES_WRITE` | on | Create, update, close, reopen issues |
| `ENABLE_PULL_REQUESTS` | on | List, get PRs, get PR files, get PR diff |
| `ENABLE_PULL_REQUESTS_WRITE` | on | Create PRs, request reviewers, update PRs |
| `ENABLE_WORKFLOWS` | on | List/get workflows, list/get workflow runs, view run logs |
| `ENABLE_WORKFLOWS_WRITE` | on | Trigger workflow dispatch, cancel/rerun workflow runs |

**Workflow rule:**
```
github_create_branch → github_write_file → github_create_pull_request
```
No direct writes to `main`. No merge function.

---

### `confirm_destructive_actions.py` — Destructive Action Guard

Shows a confirmation dialog before irreversible operations (file/directory deletion, etc.). The model must call this and receive `"confirmed"` before proceeding.

---

### `github_search.py` — GitHub Global Search

Searches the entire GitHub universe — public repositories, code, issues, pull requests, commits, users and topics — right from Open WebUI. Works without a token for public searches, or with a token for higher rate limits and private repository access.

**Functions:**

| Function | Description |
|----------|-------------|
| `github_search(query, search_type)` | Main search across all types |
| `github_search_repos(query)` | Find repositories |
| `github_search_code(query)` | Search source code |
| `github_search_issues(query)` | Search issues and pull requests |
| `github_search_commits(query)` | Search commit messages |
| `github_search_users(query)` | Find users and organizations |

**User Valves:**
| Valve | Default | Description |
|-------|---------|-------------|
| `GITHUB_TOKEN` | (empty) | Optional PAT for 30 req/min and private repos. Leave empty for anonymous public searches (10 req/min). |

See the [GitHub Search PR](https://github.com/Piggidragon/OpenWebUI-plugins/pull/1) for details.

---

## Pipelines

Pipelines are custom processing flows that run server-side in Open WebUI. Each pipeline is a single Python file defining a `Pipe` (or `Pipeline`) class.

---

### `helix_agent.py` — Helix Agent (v0.24.2)

A single-model **Plan → Execute → Review → Output** loop with a **Replan** phase for course correction. Per-phase tool filtering controls which tools the LLM sees at each step.

**How it works:**
1. **Plan** — The model analyses the request and creates a numbered task list, then calls `confirm_plan` to present it.
2. **Execute** — The model works through tasks one at a time, calling tools and marking them `complete_task` or `fail_task`. It may call `replan` or `fix_plan` if the approach needs adjusting.
3. **Review** — After all tasks are done, the model calls one of: `proceed_to_output()` (move to Output), `fix_plan(reason, updated_tasks)` (minor fixes), or `replan(reason)` (major rework → enters Replan phase).
4. **Replan** — If `replan` is called, the model enters Replan mode and must call `confirm_plan` with the revised task list before returning to Execute.
5. **Output** — Two-turn output phase:
   - **Turn 1 (Rendering)** — The model may call rendering/visualisation tools (e.g. `display_file`) to illustrate results. No summary text yet.
   - **Turn 2 (Final Summary)** — The model produces a structured JSON summary of what was accomplished, files created/modified, failed tasks, and overall status.

**Internal control tools** (always available, phase-relevant subset shown):

| Tool | Available in | Purpose |
|------|--------------|---------|
| `confirm_plan` | Plan, Replan | Present the task plan for approval |
| `complete_task` | Execute | Mark a task as done |
| `fail_task` | Execute | Mark a task as failed with a reason |
| `fix_plan` | Execute, Review | Request minor plan corrections |
| `replan` | Execute, Review | Enter Replan phase for major strategy changes |
| `proceed_to_output` | Review | Move to the Output phase |
| `run_tools_parallel` | Plan, Execute, Review, Replan | Call multiple independent tools at once |
| `ask_user` | Plan, Replan | Interactive clarification questions with selectable options |
| `terminate` | Plan | Signal an inappropriate or impossible request |

**Key features:**
- **Per-phase tool filtering** — only expose relevant tools to the LLM at each phase (Plan / Execute / Review / Output / Replan). Configurable via Valves.
- **Native OWUI task list UI** — real-time task progress via `chat:message:tasks` events (pending, in_progress, completed, cancelled). No HTML hacks.
- **Task finalization** — on termination, remaining tasks are marked completed so the task list UI dismisses cleanly.
- **Plan confirmation** — optional modal popup (UserValves: `ENABLE_PLAN_APPROVAL`, `YOLO_MODE`).
- **LLM-based context compression** — history and goal are compressed independently using a configurable model (`CONTEXT_COMPRESSION_MODEL`). History is compressed mid-loop when token-based thresholds (derived from `CONTEXT_LENGTH`) are exceeded; goal is compressed asynchronously after each run.
- **Single CONTEXT_LENGTH valve** — one token-based setting drives all adaptive compression thresholds via `CHARS_PER_TOKEN_ESTIMATE`.
- **MCP support** — MCP server tools provided via the `__tools__` parameter.
- **Skills support** — resolves user skills from model metadata and injects them into the system prompt.
- **Debug mode** — message "show tools" returns the available items directly without any LLM call.
- **Context window management** — adaptive history truncation preserving tool-call pair integrity, plus `MAX_HISTORY_MESSAGES` cap.
- **Token tracking** — session-wide input/output token counts are tracked and reported.
- **Iteration limit** — configurable max loop iterations with a Continue/Cancel dialog and `ITERATION_LIMIT_TIMEOUT`.
- **System prompt refresh** — the LLM always sees up-to-date task state after mutations.
- **DB-backed state persistence** — agent state is serialized to a JSON file attachment and synced to the OpenWebUI chat/message DB. Recovers from deep DB history scan across parent message chains.
- **Loop count persistence** — iteration counter is saved and restored across sessions.
- **Exponential backoff file sync** — robust DB persistence under heavy load.
- **Knowledge bases** — native OpenWebUI vector search via model metadata.
- **File handling** — `add_file_context` + `chat_completion_files_handler` for native multimodal/text file injection.
- **Graceful shutdown** — handles `GeneratorExit`/`CancelledError`, saves state, and syncs files before exiting.
- **Duplicate tool call detection** — prevents repeating identical failed tool calls.
- **Interactive ask_user UI** — custom JS overlay with selectable options and free-text input.

**Admin Valves (Valves):**

| Valve | Default | Description |
|-------|---------|-------------|
| `DEBUG_MODE` | false | Enable debug mode. Messages "show tools" return available items directly without any LLM call. |
| `AGENT_MODEL` | (empty) | Model ID for Helix Agent. Must support function calling. Leave empty to use the selected model. |
| `CONTEXT_COMPRESSION_MODEL` | (empty) | Model ID for context compression. Falls back to `AGENT_MODEL`. Consider smaller models for cost efficiency. |
| `MAX_ITERATIONS` | 100 | Maximum Helix Agent iterations before stopping. |
| `MAX_REPLAN_LOOPS` | 3 | Safety cap: after this many REPLAN loops the agent falls back to single-task EXECUTE. |
| `ENABLE_HARD_STOP_ON_ERRORS` | false | If True, hard-stop after `MAX_CONSECUTIVE_ERRORS` consecutive tool call failures. If False, the model receives the error and can self-correct. |
| `MAX_CONSECUTIVE_ERRORS` | 3 | Number of consecutive errors that triggers a hard stop when `ENABLE_HARD_STOP_ON_ERRORS` is True. |
| `LLM_RETRY_COUNT` | 1 | Number of retries for transient LLM API errors (ConnectionError, TimeoutError). Set to 0 to disable retries. |
| `TOOL_TIMEOUT` | 90 | Timeout in seconds for individual tool execution. 0 to disable. |
| `CONTEXT_LENGTH` | 128000 | Context window length in tokens. Drives all adaptive compression thresholds. |
| `CHARS_PER_TOKEN_ESTIMATE` | 3.5 | Estimated characters per token for the active model. Converts token limits into character-based internal thresholds. |
| `COMPRESSION_INTERVAL` | 5 | Minimum loop iterations between consecutive history compressions. |
| `KEEP_RECENT_MESSAGES` | 6 | Number of recent messages to always keep uncompressed. Older messages are candidates for compression. |
| `MAX_HISTORY_MESSAGES` | 100 | Maximum total conversation messages retained in context. Older messages are dropped while keeping tool-call pairs intact. |
| `PLAN_TOOLS` | (read-only set) | Comma-separated tools allowed in PLAN phase. Empty = all tools. |
| `EXECUTE_TOOLS` | (broad set) | Comma-separated tools allowed in EXECUTE phase. Empty = all tools. |
| `REVIEW_TOOLS` | (read-only + exec set) | Comma-separated tools allowed in REVIEW phase. Empty = all tools. |
| `OUTPUT_TOOLS` | `display_file` | Comma-separated rendering/visualization tools allowed in OUTPUT phase Turn 1. Empty = none. |
| `PLAN_PROMPT` | (built-in) | Custom PLAN system prompt. |
| `EXECUTE_PROMPT` | (built-in) | Custom EXECUTE system prompt. Placeholder: `{task_state}`. |
| `REVIEW_PROMPT` | (built-in) | Custom REVIEW system prompt. Placeholders: `{goal}`, `{task_state}`. |
| `OUTPUT_PROMPT` | (built-in) | Custom OUTPUT system prompt — Turn 1 (rendering/visualization). Placeholders: `{goal}`, `{task_state}`. |
| `ENABLE_TOOL_TRUNCATION` | true | If True, tool results are truncated to `MAX_TOOL_RESULT_CHARS`. If False, truncation is completely disabled. |
| `MAX_TOOL_RESULT_CHARS` | 12000 | Max characters for tool results before truncation. |
| `MAX_ATTACHMENT_SIZE_MB` | 5 | Maximum allowed size of individual attached files in MB. 0 to disable the size check. |
| `PLAN_APPROVAL_TIMEOUT` | 600 | Timeout in seconds for the plan approval modal. After this time the plan is auto-approved. |
| `ITERATION_LIMIT_TIMEOUT` | 300 | Timeout in seconds for the iteration limit Continue/Cancel modal. After this time the agent auto-stops. |

**User Valves (UserValves):**

| Valve | Default | Description |
|-------|---------|-------------|
| `YOLO_MODE` | false | Skip all user confirmations. Auto-approve plans and ignore iteration limits. |
| `ENABLE_PLAN_APPROVAL` | true | Show plan confirmation popup before execution. When off, plans are auto-approved without asking the user. |
| `SKIP_PLAN_ON_RESUME` | true | When the previous session is finished, skip the full PLAN phase for a new user request and jump straight to Replan. Set to False to always start fresh with a full PLAN phase. |
| `SKIP_OUTPUT_RENDERING` | true | If True, skip the OUTPUT phase Turn 1 (rendering/visualization) and go straight to the final summary Turn 2. |
| `MAX_PLAN_QUESTIONS` | 3 | Maximum number of clarification questions (`ask_user`) the agent may ask per planning phase before it is forced to finalise the plan. |
| `MAX_TOOL_RESULT_CHARS` | 12000 | Max characters for individual tool results before truncation. Set to -1 to use admin default, 0 to disable truncation entirely. |

---

### `planner_v3.py` — Planner v3 (v3.10.3, by Haervwe)

Multi-agent orchestrator with subagents, MCP support, and plan approval UI. See the [planner_v3.py](pipelines/planner_v3.py) header for full documentation.

---

### Option A — Workspace Import (recommended)

1. Open your Open WebUI Workspace
2. Go to **Tools** → **Import Tool**
3. Paste the raw URL to the tool file (e.g. `https://raw.githubusercontent.com/Piggidragon/OpenWebUI-plugins/main/tools/github_access.py`)
4. Click **Import**

### Option B — Manual Upload

1. Download the `.py` file from this repository
2. In Open WebUI, go to **Workspace** → **Tools** → **Create Tool**
3. Paste the content and save

---

## System Prompt

For the best experience, add this to your model's system prompt:

```
When interacting with GitHub, use the "github-workflow" skill.
Never write directly to main. Always follow: github_create_branch → github_write_file → github_create_pull_request.
```

---

## Skill

The `github-workflow` skill enforces consistent naming conventions:

| Branch Type | Purpose |
|-------------|---------|
| `feature/*` | New features and tools |
| `fix/*` | Bugfixes |
| `refactor/*` | Code restructuring |
| `docs/*` | Documentation |
| `chore/*` | Config, CI, dependencies |

PR titles follow the format: `<Type>: <Short description>` (e.g. `Feature: GitHub Global Search Tool`).

---

## License

This repository is licensed under the terms in [LICENSE](./LICENSE).
