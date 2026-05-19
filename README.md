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

### `helix_agent.py` — Helix Agent (v0.25.0)

A single-model **Plan → Execute → Review → Output** loop with a **Replan** phase for course correction. Per-phase tool filtering controls which tools the LLM sees at each step.

#### Architecture

The Helix Agent runs a single model in a structured, repeating loop. The LLM is never given the full tool set at once; instead, its view of available tools is filtered per phase, which forces discipline and improves reliability.

```
User Request
     v
┌─────────────┐
│ PLAN        │ ━━ numbered task list → confirm_plan → user approval?
└─────────────┘                          (skipped if YOLO_MODE / resume)
     v
┌─────────────┐
│ EXECUTE     │ ━━ one task at a time: tool calls → complete_task / fail_task
└─────────────┘     replan available at any time; fix_plan for minor corrections
     v
┌─────────────┐
│ REVIEW      │ ━━ read-only verification → decide: proceed? fix_plan? replan?
└─────────────┘
     v (if replan) back to REPLAN → confirm_plan → EXECUTE again
     v
┌─────────────┐
│ OUTPUT      │ ━━ RENDER: optional visualisation tools (e.g., display_file)
│             │       SUMMARY: mandatory structured JSON summary of results
└─────────────┘
```

**Phases explained**

1. **Plan** — The model analyses the request and creates a concise, numbered task list (usually 3–7 tasks). It **cannot** call execution tools in this phase; only internal control tools (`confirm_plan`, `ask_user`, `terminate`) are exposed.
2. **Execute** — The model works through tasks sequentially, calling real tools and marking each with `complete_task` or `fail_task`. `run_tools_parallel` is available for independent tool calls. It may call `fix_plan` (minor corrections) or `replan` (major strategy change) at any time.
3. **Review** — A dedicated quality-gate phase. The model must use read-only tools to verify the completed work before deciding: move to Output (`proceed_to_output`), add small fixes (`fix_plan`), or restart the plan entirely (`replan` → enters Replan phase).
4. **Replan** — Entered when `replan` is called. The model creates a fresh, minimal task list (1–3 tasks) based on what went wrong, then must call `confirm_plan` to return to Execute.
5. **Output** — A two-turn output phase:
   - **RENDER** — the model may call rendering/visualisation tools (e.g., `display_file`) to illustrate results. No summary text yet.
   - **SUMMARY** — the model produces a structured JSON summary of what was accomplished, files created/modified, failed tasks, and overall status.

**Internal control tools** (only a phase-relevant subset is injected into the prompt at each step):

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

#### Admin Valves (global / per-model settings)

| Valve | Default | Description |
|-------|---------|-------------|
| `DEBUG_MODE` | `false` | Enable debug mode. Sending the literal message `show tools` returns the currently available tool list directly, without triggering any LLM call. Useful for checking which tools are exposed per phase. |
| `AGENT_MODEL` | *(empty)* | Model ID used for the Helix Agent loop. Must support function calling. Leave empty to use whatever model the user selected in the chat. |
| `CONTEXT_COMPRESSION_MODEL` | *(empty)* | Model ID used for LLM-based context compression (history and goal). Falls back to `AGENT_MODEL` if unset. Using a smaller model (e.g., 3.5‑class) here saves cost. |
| `MAX_ITERATIONS` | `100` | Hard limit on how many agent-loop iterations may run before a Continue/Cancel dialog pops up. Stops runaway sessions. |
| `MAX_REPLAN_LOOPS` | `3` | Safety cap: after this many full REPLAN cycles the agent switches to a simplified single-task EXECUTE mode to avoid infinite loops. |
| `ENABLE_HARD_STOP_ON_ERRORS` | `false` | If `true`, the agent immediately stops after `MAX_CONSECUTIVE_ERRORS` consecutive tool-call failures. If `false`, the error is returned to the model so it can self-correct. |
| `MAX_CONSECUTIVE_ERRORS` | `3` | Consecutive errors that trigger a hard stop when `ENABLE_HARD_STOP_ON_ERRORS` is `true`. |
| `LLM_RETRY_COUNT` | `1` | Retries for transient LLM API errors (`ConnectionError`, `TimeoutError`). Set to `0` to disable retries. |
| `TOOL_TIMEOUT` | `90` | Timeout in seconds for individual tool execution (`0` = disabled). |
| `CONTEXT_LENGTH` | `128000` | Effective context window length in **tokens**. This single value drives all adaptive compression thresholds via `CHARS_PER_TOKEN_ESTIMATE`. |
| `CHARS_PER_TOKEN_ESTIMATE` | `3.5` | Characters per token estimate. Converts token limits into internal character thresholds. Adjust if you know the exact model ratio (e.g., ~4 for many English texts). |
| `COMPRESSION_INTERVAL` | `5` | Minimum loop iterations between consecutive history compressions. Prevents back-to-back compression. |
| `KEEP_RECENT_MESSAGES` | `6` | Number of recent messages always kept uncompressed. Older messages become candidates for summarisation. |
| `MAX_HISTORY_MESSAGES` | `100` | Absolute cap on total conversation messages retained in context. Older messages are dropped entirely while keeping tool-call pairs intact. |
| `PLAN_TOOLS` | *(empty)* | Comma-separated tool names allowed in PLAN phase. **Empty = all tools** (not recommended; PLAN should stay tool-free). |
| `EXECUTE_TOOLS` | *(empty)* | Comma-separated tool names allowed in EXECUTE phase. Empty = all registered tools. Use this to restrict the tool set for speed or safety. |
| `REVIEW_TOOLS` | *(empty)* | Comma-separated tool names allowed in REVIEW phase. Empty = all registered tools. Typically configured to read-only + verification tools. |
| `OUTPUT_TOOLS` | `display_file` | Comma-separated rendering/visualisation tools for the OUTPUT RENDER turn. Empty = none. |
| `PLAN_PROMPT` | *(built-in)* | Custom system prompt override for the PLAN phase. |
| `EXECUTE_PROMPT` | *(built-in)* | Custom system prompt override for the EXECUTE phase. Placeholder `{task_state}` is injected at runtime. |
| `REVIEW_PROMPT` | *(built-in)* | Custom system prompt override for the REVIEW phase. Placeholders: `{goal}`, `{task_state}`. |
| `OUTPUT_PROMPT` | *(built-in)* | Custom system prompt override for the OUTPUT RENDER turn. Placeholders: `{goal}`, `{task_state}`. |
| `ENABLE_TOOL_TRUNCATION` | `true` | Whether tool results exceeding `MAX_TOOL_RESULT_CHARS` should be truncated. |
| `MAX_TOOL_RESULT_CHARS` | `12000` | Character limit for individual tool results before truncation. |
| `MAX_ATTACHMENT_SIZE_MB` | `5` | Maximum size of individual attached files in MB (`0` = no check). |
| `PLAN_APPROVAL_TIMEOUT` | `600` | Seconds before the plan-approval modal auto-approves itself. |
| `ITERATION_LIMIT_TIMEOUT` | `300` | Seconds before the iteration-limit Continue/Cancel modal auto-stops the agent. |
| `MAX_SESSION_SECONDS` | `1200` | Hard session lifetime cap (default 20 min). `0` = disabled. |
| `MAX_ITERATION_SECONDS` | `300` | Hard per-iteration cap (default 5 min). `0` = disabled. |
| `SSE_CHUNK_TIMEOUT_SECONDS` | `60` | SSE stream chunk timeout; aborts stuck streams if no data arrives. `0` = disabled. |
| `MAX_LLM_CALLS` | `50` | Maximum `generate_chat_completion` calls per session (`0` = disabled). Acts as an API-budget guard. |

#### User Valves (per-user settings)

| Valve | Default | Description |
|-------|---------|-------------|
| `YOLO_MODE` | `false` | **Skip all confirmations.** Auto-approves plans and ignores iteration/session limits. Use with caution — essentially disables all gates. |
| `ENABLE_PLAN_APPROVAL` | `true` | Show the modal plan-approval popup before execution. Set to `false` to auto-approve all plans silently. |
| `SKIP_PLAN_ON_RESUME` | `true` | When a previous session already finished, start the next user request in Replan rather than full Plan. Set to `false` to always begin with a fresh PLAN phase. |
| `SKIP_OUTPUT_RENDERING` | `true` | Skip the OUTPUT RENDER turn and go straight to SUMMARY. Set to `false` if you want the agent to call rendering tools (e.g., `display_file`) before the final summary. |
| `MAX_PLAN_QUESTIONS` | `3` | Max clarification questions (`ask_user`) per planning phase. After this cap the model is forced to finalise the plan. |
| `MAX_TOOL_RESULT_CHARS` | `12000` | User-level override for tool-result truncation. `-1` = use admin default, `0` = disable truncation entirely. |

#### Pros & Cons

**Pros**
- **Structured control loop** — The Plan/Execute/Review/Replan phases enforce a deliberate workflow. The model cannot "just wing it"; it must create a plan, execute step-by-step, and verify before delivering results.
- **Per-phase tool filtering** — Only relevant tools are visible to the LLM at each step. This reduces confusion, hallucinated tool calls, and token waste.
- **Native OpenWebUI integration** — Uses built-in `chat:message:tasks` events for real-time task progress, DB-backed file attachments for state persistence, and standard `__tools__` / `__event_emitter__` / `__event_call__` infrastructure. No HTML hacks.
- **Resilient state recovery** — Agent state is saved as JSON file attachments and synced to the OpenWebUI DB. On reload the agent scans the entire parent message chain to recover its previous task list, loop count, and history.
- **Context management** — Adaptive history truncation, LLM-based compression of history and goal, plus `MAX_HISTORY_MESSAGES` and `CHARS_PER_TOKEN_ESTIMATE` let you tune context usage precisely.
- **Safety budgets** — Multiple independent guards: `MAX_ITERATIONS`, `MAX_REPLAN_LOOPS`, `MAX_SESSION_SECONDS`, `MAX_ITERATION_SECONDS`, `MAX_LLM_CALLS`, `MAX_CONSECUTIVE_ERRORS`, `ITERATION_LIMIT_TIMEOUT`.
- **Parallel execution** — `run_tools_parallel` lets the model batch independent tool calls for faster execution.
- **Skills & Knowledge** — Automatically injects user skills from model metadata and supports native OpenWebUI knowledge bases via model metadata.

**Cons**
- **Single-model bottleneck** — Everything runs through one LLM. Complex tasks still require many sequential turns because the model must call `complete_task` after every step.
- **Token-heavy** — The system prompt is re-injected each loop and includes the full task state, loop info, tool catalog, and custom prompts. This can burn through context windows quickly on long sessions.
- **Not truly multi-agent** — Unlike multi-agent orchestrators (e.g. Planner v3), there is no specialised sub-agent for planning, coding, or review. One model does it all.
- **Output phase overhead** — The two-turn Output (RENDER then SUMMARY) adds extra LLM calls. Most users leave `SKIP_OUTPUT_RENDERING = true`.
- **File-only persistence** — All results are expected to exist as files under `[USER_HOME]/agent/<project>/` (see Requirement below). If no terminal/file-writing tools are available, the agent has nowhere to persist its work.

#### Important: Required Tools & Disabled Built-ins

**Tools that MUST be enabled** (or the agent is crippled):
- **Terminal / shell execution** (e.g., `openterminal` or equivalent) — The Helix Agent stores all generated artefacts (code, configs, results) as files under `[USER_HOME]/agent/<project>/`. Without a terminal tool the agent cannot create, edit, or verify files. This is effectively a **hard requirement**.

**Tools that MUST be disabled** in the Helix Agent model configuration to avoid conflicts:
- **Code Interpreter** — The agent handles tool execution itself; a built-in code interpreter bypasses the loop and breaks the task-tracking / review logic.
- **Task Management** — Native OWUI task management interferes with Helix's own task-state tracking and `chat:message:tasks` events.
- **Adaptive Memories / Context Compression / Auto Tool Routing** — Any built-in OpenWebUI filter that automatically compresses context, routes tools, or manages memory will clash with Helix's explicit per-phase filtering and manual context-management valves.
- **Parallel Tools** — The Helix Agent already provides `run_tools_parallel`; a second parallel-tool layer causes race conditions and double-counted tool calls.
- **Question Ask** — The agent has a built-in `ask_user` tool with selectable options and custom text input. Enabling another question-ask mechanism leads to duplicate or conflicting user prompts.

In short: **disable all "smart" built-in OWUI features** for the model running Helix, and point the `OUTPUT_TOOLS` valve to your rendering/display tool of choice (e.g. `display_file`).

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
