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

### `helix_agent.py` — Helix Agent (v4.6.1)

A single-model Plan → Execute → Review → Replan loop with per-phase tool control.

**How it works:**
1. **Plan** — The model creates a task list and calls `confirm_plan`.
2. **Execute** — The model works through tasks, calling tools and marking them complete/failed.
3. **Review** — Once all tasks are done, the model calls `terminate` to deliver the final result.
4. **Replan** — If tasks need adjusting, the model calls `replan` or `fix_plan` and loops back to Execute.

**Key features:**
- **Per-phase tool filtering** — only expose relevant tools to the LLM at each phase (configurable via Valves).
- **Native OWUI task list UI** — real-time task progress via `chat:message:tasks` events (pending, in_progress, completed, cancelled). No HTML hacks.
- **Task finalization** — on termination, remaining tasks are marked completed so the task list UI dismisses cleanly.
- **Plan confirmation** — optional modal popup (UserValves: `ENABLE_PLAN_APPROVAL`, `YOLO_MODE`).
- **Context window management** — adaptive history truncation preserving tool-call pair integrity.
- **Iteration limit** — configurable max loop iterations with a Continue/Cancel dialog.
- **System prompt refresh** — the LLM always sees up-to-date task state after mutations.
- **DB-backed state persistence** — agent state is serialized to a JSON file attachment and synced to the OpenWebUI chat/message DB. Recovers from deep DB history scan across parent message chains.
- **Robust file persistence** — tool-generated files are tracked, deduplicated, and synced to the DB with exponential backoff retry.
- **RAG search** — built-in semantic search over attached large files.
- **Graceful shutdown** — handles `GeneratorExit`/`CancelledError`, saves state, and syncs files before exiting.

**Admin Valves (Valves):**
| Valve | Default | Description |
|-------|---------|-------------|
| `AGENT_MODEL` | (empty) | Model ID for Helix Agent. Leave empty to use the selected model. Must support function calling. |
| `MAX_ITERATIONS` | 100 | Maximum Helix Agent iterations before stopping. |
| `MAX_TOOL_RESULT_CHARS` | 4200 | Max characters for tool results before truncation. |
| `TOOL_TIMEOUT` | 90 | Timeout in seconds for individual tool execution. 0 to disable. |
| `PLAN_TOOLS` | (empty) | Comma-separated tools allowed in PLAN phase. Empty = all tools. |
| `EXECUTE_TOOLS` | (empty) | Comma-separated tools allowed in EXECUTE phase. Empty = all tools. |
| `REVIEW_TOOLS` | (empty) | Comma-separated tools allowed in REVIEW phase. Empty = all tools. |
| `PLAN_PROMPT` | (built-in) | Custom PLAN system prompt. Placeholders: `{tool_names}`. Leave empty to use the built-in default. |
| `EXECUTE_PROMPT` | (built-in) | Custom EXECUTE system prompt. Placeholders: `{tool_names}`, `{task_state}`. Leave empty to use the built-in default. |
| `REVIEW_PROMPT` | (built-in) | Custom REVIEW system prompt. Placeholders: `{goal}`, `{task_state}`, `{tool_names}`. Leave empty to use the built-in default. |

**User Valves (UserValves):**
| Valve | Default | Description |
|-------|---------|-------------|
| `ENABLE_PLAN_APPROVAL` | true | Show plan confirmation popup before execution. When off, plans are auto-approved without asking the user. |
| `YOLO_MODE` | false | Skip all user confirmations. Auto-approve plans and ignore iteration limits. |

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
