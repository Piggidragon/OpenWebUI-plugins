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
- Create branches, write and delete files
- Create, read and comment on issues
- List, create and review pull requests
- Trigger and monitor GitHub Actions workflows

**User Valves:**
| Valve | Default | Controls |
|-------|---------|----------|
| `ENABLE_CONTENT` | on | Read access to repos, files, branches, commits |
| `ENABLE_CONTENT_WRITE` | on | Create branches, write/delete files |
| `ENABLE_ISSUES` | on | Read, create, update issues |
| `ENABLE_PULL_REQUESTS` | on | Read, create PRs, request reviewers |
| `ENABLE_WORKFLOWS` | on | List, trigger, cancel workflow runs |

**Workflow rule:**
```
github_create_branch → github_create_or_update_file → github_create_pull_request
```
No direct writes to `main`. No merge function.

---

### `confirm_destructive_actions.py` — Destructive Action Guard

Shows a confirmation dialog before irreversible operations (file deletion, branch deletion, etc.). The model must call this and receive `"confirmed"` before proceeding.

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

### `agent_loop.py` — Agent Loop (v4.2.0)

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
- **Silent mode** — strips tool call details, reasoning blocks, and intermediate status from output (UserValves: `SILENT_MODE`).
- **Context window management** — adaptive history truncation preserving tool-call pair integrity.
- **Iteration limit** — configurable max loop iterations with a Continue/Cancel dialog.
- **System prompt refresh** — the LLM always sees up-to-date task state after mutations.
- **Graceful shutdown** — handles `GeneratorExit`/`CancelledError` and saves state.

**Admin Valves (AgentValves):**
| Valve | Default | Description |
|-------|---------|-------------|
| `AGENT_MODEL` | (empty) | Model ID for the loop. Leave empty to use the selected model. Must support function calling. |
| `MAX_ITERATIONS` | 24 | Maximum agent loop iterations. |
| `MAX_TOOL_RESULT_CHARS` | 4200 | Max characters for tool results before truncation. |
| `TOOL_TIMEOUT` | 90 | Timeout in seconds for individual tool execution. 0 to disable. |
| `PLAN_TOOLS` | (empty) | Comma-separated tools allowed in PLAN phase. Empty = all tools. |
| `EXECUTE_TOOLS` | (empty) | Comma-separated tools allowed in EXECUTE phase. Empty = all tools. |
| `REVIEW_TOOLS` | (empty) | Comma-separated tools allowed in REVIEW phase. Empty = all tools. |
| `TOOLS_DENYLIST` | (empty) | Tools never available in any phase. |
| `PLAN_PROMPT` | (empty) | Custom PLAN system prompt. Placeholders: `{tool_names}`. |
| `EXECUTE_PROMPT` | (empty) | Custom EXECUTE system prompt. Placeholders: `{tool_names}`, `{task_state}`. |
| `REVIEW_PROMPT` | (empty) | Custom REVIEW system prompt. Placeholders: `{goal}`, `{task_state}`, `{tool_names}`. |

**User Valves (AgentUserValves):**
| Valve | Default | Description |
|-------|---------|-------------|
| `ENABLE_PLAN_APPROVAL` | false | Show plan confirmation popup before execution. |
| `YOLO_MODE` | false | Skip all user confirmations. Auto-approve plans and ignore iteration limits. |
| `SILENT_MODE` | false | Hide tool call details, reasoning blocks, and intermediate status. Show only plan approvals, final results, and errors. |

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
Never write directly to main. Always follow: github_create_branch → github_create_or_update_file → github_create_pull_request.
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
