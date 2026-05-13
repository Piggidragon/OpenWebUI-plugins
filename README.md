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
| `ENABLE_ISSUES` | on | Read, create, update, close issues; add/list/update/delete comments |
| `ENABLE_PULL_REQUESTS` | on | Read, create, update PRs, request reviewers, diff |
| `ENABLE_WORKFLOWS` | on | List, trigger, cancel workflow runs |

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

## Installation

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
