# GitHub Workflow & Python Doc Standard

## 0️⃣ Hard Rules
**Rule 1:** Never push or write directly to `main`. Every change must follow:
```
Create branch → Write code locally → Create pull request
```

**Rule 2:** All files MUST be written locally FIRST in the `github/` directory with correct subfolder structure, never directly to `/home/` or anywhere else.

```
github/
├── tools/       ← Tool plugins (Tools class)
├── pipelines/   ← Pipe plugins (Pipe/Pipeline class)
├── prompts/     ← Markdown prompt templates
└── README.md    ← Repository documentation
```

**Rule 3:** Every Python file must have the standard docstring header (see §4).

**Rule 4:** All pull request content, commit messages, issue comments, code comments, and documentation MUST be written in **English** only. No German or other languages in repository-facing content.

---

## 1️⃣ Branch Naming

Format: `<type>/<short-description>`

| Type | When | Example |
|------|------|---------|
| `feature/` | New tool or feature | `feature/github-search-tool` |
| `fix/` | Bugfix | `fix/rate-limit-handling` |
| `refactor/` | Code restructuring, no new features | `refactor/cleanup-helpers` |
| `docs/` | Documentation only | `docs/add-workflow-examples` |
| `chore/` | Config, CI, dependencies | `chore/update-valves-defaults` |

Description: **kebab-case**, max 5 words, English.

---

## 2️⃣ Write Code — Local First

1. Create the file under `github/` with the correct subfolder:
   - Tool → `github/tools/name.py`
   - Pipeline → `github/pipelines/name.py`
   - Prompt → `github/prompts/name.md`

2. Write the file with the standard docstring header (see §4).

3. Commit to the branch using `github_create_file` (for new files) or `github_write_file` (to overwrite existing files) with `branch` parameter.

---

## 3️⃣ Pull Request

### PR Title Format
```
<Type>: <Short description>
```

Examples:
- `Feature: GitHub Global Search Tool`
- `Fix: Rate limit handling in search`
- `Docs: Add workflow examples`
- `Refactor: Remove duplicate helpers`
- `Chore: Update valve defaults`

### PR Body Structure

Always include these sections in English:

```
## What does this change?

[2-3 sentences explaining what changed and why]

## Changes

- [Point 1]
- [Point 2]
- [Point 3]

## How was this tested?

- Test A
- Test B

## Notes

[Any additional context]
```

---

## 4️⃣ Python Doc Standard (Tools & Pipelines)

### 4.1 File Header Docstring

Every Python file starts with a triple-quoted docstring with these fields:

```python
"""
title: <Short Display Name>
author: Piggidragon
version: <semver>
description: >
  One or two lines describing what it does.
  Can wrap to multiple lines with `>` prefix.
  Tools: explain what the model can use it for.
  Pipelines: explain architecture and what it does/doesn't do.
requirements: <pip-packages>  (optional; pipelines only)
"""
```

**Existing examples:**

- **Tools** (`github_search.py`): `title`, `author`, `description`, `version`
- **Pipelines** (`agent-pipeline-deprecated.py`): `title`, `author`, `version`, `description`, `requirements`

**Tool-specific:** Add `requirements: <packages>` only if the tool has pip dependencies.

### 4.2 Class Structure

**Tools** follow this pattern:

```python
class Tools:
    class Valves(BaseModel):
        """Global / admin settings — leave empty if not needed."""
        pass

    class UserValves(BaseModel):
        SOME_TOKEN: str = Field(default="", description="...")

    def __init__(self):
        self.valves = self.Valves()
```

**Pipelines** follow this pattern:

```python
class Pipe:
    class Valves(BaseModel):
        SETTING: str = Field(default="", description="...")

    def __init__(self):
        self.type = "manifold"  # or "pipe"
        self.valves = self.Valves()
```

### 4.3 Function Docstrings

Every public async method needs a docstring:

```python
async def my_tool(self, param1: str, param2: int = 10, __user__: Optional[dict] = None) -> str:
    """
    Short description of what this tool does.

    :param param1: What this parameter controls
    :param param2: What this parameter controls (default: 10)
    :param __user__: Injected by OWUI (not user-provided)
    :return: What the model gets back
    """
```

Rules:
- First line = short description (one sentence)
- `:param` annotations for every parameter the model will see
- `__user__`, `__event_emitter__`, `__event_call__` get a brief note that they're injected
- No need to document internal helpers (start with `_`)

### 4.4 Internal Helpers

Prefix with underscore `_`. No docstring needed unless complex logic.

```python
def _get_uv(self, __user__: Optional[dict]) -> "Tools.UserValves":
    """Extract UserValves from the __user__ dict injected by OpenWebUI."""
    ...
```

---

## 5️⃣ Folder Structure Reference

```
github/                                      ← Local working copy
├── tools/
│   ├── github_access.py                     ← Full GitHub CRUD (47KB)
│   ├── github_search.py                     ← GitHub global search (14KB)
│   └── confirm_destructive_actions.py        ← Confirmation dialog (7KB)
├── pipelines/
│   ├── agent-pipeline-deprecated.py          ← Old agent pipeline (138KB)
│   └── agent_loop.py                         ← New minimal agent loop (666 lines)
├── prompts/
│   └── model-systemprompt.md                 ← System prompt template
├── .gitignore
├── LICENSE
└── README.md
```

When creating new files:
- Tool with a `Tools` class → `github/tools/`
- Pipeline with a `Pipe` or `Pipeline` class → `github/pipelines/`
- Markdown templates → `github/prompts/`

---

## 6️⃣ Commit Message Convention

```
<type>: <short description>

<optional body explaining why>
```

Examples:
- `feat: add GitHub global search tool`
- `refactor: simplify agent loop pipe v2.0.0`
- `docs: add workflow examples to README`
- `fix: handle rate limit errors in search`
- `chore: update valve defaults`