# GitHub Workflow & Python Doc Standard

## 0️⃣ Hard Rules

**Rule 1:** Never write directly to `main`. Every change must follow:
```
github_create_branch → github_create_file / github_write_file → github_create_pull_request
```

**Rule 2:** When using `github_write_file`, you MUST provide the COMPLETE file content. Never send only the changed parts — partial content will overwrite the entire file and cause data loss.

**Rule 3:** All commit messages, PR titles, PR bodies, issue comments, and documentation MUST be in **English** only.

**Rule 4:** Every Python file must have the standard docstring header (see §3).

---

## 1️⃣ Branch Naming

Format: `<type>/<short-description>` (kebab-case, max 5 words)

| Type | When |
|------|------|
| `feature/` | New tool or feature |
| `fix/` | Bugfix |
| `refactor/` | Code restructuring |
| `docs/` | Documentation only |
| `chore/` | Config, CI, dependencies |

---

## 2️⃣ Editing Existing Files

To edit a file, always read it first, then write the full content back:
1. `github_get_file(repo, path, branch)` — read current content
2. `github_write_file(repo, path, COMPLETE_content, branch)` — write entire updated file

Use `github_create_file` only for files that do not yet exist. It will refuse if the file already exists.

---

## 3️⃣ Python Doc Standard

### File Header

```python
"""
title: <Short Display Name>
author: Piggidragon
version: <semver>
description: >
  One or two lines describing what it does.
requirements: <pip-packages>  (optional; pipelines only)
"""
```

### Class Structure

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

### Function Docstrings

```python
async def my_tool(self, param1: str, __user__: Optional[dict] = None) -> str:
    """
    Short description of what this tool does.

    :param param1: What this parameter controls
    :param __user__: Injected by OWUI (not user-provided)
    """
```

Internal helpers start with `_` and only need a docstring if the logic is complex.

---

## 4️⃣ Commit & PR Conventions

**Commit messages:** `<type>: <short description>`
**PR titles:** `<Type>: <Short description>`

```
feat: add GitHub global search tool
fix: handle rate limit errors
docs: add workflow examples
refactor: remove duplicate helpers
chore: update valve defaults
```

**PR body** must include:
- What does this change?
- Changes (bullet points)
- How was this tested?
- Notes (optional)