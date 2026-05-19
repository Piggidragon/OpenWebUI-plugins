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

To edit a file, always read it first. For large files, use `limit` and `offset` to read only a slice:
```python
github_read_file(repo, path, branch, limit=50, offset=1)  # lines 1-50
github_read_file(repo, path, branch, limit=20, offset=51) # lines 51-70
```

Then choose the right edit tool based on the change size:

| Tool | When to use |
|------|-------------|
| `github_replace_string` | Small, targeted edits (rename variable, fix typo, change a value). Replaces an exact substring — no need to rewrite the whole file. |
| `github_insert_at_line` | Insert new lines at a specific line number (1-based). Existing content shifts down. |
| `github_delete_lines` | Remove a contiguous range of lines (inclusive, 1-based). |
| `github_write_file` | Large refactors or rewrites. **MUST provide the COMPLETE file content** — partial content will overwrite everything and cause data loss. |

**Rule:** Prefer `replace_string`, `insert_at_line`, or `delete_lines` whenever possible. Only use `github_write_file` when you truly need to rewrite the entire file.

**Rule:** Never write directly to `main`. Every change must follow:
```
github_create_branch → <edit_tool> → github_create_pull_request
```

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