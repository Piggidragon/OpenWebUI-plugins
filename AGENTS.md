# AGENTS.md

## Repository Overview

OpenWebUI plugins: tools, pipelines, and prompts that extend [Open WebUI](https://github.com/open-webui/open-webui). Each plugin is a standalone Python file — no build system, no package manager, no tests.

## Structure

- **`tools/`** — Tool plugins. Each file defines a `class Tools` with `Valves` (admin) and `UserValves` (per-user) nested Pydantic models. Public `async` methods become LLM-callable functions.
- **`pipelines/`** — Pipeline plugins. Each file defines a `class Pipe` (or `class Pipeline` for the deprecated one). These are complex orchestration engines that run multi-step LLM loops.
- **`prompts/`** — Prompt templates consumed by pipelines (e.g., system prompts).

## Plug-in Conventions

- **Module docstring is metadata**: `title`, `author`, `version`, `description` (and optionally `requirements` / `required_open_webui_version`) are parsed from the top-level docstring by Open WebUI.
- **Valves pattern**: `class Tools` → nested `class Valves(BaseModel)` (global/admin) and `class UserValves(BaseModel)` (per-user). Pipelines use `class Pipe` with the same nested pattern.
- **No `main` / no CLI**: plugins are imported by Open WebUI at runtime. Do not add `if __name__` blocks or standalone entrypoints.
- **Dependencies**: declared in the docstring `requirements:` field, not in a requirements.txt. Open WebUI resolves them. Only `pydantic` and `httpx` are common across tools; pipelines also depend on `open_webui` internals.
- **`__user__` injection**: OpenWebUI injects a `__user__` dict into tool/pipeline calls. Access user valves via `__user__["valves"]`.

## Key Design Rules

- **GitHub tools enforce branch→file→PR workflow**: `github_create_branch` → `github_create_or_update_file` → `github_create_pull_request`. No direct writes to `main`. No merge function exists.
- **Destructive actions must be guarded**: `confirm_destructive_action` must be called before any irreversible operation, returning `"confirmed"` or `"cancelled"`.
- **Parallel tool calls**: the system prompt (`prompts/model-systemprompt.md`) instructs the model to use `run_tools_parallel` for independent calls.

## Pipelines

- **`helix_agent.py`** — Helix Agent: single-model Plan→Execute→Review→Replan loop with per-phase tool filtering via Valves. Requires `open-webui>=0.9.1`.
- **`planner_v3.py`** (v3.10.3, by Haervwe) — multi-agent orchestrator with subagents, MCP support, plan approval UI. Requires `open_webui>=0.9.1`.
- **`agent-pipeline-deprecated.py`** — legacy v0.24.0 pipeline. Do not modify unless fixing a bug.

## Editing Guidelines

- Each plugin file is self-contained. Changing one file does not affect others.
- Preserve the OpenWebUI docstring header format (`title`, `author`, `description`, `version`, `requirements`) — it is parsed at runtime.
- Use Pydantic `BaseModel` / `Field` for all configuration schemas.
- Keep async signatures on tool methods — Open WebUI expects `async def`.
- i18n strings live in module-level dicts (see `confirm_destructive_actions.py` for the pattern).