# AGENTS.md

## Repository Overview

OpenWebUI plugins: standalone Python files extending [Open WebUI](https://github.com/open-webui/open-webui). No build system, no package manager, no tests.

## Structure

- **`tools/`** — Tool plugins. Each file defines a `class Tools` with nested `Valves` (admin) and `UserValves` (per-user) Pydantic models. Public `async` methods become LLM-callable functions.
- **`pipelines/`** — Pipeline plugins. Each file defines a `class Pipe` (or `class Pipeline` for the deprecated one). Complex orchestration engines running multi-step LLM loops.
- **`prompts/`** — Prompt templates consumed by pipelines (e.g. `model-systemprompt.md`).
- **`skills/`** — Workflow and convention docs referenced by pipelines (e.g. `github_workflow.md`).

## Plug-in Conventions

- **Module docstring is runtime metadata** — OpenWebUI parses the top-level docstring. Required fields:
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
- **Valves boilerplate** — Tools and pipelines both use:
  ```python
  class Tools:  # or class Pipe
      class Valves(BaseModel):
          """Global / admin settings — leave empty if not needed."""
          pass
      class UserValves(BaseModel):
          SOME_TOKEN: str = Field(default="", description="...")
      def __init__(self):
          self.valves = self.Valves()
  ```
- **No `main` / no CLI** — Plugins are imported by OpenWebUI at runtime. Never add `if __name__` blocks or standalone entrypoints.
- **Dependencies** — Declared only in the docstring `requirements:` field (pipelines only). OpenWebUI resolves them. Do not create a `requirements.txt`.
- **Injected parameters** — OpenWebUI injects special params into tool/pipeline calls:
  - `__user__: Optional[dict]` — access user valves via `__user__["valves"]`.
  - `__event_emitter__` / `__event_call__` — native UI events (confirmation dialogs, status updates).
  - `__tools__` — available tool list for pipelines.
- **Async signatures** — Public tool methods must be `async def`.

## Workflow & Design Rules

- **GitHub branch→file→PR workflow** — `github_create_branch` → `github_create_or_update_file` → `github_create_pull_request`. No direct writes to `main`. No merge function exists.
- **Branch naming** — `<type>/<short-description>` (kebab-case, max 5 words). Types: `feature/`, `fix/`, `refactor/`, `docs/`, `chore/`.
- **Commit / PR format** — `<type>: <short description>` (e.g. `feat: add GitHub global search tool`). PR body must list changes, testing, and optional notes.
- **Destructive actions must be guarded** — Call `confirm_destructive_action` before any irreversible operation. It returns `"confirmed"` or `"cancelled"`.
- **Parallel tool calls** — The system prompt (`prompts/model-systemprompt.md`) instructs the model to use `run_tools_parallel` for independent calls.

## Pipelines

- **`helix_agent.py`** (v0.24.2) — Single-model Plan→Execute→Review→Replan loop. Per-phase tool filtering via Valves. Requires `open-webui>=0.9.1`.
- **`planner_v3.py`** (v3.10.3, by Haervwe) — Multi-agent orchestrator with subagents, MCP support, plan approval UI. Requires `open_webui>=0.9.1`.
- **`agent-pipeline-deprecated.py`** — Legacy v0.24.0 pipeline. Do not modify unless fixing a bug.

## Editing Guidelines

- Each plugin file is self-contained. Changing one file does not affect others.
- Preserve the docstring header format exactly — it is parsed at runtime.
- Use Pydantic `BaseModel` / `Field` for all configuration schemas.
- i18n strings live in module-level dicts keyed by language code (see `confirm_destructive_actions.py` for the pattern).
