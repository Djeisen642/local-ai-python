## Summary

This repository is a small Python 3.13+ project called `local_ai`. It contains a minimal library in `src/local_ai` and a single unit test in `tests/`. The project uses `pyproject.toml` with `uv_build` as the build backend and development tools configured (pytest, mypy, ruff, etc.).

Here are the goals of this project:

- Speech to text
- Text to speech
- Read emails to determine which ones require a push notification
- Read calendar to determine which events require a push notification
- Memorize and adapt what is important to the user
- Interact with smart home devices

## Key facts

- Languages: Python (target 3.13)
- Layout: small; main package under `src/local_ai` and tests under `tests`
- Build system: PEP 517 build with `uv_build`
- Test runner: `pytest` (configured via `pyproject.toml`)
- Lint/type tools: `ruff`, `mypy` (strict), plus docformatter, bandit in dev group

## When to trust this file

Treat these instructions as authoritative. Only search the repo if a step fails, or the file indicates missing/changed configuration.

## Bootstrap / environment

Always start from a clean virtual environment. Recommended Python: 3.13 (the project requires >=3.13).

Typical setup (uv-only)

This repository uses `uv` to manage Python environments and includes `uv.lock`. Prefer running commands with `uv run <cmd>` so they execute inside the project's uv environment. Example install and setup:

```bash
# upgrade pip inside uv environment and install editable package + dev deps
uv run pip install --upgrade pip
# quote the dependency group to avoid zsh globbing: '.[dev]'
uv run pip install -e '.[dev]'
```

## Build / Run / Test / Lint

Use these commands via `uv run <cmd>` after creating/activating the `uv` environment.

- Run tests:

```bash
uv run pytest tests
```

- Run the module (manual run):

```bash
uv run python -m local_ai.main
```

- Lint (ruff):

```bash
uv run ruff check src tests
```

- Type check (mypy strict):

```bash
uv run mypy
```

## Known working order

1. Activate virtualenv.
2. Install dev deps (`pip install -e .[dev]` or manual installs).
3. Run `ruff` then `mypy` then test.

If anything fails, run tests or `mypy` again after fixing issues.

## Common failures & workarounds

- Missing Python 3.13: use a compatible interpreter or update CI to provide 3.13.
- If `pip install -e .[dev]` fails due to resolver/PEP issues, install dev packages individually: `pip install pytest pytest-cov mypy ruff docformatter bandit`.
- If `ruff` or `mypy` report configuration issues, rely on `pyproject.toml` values; only change config if tests or CI require it.

## Project layout (important paths)

- `pyproject.toml` — build and tool configuration (root).
- `README.md` — high level project goals.
- `src/local_ai/main.py` — primary module with `hello_world()` and `main()`.
- `tests/test_local_ai.py` — unit tests.
- `.github/` — CI and workflow files (may be absent or minimal).

## Checks run pre-commit/CI

This repo does not include explicit GitHub Actions workflows in the visible tree; assume standard checks the maintainer expects locally: ruff, mypy, pytest. If a workflow exists in `.github/workflows/`, prefer reproducing its steps locally in the same order.

## How to make a safe change

- Keep changes minimal and focused to the module under `src/local_ai` and update or add tests in `tests/`.
- Run `ruff` and `mypy` before tests; fix lint/type issues first. Then run `pytest` to validate behavior.
- Use `pytest -q` for quick feedback; use `pytest -k <pattern>` to run subset tests while iterating.

## What to edit and where to look

- Implement library code in `src/local_ai/`.
- Tests live in `tests/` and use direct imports (`from local_ai.main import ...`).
- Update `pyproject.toml` only when adding dependencies or changing tool configs.

## Developer tips for the coding agent

- Make a single logical change per branch/PR.
- Run linting and type checks locally before opening a PR.
- Keep any new external dependencies small and add them to `pyproject.toml` with justification.
- When editing tests, ensure `python -m pytest tests` passes locally.

## Final validation checklist (run before proposing PR)

Run these from the repository root using `uv run`:

1. `uv run pip install -e '.[dev]'`
2. `uv run ruff check src tests`
3. `uv run mypy`
4. `uv run pytest`

If all pass, the change is safe to propose.

## Extra: repository root listing

Files in repo root (important): `Dockerfile`, `LICENSE`, `pyproject.toml`, `README.md`, `uv.lock`, `src/`, `tests/`.
