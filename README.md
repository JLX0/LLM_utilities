# LLM Utilities

Misc. utilities for interacting with LLMs.

> ⚠️ **Currently under active development** — Documentation and more examples will be added soon.

---

## Requirements

- **OS:** Linux-based
- **Python:** >= 3.12

---

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/JLX0/LLM_utilities
   cd LLM_utilities
   ```

2. **Create virtual environment and install**

   ```bash
   uv venv .venv && source .venv/bin/activate
   uv pip install -e .
   ```

3. **Install code-checking and linting tools (for development)**

   ```bash
   uv pip install -e ".[checking]"
   pre-commit install
   ```

   > 💡 **Tip 1:** The pre-commit checks are more thorough but slower. Use `./format.sh` for a quicker, more focused check.

   > 💡 **Tip 2:** Always run the pre-commit checks (`pre-commit run --all-files`) before committing your changes to ensure consistent formatting and style.

4. **Install documentation tools (optional, for development)**

   ```bash
   uv pip install -e ".[document]"
   ```

---

## Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting (replacing black, flake8, and isort), along with [MyPy](https://mypy-lang.org/) for static type checking.

### Using format.sh

LLM_utils includes a helper script, `format.sh`, to automatically check and format source code.

#### Full formatting (default)

Format and lint **all source files**:

```bash
./format.sh
```

#### Check-only mode

Run all checks **without modifying any files**:

```bash
./format.sh -n
```

#### Format or check specific files or directories

Apply formatting and linting only to a specific target:

```bash
./format.sh -f LLM_utils/some_module.py
```

### Using pre-commit

Run all pre-commit checks:

```bash
pre-commit run --all-files
```

---