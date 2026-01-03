#!/bin/bash
# This script checks and formats Python source code using Ruff (formatter + linter + isort),
# Blackdoc (docstring/doc code blocks), and mypy (type checking).
# - With `-n`, the script performs checks without modifying files.
# - With `-f <filepath>`, it targets the specified path only.
# - With `-c`, it cleans up the dmypy daemon after running.

# Initialize defaults
update=1   # default: modify files if needed
target=""
mypy_target=""
missing_dependencies=()
single_file_mode=0  # track if we're targeting a single file
cleanup_daemon=0    # whether to stop dmypy daemon after running

# Define exclusion patterns (matching pre-commit config)
# Ruff exclusions
RUFF_EXCLUDE_PATTERNS=(
  "tutorial"
  "docs/visualization_examples"
  "docs/visualization_matplotlib_examples"
)

# Mypy exclusions
MYPY_EXCLUDE_PATTERNS=(
  "docs"
  "tutorial"
)

# Build ruff exclude arguments
RUFF_EXCLUDE_ARGS=""
for pattern in "${RUFF_EXCLUDE_PATTERNS[@]}"; do
  RUFF_EXCLUDE_ARGS="$RUFF_EXCLUDE_ARGS --exclude $pattern"
done

# Build mypy exclude pattern (uses regex format)
MYPY_EXCLUDE_PATTERN=$(IFS="|"; echo "${MYPY_EXCLUDE_PATTERNS[*]}")

# --- Dependency checks ---
# Core tools
command -v ruff &> /dev/null || missing_dependencies+=(ruff)
command -v blackdoc &> /dev/null || missing_dependencies+=(blackdoc)
command -v mypy &> /dev/null || missing_dependencies+=(mypy)

# Type stub dependencies (matching pre-commit additional_dependencies)
TYPE_STUBS=(
  "alembic>=1.5.0"
  "colorlog"
  "numpy"
  "packaging>=20.0"
  "sqlalchemy>=1.3.0"
  "tqdm"
  "PyYAML"
  "mypy_boto3_s3"
  "types-PyYAML"
  "types-redis"
  "types-setuptools"
  "types-tqdm"
  "typing_extensions>=3.10.0.0"
)

# Check if type stubs are installed using importlib.metadata (works with any package manager)
missing_type_stubs=()
for stub in "${TYPE_STUBS[@]}"; do
  # Extract package name (before >= or ==)
  pkg_name=$(echo "$stub" | sed 's/[>=<].*//')
  # Use Python's importlib.metadata (works with pip, uv, poetry, etc.)
  python -c "import importlib.metadata; importlib.metadata.version('$pkg_name')" &>/dev/null || missing_type_stubs+=("$stub")
done

if [ ! ${#missing_dependencies[@]} -eq 0 ] || [ ! ${#missing_type_stubs[@]} -eq 0 ]; then
  echo "Missing dependencies detected:"
  [ ! ${#missing_dependencies[@]} -eq 0 ] && echo "  Tools: ${missing_dependencies[@]}"
  [ ! ${#missing_type_stubs[@]} -eq 0 ] && echo "  Type stubs: ${missing_type_stubs[@]}"

  read -p "Would you like to install the missing dependencies? (y/N): " yn
  case "$yn" in
    [yY]*)
      if [ ! ${#missing_dependencies[@]} -eq 0 ]; then
        # Try uv first, fall back to pip
        if command -v uv &> /dev/null; then
          uv pip install "${missing_dependencies[@]}" || { echo "Tool installation failed."; exit 1; }
        else
          pip install "${missing_dependencies[@]}" || { echo "Tool installation failed."; exit 1; }
        fi
      fi
      if [ ! ${#missing_type_stubs[@]} -eq 0 ]; then
        # Try uv first, fall back to pip
        if command -v uv &> /dev/null; then
          uv pip install "${missing_type_stubs[@]}" || { echo "Type stub installation failed."; exit 1; }
        else
          pip install "${missing_type_stubs[@]}" || { echo "Type stub installation failed."; exit 1; }
        fi
      fi
      ;;
    *)
      echo "Warning: Running without all dependencies may produce inconsistent results."
      echo "For full consistency with pre-commit, run: uv pip install -e \".[checking]\""
      ;;
  esac
fi

# --- Parse options ---
while getopts "nf:c" OPT; do
  case $OPT in
    n) update=0 ;;               # check-only mode
    f) target="$OPTARG"; mypy_target="$OPTARG"; single_file_mode=1 ;;
    c) cleanup_daemon=1 ;;       # cleanup dmypy daemon after running
    *) ;;
  esac
done

# Default to current directory if no target provided
if [ -z "$target" ]; then
  target="."
fi
if [ -z "$mypy_target" ]; then
  mypy_target="."
fi

res_all=0

# --- Ruff: formatting and linting ---
# Matching pre-commit: ruff-format and ruff with --fix flag

if [ $update -eq 1 ]; then
  echo "Running ruff format (apply changes)…"
  ruff format $RUFF_EXCLUDE_ARGS "$target"
  if [ $? -ne 0 ]; then
    echo "ruff format failed."
    res_all=1
  else
    echo "ruff format succeeded."
  fi

  echo "Running ruff check --fix (matching pre-commit)…"
  ruff check --fix $RUFF_EXCLUDE_ARGS "$target"
  if [ $? -ne 0 ]; then
    echo "ruff check reported issues (some may require manual fixes)."
    res_all=1
  else
    echo "ruff check succeeded."
  fi
else
  echo "Running ruff format --check --diff…"
  ruff format --check --diff $RUFF_EXCLUDE_ARGS "$target"
  if [ $? -ne 0 ]; then
    echo "ruff format check failed."
    res_all=1
  else
    echo "ruff format check succeeded."
  fi

  echo "Running ruff check…"
  ruff check $RUFF_EXCLUDE_ARGS "$target"
  if [ $? -ne 0 ]; then
    echo "ruff check failed."
    res_all=1
  else
    echo "ruff check succeeded."
  fi
fi

# --- Blackdoc: format code blocks in docstrings ---
if [ $update -eq 1 ]; then
  echo "Running blackdoc (apply changes)…"
  blackdoc_output=$(blackdoc "$target" 2>&1)
  if echo "$blackdoc_output" | grep -q "left unchanged"; then
    echo "blackdoc succeeded."
  else
    if echo "$blackdoc_output" | grep -qi "error"; then
      echo "$blackdoc_output"
      echo "blackdoc failed."
      res_all=1
    else
      echo "$blackdoc_output"
      echo "blackdoc completed with changes."
    fi
  fi
else
  echo "Running blackdoc --check --diff…"
  res_blackdoc=$(blackdoc "$target" --check --diff 2>&1)
  if [ $? -ne 0 ]; then
    echo "$res_blackdoc"
    echo "blackdoc check failed."
    res_all=1
  else
    echo "blackdoc check succeeded."
  fi
fi

# --- Mypy: strict type checking ---
# Using explicit flags to match pre-commit config exactly

# Build mypy arguments matching pre-commit
MYPY_ARGS=(
  --warn-unused-configs
  --disallow-untyped-calls
  --disallow-untyped-defs
  --disallow-incomplete-defs
  --check-untyped-defs
  --no-implicit-optional
  --warn-redundant-casts
  --strict-equality
  --extra-checks
  --no-implicit-reexport
  --ignore-missing-imports
  --enable-incomplete-feature=NewGenericSyntax
)

used_dmypy=0  # Track if we used dmypy

if [ $single_file_mode -eq 1 ] && [ -f "$mypy_target" ]; then
  # Single file mode: Check if file should be excluded
  should_skip=0
  for pattern in "${MYPY_EXCLUDE_PATTERNS[@]}"; do
    if [[ "$mypy_target" == *"$pattern"* ]]; then
      should_skip=1
      echo "Skipping mypy for $mypy_target (matches exclusion pattern: $pattern)"
      break
    fi
  done

  if [ $should_skip -eq 0 ]; then
    # Use dmypy daemon for speed WITH strict checks for accuracy
    echo "Running mypy on single file (strict + fast with caching)…"

    if command -v dmypy &> /dev/null; then
      used_dmypy=1
      # Use dmypy with full strict flags - first run may be slow, subsequent runs are fast
      res_mypy=$(dmypy run -- "${MYPY_ARGS[@]}" "$mypy_target" 2>&1)
      mypy_exit=$?

      # If dmypy had issues, fall back to regular mypy
      if echo "$res_mypy" | grep -q "Daemon crashed!"; then
        echo "Daemon crashed, restarting and retrying..."
        dmypy stop &>/dev/null
        res_mypy=$(dmypy run -- "${MYPY_ARGS[@]}" "$mypy_target" 2>&1)
        mypy_exit=$?
      fi
    else
      # Fall back to regular mypy with incremental cache
      res_mypy=$(mypy "${MYPY_ARGS[@]}" "$mypy_target" 2>&1)
      mypy_exit=$?
    fi
  else
    mypy_exit=0
  fi
else
  # Full codebase mode: MATCHES PRE-COMMIT EXACTLY
  echo "Running mypy (full check matching pre-commit)…"
  res_mypy=$(mypy "${MYPY_ARGS[@]}" --exclude "$MYPY_EXCLUDE_PATTERN" "$mypy_target" 2>&1)
  mypy_exit=$?
fi

if [ $mypy_exit -ne 0 ]; then
  echo "$res_mypy"
  echo "mypy failed."
  res_all=1
else
  echo "mypy succeeded."
fi

# --- Cleanup: Stop dmypy daemon if requested or after single-file checks ---
if [ $used_dmypy -eq 1 ]; then
  if [ $cleanup_daemon -eq 1 ] || [ $single_file_mode -eq 1 ]; then
    echo "Cleaning up dmypy daemon..."
    dmypy stop &>/dev/null
    if [ $? -eq 0 ]; then
      echo "dmypy daemon stopped successfully."
    fi
  else
    echo "(dmypy daemon is still running for faster subsequent checks. Use -c flag to stop it.)"
  fi
fi

# --- Final exit status ---
if [ $res_all -eq 1 ]; then
  echo ""
  echo "❌ Checks failed. For guaranteed consistency with pre-commit, consider:"
  echo "   pre-commit run --all-files"
  exit 1
else
  echo ""
  echo "✅ All checks passed!"
fi