# LLM Utilities

Misc. utilities for interacting with LLMs.

---

## Requirements

- **OS:** Linux-based (tested with Ubuntu 20.04.6 LTS)
- **uv* (tested with 0.8.8)

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
5. **Set API keys**

   ```bash
   # By default, if the OpenRouter key is provided, it will override the API key for the original provider
    export OPENROUTER_API_KEY=""
    
    export GEMINI_API_KEY=""
    export ANTHROPIC_API_KEY=""
    export DEEPSEEK_API_KEY=""
    export OPENAI_API_KEY=""
   
   # You only need to provide the keys for LLMs that you will call
   ```

---

## LLM requests

### Routing Logic

**Model Selection:**
- **Gemini Pro 3.0**: Always uses direct API (not available on OpenRouter)
- **GPT-5.2, Claude Sonnet 4.5, DeepSeek V3.2**: Prefer OpenRouter if `OPENROUTER_API_KEY` is set, otherwise fall back to direct provider APIs
- **`force_direct=True`**: Overrides OpenRouter preference, forcing direct API use

### Reasoning/Thinking Token Control

When `reasoning_effort` is specified ("low", "medium", or "high"):

| Provider | Direct API | OpenRouter |
|----------|-----------|------------|
| **OpenAI** | `reasoning_effort=<level>`, `max_completion_tokens`, `temperature=1.0` | `reasoning={"effort": <level>}`, `temperature=1.0` |
| **Anthropic** | `thinking={"type": "enabled", "budget_tokens": X}`, `temperature=1.0` | `reasoning={"max_tokens": X}` |
| **Gemini** | `reasoning_effort=<level>` (no temperature) | N/A (always direct) |
| **DeepSeek** | Switches model from `deepseek-chat` → `deepseek-reasoner` | `reasoning={"enabled": True, "max_tokens": X}` |

**Notes:**
- **OpenAI Direct**: Uses native `reasoning_effort` parameter with `temperature=1.0` (required by OpenAI for reasoning). Uses `max_completion_tokens` instead of `max_tokens`.
- **OpenAI via OpenRouter**: Uses `reasoning={"effort": <level>}` object with `temperature=1.0`.
- **Anthropic Direct**: Uses native `thinking` parameter with `budget_tokens` and `temperature=1.0` (required for thinking mode). Automatically increases `max_tokens` by `budget_tokens + 1000` to accommodate both thinking and response.
- **Anthropic via OpenRouter**: Uses `reasoning={"max_tokens": X}` since OpenRouter doesn't support the `thinking` parameter directly. Also increases `max_tokens`. Does not set temperature.
- **Gemini**: Cannot disable thinking/reasoning (always active). Defaults to `reasoning_effort="high"` if not specified. Minimum `max_tokens=200` enforced. Temperature is not supported with Gemini reasoning. `drop_params=False` is always set.
- **DeepSeek Direct**: Reasoning is enabled by switching to the `deepseek-reasoner` model rather than passing parameters.
- **DeepSeek via OpenRouter**: Uses `reasoning={"enabled": True, "max_tokens": X}` object.

**Gemini-Specific Behavior:**
- Gemini 3 Pro **cannot disable thinking** - reasoning tokens are always used
- If `reasoning_effort` is not specified, it defaults to `"high"`
- Minimum `max_tokens` of 200 is enforced to ensure room for both reasoning and response

**Token Budget Mapping (for Anthropic/DeepSeek):**

| Effort Level | Anthropic | DeepSeek |
|--------------|-----------|----------|
| Low | 1,024 tokens | 1,024 tokens |
| Medium | 2,048 tokens | 8,192 tokens |
| High | 4,096 tokens | 30,000 tokens |

### API Key Requirements

| Model | Direct API Key | OpenRouter Key |
|-------|---------------|----------------|
| GPT-5.2 | `OPENAI_API_KEY` | `OPENROUTER_API_KEY` |
| Claude Sonnet 4.5 | `ANTHROPIC_API_KEY` | `OPENROUTER_API_KEY` |
| Gemini Pro 3.0 | `GEMINI_API_KEY` | N/A |
| DeepSeek V3.2 | `DEEPSEEK_API_KEY` | `OPENROUTER_API_KEY` |

### Query Flow

1. Resolve model alias to canonical name (e.g., "sonnet" → "claude-sonnet-4.5")
2. Determine routing: OpenRouter preferred (if key available and not Gemini), else direct
3. Map canonical name to provider-specific model ID
4. **For Gemini**: Auto-default `reasoning_effort="high"` if not specified
5. Build API params with provider-appropriate reasoning configuration
6. **For Gemini**: Enforce minimum `max_tokens=200`, remove temperature, set `drop_params=False`
7. **For providers with token budgets**: Automatically increase `max_tokens` to accommodate reasoning + response
8. Set correct API key environment variable
9. Call LiteLLM with constructed parameters
10. Calculate cost using routing-appropriate pricing

### Model ID Mappings

**Direct API:**
| Canonical Name | LiteLLM Model ID |
|----------------|------------------|
| claude-sonnet-4.5 | `anthropic/claude-sonnet-4-5-20250929` |
| gpt-5.2 | `openai/gpt-5.2` |
| gemini-pro-3.0 | `gemini/gemini-3-pro-preview` |
| deepseek-v3.2 | `deepseek/deepseek-chat` |
| deepseek-v3.2 (reasoning) | `deepseek/deepseek-reasoner` |

**OpenRouter:**
| Canonical Name | LiteLLM Model ID |
|----------------|------------------|
| claude-sonnet-4.5 | `openrouter/anthropic/claude-sonnet-4.5` |
| gpt-5.2 | `openrouter/openai/gpt-5.2` |
| deepseek-v3.2 | `openrouter/deepseek/deepseek-v3.2` |

### Constants
```python
GEMINI_MIN_MAX_TOKENS = 2048  # Minimum max tokens for Gemini to ensure response space

EFFORT_TO_TOKENS = {
    "anthropic": {"low": 1024, "medium": 2048, "high": 4096},
    "deepseek": {"low": 1024, "medium": 8192, "high": 30000},
}

VALID_EFFORTS = ("low", "medium", "high")
```
---

## Testing

After the relevant keys are set, run

```bash
pytest tests/ -v -s
```
In general it should complete in 10 minutes

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