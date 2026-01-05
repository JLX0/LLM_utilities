"""
LiteLLM-based LLM interface for unified API access across multiple providers.

Supported models:
- claude-sonnet-4.5 (direct via Anthropic API or via OpenRouter)
- gpt-5.2 (direct via OpenAI API or via OpenRouter)
- gemini-pro-3.0 (direct via Google API only - no OpenRouter support)
- deepseek-v3.2 (direct via DeepSeek API or via OpenRouter, both reasoning and non-reasoning modes)

Routing logic:
- Gemini: Always uses direct Google API
- Other models: Prefer OpenRouter if OPENROUTER_API_KEY is set, otherwise use direct API

Dependencies:
- litellm
- tenacity (required for reasoning/thinking features)
"""

from __future__ import annotations

import ast
from collections.abc import Callable
import json
import logging
import os
import traceback
from typing import Any
import warnings

import litellm
from litellm import completion

from LLM_utils.cost import Calculator
from LLM_utils.fault_tolerance import retry_overtime_kill


# Configure logging
logger = logging.getLogger("LiteLLM")
logger.setLevel(logging.WARNING)
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.success_callback = []
litellm.failure_callback = []

warnings.filterwarnings(
    "ignore",
    message=".*is bound to a different event loop.*",
    category=RuntimeWarning,
)
# Suppress Pydantic serialization warnings from litellm's response handling
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
# Suppress fork deprecation warning in multi-threaded process
warnings.filterwarnings(
    "ignore",
    message=".*use of fork\\(\\) may lead to deadlocks.*",
    category=DeprecationWarning,
)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


# =============================================================================
# Model Mappings
# =============================================================================

# Canonical model names (user-facing)
CANONICAL_MODELS = [
    "claude-sonnet-4.5",
    "gpt-5.2",
    "gemini-pro-3.0",
    "deepseek-v3.2",
]

# User-friendly aliases to canonical names
MODEL_ALIASES: dict[str, str] = {
    # Claude aliases
    "claude-sonnet-4.5": "claude-sonnet-4.5",
    "claude-4-5-sonnet": "claude-sonnet-4.5",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude": "claude-sonnet-4.5",
    # GPT aliases
    "gpt-5.2": "gpt-5.2",
    "gpt-5": "gpt-5.2",
    "gpt": "gpt-5.2",
    # Gemini aliases
    "gemini-pro-3.0": "gemini-pro-3.0",
    "gemini-3-pro": "gemini-pro-3.0",
    "gemini-pro-3": "gemini-pro-3.0",
    "gemini": "gemini-pro-3.0",
    # DeepSeek aliases
    "deepseek-v3.2": "deepseek-v3.2",
    "deepseek-3.2": "deepseek-v3.2",
    "deepseek": "deepseek-v3.2",
}

# Direct API model mappings (native provider APIs)
DIRECT_MODEL_MAPPINGS: dict[str, str] = {
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4-5-20250929",
    "gpt-5.2": "openai/gpt-5.2",
    "gemini-pro-3.0": "gemini/gemini-3-pro-preview",
    "deepseek-v3.2": "deepseek/deepseek-chat",  # Non-thinking mode by default
}

# DeepSeek thinking mode model (direct API)
DEEPSEEK_REASONING_MODEL = "deepseek/deepseek-reasoner"

# OpenRouter model mappings
OPENROUTER_MODEL_MAPPINGS: dict[str, str] = {
    "claude-sonnet-4.5": "openrouter/anthropic/claude-sonnet-4.5",
    "gpt-5.2": "openrouter/openai/gpt-5.2",
    "deepseek-v3.2": "openrouter/deepseek/deepseek-v3.2",
    # Note: Gemini is not available on OpenRouter, always uses direct
}

# Valid reasoning effort levels
VALID_EFFORTS = ("low", "medium", "high")

# Reasoning effort to token budget mapping for providers that need explicit budgets
EFFORT_TO_TOKENS: dict[str, dict[str, int]] = {
    "anthropic": {
        "low": 1024,
        "medium": 2048,
        "high": 4096,
    },
    "deepseek": {
        "low": 1024,
        "medium": 8192,
        "high": 30000,
    },
}

# Environment variable names for API keys
API_KEY_ENV_VARS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Minimum max_tokens for Gemini models to ensure room for both reasoning and response
GEMINI_MIN_MAX_TOKENS = 200


# =============================================================================
# Helper Functions
# =============================================================================


def _check_tenacity_available() -> bool:
    """Check if tenacity module is available."""
    try:
        import tenacity  # noqa: F401

        return True
    except ImportError:
        return False


def _has_openrouter_key() -> bool:
    """Check if OpenRouter API key is available."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _is_gemini_model(canonical_name: str) -> bool:
    """Check if the canonical model name is a Gemini model."""
    return canonical_name == "gemini-pro-3.0"


def _resolve_to_canonical(model: str) -> str:
    """Resolve a model name or alias to its canonical form."""
    return MODEL_ALIASES.get(model.lower(), model)


def _get_litellm_model_id(
    canonical_name: str,
    use_openrouter: bool,
    reasoning_enabled: bool = False,
) -> str:
    """
    Get the LiteLLM model identifier based on routing preference.

    Args:
        canonical_name: The canonical model name (e.g., "claude-sonnet-4.5")
        use_openrouter: Whether to use OpenRouter routing
        reasoning_enabled: Whether reasoning/thinking mode is enabled (affects DeepSeek)

    Returns:
        The LiteLLM model identifier string
    """
    # Gemini always uses direct API
    if _is_gemini_model(canonical_name):
        return DIRECT_MODEL_MAPPINGS[canonical_name]

    # DeepSeek with reasoning enabled uses different model in direct mode
    if canonical_name == "deepseek-v3.2" and reasoning_enabled and not use_openrouter:
        return DEEPSEEK_REASONING_MODEL

    # Use OpenRouter or direct based on preference
    if use_openrouter and canonical_name in OPENROUTER_MODEL_MAPPINGS:
        return OPENROUTER_MODEL_MAPPINGS[canonical_name]

    return DIRECT_MODEL_MAPPINGS.get(canonical_name, canonical_name)


def _get_provider_from_model_id(model_id: str) -> str:
    """
    Extract the provider from a LiteLLM model identifier.

    Returns one of: "openrouter", "anthropic", "openai", "gemini", "deepseek", "unknown"
    """
    model_lower = model_id.lower()

    if model_lower.startswith("openrouter/"):
        return "openrouter"
    elif model_lower.startswith("anthropic/"):
        return "anthropic"
    elif model_lower.startswith("openai/"):
        return "openai"
    elif model_lower.startswith("gemini/"):
        return "gemini"
    elif model_lower.startswith("deepseek/"):
        return "deepseek"
    else:
        # Try to infer from model name
        if "claude" in model_lower:
            return "anthropic"
        elif "gpt" in model_lower:
            return "openai"
        elif "gemini" in model_lower:
            return "gemini"
        elif "deepseek" in model_lower:
            return "deepseek"
        return "unknown"


def _get_openrouter_subprovider(model_id: str) -> str | None:
    """
    Parse the sub-provider from an OpenRouter model ID.

    Example: "openrouter/anthropic/claude-sonnet-4.5" -> "anthropic"
    """
    if not model_id.lower().startswith("openrouter/"):
        return None
    parts = model_id.split("/", 2)
    if len(parts) >= 2:
        return parts[1].lower()
    return None


def _get_token_budget(provider: str, effort: str) -> int | None:
    """Get token budget for a provider and effort level."""
    provider_map = EFFORT_TO_TOKENS.get(provider)
    if provider_map is None:
        return None
    return provider_map.get(effort)


def _extract_response_content(response: Any, debug: bool = False) -> str | None:
    """
    Extract text content from a LiteLLM response object.

    Handles various response formats from different providers.

    Args:
        response: The LiteLLM response object
        debug: Whether to print debug information

    Returns:
        The extracted text content or None if not found
    """
    if response is None:
        if debug:
            print("_extract_response_content: response is None")
        return None

    # Check if response has choices
    if not hasattr(response, "choices") or not response.choices:
        if debug:
            print(
                f"_extract_response_content: Response has no choices. Response attrs: {dir(response)}"
            )
        return None

    choice = response.choices[0]

    # Try to get message content
    if hasattr(choice, "message") and choice.message is not None:
        message = choice.message

        # Standard content field
        if hasattr(message, "content"):
            content = message.content
            # Handle case where content is an empty string (valid) vs None (invalid)
            if content is not None:
                # Content could be empty string which is technically valid
                if isinstance(content, str):
                    return content
                # Some providers might return content as a list
                elif isinstance(content, list):
                    # Try to extract text from content blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            text_parts.append(block["text"])
                        elif isinstance(block, str):
                            text_parts.append(block)
                    if text_parts:
                        return "".join(text_parts)
                    if debug:
                        print(
                            f"_extract_response_content: content is list but no text found: {content}"
                        )
            elif debug:
                print("_extract_response_content: message.content is None")

        # Some providers put content in different fields
        if hasattr(message, "text") and message.text is not None:
            return message.text

        # Check for tool calls or function calls that might have content
        if hasattr(message, "tool_calls") and message.tool_calls:
            if debug:
                print("_extract_response_content: Response has tool_calls instead of content")
            return None

    # Try delta for streaming responses
    if hasattr(choice, "delta") and choice.delta is not None:
        delta = choice.delta
        if hasattr(delta, "content") and delta.content is not None:
            return delta.content

    # Try text field directly on choice
    if hasattr(choice, "text") and choice.text is not None:
        return choice.text

    if debug:
        print(f"_extract_response_content: Could not extract content. Choice attrs: {dir(choice)}")
        if hasattr(choice, "message"):
            print(f"_extract_response_content: Message attrs: {dir(choice.message)}")

    return None


def check_and_read_key_file(file_path: str, target_key: str) -> Any:
    """
    Check and read a key from a JSON file.

    Checks if a file named `key.json` exists in the specified path, validates if
    it contains a Python dictionary, and retrieves the value associated with the
    specified key.

    Args:
        file_path (str): The path where the `key.json` file is expected to be located.
        target_key (str): The key in the dictionary whose value needs to be retrieved.

    Returns:
        Any: The value associated with the specified key if all checks pass,
            or -1 if any validation fails.

    Example:
        >>> # Assuming key.json contains {"api_key": "abc123"}
        >>> value = check_and_read_key_file("/path/to/file", "api_key")
        >>> if value != -1:
        ...     print(f"Found key: {value}")
        ... else:
        ...     print("Key not found or invalid file")
        ...
    """
    full_path = os.path.join(file_path, "key.json")

    if not os.path.exists(full_path):
        return -1

    try:
        with open(full_path, encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return -1

    if not isinstance(data, dict):
        return -1

    return data.get(target_key, -1)


def get_api_key(
    base_path: str, target_key: str, default_key: str = "type_your_key_here_or_use_key.json"
) -> str:
    """
    Retrieve the API key from a file or use a default value.

    Args:
        base_path (str): Base path to search for the key file.
        target_key (str): Target key to retrieve from the file.
        default_key (str): Default key to use if file reading fails.

    Returns:
        str: The API key string.

    Example:
        # Sample usage:
        key = get_api_key("../", "default_key")
        # Returns either the key from file or the default key
    """
    key = check_and_read_key_file(base_path, target_key)
    return default_key if key == -1 else key


def get_supported_models() -> list[str]:
    """Return a list of supported model names/aliases."""
    return list(MODEL_ALIASES.keys())


# =============================================================================
# LLM Base Class
# =============================================================================


class LLMBase:
    """
    Base class for all LLMs with shared functionality.

    This class provides common methods for interacting with language models,
    including debug printing, timeout handling, and test-based generation.

    Attributes:
        api_key (Optional[str]): The API key for authentication.
        model (str): The LLM model identifier being used.
        timeout (float): Maximum time limit for API calls.
        maximum_generation_attempts (int): Max attempts for generation with tests.
        maximum_timeout_attempts (int): Max retry attempts for timeouts/throttling.
        debug (bool): Flag indicating if debug mode is enabled.
        max_tokens (int): Maximum tokens for completion.
        reasoning_effort (Optional[str]): Reasoning effort level ("low", "medium", "high").

    Example:
        >>> base_llm = LLMBase(
        ...     api_key="your-key", model="claude-sonnet-4.5", debug=True
        ... )
        >>> print(base_llm.model)
        'claude-sonnet-4.5'
    """

    def __init__(
        self,
        api_key: str | None,
        model: str = "claude-sonnet-4.5",
        timeout: float = 60,
        maximum_generation_attempts: int = 3,
        maximum_timeout_attempts: int = 3,
        debug: bool = False,
        max_tokens: int = 8192,
        reasoning_effort: str | None = None,
    ) -> None:
        """
        Initialize the base LLM.

        Args:
            api_key (Optional[str]): The API key for authentication.
            model (str, optional): The LLM model identifier to use. Defaults to 'claude-sonnet-4.5'.
            timeout (float, optional): Maximum time limit for API calls. Defaults to 60.
            maximum_generation_attempts (int, optional): Max attempts for generation. Defaults to 3.
            maximum_timeout_attempts (int, optional): Max retry attempts. Defaults to 3.
            debug (bool, optional): Enable debug mode for detailed logging. Defaults to False.
            max_tokens (int, optional): Maximum tokens for completion. Defaults to 8192.
            reasoning_effort (Optional[str], optional): Reasoning effort level. Defaults to None.
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.maximum_generation_attempts = maximum_generation_attempts
        self.maximum_timeout_attempts = maximum_timeout_attempts
        self.debug = debug
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    @staticmethod
    def print_prompt(messages: list[dict[str, str]]) -> None:
        """
        Print each segment of a message prompt.

        Args:
            messages (list[dict[str, str]]): List of message segments to print.

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Hello!"},
            ... ]
            >>> LLMBase.print_prompt(messages)
        """
        for message in messages:
            if isinstance(message.get("content"), str):
                print(message["content"])

    def _print_debug_prompt(self, messages: list[dict[str, str]]) -> None:
        """Print prompt if debug mode is enabled."""
        if self.debug:
            print("---Prompt beginning marker---")
            self.print_prompt(messages)
            print("---Prompt ending marker---")

    def _print_debug_response(self, response_text: str) -> None:
        """Print response if debug mode is enabled."""
        if self.debug:
            print("---Response beginning marker---")
            print(response_text)
            print("---Response ending marker---")

    def ask_base(
        self,
        messages: list[dict[str, str]],
        ret_dict: dict[str, Any] | None = None,
    ) -> tuple[str | None, float]:
        """
        Base method to send a message to the LLM. Must be implemented by subclasses.

        Args:
            messages (list[dict[str, str]]): The messages to be sent.
            ret_dict (Optional[dict[str, Any]], optional): A dictionary to capture the
                method's return value. Defaults to None.

        Returns:
            tuple[Optional[str], float]: The response text and the cost,
                or (None, 0.0) if the request fails.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement ask_base()")

    def ask(
        self,
        messages: list[dict[str, str]],
        ret_dict: dict[str, Any] | None = None,
    ) -> tuple[str | None, float]:
        """
        Send a message to the LLM with retry functionality for handling timeouts.

        This method wraps ask_base() with timeout handling using retry_overtime_kill.

        Args:
            messages (list[dict[str, str]]): The messages to be sent.
            ret_dict (Optional[dict[str, Any]], optional): A dictionary to capture the
                method's return value. Defaults to None.

        Returns:
            tuple[Optional[str], float]: The response text and cost, or
                ("termination_signal", cost) if timeouts are exceeded.
        """

        def target_function(inner_ret_dict: dict[str, Any], *args: Any) -> None:
            result = self.ask_base(*args)
            inner_ret_dict["result"] = result

        exceeded, result = retry_overtime_kill(
            target_function=target_function,
            target_function_args=(messages,),
            time_limit=int(self.timeout),
            maximum_retry=self.maximum_timeout_attempts,
            ret=True,
        )

        response_text, cost = result.get("result", (None, 0.0))

        if not exceeded and response_text:
            return response_text, cost
        else:
            return "termination_signal", cost

    def ask_with_test(
        self,
        messages: list[dict[str, str]],
        tests: Callable[[str], Any],
    ) -> tuple[Any, float]:
        """
        Send a message with testing function and retry on test failures.

        This method is for simple testing functions with retry, such as testing general
        strings or Python objects (instead of multiple lines of Python code).
        Tests are also supposed to convert the response to the expected type.

        Args:
            messages (list[dict[str, str]]): The messages to send.
            tests (Callable[[str], Any]): A function to test and convert the response.

        Returns:
            tuple[Any, float]: The tested/converted response and the accumulated cost,
                or ("termination_signal", accumulated_cost) if all attempts fail.
        """
        cost_accumulation = 0.0

        def target_function(inner_ret_dict: dict[str, Any], *args: Any) -> None:
            response, cost = self.ask_base(*args)
            inner_ret_dict["response"] = response
            inner_ret_dict["cost"] = cost

        for trial_count in range(self.maximum_generation_attempts):
            print(
                f"Sequence generation under testing: attempt {trial_count + 1} "
                f"of {self.maximum_generation_attempts}"
            )

            exceeded, result = retry_overtime_kill(
                target_function=target_function,
                target_function_args=(messages,),
                time_limit=int(self.timeout),
                maximum_retry=self.maximum_timeout_attempts,
                ret=True,
            )

            if exceeded:
                print(f"Inquiry timed out for {self.maximum_timeout_attempts} times, retrying...")
                continue

            response: str | None = result.get("response")
            cost = result.get("cost", 0.0)
            cost_accumulation += cost

            # Check if response is None before calling tests
            if response is None:
                print("Response is None, retrying...")
                continue

            try:
                tested_response = tests(response)
                print("Test passed")
                return tested_response, cost_accumulation
            except Exception:
                print("Test failed, reason:")
                print(traceback.format_exc())
                print("Trying again")

        print("Maximum trial reached for sequence generation under testing")
        return "termination_signal", cost_accumulation


# =============================================================================
# Main LiteLLM Interface
# =============================================================================


class LiteLLM_interface(LLMBase):
    """
    A unified client for interacting with multiple LLM providers via LiteLLM.

    Supports:
    - claude-sonnet-4.5 (direct via Anthropic API or via OpenRouter)
    - gpt-5.2 (direct via OpenAI API or via OpenRouter)
    - gemini-pro-3.0 (direct via Google API only)
    - deepseek-v3.2 (direct via DeepSeek API or via OpenRouter, with optional reasoning)

    Routing Logic:
    - Gemini models always use direct Google API (not available on OpenRouter)
    - For other models: prefer OpenRouter if OPENROUTER_API_KEY is set
    - Fall back to direct API if OpenRouter key is not available

    Note on Gemini:
    - Gemini 3 Pro cannot disable thinking/reasoning - it's always active
    - If reasoning_effort is not specified, it defaults to "high"
    - LiteLLM maps reasoning_effort to Gemini's thinking_level parameter

    Attributes:
        canonical_model (str): The canonical model name (e.g., "claude-sonnet-4.5")
        resolved_model (str): The resolved LiteLLM model identifier
        use_openrouter (bool): Whether OpenRouter routing is being used
        temperature (float): Sampling temperature

    Example:
        >>> # With OpenRouter key set, will use OpenRouter
        >>> llm = LiteLLM_interface(model="claude-sonnet-4.5")
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response, cost = llm.ask(messages)

        >>> # Force direct API by not setting OPENROUTER_API_KEY
        >>> # or by setting specific provider key
        >>> import os
        >>> os.environ["ANTHROPIC_API_KEY"] = "your-key"
        >>> llm = LiteLLM_interface(model="claude-sonnet-4.5")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4.5",
        timeout: float = 120,
        maximum_generation_attempts: int = 3,
        maximum_timeout_attempts: int = 5,
        debug: bool = False,
        max_tokens: int = 8192,
        reasoning_effort: str | None = None,
        temperature: float = 0.7,
        force_direct: bool = False,
    ) -> None:
        """
        Initialize the LiteLLM client.

        Args:
            api_key (Optional[str]): API key. If provided, it will be set in the
                appropriate environment variable based on the model and routing.
            model (str, optional): Model identifier or alias. Defaults to 'claude-sonnet-4.5'.
            timeout (float, optional): Maximum time limit for API calls. Defaults to 120.
            maximum_generation_attempts (int, optional): Max attempts for generation. Defaults to 3.
            maximum_timeout_attempts (int, optional): Max retry attempts. Defaults to 5.
            debug (bool, optional): Enable debug mode for detailed logging. Defaults to False.
            max_tokens (int, optional): Maximum tokens for completion. Defaults to 8192.
            reasoning_effort (Optional[str], optional): Reasoning effort level
                ("low", "medium", "high"). For Gemini models, defaults to "high" if not specified.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            force_direct (bool, optional): Force direct API even if OpenRouter key is available.
                Defaults to False.
        """
        # Resolve the canonical model name first
        canonical_model = _resolve_to_canonical(model)

        # For Gemini models, thinking cannot be disabled, so default to "high" if not specified
        # This ensures sufficient tokens are allocated for both reasoning and response
        if _is_gemini_model(canonical_model) and reasoning_effort is None:
            reasoning_effort = "high"
            if debug:
                print("Gemini model detected: defaulting reasoning_effort to 'high'")

        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            maximum_generation_attempts=maximum_generation_attempts,
            maximum_timeout_attempts=maximum_timeout_attempts,
            debug=debug,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

        self.temperature = temperature
        self.force_direct = force_direct

        # Store canonical model
        self.canonical_model = canonical_model

        # Determine routing: OpenRouter vs Direct
        # Gemini always uses direct; others prefer OpenRouter if key available
        reasoning_enabled = reasoning_effort is not None and reasoning_effort in VALID_EFFORTS

        if _is_gemini_model(self.canonical_model):
            self.use_openrouter = False
        elif force_direct:
            self.use_openrouter = False
        else:
            self.use_openrouter = _has_openrouter_key()

        # Get the resolved LiteLLM model identifier
        self.resolved_model = _get_litellm_model_id(
            self.canonical_model,
            self.use_openrouter,
            reasoning_enabled,
        )

        # Set API key in environment if provided
        if api_key:
            self._set_api_key(api_key)

        if self.debug:
            print(f"Model routing: {self.canonical_model} -> {self.resolved_model}")
            print(f"Using OpenRouter: {self.use_openrouter}")

    def _set_api_key(self, api_key: str) -> None:
        """Set the API key in the appropriate environment variable."""
        if self.use_openrouter:
            os.environ["OPENROUTER_API_KEY"] = api_key
        else:
            # Set direct API key based on canonical model
            if self.canonical_model == "claude-sonnet-4.5":
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif self.canonical_model == "gpt-5.2":
                os.environ["OPENAI_API_KEY"] = api_key
            elif self.canonical_model == "gemini-pro-3.0":
                os.environ["GEMINI_API_KEY"] = api_key
            elif self.canonical_model == "deepseek-v3.2":
                os.environ["DEEPSEEK_API_KEY"] = api_key

    def _build_api_params(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Build the API parameters for the LiteLLM call.

        Handles provider-specific parameter requirements for both direct and
        OpenRouter routing.

        Args:
            messages (list[dict[str, str]]): The messages to be sent.

        Returns:
            dict[str, Any]: The API parameters.
        """
        litellm.modify_params = True

        api_params: dict[str, Any] = {
            "model": self.resolved_model,
            "messages": messages,
            "stream": False,
            "num_retries": self.maximum_timeout_attempts,
            "drop_params": True,
        }

        provider = _get_provider_from_model_id(self.resolved_model)

        # Handle max tokens based on provider
        if self.max_tokens is not None:
            if provider == "openai" or (
                provider == "openrouter"
                and _get_openrouter_subprovider(self.resolved_model) == "openai"
            ):
                # OpenAI GPT-5.x uses max_completion_tokens
                api_params["max_completion_tokens"] = self.max_tokens
            else:
                api_params["max_tokens"] = self.max_tokens

        # Handle temperature (may be overridden for reasoning modes)
        if self.temperature is not None:
            api_params["temperature"] = self.temperature

        # Handle Gemini-specific requirements
        # Gemini 3 Pro cannot disable thinking, so we always need reasoning params
        if provider == "gemini":
            # Ensure minimum max_tokens for Gemini to have room for both reasoning and response
            current_max = api_params.get("max_tokens", self.max_tokens)
            if current_max < GEMINI_MIN_MAX_TOKENS:
                api_params["max_tokens"] = GEMINI_MIN_MAX_TOKENS
                if self.debug:
                    print(
                        f"Gemini: increased max_tokens from {current_max} to {GEMINI_MIN_MAX_TOKENS}"
                    )

            # Always set reasoning_effort for Gemini (it's already defaulted to "high" in __init__)
            effort = self.reasoning_effort if self.reasoning_effort else "high"
            api_params["reasoning_effort"] = effort
            # Gemini reasoning doesn't support temperature parameter
            api_params.pop("temperature", None)
            # Don't drop params for Gemini - reasoning params are essential
            api_params["drop_params"] = False

            if self.debug:
                print(f"Gemini: set reasoning_effort to '{effort}'")

        # Handle reasoning/thinking for non-Gemini providers
        elif self.reasoning_effort and self.reasoning_effort in VALID_EFFORTS:
            self._add_reasoning_params(api_params, provider)

        return api_params

    def _add_reasoning_params(self, api_params: dict[str, Any], provider: str) -> None:
        """
        Add reasoning/thinking parameters based on provider.

        Args:
            api_params: The API parameters dict to modify
            provider: The provider identifier
        """
        effort = self.reasoning_effort

        if provider == "openrouter":
            # OpenRouter routing - handle based on sub-provider
            subprovider = _get_openrouter_subprovider(self.resolved_model)

            if subprovider == "openai":
                # GPT via OpenRouter: pass effort directly
                api_params["reasoning"] = {"effort": effort}
                api_params["temperature"] = 1.0  # Required for OpenAI reasoning
                api_params["drop_params"] = False

            elif subprovider == "anthropic":
                # Claude via OpenRouter: use reasoning parameter with max_tokens
                # OpenRouter doesn't support the 'thinking' parameter directly
                token_budget = _get_token_budget("anthropic", effort)
                if token_budget:
                    api_params["reasoning"] = {"max_tokens": token_budget}
                    # Increase max_tokens to accommodate reasoning tokens + response
                    current_max = api_params.get("max_tokens", 8192)
                    api_params["max_tokens"] = max(current_max, token_budget + 1000)
                    api_params["drop_params"] = False

            elif subprovider == "deepseek":
                # DeepSeek via OpenRouter: enable reasoning with token budget
                token_budget = _get_token_budget("deepseek", effort)
                if token_budget:
                    api_params["reasoning"] = {"enabled": True, "max_tokens": token_budget}
                    api_params["drop_params"] = False

        elif provider == "openai":
            # Direct OpenAI: pass reasoning_effort directly
            api_params["reasoning_effort"] = effort
            api_params["temperature"] = 1.0  # Required for OpenAI reasoning
            api_params["drop_params"] = False

        elif provider == "anthropic":
            # Direct Anthropic: use thinking parameter with budget_tokens
            token_budget = _get_token_budget("anthropic", effort)
            if token_budget:
                api_params["thinking"] = {"type": "enabled", "budget_tokens": token_budget}
                # Increase max_tokens to accommodate thinking tokens + response
                current_max = api_params.get("max_tokens", 8192)
                api_params["max_tokens"] = max(current_max, token_budget + 1000)
                api_params["drop_params"] = False

        elif provider == "deepseek":
            # Direct DeepSeek: model is already switched to deepseek-reasoner
            # in _get_litellm_model_id, so no additional params needed
            # The reasoning is implicit in the model choice
            pass

    def ask_base(
        self,
        messages: list[dict[str, str]],
        ret_dict: dict[str, Any] | None = None,
    ) -> tuple[str | None, float]:
        """
        Base method to send a message to the LLM via LiteLLM and capture the response.

        Args:
            messages (list[dict[str, str]]): The messages to be sent to the chat model.
            ret_dict (Optional[dict[str, Any]], optional): A dictionary to capture the
                method's return value. Defaults to None.

        Returns:
            tuple[Optional[str], float]: The chat model's response text and the cost,
                or (None, 0.0) if the request fails.
        """
        self._print_debug_prompt(messages)

        api_params = self._build_api_params(messages)

        if self.debug:
            # Print params without messages for clarity
            debug_params = {k: v for k, v in api_params.items() if k != "messages"}
            print(f"API params: {debug_params}")

        try:
            response = completion(**api_params)
        except Exception as e:
            if self.debug:
                print(f"LiteLLM API error: {e}")
                import traceback

                traceback.print_exc()
            if ret_dict is not None:
                ret_dict["result"] = (None, 0.0)
            return None, 0.0

        # Debug: print raw response structure
        if self.debug:
            print(f"Raw response object type: {type(response).__name__}")
            if hasattr(response, "choices"):
                print(f"Number of choices: {len(response.choices) if response.choices else 0}")
                if response.choices:
                    choice = response.choices[0]
                    print(f"Choice type: {type(choice).__name__}")
                    if hasattr(choice, "message"):
                        msg = choice.message
                        print(f"Message type: {type(msg).__name__ if msg else None}")
                        if msg:
                            print(
                                f"Message content type: {type(msg.content).__name__ if hasattr(msg, 'content') else 'N/A'}"
                            )
                            print(
                                f"Message content value: {repr(msg.content)[:200] if hasattr(msg, 'content') else 'N/A'}"
                            )

        # Extract response text using the helper function
        response_text = _extract_response_content(response, debug=self.debug)

        if response_text is None:
            if self.debug:
                print("Warning: Could not extract response content")
                # Try to dump more info about the response
                try:
                    import json

                    if hasattr(response, "model_dump"):
                        print(
                            f"Response dump: {json.dumps(response.model_dump(), indent=2, default=str)[:1000]}"
                        )
                except Exception as dump_err:
                    print(f"Could not dump response: {dump_err}")
            if ret_dict is not None:
                ret_dict["result"] = (None, 0.0)
            return None, 0.0

        self._print_debug_response(response_text)

        # Calculate cost from usage data
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

            if self.debug:
                print(
                    f"Tokens: {input_tokens} in + {output_tokens} out = "
                    f"{input_tokens + output_tokens} total"
                )

            # Calculate cost using the Calculator
            # Pass routing info for accurate pricing
            calculator = Calculator(
                self.canonical_model,
                use_openrouter=self.use_openrouter,
            )
            calculator.input_token_length = input_tokens
            calculator.output_token_length = output_tokens
            cost = calculator.calculate_cost_from_tokens()

            if self.debug:
                print(f"Cost: ${cost:.6f}")
        else:
            cost = 0.0

        if ret_dict is not None:
            ret_dict["result"] = (response_text, cost)

        return response_text, cost


# Backward compatibility alias
OpenAI_interface = LiteLLM_interface


# =============================================================================
# Utility Functions
# =============================================================================


def extract_code_base(raw_sequence: str, language: str = "python") -> str:
    """
    Extract code from markdown code blocks.

    Args:
        raw_sequence (str): Raw text containing code blocks.
        language (str, optional): Programming language identifier. Defaults to "python".

    Returns:
        str: Extracted code or original sequence if no code blocks found.
    """
    try:
        sub1 = f"```{language}"
        idx1 = raw_sequence.index(sub1)
    except ValueError:
        try:
            sub1 = f"``` {language}"
            idx1 = raw_sequence.index(sub1)
        except ValueError:
            try:
                sub1 = "```"
                idx1 = raw_sequence.index(sub1)
            except ValueError:
                return raw_sequence
    sub2 = "```"
    idx2 = raw_sequence.index(
        sub2,
        idx1 + 1,
    )
    extraction = raw_sequence[idx1 + len(sub1) + 1 : idx2]
    return extraction


def extract_code(raw_sequence: str, language: str = "python", mode: str = "code") -> Any:
    """
    Extract code from markdown and optionally evaluate as Python object.

    Args:
        raw_sequence (str): Raw text containing code blocks.
        language (str, optional): Programming language identifier. Defaults to "python".
        mode (str, optional): "code" for raw code, "python_object" to evaluate. Defaults to "code".

    Returns:
        str or Any: Extracted code string, or evaluated Python object if mode="python_object".
    """
    extraction = extract_code_base(raw_sequence, language)
    if mode == "code":
        return extraction
    if mode == "python_object":
        return ast.literal_eval(extraction)
