"""
LiteLLM-based LLM interface for unified API access across multiple providers.

Supported models:
- claude-sonnet-4.5 (via OpenRouter)
- gpt-5.2 (via OpenRouter)
- gemini-pro-3.0 (direct via LiteLLM)
- deepseek-v3.2 (via OpenRouter, both reasoning and non-reasoning modes)
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
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


# Supported model aliases and their LiteLLM model identifiers
MODEL_ALIASES: dict[str, str] = {
    # Claude models (via OpenRouter)
    "claude-sonnet-4.5": "openrouter/anthropic/claude-sonnet-4.5",
    "claude-4-5-sonnet": "openrouter/anthropic/claude-sonnet-4.5",
    "claude-sonnet-4-5": "openrouter/anthropic/claude-sonnet-4.5",
    # GPT models (via OpenRouter)
    "gpt-5.2": "openrouter/openai/gpt-5.2",
    "gpt-5": "openrouter/openai/gpt-5.2",
    # Gemini models (direct via LiteLLM)
    "gemini-pro-3.0": "gemini/gemini-3-pro-preview",
    "gemini-3-pro": "gemini/gemini-3-pro-preview",
    "gemini-pro-3": "gemini/gemini-3-pro-preview",
    # DeepSeek v3.2 (via OpenRouter)
    "deepseek-v3.2": "openrouter/deepseek/deepseek-v3.2",
    "deepseek-3.2": "openrouter/deepseek/deepseek-v3.2",
    "deepseek": "openrouter/deepseek/deepseek-v3.2",
}

# Valid reasoning effort levels
VALID_EFFORTS = ("low", "medium", "high")

# Reasoning effort to token budget mapping for providers that need it
EFFORT_TO_TOKENS: dict[str, dict[str, int]] = {
    "anthropic": {
        "low": 1024,
        "medium": 8192,
        "high": 30000,
    },
    "deepseek": {
        "low": 1024,
        "medium": 8192,
        "high": 30000,
    },
}


def _setup_openrouter_env() -> None:
    """Set up OpenRouter environment variables if not already configured."""
    if "OPENROUTER_API_BASE" not in os.environ:
        os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"

    if "OPENROUTER_API_KEY" not in os.environ:
        warnings.warn(
            "OPENROUTER_API_KEY environment variable is not set. OpenRouter queries will fail.",
            UserWarning,
        )


_setup_openrouter_env()


def _resolve_model_name(model: str) -> str:
    """Resolve a model alias to its full LiteLLM model identifier."""
    return MODEL_ALIASES.get(model, model)


def _is_openrouter_model(model_name: str) -> bool:
    """Check if model uses OpenRouter."""
    return (model_name or "").lower().startswith("openrouter/")


def _is_gemini_direct_model(model_name: str) -> bool:
    """Check if model uses direct Gemini API (not OpenRouter)."""
    return (model_name or "").lower().startswith("gemini/")


def _is_openai_model(model_name: str) -> bool:
    """Check if model is an OpenAI model (for max_completion_tokens handling)."""
    lower = model_name.lower()
    return "gpt-5" in lower or "openai/gpt" in lower


def _parse_openrouter_provider(model_name: str) -> str | None:
    """Parse the OpenRouter provider from: openrouter/<provider>/<model-id>"""
    name = (model_name or "").strip()
    if not name.lower().startswith("openrouter/"):
        return None
    parts = name.split("/", 2)
    if len(parts) < 3:
        return None
    return parts[1].lower()


def _get_token_budget(provider: str, effort: str) -> int | None:
    """Get token budget for a provider and effort level."""
    provider_map = EFFORT_TO_TOKENS.get(provider)
    if provider_map is None:
        return None
    return provider_map.get(effort)


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


class LiteLLM_interface(LLMBase):
    """
    A unified client for interacting with multiple LLM providers via LiteLLM.

    Supports:
    - claude-sonnet-4.5 (via OpenRouter)
    - gpt-5.2 (via OpenRouter)
    - gemini-pro-3.0 (direct via LiteLLM)
    - deepseek-v3.2 (via OpenRouter, with optional reasoning)

    Attributes:
        resolved_model (str): The resolved LiteLLM model identifier.

    Example:
        >>> llm = LiteLLM_interface(model="claude-sonnet-4.5")
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response, cost = llm.ask(messages)
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
    ) -> None:
        """
        Initialize the LiteLLM client.

        Args:
            api_key (Optional[str]): API key (used for setting environment variables if needed).
            model (str, optional): Model identifier or alias. Defaults to 'claude-sonnet-4.5'.
            timeout (float, optional): Maximum time limit for API calls. Defaults to 120.
            maximum_generation_attempts (int, optional): Max attempts for generation. Defaults to 3.
            maximum_timeout_attempts (int, optional): Max retry attempts. Defaults to 5.
            debug (bool, optional): Enable debug mode for detailed logging. Defaults to False.
            max_tokens (int, optional): Maximum tokens for completion. Defaults to 8192.
            reasoning_effort (Optional[str], optional): Reasoning effort level. Defaults to None.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
        """
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

        self.resolved_model = _resolve_model_name(model)
        self.temperature = temperature

        # Set API key in environment if provided
        if api_key:
            if _is_openrouter_model(self.resolved_model):
                os.environ["OPENROUTER_API_KEY"] = api_key
            elif _is_gemini_direct_model(self.resolved_model):
                os.environ["GEMINI_API_KEY"] = api_key

    def _build_api_params(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Build the API parameters for the LiteLLM call.

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

        # Handle max tokens
        if self.max_tokens is not None:
            if _is_openai_model(self.resolved_model):
                api_params["max_completion_tokens"] = self.max_tokens
            else:
                api_params["max_tokens"] = self.max_tokens

        # Handle temperature (may be overridden for reasoning modes)
        if self.temperature is not None:
            api_params["temperature"] = self.temperature

        # Handle reasoning_effort based on model routing
        is_gemini_direct = _is_gemini_direct_model(self.resolved_model)
        is_openrouter = _is_openrouter_model(self.resolved_model)
        or_provider = _parse_openrouter_provider(self.resolved_model) if is_openrouter else None

        if self.reasoning_effort and self.reasoning_effort in VALID_EFFORTS:
            if is_gemini_direct:
                # Gemini direct: pass reasoning_effort as parameter
                api_params["reasoning_effort"] = self.reasoning_effort
                # Gemini reasoning doesn't support temperature
                api_params.pop("temperature", None)
                api_params["drop_params"] = False

            elif is_openrouter:
                if or_provider == "openai":
                    # GPT via OpenRouter: pass effort directly
                    api_params["reasoning"] = {"effort": self.reasoning_effort}
                    api_params["temperature"] = 1.0  # Required for OpenAI reasoning
                    api_params["drop_params"] = False

                elif or_provider == "anthropic":
                    # Claude via OpenRouter: convert effort to token budget
                    token_budget = _get_token_budget("anthropic", self.reasoning_effort)
                    if token_budget:
                        api_params["reasoning"] = {"max_tokens": token_budget}
                        api_params["drop_params"] = False

                elif or_provider == "deepseek":
                    # DeepSeek via OpenRouter: enable reasoning with token budget
                    token_budget = _get_token_budget("deepseek", self.reasoning_effort)
                    if token_budget:
                        api_params["reasoning"] = {"enabled": True, "max_tokens": token_budget}
                        api_params["drop_params"] = False

        return api_params

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

        try:
            response = completion(**api_params)
        except Exception as e:
            if self.debug:
                print(f"LiteLLM API error: {e}")
            if ret_dict is not None:
                ret_dict["result"] = (None, 0.0)
            return None, 0.0

        # Extract response text
        response_text = None
        if response.choices and response.choices[0].message:
            response_text = response.choices[0].message.content

        if response_text is None:
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
            calculator = Calculator(self.model)
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
