from __future__ import annotations

import ast
import json
import os
import random
import time
import traceback
from typing import Any
from typing import Callable
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessageParam

from LLM_utils.cost import Calculator
from LLM_utils.fault_tolerance import retry_overtime_kill


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
        with open(full_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, IOError):
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

    Example:
        >>> base_llm = LLMBase(api_key="your-key", model="gpt-4", debug=True)
        >>> print(base_llm.model)
        'gpt-4'
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gpt-4-mini",
        timeout: float = 60,
        maximum_generation_attempts: int = 3,
        maximum_timeout_attempts: int = 3,
        debug: bool = False,
    ) -> None:
        """
        Initialize the base LLM.

        Args:
            api_key (Optional[str]): The API key for authentication.
            model (str, optional): The LLM model identifier to use. Defaults to 'gpt-4-mini'.
            timeout (float, optional): Maximum time limit for API calls. Defaults to 60.
            maximum_generation_attempts (int, optional): Max attempts for generation. Defaults to 3.
            maximum_timeout_attempts (int, optional): Max retry attempts. Defaults to 3.
            debug (bool, optional): Enable debug mode for detailed logging. Defaults to False.
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.maximum_generation_attempts = maximum_generation_attempts
        self.maximum_timeout_attempts = maximum_timeout_attempts
        self.debug = debug

    @staticmethod
    def print_prompt(messages: list[ChatCompletionMessageParam]) -> None:
        """
        Print each segment of a message prompt.

        Args:
            messages (list[ChatCompletionMessageParam]): List of message segments to print.

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Hello!"},
            ... ]
            >>> LLMBase.print_prompt(messages)
        """
        for message in messages:
            if isinstance(message["content"], str):
                print(message["content"])

    def _print_debug_prompt(self, messages: list[ChatCompletionMessageParam]) -> None:
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
        messages: list[ChatCompletionMessageParam],
        ret_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[str], float]:
        """
        Base method to send a message to the LLM. Must be implemented by subclasses.

        Args:
            messages (list[ChatCompletionMessageParam]): The messages to be sent.
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
        messages: list[ChatCompletionMessageParam],
        ret_dict: Optional[dict[str, any]] = None,
    ) -> tuple[Optional[str], float]:
        """
        Send a message to the LLM with retry functionality for handling timeouts.

        This method wraps ask_base() with timeout handling using retry_overtime_kill.

        Args:
            messages (list[ChatCompletionMessageParam]): The messages to be sent.
            ret_dict (Optional[dict[str, any]], optional): A dictionary to capture the
                method's return value. Defaults to None.

        Returns:
            tuple[Optional[str], float]: The response text and cost, or
                ("termination_signal", cost) if timeouts are exceeded.
        """

        def target_function(ret_dict: dict[str, Any], *args: Any) -> None:
            self.ask_base(*args, ret_dict=ret_dict)

        exceeded, result = retry_overtime_kill(
            target_function=target_function,
            target_function_args=(messages,),
            time_limit=self.timeout,
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
        messages: list[ChatCompletionMessageParam],
        tests: Callable[[str], str],
    ) -> tuple[Any, float]:
        """
        Send a message with testing function and retry on test failures.

        This method is for simple testing functions with retry, such as testing general
        strings or Python objects (instead of multiple lines of Python code).
        Tests are also supposed to convert the response to the expected type.

        Args:
            messages (list[ChatCompletionMessageParam]): The messages to send.
            tests (Callable[[str], str]): A function to test and convert the response.

        Returns:
            tuple[Any, float]: The tested/converted response and the accumulated cost,
                or ("termination_signal", accumulated_cost) if all attempts fail.
        """
        cost_accumulation = 0.0

        def target_function(ret_dict: dict[str, Any], *args: Any) -> None:
            response, cost = self.ask_base(*args, ret_dict=ret_dict)
            ret_dict["response"] = response
            ret_dict["cost"] = cost

        for trial_count in range(self.maximum_generation_attempts):
            print(
                f"Sequence generation under testing: attempt {trial_count + 1} "
                f"of {self.maximum_generation_attempts}"
            )

            exceeded, result = retry_overtime_kill(
                target_function=target_function,
                target_function_args=(messages,),
                time_limit=self.timeout,
                maximum_retry=self.maximum_timeout_attempts,
                ret=True,
            )

            if exceeded:
                print(f"Inquiry timed out for {self.maximum_timeout_attempts} times, retrying...")
                continue

            response = result.get("response")
            cost = result.get("cost", 0.0)
            cost_accumulation += cost

            try:
                response = tests(response)
                print("Test passed")
                return response, cost_accumulation
            except Exception as e:
                print("Test failed, reason:")
                print(traceback.format_exc())
                print("Trying again")

        print("Maximum trial reached for sequence generation under testing")
        return "termination_signal", cost_accumulation


class OpenAI_interface(LLMBase):
    """
    A client for interacting with OpenAI's interface.

    This class provides methods to communicate with models through OpenAI's API,
    with built-in retry functionality for handling timeouts.

    Attributes:
        client (OpenAI): The OpenAI client instance for making API calls.

    Example:
        >>> gpt = OpenAI_interface(api_key="your-key", model="gpt-4")
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response, cost = gpt.ask(messages)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-mini",
        timeout: float = 60,
        maximum_generation_attempts: int = 3,
        maximum_timeout_attempts: int = 3,
        debug: bool = False,
    ) -> None:
        """
        Initialize the OpenAI client.

        Args:
            api_key (str): The OpenAI API key for authentication.
            model (str, optional): The model identifier to use. Defaults to 'gpt-4-mini'.
            timeout (float, optional): Maximum time limit for API calls. Defaults to 60.
            maximum_generation_attempts (int, optional): Max attempts for generation. Defaults to 3.
            maximum_timeout_attempts (int, optional): Max retry attempts. Defaults to 3.
            debug (bool, optional): Enable debug mode for detailed logging. Defaults to False.
        """
        super().__init__(
            api_key, model, timeout, maximum_generation_attempts, maximum_timeout_attempts, debug
        )

        if self.model == "deepseek-chat" or self.model == "deepseek-reasoner":
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            self.client = OpenAI(api_key=api_key)

    def ask_base(
        self,
        messages: list[ChatCompletionMessageParam],
        ret_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[str], float]:
        """
        Base method to send a message to the chat model and capture the response.

        Args:
            messages (list[ChatCompletionMessageParam]): The messages to be sent to the chat model.
            ret_dict (Optional[dict[str, Any]], optional): A dictionary to capture the
                method's return value. Defaults to None.

        Returns:
            tuple[Optional[str], float]: The chat model's response text and the cost,
                or (None, 0.0) if the request fails.
        """
        self._print_debug_prompt(messages)

        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        if response.choices[0].message.content is None:
            return None, 0.0

        response_text: str = response.choices[0].message.content
        self._print_debug_response(response_text)

        # Calculate cost
        calculator_instance = Calculator(self.model, messages, response_text)

        if self.model == "deepseek-chat" or self.model == "deepseek-reasoner":
            cost = calculator_instance.calculate_cost_DeepSeek()
        else:
            cost = calculator_instance.calculate_cost_GPT()

        if ret_dict is not None:
            ret_dict["result"] = (response_text, cost)

        return response_text, cost


class Anthropic_Bedrock_interface(LLMBase):
    """
    A client for interacting with Claude models through AWS Bedrock.

    Provides methods to communicate with Anthropic models through AWS Bedrock API,
    with built-in retry functionality for handling throttling and timeouts.
    Supports Extended Thinking mode for complex reasoning tasks.

    Attributes:
        client: The Bedrock runtime client.
        bedrock_model_id (str): The full Bedrock model identifier.
        region_name (str): AWS region for Bedrock.
        enable_thinking (bool): Whether extended thinking is enabled.
        thinking_budget_tokens (int): Token budget for thinking (>= 1024).
        max_tokens (int): Maximum output tokens.
        temperature (float): Temperature (not used when thinking is enabled).

    Example:
        >>> # Without thinking
        >>> claude = Anthropic_Bedrock_interface(model="claude-sonnet-4.5")
        >>> # With thinking enabled
        >>> claude_thinking = Anthropic_Bedrock_interface(
        ...     model="claude-sonnet-4.5",
        ...     thinking_budget_tokens=4000,
        ...     max_tokens=12000
        ... )
    """

    # Model ID mapping - ALL models now use inference profiles for Bedrock compatibility
    MODEL_IDS = {
        # Map friendly names to INFERENCE PROFILE IDs (not direct model IDs)
        # Using "global." prefix for best availability (auto-routing across regions)
        # Haiku family
        "claude-haiku-3": "anthropic.claude-3-haiku-20240307-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "claude-haiku-3.5": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-3-5-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-4-5-haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        # Sonnet family - ALL require inference profiles
        "claude-sonnet-3.5": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-sonnet-3.7": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "claude-3-7-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "claude-sonnet-4": "global.anthropic.claude-sonnet-4-20250514-v1:0",
        "claude-4-sonnet": "global.anthropic.claude-sonnet-4-20250514-v1:0",
        "claude-sonnet-4.5": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-4-5-sonnet": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        # Opus family
        "claude-opus-3": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-opus-4": "us.anthropic.claude-opus-4-20250514-v1:0",
        "claude-4-opus": "us.anthropic.claude-opus-4-20250514-v1:0",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4.5",
        region_name: str = "us-east-1",
        timeout: float = 300,
        maximum_generation_attempts: int = 3,
        maximum_timeout_attempts: int = 10,
        debug: bool = False,
        *,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 0,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize the Anthropic Bedrock client with optional Extended Thinking.

        Args:
            api_key (Optional[str]): Not used (Bedrock uses AWS credentials).
            model (str, optional): Model name. Defaults to 'claude-sonnet-4.5'.
            region_name (str, optional): AWS region. Defaults to 'us-east-1'.
            timeout (float, optional): Max time for API calls. Defaults to 300.
            maximum_generation_attempts (int, optional): Max attempts. Defaults to 3.
            maximum_timeout_attempts (int, optional): Max retries. Defaults to 10.
            debug (bool, optional): Enable debug logging. Defaults to False.
            enable_thinking (bool, optional): Enable extended thinking. Defaults to False.
            thinking_budget_tokens (int, optional): Thinking token budget (>= 1024).
                Defaults to 0 (disabled).
            max_tokens (int, optional): Max output tokens. Defaults to 8192.
            temperature (float, optional): Temperature (unused with thinking). Defaults to 0.7.
        """
        super().__init__(
            api_key, model, timeout, maximum_generation_attempts, maximum_timeout_attempts, debug
        )

        self.region_name = region_name

        sdk_config = Config(
            connect_timeout=5,
            read_timeout=int(timeout),
            retries={"total_max_attempts": 8, "mode": "adaptive"},
        )
        self.client = boto3.client("bedrock-runtime", region_name=region_name, config=sdk_config)

        self.bedrock_model_id = self.MODEL_IDS.get(model, model)

        self.enable_thinking = enable_thinking or (thinking_budget_tokens >= 1024)
        self.thinking_budget_tokens = thinking_budget_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _convert_messages(
        self, messages: list[ChatCompletionMessageParam]
    ) -> tuple[Optional[str], list[dict]]:
        """
        Convert OpenAI-style messages to Bedrock format.

        Args:
            messages (list[ChatCompletionMessageParam]): OpenAI-formatted messages.

        Returns:
            tuple[Optional[str], list[dict]]: (system_message, converted_messages)
        """
        system_message = None
        converted_messages = []

        for m in messages:
            if m["role"] == "system":
                system_message = m["content"]
                continue
            converted_messages.append({"role": m["role"], "content": [{"text": m["content"]}]})

        return system_message, converted_messages

    def ask_base(
        self,
        messages: list[ChatCompletionMessageParam],
        ret_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[str], float]:
        """
        Base method to send a message to Claude via Bedrock with optional Extended Thinking.

        NOTE: We calculate cost directly from Bedrock's response usage data.
        We do NOT use the Calculator class for Bedrock token counting.
        """
        self._print_debug_prompt(messages)

        system_message, converted_messages = self._convert_messages(messages)

        inf_cfg: dict[str, Any] = {"maxTokens": int(self.max_tokens)}

        addl_fields: Optional[dict] = None
        if self.enable_thinking:
            budget = max(1024, int(self.thinking_budget_tokens))
            if budget >= inf_cfg["maxTokens"]:
                inf_cfg["maxTokens"] = budget + 2048
            addl_fields = {"thinking": {"type": "enabled", "budget_tokens": budget}}

            if self.debug:
                print(
                    f"Extended thinking enabled: budget={budget}, max_tokens={inf_cfg['maxTokens']}"
                )
        else:
            inf_cfg["temperature"] = float(self.temperature)

        params: dict[str, Any] = {
            "modelId": self.bedrock_model_id,
            "messages": converted_messages,
            "inferenceConfig": inf_cfg,
        }

        if system_message is not None:
            params["system"] = [{"text": system_message}]

        if addl_fields is not None:
            params["additionalModelRequestFields"] = addl_fields

        for attempt in range(1, self.maximum_timeout_attempts + 1):
            try:
                response = self.client.converse(**params)

                blocks = response.get("output", {}).get("message", {}).get("content", []) or []
                texts: list[str] = []
                thinking_texts: list[str] = []

                for b in blocks:
                    if isinstance(b, dict):
                        if "text" in b and isinstance(b["text"], str):
                            texts.append(b["text"])
                        elif "thinking" in b and isinstance(b["thinking"], str):
                            thinking_texts.append(b["thinking"])

                response_text = "\n".join(texts).strip() if texts else ""

                if self.debug and thinking_texts:
                    print(f"--- Thinking content ({len(thinking_texts)} blocks) ---")
                    for i, think in enumerate(thinking_texts, 1):
                        print(f"Thinking block {i}: {think[:200]}...")

                self._print_debug_response(response_text)

                usage = response.get("usage", {}) or {}
                input_tokens = int(usage.get("inputTokens", 0))
                output_tokens = int(usage.get("outputTokens", 0))

                if self.debug:
                    print(
                        f"Bedrock tokens: {input_tokens} in + {output_tokens} out = "
                        f"{input_tokens + output_tokens} total"
                    )

                input_price = Calculator.Anthropic_input_pricing.get(self.model, 3.0)
                output_price = Calculator.Anthropic_output_pricing.get(self.model, 15.0)
                cost = (input_tokens * input_price + output_tokens * output_price) / 1e6

                if self.debug:
                    print(f"Cost: ${cost:.6f}")

                if ret_dict is not None:
                    ret_dict["result"] = (response_text, cost)

                return response_text, cost

            except ClientError as exc:
                error_code = exc.response["Error"]["Code"]

                if error_code != "ThrottlingException":
                    print(f"Bedrock API error: {error_code}")
                    raise

                if attempt < self.maximum_timeout_attempts:
                    sleep_for = min(1.0 * 2 ** (attempt - 1) + random.uniform(0, 0.5), 20)
                    print(
                        f"Throttled, retrying in {sleep_for:.2f}s "
                        f"(attempt {attempt}/{self.maximum_timeout_attempts})"
                    )
                    time.sleep(sleep_for)
                else:
                    raise RuntimeError(
                        f"Exceeded {self.maximum_timeout_attempts} attempts due to throttling."
                    )

        return None, 0.0


def extract_code_base(raw_sequence, language="python"):
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
    except:
        try:
            sub1 = f"``` {language}"
            idx1 = raw_sequence.index(sub1)
        except:
            try:
                sub1 = "```"
                idx1 = raw_sequence.index(sub1)
            except:
                return raw_sequence
    sub2 = "```"
    idx2 = raw_sequence.index(
        sub2,
        idx1 + 1,
    )
    extraction = raw_sequence[idx1 + len(sub1) + 1 : idx2]
    return extraction


def extract_code(raw_sequence, language="python", mode="code"):
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
