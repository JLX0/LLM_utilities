from __future__ import annotations

import json
import os
from typing import Any
from typing import Optional
from typing import Callable
import ast

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessageParam
import traceback

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

def get_api_key(base_path: str, target_key: str, default_key: str = "type_your_key_here_or_use_key.json") -> str:
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
    Base class for all LLMs.

    This class serves as the foundation for different language model implementations,
    providing common functionality and attributes.

    Attributes:
        api_key (Optional[str]): The API key for authentication.
        model (str): The LLM model identifier being used.
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
            debug (bool, optional): Enable debug mode for detailed logging. Defaults to False.
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.maximum_generation_attempts = maximum_generation_attempts
        self.maximum_timeout_attempts = maximum_timeout_attempts
        self.debug = debug


class OpenAI_interface(LLMBase):
    """
    A client for interacting with OpenAI's interface

    This class provides methods to communicate with models through OpenAI's API,
    with built-in retry functionality for handling timeouts.

    Attributes:
        timeout (int): Maximum time limit for API calls.
        maximum_retry (int): Maximum number of retry attempts.
        client (OpenAI): The OpenAI client instance for making API calls.

    Example:
        >>> gpt = OpenAI(api_key="your-key", model="gpt-4")
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = gpt.ask(messages)
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
            debug (bool, optional): Enable debug mode for detailed logging. Defaults to False.
        """
        super().__init__(api_key, model, timeout, maximum_generation_attempts, maximum_timeout_attempts, debug)

        if self.model =="deepseek-chat":
            self.client = OpenAI(api_key=api_key , base_url="https://api.deepseek.com")
        else:
            self.client = OpenAI(api_key=api_key)

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
            >>> OpenAI_interface.print_prompt(messages)
        """
        for message in messages:
            if isinstance(message["content"], str):
                print(message["content"])

    def ask_base(
            self ,
            messages: list[ChatCompletionMessageParam] ,
            ret_dict: Optional[dict[str , Any]] = None ,
            ) -> tuple[Optional[str] , float] :
        """
        Base method to send a message to the chat model and capture the response.

        Args:
            messages (list[ChatCompletionMessageParam]): The messages to be sent to the chat model.
            ret_dict (Optional[dict[str, str]], optional): A dictionary to capture the
                method's return value. Defaults to None.

        Returns:
            tuple[Optional[str], float]: The chat model's response text and the cost,
                or (None, 0.0) if the request fails.
        """
        if self.debug :
            print("---Prompt beginning marker---")
            self.print_prompt(messages)
            print("---Prompt ending marker---")

        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model ,
            messages=messages ,
            )

        if response.choices[0].message.content is None :
            return None , 0.0

        response_text: str = response.choices[0].message.content

        if self.debug :
            print("---Response beginning marker---")
            print(response_text)
            print("---Response ending marker---")

        calculator_instance = Calculator(self.model , messages , response_text)

        if self.model == "deepseek-chat" :
            cost = calculator_instance.calculate_cost_DeepSeek()
        else :
            cost = calculator_instance.calculate_cost_GPT()

        if ret_dict is not None :
            ret_dict["result"] = (response_text , cost)

        return response_text , cost

    def ask(
            self ,
            messages: list[ChatCompletionMessageParam] ,
            ret_dict: Optional[dict[str , any]] = None ,
            ) -> tuple[Optional[str] , float] :
        """
        Send a message to the chat model with retry functionality for handling timeouts.

        Args:
            messages (list[ChatCompletionMessageParam]): The messages to be sent to the chat model.
            ret_dict (Optional[dict[str, str]], optional): A dictionary to capture the
                method's return value. Defaults to None.

        Returns:
            Optional[str]: The chat model's response text, or None if the request fails.
        """

        def target_function(ret_dict: dict[str , Any] , *args: Any) -> None :
            self.ask_base(*args , ret_dict=ret_dict)

        exceeded , result = retry_overtime_kill(
            target_function=target_function ,
            target_function_args=(messages ,) ,
            time_limit=self.timeout ,
            maximum_retry=self.maximum_timeout_attempts ,
            ret=True ,
            )

        response_text , cost=result.get("result")

        if not exceeded :
            return response_text,cost
        else :
            return "termination_signal" , cost

    def ask_with_test(
            self ,
            messages: list[ChatCompletionMessageParam] ,
            tests: Callable[[str] , str] ,
            ) -> tuple[Any , float] :
        """
        This method is only for simple testing functions with retry, such as testing general
        strings or Python objects (instead of multiple lines of Python code).

        Tests are also supposed to convert the response to the expected type.

        Args:
            messages: The messages to be sent to the chat model.
            tests: A function to test the response from the chat model.

        Returns:
            tuple[Any, float]: The tested response and the accumulated cost.
        """
        cost_accumulation = 0.0

        def target_function(ret_dict: dict[str , Any] , *args: Any) -> None :
            response , cost = self.ask_base(*args , ret_dict=ret_dict)
            ret_dict["response"] = response
            ret_dict["cost"] = cost

        for trial_count in range(self.maximum_generation_attempts) :
            print(
                f"Sequence generation under testing: attempt {trial_count + 1} of {self.maximum_generation_attempts}")
            exceeded , result = retry_overtime_kill(
                target_function=target_function ,
                target_function_args=(messages ,) ,
                time_limit=self.timeout ,
                maximum_retry=self.maximum_timeout_attempts ,
                # This retry is for timeout, instead of tests
                ret=True ,
                )

            if exceeded :
                print(f"Inquiry timed out for {self.maximum_timeout_attempts} times, retrying...")
                continue

            response = result.get("response")
            cost = result.get("cost" , 0.0)
            cost_accumulation += cost

            try :
                response = tests(response)
                print("Test passed")
                return response , cost_accumulation
            except Exception as e :
                print("Test failed, reason:")
                print(traceback.format_exc())
                print("Trying again")

        print("Maximum trial reached for sequence generation under testing")
        return "termination_signal" , cost_accumulation


def extract_code_base(raw_sequence, language="python"):
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
    idx2 = raw_sequence.index(sub2 , idx1 + 1 , )
    extraction = raw_sequence[idx1 + len(sub1) + 1 : idx2]
    return extraction

def extract_code(raw_sequence, language="python", mode="code"):
    extraction= extract_code_base(raw_sequence, language)
    if mode == "code":
        return extraction
    if mode == "python_object":
        return ast.literal_eval(extraction)