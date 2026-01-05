"""
Cost calculation utilities for LLM API usage.

Supports both direct API and OpenRouter pricing:
- claude-sonnet-4.5
- gpt-5.2
- gemini-pro-3.0
- deepseek-v3.2
"""

from __future__ import annotations

import tiktoken

from LLM_utils.prompter import PromptBase


class Calculator:
    """
    Calculator for estimating and tracking LLM API costs.

    Supports multiple providers with both direct and OpenRouter pricing:
    - OpenAI (GPT models)
    - Anthropic (Claude models)
    - Google (Gemini models)
    - DeepSeek models

    Attributes:
        model (str): The canonical model identifier.
        use_openrouter (bool): Whether OpenRouter routing is being used.
        formatted_input_sequence: The formatted input messages.
        output_sequence_string: The output text string.
        input_token_length (int): Number of input tokens.
        output_token_length (int): Number of output tokens.
    """

    # ==========================================================================
    # Direct API Pricing (per 1M tokens in USD)
    # ==========================================================================

    # OpenAI Direct Pricing
    OPENAI_DIRECT_INPUT_PRICING = {
        "gpt-5.2": 1.75,
        "gpt-5": 1.25,
    }

    OPENAI_DIRECT_OUTPUT_PRICING = {
        "gpt-5.2": 14.00,
        "gpt-5": 10.00,
    }

    # Anthropic Direct Pricing
    ANTHROPIC_DIRECT_INPUT_PRICING = {
        "claude-sonnet-4.5": 3.00,
        "claude-4-5-sonnet": 3.00,
        "claude-sonnet-4-5": 3.00,
    }

    ANTHROPIC_DIRECT_OUTPUT_PRICING = {
        "claude-sonnet-4.5": 15.00,
        "claude-4-5-sonnet": 15.00,
        "claude-sonnet-4-5": 15.00,
    }

    # Gemini Direct Pricing (always direct, no OpenRouter)
    GEMINI_DIRECT_INPUT_PRICING = {
        "gemini-pro-3.0": 2.00,
        "gemini-3-pro": 2.00,
        "gemini-pro-3": 2.00,
    }

    GEMINI_DIRECT_OUTPUT_PRICING = {
        "gemini-pro-3.0": 12.00,
        "gemini-3-pro": 12.00,
        "gemini-pro-3": 12.00,
    }

    # DeepSeek Direct Pricing
    DEEPSEEK_DIRECT_INPUT_PRICING = {
        "deepseek-v3.2": 0.28,
        "deepseek-3.2": 0.28,
        "deepseek": 0.28,
    }

    DEEPSEEK_DIRECT_OUTPUT_PRICING = {
        "deepseek-v3.2": 0.42,
        "deepseek-3.2": 0.42,
        "deepseek": 0.42,
    }

    # ==========================================================================
    # OpenRouter Pricing (per 1M tokens in USD)
    # Note: OpenRouter prices are model-specific and may differ from (or match) direct provider rates.
    # ==========================================================================

    # GPT via OpenRouter
    OPENROUTER_GPT_INPUT_PRICING = {
        "gpt-5.2": 1.75,
        "gpt-5": 1.25,
    }

    OPENROUTER_GPT_OUTPUT_PRICING = {
        "gpt-5.2": 14.00,
        "gpt-5": 10.00,
    }

    # Anthropic via OpenRouter
    OPENROUTER_ANTHROPIC_INPUT_PRICING = {
        "claude-sonnet-4.5": 3.00,
        "claude-4-5-sonnet": 3.00,
        "claude-sonnet-4-5": 3.00,
    }

    OPENROUTER_ANTHROPIC_OUTPUT_PRICING = {
        "claude-sonnet-4.5": 15.00,
        "claude-4-5-sonnet": 15.00,
        "claude-sonnet-4-5": 15.00,
    }

    # DeepSeek via OpenRouter
    OPENROUTER_DEEPSEEK_INPUT_PRICING = {
        "deepseek-v3.2": 0.25,
        "deepseek-3.2": 0.25,
        "deepseek": 0.25,
    }

    OPENROUTER_DEEPSEEK_OUTPUT_PRICING = {
        "deepseek-v3.2": 0.38,
        "deepseek-3.2": 0.38,
        "deepseek": 0.38,
    }

    def __init__(
        self,
        model: str,
        formatted_input_sequence: list[dict[str, str]] | None = None,
        output_sequence_string: str | None = None,
        use_openrouter: bool = False,
    ):
        """
        Initialize the Calculator.

        Args:
            model (str): The canonical model identifier.
            formatted_input_sequence: Optional formatted input messages.
            output_sequence_string: Optional output text string.
            use_openrouter (bool): Whether OpenRouter routing is being used.
        """
        self.model = model
        self.formatted_input_sequence = formatted_input_sequence
        self.output_sequence_string = output_sequence_string
        self.use_openrouter = use_openrouter
        self.input_token_length = 0
        self.output_token_length = 0

    def _get_provider(self) -> str:
        """Determine the provider from the model name."""
        model_lower = self.model.lower()

        if "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        elif "gemini" in model_lower:
            return "gemini"
        elif "deepseek" in model_lower:
            return "deepseek"
        else:
            return "unknown"

    def _get_input_price(self) -> float:
        """Get the input price per 1M tokens for the model."""
        provider = self._get_provider()

        if self.use_openrouter:
            # OpenRouter pricing
            if provider == "openai" and self.model in self.OPENROUTER_GPT_INPUT_PRICING:
                return self.OPENROUTER_GPT_INPUT_PRICING[self.model]
            elif provider == "anthropic" and self.model in self.OPENROUTER_ANTHROPIC_INPUT_PRICING:
                return self.OPENROUTER_ANTHROPIC_INPUT_PRICING[self.model]
            elif provider == "deepseek" and self.model in self.OPENROUTER_DEEPSEEK_INPUT_PRICING:
                return self.OPENROUTER_DEEPSEEK_INPUT_PRICING[self.model]
        else:
            # Direct API pricing
            if provider == "openai" and self.model in self.OPENAI_DIRECT_INPUT_PRICING:
                return self.OPENAI_DIRECT_INPUT_PRICING[self.model]
            elif provider == "anthropic" and self.model in self.ANTHROPIC_DIRECT_INPUT_PRICING:
                return self.ANTHROPIC_DIRECT_INPUT_PRICING[self.model]
            elif provider == "gemini" and self.model in self.GEMINI_DIRECT_INPUT_PRICING:
                return self.GEMINI_DIRECT_INPUT_PRICING[self.model]
            elif provider == "deepseek" and self.model in self.DEEPSEEK_DIRECT_INPUT_PRICING:
                return self.DEEPSEEK_DIRECT_INPUT_PRICING[self.model]

        # Fallback pricing based on provider
        fallback_prices = {
            "openai": 2.50,
            "anthropic": 3.00,
            "gemini": 1.25,
            "deepseek": 0.27,
        }
        return fallback_prices.get(provider, 3.00)

    def _get_output_price(self) -> float:
        """Get the output price per 1M tokens for the model."""
        provider = self._get_provider()

        if self.use_openrouter:
            # OpenRouter pricing
            if provider == "openai" and self.model in self.OPENROUTER_GPT_OUTPUT_PRICING:
                return self.OPENROUTER_GPT_OUTPUT_PRICING[self.model]
            elif (
                provider == "anthropic" and self.model in self.OPENROUTER_ANTHROPIC_OUTPUT_PRICING
            ):
                return self.OPENROUTER_ANTHROPIC_OUTPUT_PRICING[self.model]
            elif provider == "deepseek" and self.model in self.OPENROUTER_DEEPSEEK_OUTPUT_PRICING:
                return self.OPENROUTER_DEEPSEEK_OUTPUT_PRICING[self.model]
        else:
            # Direct API pricing
            if provider == "openai" and self.model in self.OPENAI_DIRECT_OUTPUT_PRICING:
                return self.OPENAI_DIRECT_OUTPUT_PRICING[self.model]
            elif provider == "anthropic" and self.model in self.ANTHROPIC_DIRECT_OUTPUT_PRICING:
                return self.ANTHROPIC_DIRECT_OUTPUT_PRICING[self.model]
            elif provider == "gemini" and self.model in self.GEMINI_DIRECT_OUTPUT_PRICING:
                return self.GEMINI_DIRECT_OUTPUT_PRICING[self.model]
            elif provider == "deepseek" and self.model in self.DEEPSEEK_DIRECT_OUTPUT_PRICING:
                return self.DEEPSEEK_DIRECT_OUTPUT_PRICING[self.model]

        # Fallback pricing based on provider
        fallback_prices = {
            "openai": 10.00,
            "anthropic": 15.00,
            "gemini": 5.00,
            "deepseek": 1.10,
        }
        return fallback_prices.get(provider, 15.00)

    def calculate_cost_from_tokens(self) -> float:
        """
        Calculate cost based on pre-set input_token_length and output_token_length.

        Returns:
            float: The calculated cost in USD.
        """
        input_price = self._get_input_price()
        output_price = self._get_output_price()

        input_cost = self.input_token_length * input_price / 1e6
        output_cost = self.output_token_length * output_price / 1e6

        return input_cost + output_cost

    def calculate_input_token_length_tiktoken(self) -> int:
        """
        Calculate the number of input tokens using tiktoken.

        This is an approximation that works for most models.

        Returns:
            int: Number of input tokens.
        """
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        if self.formatted_input_sequence:
            for message in self.formatted_input_sequence:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def calculate_output_token_length_tiktoken(self) -> int:
        """
        Calculate the number of output tokens using tiktoken.

        Returns:
            int: Number of output tokens.
        """
        try:
            tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            tokenizer = tiktoken.get_encoding("cl100k_base")

        if self.output_sequence_string:
            tokens = tokenizer.encode(self.output_sequence_string)
            return len(tokens)
        return 0

    def calculate_cost(self) -> float:
        """
        Calculate the total cost based on input and output sequences.

        This method uses tiktoken for token counting (approximation).

        Returns:
            float: The calculated cost in USD.
        """
        if self.formatted_input_sequence is not None:
            self.input_token_length = self.calculate_input_token_length_tiktoken()
        if self.output_sequence_string is not None:
            self.output_token_length = self.calculate_output_token_length_tiktoken()

        return self.calculate_cost_from_tokens()

    def calculate_input_token_length(self, input_sequence: list[str], form: str = "list") -> int:
        """
        Calculate input token length from various input formats.

        Args:
            input_sequence: The input sequence as a list of strings.
            form (str): Format of input - "list" for list of strings, "formatted" for
                list of message dicts.

        Returns:
            int: Number of input tokens.
        """
        if form == "list":
            self.formatted_input_sequence = PromptBase.list_to_formatted_OpenAI(input_sequence)
        elif form == "formatted":
            # For formatted input, we expect list[dict[str, str]] but accept list[str] signature
            # The caller is responsible for passing the correct type
            self.formatted_input_sequence = input_sequence  # type: ignore[assignment]
        else:
            raise ValueError("Invalid form. Use 'list' or 'formatted'.")

        self.input_token_length = self.calculate_input_token_length_tiktoken()
        return self.input_token_length

    def length_limiter(
        self,
        input_sequence: list[str],
        limit: int,
        truncation: bool = True,
        include_truncation_warning: bool = True,
    ) -> list[str]:
        """
        Limit the input sequence to a certain token length.

        Args:
            input_sequence (list[str]): The input sequence as a list of strings.
            limit (int): Maximum token limit.
            truncation (bool): Whether to truncate if over limit. Defaults to True.
            include_truncation_warning (bool): Whether to add a warning message. Defaults to True.

        Returns:
            list[str]: The (possibly truncated) input sequence.
        """
        self.calculate_input_token_length(input_sequence, form="list")

        if self.input_token_length > limit:
            print(f"Warning: Input sequence is longer than {limit} tokens.")

            if truncation:
                # Calculate the token length of the warning message if it will be included
                warning_message = "---Warning, this information is too long and is truncated---"
                warning_token_length = (
                    self.calculate_input_token_length([warning_message], form="list")
                    if include_truncation_warning
                    else 0
                )

                # Adjust the limit to account for the warning message
                adjusted_limit = (
                    limit - warning_token_length if include_truncation_warning else limit
                )

                # Convert the input sequence to a single string for truncation
                input_string = " ".join(input_sequence)

                # Estimate the required reduction percentage
                required_reduction_ratio = adjusted_limit / self.input_token_length

                # Apply the estimated reduction
                truncated_length = int(len(input_string) * required_reduction_ratio)

                # Split the truncated string back into a list of strings
                truncated_input_sequence = []
                current_length = 0
                for sentence in input_sequence:
                    if current_length + len(sentence) + 1 <= truncated_length:
                        truncated_input_sequence.append(sentence)
                        current_length += len(sentence) + 1  # +1 for the space
                    else:
                        # Truncate the last sentence to fit within the limit
                        remaining_length = truncated_length - current_length
                        if remaining_length > 0:
                            truncated_sentence = sentence[:remaining_length]
                            truncated_input_sequence.append(truncated_sentence)
                        break

                # Re-calculate the token length to ensure it's within the adjusted limit
                self.calculate_input_token_length(truncated_input_sequence, form="list")

                # Iteratively adjust if still over limit
                truncated_string = " ".join(truncated_input_sequence)
                while self.input_token_length > adjusted_limit:
                    required_reduction_ratio = adjusted_limit / self.input_token_length
                    truncated_length = int(len(truncated_string) * required_reduction_ratio)
                    truncated_string = truncated_string[:truncated_length]

                    truncated_input_sequence = []
                    current_length = 0
                    for sentence in input_sequence:
                        if current_length + len(sentence) + 1 <= truncated_length:
                            truncated_input_sequence.append(sentence)
                            current_length += len(sentence) + 1
                        else:
                            remaining_length = truncated_length - current_length
                            if remaining_length > 0:
                                truncated_sentence = sentence[:remaining_length]
                                truncated_input_sequence.append(truncated_sentence)
                            break

                    self.calculate_input_token_length(truncated_input_sequence, form="list")

                # Append the truncation warning if required
                if include_truncation_warning:
                    truncated_input_sequence.append(warning_message)

                # Re-calculate the final token length including the warning
                self.calculate_input_token_length(truncated_input_sequence, form="list")

                print(f"Warning: Input sequence is truncated to be about {limit} tokens.")
                return truncated_input_sequence
            else:
                return input_sequence
        else:
            return input_sequence


def get_supported_models_pricing(
    use_openrouter: bool = False,
) -> dict[str, dict[str, float | str]]:
    """
    Get pricing information for all supported models.

    Args:
        use_openrouter (bool): Whether to return OpenRouter pricing or direct API pricing.

    Returns:
        dict: A dictionary with model names as keys and pricing info as values.
            Each value dict contains 'input_per_1m' (float), 'output_per_1m' (float),
            and 'routing' (str).
    """
    models: dict[str, dict[str, float | str]] = {}

    if use_openrouter:
        # OpenRouter pricing
        for model in Calculator.OPENROUTER_GPT_INPUT_PRICING:
            models[model] = {
                "input_per_1m": Calculator.OPENROUTER_GPT_INPUT_PRICING[model],
                "output_per_1m": Calculator.OPENROUTER_GPT_OUTPUT_PRICING.get(model, 11.0),
                "routing": "openrouter",
            }

        for model in Calculator.OPENROUTER_ANTHROPIC_INPUT_PRICING:
            models[model] = {
                "input_per_1m": Calculator.OPENROUTER_ANTHROPIC_INPUT_PRICING[model],
                "output_per_1m": Calculator.OPENROUTER_ANTHROPIC_OUTPUT_PRICING.get(model, 16.5),
                "routing": "openrouter",
            }

        for model in Calculator.OPENROUTER_DEEPSEEK_INPUT_PRICING:
            models[model] = {
                "input_per_1m": Calculator.OPENROUTER_DEEPSEEK_INPUT_PRICING[model],
                "output_per_1m": Calculator.OPENROUTER_DEEPSEEK_OUTPUT_PRICING.get(model, 2.19),
                "routing": "openrouter",
            }

        # Gemini is always direct
        for model in Calculator.GEMINI_DIRECT_INPUT_PRICING:
            models[model] = {
                "input_per_1m": Calculator.GEMINI_DIRECT_INPUT_PRICING[model],
                "output_per_1m": Calculator.GEMINI_DIRECT_OUTPUT_PRICING.get(model, 5.0),
                "routing": "direct",
            }
    else:
        # Direct API pricing
        for model in Calculator.OPENAI_DIRECT_INPUT_PRICING:
            models[model] = {
                "input_per_1m": Calculator.OPENAI_DIRECT_INPUT_PRICING[model],
                "output_per_1m": Calculator.OPENAI_DIRECT_OUTPUT_PRICING.get(model, 10.0),
                "routing": "direct",
            }

        for model in Calculator.ANTHROPIC_DIRECT_INPUT_PRICING:
            models[model] = {
                "input_per_1m": Calculator.ANTHROPIC_DIRECT_INPUT_PRICING[model],
                "output_per_1m": Calculator.ANTHROPIC_DIRECT_OUTPUT_PRICING.get(model, 15.0),
                "routing": "direct",
            }

        for model in Calculator.GEMINI_DIRECT_INPUT_PRICING:
            models[model] = {
                "input_per_1m": Calculator.GEMINI_DIRECT_INPUT_PRICING[model],
                "output_per_1m": Calculator.GEMINI_DIRECT_OUTPUT_PRICING.get(model, 5.0),
                "routing": "direct",
            }

        for model in Calculator.DEEPSEEK_DIRECT_INPUT_PRICING:
            models[model] = {
                "input_per_1m": Calculator.DEEPSEEK_DIRECT_INPUT_PRICING[model],
                "output_per_1m": Calculator.DEEPSEEK_DIRECT_OUTPUT_PRICING.get(model, 1.10),
                "routing": "direct",
            }

    return models


if __name__ == "__main__":
    # Test the Calculator with different models
    print("Testing cost calculation for supported models:\n")

    test_models = [
        "claude-sonnet-4.5",
        "gpt-5.2",
        "gemini-pro-3.0",
        "deepseek-v3.2",
    ]

    print("=== Direct API Pricing ===")
    for model in test_models:
        calc = Calculator(model, use_openrouter=False)
        calc.input_token_length = 1000
        calc.output_token_length = 500
        cost = calc.calculate_cost_from_tokens()
        print(f"{model}: ${cost:.6f} for 1000 input + 500 output tokens")

    print("\n=== OpenRouter Pricing ===")
    for model in test_models:
        if model == "gemini-pro-3.0":
            print(f"{model}: N/A (Gemini always uses direct API)")
            continue
        calc = Calculator(model, use_openrouter=True)
        calc.input_token_length = 1000
        calc.output_token_length = 500
        cost = calc.calculate_cost_from_tokens()
        print(f"{model}: ${cost:.6f} for 1000 input + 500 output tokens")

    print("\n=== All supported models (direct) pricing ===")
    pricing = get_supported_models_pricing(use_openrouter=False)
    for model, prices in pricing.items():
        print(
            f"  {model}: ${prices['input_per_1m']}/1M in, "
            f"${prices['output_per_1m']}/1M out ({prices['routing']})"
        )

    print("\n=== All supported models (OpenRouter) pricing ===")
    pricing = get_supported_models_pricing(use_openrouter=True)
    for model, prices in pricing.items():
        print(
            f"  {model}: ${prices['input_per_1m']}/1M in, "
            f"${prices['output_per_1m']}/1M out ({prices['routing']})"
        )
