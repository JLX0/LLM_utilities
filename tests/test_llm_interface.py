"""
Integration tests for LiteLLM_interface with actual API calls.

These tests require the following environment variables to be set:
- OPENROUTER_API_KEY
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GEMINI_API_KEY
- DEEPSEEK_API_KEY

Run with: pytest tests/test_llm_interface.py -v -s
"""

from __future__ import annotations

import os

import pytest

from LLM_utils.cost import Calculator
from LLM_utils.cost import get_supported_models_pricing
from LLM_utils.inquiry import _check_tenacity_available
from LLM_utils.inquiry import _get_litellm_model_id
from LLM_utils.inquiry import _is_gemini_model
from LLM_utils.inquiry import _resolve_to_canonical
from LLM_utils.inquiry import LiteLLM_interface


# =============================================================================
# Test Configuration
# =============================================================================

# Simple test prompt that should work with all models
SIMPLE_PROMPT = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]

# Reasoning prompt for testing thinking/reasoning modes
REASONING_PROMPT = [
    {
        "role": "user",
        "content": "What is 15 * 17? Think step by step and show your work briefly.",
    }
]

# Models to test
TEST_MODELS = [
    "claude-sonnet-4.5",
    "gpt-5.2",
    "gemini-pro-3.0",
    "deepseek-v3.2",
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def check_openrouter_key():
    """Check if OpenRouter API key is available."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")


@pytest.fixture
def check_openai_key():
    """Check if OpenAI API key is available."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture
def check_anthropic_key():
    """Check if Anthropic API key is available."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


@pytest.fixture
def check_gemini_key():
    """Check if Gemini API key is available."""
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")


@pytest.fixture
def check_deepseek_key():
    """Check if DeepSeek API key is available."""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")


@pytest.fixture
def check_tenacity():
    """Check if tenacity is available for reasoning tests."""
    if not _check_tenacity_available():
        pytest.skip("tenacity not installed (required for reasoning features)")


@pytest.fixture
def clear_openrouter_key():
    """Temporarily clear OpenRouter key to force direct API usage."""
    original_key = os.environ.get("OPENROUTER_API_KEY")
    if original_key:
        del os.environ["OPENROUTER_API_KEY"]
    yield
    if original_key:
        os.environ["OPENROUTER_API_KEY"] = original_key


# =============================================================================
# Helper Functions
# =============================================================================


def validate_response(response: tuple[str | None, float], allow_none: bool = False) -> None:
    """Validate that response is properly formed."""
    response_text, cost = response
    if not allow_none:
        assert response_text is not None, "Response text should not be None"
        assert isinstance(response_text, str), "Response text should be a string"
        assert len(response_text) > 0, "Response text should not be empty"
    assert isinstance(cost, float), "Cost should be a float"
    assert cost >= 0, "Cost should be non-negative"


def print_response_info(
    model: str, routing: str, response_text: str | None, cost: float, reasoning: str | None = None
) -> None:
    """Print response information for debugging."""
    print(f"\n{'=' * 60}")
    print(f"Model: {model}")
    print(f"Routing: {routing}")
    if reasoning:
        print(f"Reasoning effort: {reasoning}")
    print(f"Cost: ${cost:.6f}")
    if response_text:
        print(f"Response preview: {response_text[:100]}...")
    else:
        print("Response: None")
    print(f"{'=' * 60}")


# =============================================================================
# Unit Tests - Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Test helper functions for model resolution and routing."""

    def test_resolve_to_canonical(self):
        """Test model name resolution to canonical form."""
        assert _resolve_to_canonical("claude") == "claude-sonnet-4.5"
        assert _resolve_to_canonical("claude-sonnet-4.5") == "claude-sonnet-4.5"
        assert _resolve_to_canonical("gpt") == "gpt-5.2"
        assert _resolve_to_canonical("gpt-5") == "gpt-5.2"
        assert _resolve_to_canonical("gemini") == "gemini-pro-3.0"
        assert _resolve_to_canonical("deepseek") == "deepseek-v3.2"

    def test_is_gemini_model(self):
        """Test Gemini model detection."""
        assert _is_gemini_model("gemini-pro-3.0") is True
        assert _is_gemini_model("claude-sonnet-4.5") is False
        assert _is_gemini_model("gpt-5.2") is False

    def test_get_litellm_model_id_direct(self):
        """Test LiteLLM model ID generation for direct routing."""
        assert (
            _get_litellm_model_id("claude-sonnet-4.5", use_openrouter=False)
            == "anthropic/claude-sonnet-4-5-20250929"
        )
        assert _get_litellm_model_id("gpt-5.2", use_openrouter=False) == "openai/gpt-5.2"
        assert (
            _get_litellm_model_id("gemini-pro-3.0", use_openrouter=False)
            == "gemini/gemini-3-pro-preview"
        )
        assert (
            _get_litellm_model_id("deepseek-v3.2", use_openrouter=False)
            == "deepseek/deepseek-chat"
        )

    def test_get_litellm_model_id_openrouter(self):
        """Test LiteLLM model ID generation for OpenRouter routing."""
        assert (
            _get_litellm_model_id("claude-sonnet-4.5", use_openrouter=True)
            == "openrouter/anthropic/claude-sonnet-4.5"
        )
        assert _get_litellm_model_id("gpt-5.2", use_openrouter=True) == "openrouter/openai/gpt-5.2"
        # Gemini always uses direct, even when OpenRouter is requested
        assert (
            _get_litellm_model_id("gemini-pro-3.0", use_openrouter=True)
            == "gemini/gemini-3-pro-preview"
        )
        assert (
            _get_litellm_model_id("deepseek-v3.2", use_openrouter=True)
            == "openrouter/deepseek/deepseek-v3.2"
        )

    def test_get_litellm_model_id_deepseek_reasoning(self):
        """Test DeepSeek model ID with reasoning enabled (direct only)."""
        # With reasoning enabled, direct DeepSeek uses deepseek-reasoner
        assert (
            _get_litellm_model_id("deepseek-v3.2", use_openrouter=False, reasoning_enabled=True)
            == "deepseek/deepseek-reasoner"
        )
        # OpenRouter still uses the same model with reasoning params
        assert (
            _get_litellm_model_id("deepseek-v3.2", use_openrouter=True, reasoning_enabled=True)
            == "openrouter/deepseek/deepseek-v3.2"
        )


# =============================================================================
# Direct API Tests
# =============================================================================


class TestDirectAPIOpenAI:
    """Test direct OpenAI API calls."""

    def test_gpt_direct_simple(self, check_openai_key, clear_openrouter_key):
        """Test GPT-5.2 direct API with simple prompt."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            force_direct=True,
            debug=True,
            max_tokens=100,
        )

        assert llm.use_openrouter is False
        assert "openai/" in llm.resolved_model

        response_text, cost = llm.ask_base(SIMPLE_PROMPT)
        validate_response((response_text, cost))
        print_response_info("gpt-5.2", "direct", response_text, cost)

    def test_gpt_direct_reasoning(self, check_openai_key, clear_openrouter_key):
        """Test GPT-5.2 direct API with reasoning effort."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            force_direct=True,
            debug=True,
            max_tokens=500,
            reasoning_effort="medium",
        )

        assert llm.use_openrouter is False

        response_text, cost = llm.ask_base(REASONING_PROMPT)
        validate_response((response_text, cost))
        print_response_info("gpt-5.2", "direct", response_text, cost, reasoning="medium")


class TestDirectAPIAnthropic:
    """Test direct Anthropic API calls."""

    def test_claude_direct_simple(self, check_anthropic_key, clear_openrouter_key):
        """Test Claude Sonnet 4.5 direct API with simple prompt."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            force_direct=True,
            debug=True,
            max_tokens=100,
        )

        assert llm.use_openrouter is False
        assert "anthropic/" in llm.resolved_model

        response_text, cost = llm.ask_base(SIMPLE_PROMPT)
        validate_response((response_text, cost))
        print_response_info("claude-sonnet-4.5", "direct", response_text, cost)

    def test_claude_direct_thinking(
        self, check_anthropic_key, clear_openrouter_key, check_tenacity
    ):
        """Test Claude Sonnet 4.5 direct API with thinking/reasoning."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            force_direct=True,
            debug=True,
            max_tokens=500,
            reasoning_effort="medium",
        )

        assert llm.use_openrouter is False

        response_text, cost = llm.ask_base(REASONING_PROMPT)
        validate_response((response_text, cost))
        print_response_info("claude-sonnet-4.5", "direct", response_text, cost, reasoning="medium")


class TestDirectAPIGemini:
    """Test direct Gemini API calls.

    Note: Gemini 3 Pro cannot disable thinking - it's always active.
    The interface defaults to reasoning_effort='high' for Gemini models.
    """

    def test_gemini_direct_simple(self, check_gemini_key):
        """Test Gemini Pro 3.0 direct API with simple prompt."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=300,  # Increased for reasoning + response
        )

        # Gemini should always use direct, regardless of OpenRouter key
        assert llm.use_openrouter is False
        assert "gemini/" in llm.resolved_model
        # Should auto-default to 'high' reasoning
        assert llm.reasoning_effort == "high"

        response_text, cost = llm.ask_base(SIMPLE_PROMPT)

        # Gemini API can be flaky - skip if None
        if response_text is None:
            pytest.skip("Gemini API returned None (may be temporarily unavailable)")

        validate_response((response_text, cost))
        print_response_info("gemini-pro-3.0", "direct", response_text, cost)

    def test_gemini_direct_reasoning(self, check_gemini_key):
        """Test Gemini Pro 3.0 direct API with reasoning effort."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=800,  # Increased for reasoning
            reasoning_effort="medium",
        )

        assert llm.use_openrouter is False

        response_text, cost = llm.ask_base(REASONING_PROMPT)
        # Gemini reasoning may not be fully supported, allow None
        if response_text is not None:
            validate_response((response_text, cost))
            print_response_info(
                "gemini-pro-3.0", "direct", response_text, cost, reasoning="medium"
            )
        else:
            pytest.skip("Gemini reasoning returned None (may need more tokens)")

    def test_gemini_always_direct_even_with_openrouter_key(
        self, check_gemini_key, check_openrouter_key
    ):
        """Test that Gemini always uses direct API even when OpenRouter key is available."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=300,  # Increased for reasoning + response
        )

        # Should still be direct despite OpenRouter key being available
        assert llm.use_openrouter is False
        assert "gemini/" in llm.resolved_model

        response_text, cost = llm.ask_base(SIMPLE_PROMPT)

        # Gemini API can be flaky - skip if None
        if response_text is None:
            pytest.skip("Gemini API returned None (may be temporarily unavailable)")

        validate_response((response_text, cost))
        print_response_info("gemini-pro-3.0", "direct (forced)", response_text, cost)


class TestDirectAPIDeepSeek:
    """Test direct DeepSeek API calls."""

    def test_deepseek_direct_simple(self, check_deepseek_key, clear_openrouter_key):
        """Test DeepSeek V3.2 direct API with simple prompt (non-thinking mode)."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            force_direct=True,
            debug=True,
            max_tokens=100,
        )

        assert llm.use_openrouter is False
        assert "deepseek/deepseek-chat" in llm.resolved_model

        response_text, cost = llm.ask_base(SIMPLE_PROMPT)
        validate_response((response_text, cost))
        print_response_info("deepseek-v3.2", "direct (non-thinking)", response_text, cost)

    def test_deepseek_direct_reasoning(self, check_deepseek_key, clear_openrouter_key):
        """Test DeepSeek V3.2 direct API with thinking mode (deepseek-reasoner)."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            force_direct=True,
            debug=True,
            max_tokens=500,
            reasoning_effort="medium",
        )

        assert llm.use_openrouter is False
        # Should switch to deepseek-reasoner for thinking mode
        assert "deepseek/deepseek-reasoner" in llm.resolved_model

        response_text, cost = llm.ask_base(REASONING_PROMPT)
        validate_response((response_text, cost))
        print_response_info(
            "deepseek-v3.2", "direct (thinking)", response_text, cost, reasoning="medium"
        )


# =============================================================================
# OpenRouter API Tests
# =============================================================================


class TestOpenRouterAPI:
    """Test OpenRouter API calls for all applicable models."""

    def test_openrouter_gpt(self, check_openrouter_key):
        """Test GPT-5.2 via OpenRouter."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            debug=True,
            max_tokens=100,
        )

        assert llm.use_openrouter is True
        assert "openrouter/openai/" in llm.resolved_model

        response_text, cost = llm.ask_base(SIMPLE_PROMPT)
        validate_response((response_text, cost))
        print_response_info("gpt-5.2", "openrouter", response_text, cost)

    def test_openrouter_claude(self, check_openrouter_key):
        """Test Claude Sonnet 4.5 via OpenRouter."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            debug=True,
            max_tokens=100,
        )

        assert llm.use_openrouter is True
        assert "openrouter/anthropic/" in llm.resolved_model

        response_text, cost = llm.ask_base(SIMPLE_PROMPT)
        validate_response((response_text, cost))
        print_response_info("claude-sonnet-4.5", "openrouter", response_text, cost)

    def test_openrouter_deepseek(self, check_openrouter_key):
        """Test DeepSeek V3.2 via OpenRouter."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            debug=True,
            max_tokens=100,
        )

        assert llm.use_openrouter is True
        assert "openrouter/deepseek/" in llm.resolved_model

        response_text, cost = llm.ask_base(SIMPLE_PROMPT)
        validate_response((response_text, cost))
        print_response_info("deepseek-v3.2", "openrouter", response_text, cost)

    def test_openrouter_gpt_reasoning(self, check_openrouter_key):
        """Test GPT-5.2 via OpenRouter with reasoning."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            debug=True,
            max_tokens=500,
            reasoning_effort="medium",
        )

        assert llm.use_openrouter is True

        response_text, cost = llm.ask_base(REASONING_PROMPT)
        validate_response((response_text, cost))
        print_response_info("gpt-5.2", "openrouter", response_text, cost, reasoning="medium")

    def test_openrouter_claude_reasoning(self, check_openrouter_key, check_tenacity):
        """Test Claude Sonnet 4.5 via OpenRouter with reasoning."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            debug=True,
            max_tokens=500,
            reasoning_effort="medium",
        )

        assert llm.use_openrouter is True

        response_text, cost = llm.ask_base(REASONING_PROMPT)
        validate_response((response_text, cost))
        print_response_info(
            "claude-sonnet-4.5", "openrouter", response_text, cost, reasoning="medium"
        )

    def test_openrouter_deepseek_reasoning(self, check_openrouter_key):
        """Test DeepSeek V3.2 via OpenRouter with reasoning."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            debug=True,
            max_tokens=500,
            reasoning_effort="medium",
        )

        assert llm.use_openrouter is True

        response_text, cost = llm.ask_base(REASONING_PROMPT)
        validate_response((response_text, cost))
        print_response_info("deepseek-v3.2", "openrouter", response_text, cost, reasoning="medium")


# =============================================================================
# Routing Logic Tests
# =============================================================================


class TestRoutingLogic:
    """Test the routing logic between direct API and OpenRouter."""

    def test_prefers_openrouter_when_key_available(self, check_openrouter_key):
        """Test that non-Gemini models prefer OpenRouter when key is available."""
        # GPT should use OpenRouter
        llm_gpt = LiteLLM_interface(model="gpt-5.2", debug=True)
        assert llm_gpt.use_openrouter is True

        # Claude should use OpenRouter
        llm_claude = LiteLLM_interface(model="claude-sonnet-4.5", debug=True)
        assert llm_claude.use_openrouter is True

        # DeepSeek should use OpenRouter
        llm_deepseek = LiteLLM_interface(model="deepseek-v3.2", debug=True)
        assert llm_deepseek.use_openrouter is True

        # Gemini should NOT use OpenRouter (always direct)
        llm_gemini = LiteLLM_interface(model="gemini-pro-3.0", debug=True)
        assert llm_gemini.use_openrouter is False

    def test_uses_direct_when_openrouter_key_not_available(self, clear_openrouter_key):
        """Test that models use direct API when OpenRouter key is not available."""
        llm_gpt = LiteLLM_interface(model="gpt-5.2", debug=True)
        assert llm_gpt.use_openrouter is False

        llm_claude = LiteLLM_interface(model="claude-sonnet-4.5", debug=True)
        assert llm_claude.use_openrouter is False

        llm_deepseek = LiteLLM_interface(model="deepseek-v3.2", debug=True)
        assert llm_deepseek.use_openrouter is False

    def test_force_direct_overrides_openrouter(self, check_openrouter_key):
        """Test that force_direct=True overrides OpenRouter preference."""
        llm = LiteLLM_interface(model="gpt-5.2", force_direct=True, debug=True)
        assert llm.use_openrouter is False
        assert "openai/" in llm.resolved_model
        assert "openrouter" not in llm.resolved_model


# =============================================================================
# Cost Calculation Tests
# =============================================================================


class TestCostCalculation:
    """Test cost calculation for different models and routing."""

    def test_cost_calculation_direct(self):
        """Test cost calculation for direct API routing."""
        for model in TEST_MODELS:
            calc = Calculator(model, use_openrouter=False)
            calc.input_token_length = 1000
            calc.output_token_length = 500
            cost = calc.calculate_cost_from_tokens()

            assert isinstance(cost, float)
            assert cost > 0
            print(f"{model} (direct): ${cost:.6f}")

    def test_cost_calculation_openrouter(self):
        """Test cost calculation for OpenRouter routing."""
        for model in TEST_MODELS:
            if model == "gemini-pro-3.0":
                # Gemini is always direct
                continue

            calc = Calculator(model, use_openrouter=True)
            calc.input_token_length = 1000
            calc.output_token_length = 500
            cost = calc.calculate_cost_from_tokens()

            assert isinstance(cost, float)
            assert cost > 0
            print(f"{model} (openrouter): ${cost:.6f}")

    def test_openrouter_pricing_higher_than_direct(self):
        """Test that OpenRouter pricing is generally higher than direct (markup)."""
        for model in ["gpt-5.2", "claude-sonnet-4.5", "deepseek-v3.2"]:
            calc_direct = Calculator(model, use_openrouter=False)
            calc_direct.input_token_length = 1000
            calc_direct.output_token_length = 500
            cost_direct = calc_direct.calculate_cost_from_tokens()

            calc_openrouter = Calculator(model, use_openrouter=True)
            calc_openrouter.input_token_length = 1000
            calc_openrouter.output_token_length = 500
            cost_openrouter = calc_openrouter.calculate_cost_from_tokens()

            # OpenRouter should have markup (higher cost)
            assert cost_openrouter >= cost_direct, (
                f"{model}: OpenRouter cost should be >= direct cost"
            )
            print(f"{model}: direct=${cost_direct:.6f}, openrouter=${cost_openrouter:.6f}")

    def test_get_supported_models_pricing(self):
        """Test get_supported_models_pricing function."""
        # Direct pricing
        direct_pricing = get_supported_models_pricing(use_openrouter=False)
        assert len(direct_pricing) > 0
        for model, prices in direct_pricing.items():
            assert "input_per_1m" in prices
            assert "output_per_1m" in prices
            assert "routing" in prices
            assert prices["routing"] == "direct"

        # OpenRouter pricing
        openrouter_pricing = get_supported_models_pricing(use_openrouter=True)
        assert len(openrouter_pricing) > 0
        for model, prices in openrouter_pricing.items():
            assert "input_per_1m" in prices
            assert "output_per_1m" in prices
            assert "routing" in prices


# =============================================================================
# Model Alias Tests
# =============================================================================


class TestModelAliases:
    """Test that all model aliases work correctly."""

    @pytest.mark.parametrize(
        "alias,expected_canonical",
        [
            ("claude", "claude-sonnet-4.5"),
            ("claude-4-5-sonnet", "claude-sonnet-4.5"),
            ("claude-sonnet-4-5", "claude-sonnet-4.5"),
            ("gpt", "gpt-5.2"),
            ("gpt-5", "gpt-5.2"),
            ("gemini", "gemini-pro-3.0"),
            ("gemini-3-pro", "gemini-pro-3.0"),
            ("gemini-pro-3", "gemini-pro-3.0"),
            ("deepseek", "deepseek-v3.2"),
            ("deepseek-3.2", "deepseek-v3.2"),
        ],
    )
    def test_alias_resolution(self, alias: str, expected_canonical: str):
        """Test that aliases resolve to correct canonical names."""
        llm = LiteLLM_interface(model=alias, debug=True)
        assert llm.canonical_model == expected_canonical


# =============================================================================
# Gemini Default Reasoning Tests
# =============================================================================


class TestGeminiDefaultReasoning:
    """Test Gemini-specific default reasoning behavior."""

    def test_gemini_defaults_to_high_reasoning(self):
        """Test that Gemini defaults to reasoning_effort='high' when not specified."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=300,
            # reasoning_effort not specified
        )

        assert llm.reasoning_effort == "high"
        assert llm.canonical_model == "gemini-pro-3.0"

    def test_gemini_respects_explicit_low_reasoning(self):
        """Test that Gemini respects explicitly set low reasoning effort."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=500,
            reasoning_effort="low",  # Explicitly set
        )

        assert llm.reasoning_effort == "low"

    def test_non_gemini_does_not_default_reasoning(self):
        """Test that non-Gemini models don't get default reasoning."""
        llm_gpt = LiteLLM_interface(model="gpt-5.2", debug=True)
        assert llm_gpt.reasoning_effort is None

        llm_claude = LiteLLM_interface(model="claude-sonnet-4.5", debug=True)
        assert llm_claude.reasoning_effort is None

        llm_deepseek = LiteLLM_interface(model="deepseek-v3.2", debug=True)
        assert llm_deepseek.reasoning_effort is None


# =============================================================================
# Integration Tests - Full Pipeline
# =============================================================================


class TestFullPipeline:
    """Integration tests for the full LLM pipeline."""

    def test_ask_with_timeout(self, check_openrouter_key):
        """Test ask() method with timeout handling."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            debug=True,
            max_tokens=100,
            timeout=60,
            maximum_timeout_attempts=2,
        )

        response_text, cost = llm.ask(SIMPLE_PROMPT)

        # Should not be termination_signal for a simple prompt
        assert response_text != "termination_signal"
        validate_response((response_text, cost))

    def test_ask_with_test_success(self, check_openrouter_key):
        """Test ask_with_test() method with a passing test."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            debug=True,
            max_tokens=50,
        )

        # Test function that checks for a number in response
        def test_contains_number(response: str) -> str:
            assert any(char.isdigit() for char in response), "Response should contain a number"
            return response

        result, cost = llm.ask_with_test(SIMPLE_PROMPT, test_contains_number)

        assert result != "termination_signal"
        assert isinstance(result, str)
        assert cost >= 0

    def test_all_models_respond(self, check_openrouter_key, check_gemini_key):
        """Test that all supported models respond successfully."""
        results = {}

        # Use different max_tokens for Gemini due to reasoning requirements
        model_configs = {
            "claude-sonnet-4.5": {"max_tokens": 100},
            "gpt-5.2": {"max_tokens": 100},
            "gemini-pro-3.0": {"max_tokens": 300},  # More tokens for Gemini
            "deepseek-v3.2": {"max_tokens": 100},
        }

        for model in TEST_MODELS:
            print(f"\nTesting {model}...")
            config = model_configs.get(model, {"max_tokens": 100})
            llm = LiteLLM_interface(
                model=model,
                debug=True,
                **config,
            )

            response_text, cost = llm.ask_base(SIMPLE_PROMPT)

            if response_text is not None:
                results[model] = {
                    "success": True,
                    "response_length": len(response_text),
                    "cost": cost,
                    "routing": "openrouter" if llm.use_openrouter else "direct",
                }
            else:
                results[model] = {"success": False}

        print("\n" + "=" * 60)
        print("Summary of all model responses:")
        print("=" * 60)
        for model, result in results.items():
            if result["success"]:
                print(
                    f"  {model}: OK (routing={result['routing']}, "
                    f"len={result['response_length']}, cost=${result['cost']:.6f})"
                )
            else:
                print(f"  {model}: FAILED")

        # All models should succeed, but Gemini can be flaky
        failed_models = []
        for model, result in results.items():
            if not result["success"]:
                if model == "gemini-pro-3.0":
                    # Gemini can be flaky, just warn
                    print(
                        f"Warning: {model} failed (Gemini API can be intermittently unavailable)"
                    )
                else:
                    failed_models.append(model)

        if failed_models:
            pytest.fail(f"Models failed to respond: {', '.join(failed_models)}")


# =============================================================================
# Reasoning Effort Level Tests
# =============================================================================


class TestReasoningEffortLevels:
    """Test different reasoning effort levels."""

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_claude_reasoning_levels(
        self, check_anthropic_key, clear_openrouter_key, check_tenacity, effort: str
    ):
        """Test Claude with different reasoning effort levels (direct)."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            force_direct=True,
            debug=True,
            max_tokens=500,
            reasoning_effort=effort,
        )

        response_text, cost = llm.ask_base(REASONING_PROMPT)
        validate_response((response_text, cost))
        print_response_info("claude-sonnet-4.5", "direct", response_text, cost, reasoning=effort)

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_gpt_reasoning_levels(self, check_openai_key, clear_openrouter_key, effort: str):
        """Test GPT with different reasoning effort levels (direct)."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            force_direct=True,
            debug=True,
            max_tokens=500,
            reasoning_effort=effort,
        )

        response_text, cost = llm.ask_base(REASONING_PROMPT)
        validate_response((response_text, cost))
        print_response_info("gpt-5.2", "direct", response_text, cost, reasoning=effort)

    @pytest.mark.parametrize("effort", ["low", "high"])  # Gemini 3 Pro only supports low/high
    def test_gemini_reasoning_levels(self, check_gemini_key, effort: str):
        """Test Gemini with different reasoning effort levels."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=800,  # Increased for reasoning
            reasoning_effort=effort,
        )

        response_text, cost = llm.ask_base(REASONING_PROMPT)
        # Gemini reasoning may not be fully supported for all effort levels
        if response_text is not None:
            validate_response((response_text, cost))
            print_response_info("gemini-pro-3.0", "direct", response_text, cost, reasoning=effort)
        else:
            pytest.skip(
                f"Gemini reasoning with effort={effort} returned None (may need more tokens)"
            )


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_model_name(self):
        """Test that invalid model names are handled gracefully."""
        llm = LiteLLM_interface(
            model="nonexistent-model-xyz",
            debug=True,
            max_tokens=100,
        )

        # Should not crash, but may return None response
        response_text, cost = llm.ask_base(SIMPLE_PROMPT)
        # Either returns None or handles the error
        assert response_text is None or isinstance(response_text, str)

    def test_empty_messages(self):
        """Test handling of empty messages list."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            debug=True,
            max_tokens=100,
        )

        response_text, cost = llm.ask_base([])
        # Should handle gracefully (likely return None)
        assert response_text is None or isinstance(response_text, str)


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
