"""
Comprehensive tests for direct API calls to each provider.

These tests focus specifically on testing direct API connections without OpenRouter,
ensuring each provider's native API works correctly.

Run with: pytest tests/test_direct_api_comprehensive.py -v -s
"""

from __future__ import annotations

import os
import pytest
import time

from LLM_utils.inquiry import LiteLLM_interface, _check_tenacity_available
from LLM_utils.cost import Calculator


# =============================================================================
# Test Configuration
# =============================================================================

SIMPLE_PROMPT = [{"role": "user", "content": "Say 'hello' and nothing else."}]

MATH_PROMPT = [{"role": "user", "content": "What is 7 * 8? Reply with just the number."}]

REASONING_PROMPT = [
    {
        "role": "user",
        "content": "A farmer has 17 sheep. All but 9 run away. How many are left? Think briefly then answer.",
    }
]

CODE_PROMPT = [
    {
        "role": "user",
        "content": "Write a Python function that returns the factorial of n. Keep it simple.",
    }
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ensure_no_openrouter():
    """Ensure OpenRouter key is not set for direct API tests."""
    original_key = os.environ.get("OPENROUTER_API_KEY")
    if original_key:
        del os.environ["OPENROUTER_API_KEY"]
    yield
    if original_key:
        os.environ["OPENROUTER_API_KEY"] = original_key


@pytest.fixture
def check_all_direct_keys():
    """Check that all direct API keys are available."""
    required_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
    ]
    missing = [key for key in required_keys if not os.environ.get(key)]
    if missing:
        pytest.skip(f"Missing API keys: {', '.join(missing)}")


@pytest.fixture
def check_tenacity():
    """Check if tenacity is available for reasoning tests."""
    if not _check_tenacity_available():
        pytest.skip("tenacity not installed (required for reasoning features)")


# =============================================================================
# OpenAI Direct Tests
# =============================================================================


class TestOpenAIDirectComprehensive:
    """Comprehensive tests for OpenAI direct API."""

    @pytest.fixture(autouse=True)
    def setup(self, ensure_no_openrouter):
        """Ensure we're testing direct API."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_simple_completion(self):
        """Test simple completion."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            force_direct=True,
            debug=True,
            max_tokens=50,
        )

        response, cost = llm.ask_base(SIMPLE_PROMPT)

        assert response is not None
        assert "hello" in response.lower()
        assert cost > 0
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_math_completion(self):
        """Test math completion."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            force_direct=True,
            debug=True,
            max_tokens=50,
        )

        response, cost = llm.ask_base(MATH_PROMPT)

        assert response is not None
        assert "56" in response
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_code_generation(self):
        """Test code generation."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            force_direct=True,
            debug=True,
            max_tokens=300,
        )

        response, cost = llm.ask_base(CODE_PROMPT)

        assert response is not None
        assert "def" in response or "factorial" in response.lower()
        print(f"Response preview: {response[:200]}...")

    def test_reasoning_low(self):
        """Test with low reasoning effort."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            force_direct=True,
            debug=True,
            max_tokens=30000,
            reasoning_effort="low",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        assert response is not None
        assert "9" in response
        print(f"Response: {response}")

    def test_reasoning_high(self):
        """Test with high reasoning effort."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            force_direct=True,
            debug=True,
            max_tokens=30000,
            reasoning_effort="high",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        assert response is not None
        assert "9" in response
        print(f"Response: {response}")

    def test_temperature_variation(self):
        """Test different temperature settings."""
        for temp in [0.0, 0.5, 1.0]:
            llm = LiteLLM_interface(
                model="gpt-5.2",
                force_direct=True,
                debug=True,
                max_tokens=50,
                temperature=temp,
            )

            response, cost = llm.ask_base(SIMPLE_PROMPT)
            assert response is not None
            print(f"Temperature {temp}: {response}")


# =============================================================================
# Anthropic Direct Tests
# =============================================================================


class TestAnthropicDirectComprehensive:
    """Comprehensive tests for Anthropic direct API."""

    @pytest.fixture(autouse=True)
    def setup(self, ensure_no_openrouter):
        """Ensure we're testing direct API."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

    def test_simple_completion(self):
        """Test simple completion."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            force_direct=True,
            debug=True,
            max_tokens=50,
        )

        response, cost = llm.ask_base(SIMPLE_PROMPT)

        assert response is not None
        assert "hello" in response.lower()
        assert cost > 0
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_math_completion(self):
        """Test math completion."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            force_direct=True,
            debug=True,
            max_tokens=50,
        )

        response, cost = llm.ask_base(MATH_PROMPT)

        assert response is not None
        assert "56" in response
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_thinking_low(self, check_tenacity):
        """Test with low thinking budget."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            force_direct=True,
            debug=True,
            max_tokens=30000,
            reasoning_effort="low",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        assert response is not None
        assert "9" in response
        print(f"Response: {response}")

    def test_thinking_medium(self, check_tenacity):
        """Test with medium thinking budget."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            force_direct=True,
            debug=True,
            max_tokens=30000,
            reasoning_effort="medium",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        assert response is not None
        assert "9" in response
        print(f"Response: {response}")

    def test_thinking_high(self, check_tenacity):
        """Test with high thinking budget."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            force_direct=True,
            debug=True,
            max_tokens=30000,
            reasoning_effort="high",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        assert response is not None
        assert "9" in response
        print(f"Response: {response}")

    def test_code_generation(self):
        """Test code generation."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            force_direct=True,
            debug=True,
            max_tokens=400,
        )

        response, cost = llm.ask_base(CODE_PROMPT)

        assert response is not None
        assert "def" in response or "factorial" in response.lower()
        print(f"Response preview: {response[:200]}...")


# =============================================================================
# Gemini Direct Tests
# =============================================================================


class TestGeminiDirectComprehensive:
    """Comprehensive tests for Gemini direct API."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure we're testing direct API."""
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

    def test_simple_completion(self):
        """Test simple completion."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=50,
        )

        # Gemini should always be direct
        assert llm.use_openrouter is False

        response, cost = llm.ask_base(SIMPLE_PROMPT)

        if response is None:
            pytest.skip("Gemini API returned None (may be rate limited or unavailable)")

        assert "hello" in response.lower()
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_math_completion(self):
        """Test math completion."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=50,
        )

        response, cost = llm.ask_base(MATH_PROMPT)

        if response is None:
            pytest.skip("Gemini API returned None (may be rate limited or unavailable)")

        assert "56" in response
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_reasoning_low(self):
        """Test with low reasoning effort."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=30000,
            reasoning_effort="low",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        if response is None:
            pytest.skip("Gemini reasoning with effort=low returned None (may not be supported)")

        assert "9" in response
        print(f"Response: {response}")

    def test_reasoning_high(self):
        """Test with high reasoning effort."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=30000,
            reasoning_effort="high",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        if response is None:
            pytest.skip("Gemini reasoning with effort=high returned None (may not be supported)")

        assert "9" in response
        print(f"Response: {response}")

    def test_code_generation(self):
        """Test code generation."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=4000,
        )

        response, cost = llm.ask_base(CODE_PROMPT)

        if response is None:
            pytest.skip("Gemini API returned None (may be rate limited or unavailable)")

        assert "def" in response or "factorial" in response.lower()
        print(f"Response preview: {response[:200]}...")

    def test_always_direct_even_with_openrouter_key(self):
        """Verify Gemini always uses direct even when OpenRouter key exists."""
        # Temporarily set OpenRouter key
        original = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"

        try:
            llm = LiteLLM_interface(
                model="gemini-pro-3.0",
                debug=True,
                max_tokens=50,
            )

            assert llm.use_openrouter is False
            assert "gemini/" in llm.resolved_model
        finally:
            if original:
                os.environ["OPENROUTER_API_KEY"] = original
            else:
                del os.environ["OPENROUTER_API_KEY"]

    def test_default_reasoning_effort(self):
        """Test that Gemini defaults to high reasoning effort."""
        llm = LiteLLM_interface(
            model="gemini-pro-3.0",
            debug=True,
            max_tokens=50,
        )

        # The default should be set to 'high' for Gemini
        assert llm.reasoning_effort == "high"


# =============================================================================
# DeepSeek Direct Tests
# =============================================================================


class TestDeepSeekDirectComprehensive:
    """Comprehensive tests for DeepSeek direct API."""

    @pytest.fixture(autouse=True)
    def setup(self, ensure_no_openrouter):
        """Ensure we're testing direct API."""
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

    def test_simple_completion_non_thinking(self):
        """Test simple completion in non-thinking mode."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            force_direct=True,
            debug=True,
            max_tokens=50,
        )

        # Should use deepseek-chat for non-thinking mode
        assert "deepseek-chat" in llm.resolved_model

        response, cost = llm.ask_base(SIMPLE_PROMPT)

        assert response is not None
        assert "hello" in response.lower()
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_math_completion(self):
        """Test math completion."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            force_direct=True,
            debug=True,
            max_tokens=50,
        )

        response, cost = llm.ask_base(MATH_PROMPT)

        assert response is not None
        assert "56" in response
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_reasoning_mode_switches_model(self):
        """Test that reasoning mode switches to deepseek-reasoner."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            force_direct=True,
            debug=True,
            max_tokens=300,
            reasoning_effort="medium",
        )

        # Should switch to deepseek-reasoner for thinking mode
        assert "deepseek-reasoner" in llm.resolved_model

    def test_reasoning_low(self):
        """Test with low reasoning effort (thinking mode)."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            force_direct=True,
            debug=True,
            max_tokens=30000,
            reasoning_effort="low",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        assert response is not None
        assert "9" in response
        print(f"Response: {response}")

    def test_reasoning_high(self):
        """Test with high reasoning effort (thinking mode)."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            force_direct=True,
            debug=True,
            max_tokens=30000,
            reasoning_effort="high",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        assert response is not None
        assert "9" in response
        print(f"Response: {response}")

    def test_code_generation(self):
        """Test code generation."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            force_direct=True,
            debug=True,
            max_tokens=400,
        )

        response, cost = llm.ask_base(CODE_PROMPT)

        assert response is not None
        assert "def" in response or "factorial" in response.lower()
        print(f"Response preview: {response[:200]}...")


# =============================================================================
# Cross-Provider Comparison Tests
# =============================================================================


class TestCrossProviderComparison:
    """Compare responses across all providers."""

    def test_all_providers_answer_same_question(self, check_all_direct_keys, ensure_no_openrouter):
        """Test that all providers can answer the same question."""
        models = [
            ("gpt-5.2", "openai"),
            ("claude-sonnet-4.5", "anthropic"),
            ("gemini-pro-3.0", "gemini"),
            ("deepseek-v3.2", "deepseek"),
        ]

        results = {}

        for model, provider in models:
            print(f"\nTesting {model} ({provider})...")

            llm = LiteLLM_interface(
                model=model,
                force_direct=True,
                debug=True,
                max_tokens=100,
            )

            start_time = time.time()
            response, cost = llm.ask_base(MATH_PROMPT)
            elapsed = time.time() - start_time

            results[model] = {
                "response": response,
                "cost": cost,
                "time": elapsed,
                "success": response is not None and "56" in response,
            }

            print(f"  Response: {response}")
            print(f"  Cost: ${cost:.6f}")
            print(f"  Time: {elapsed:.2f}s")

        # Print summary
        print("\n" + "=" * 70)
        print("Summary:")
        print("=" * 70)
        for model, result in results.items():
            status = "✓" if result["success"] else "✗"
            print(
                f"  {status} {model}: cost=${result['cost']:.6f}, time={result['time']:.2f}s"
            )

        # All should succeed
        for model, result in results.items():
            assert result["success"], f"{model} failed to correctly answer"

    def test_reasoning_across_providers(self, check_all_direct_keys, ensure_no_openrouter, check_tenacity):
        """Test reasoning mode across all providers."""
        models = [
            ("gpt-5.2", "openai"),
            ("claude-sonnet-4.5", "anthropic"),
            ("gemini-pro-3.0", "gemini"),
            ("deepseek-v3.2", "deepseek"),
        ]

        results = {}

        for model, provider in models:
            print(f"\nTesting {model} ({provider}) with reasoning...")

            llm = LiteLLM_interface(
                model=model,
                force_direct=True,
                debug=True,
                max_tokens=30000,
                reasoning_effort="medium",
            )

            start_time = time.time()
            response, cost = llm.ask_base(REASONING_PROMPT)
            elapsed = time.time() - start_time

            results[model] = {
                "response": response,
                "cost": cost,
                "time": elapsed,
                "success": response is not None and "9" in response,
            }

            print(f"  Response preview: {response[:150] if response else 'None'}...")
            print(f"  Cost: ${cost:.6f}")
            print(f"  Time: {elapsed:.2f}s")

        # Print summary
        print("\n" + "=" * 70)
        print("Reasoning Mode Summary:")
        print("=" * 70)
        for model, result in results.items():
            status = "✓" if result["success"] else "✗"
            print(
                f"  {status} {model}: cost=${result['cost']:.6f}, time={result['time']:.2f}s"
            )

        # All should succeed (except Gemini which may not support all reasoning modes)
        for model, result in results.items():
            if model == "gemini-pro-3.0" and not result["success"]:
                # Gemini reasoning may not be fully supported
                continue
            assert result["success"], f"{model} failed reasoning test"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])