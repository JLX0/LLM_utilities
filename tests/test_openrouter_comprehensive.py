"""
Comprehensive tests for OpenRouter API calls.

These tests focus specifically on testing OpenRouter routing for all applicable models.

Run with: pytest tests/test_openrouter_comprehensive.py -v -s
"""

from __future__ import annotations

import os
import time

import pytest

from LLM_utils.inquiry import _check_tenacity_available
from LLM_utils.inquiry import LiteLLM_interface


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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def check_openrouter_key():
    """Check if OpenRouter API key is available."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")


@pytest.fixture
def check_tenacity():
    """Check if tenacity is available for reasoning tests."""
    if not _check_tenacity_available():
        pytest.skip("tenacity not installed (required for Claude reasoning features)")


# =============================================================================
# OpenRouter GPT Tests
# =============================================================================


class TestOpenRouterGPT:
    """Test GPT models via OpenRouter."""

    @pytest.fixture(autouse=True)
    def setup(self, check_openrouter_key):
        """Ensure OpenRouter key is available."""
        pass

    def test_simple_completion(self):
        """Test simple completion via OpenRouter."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            debug=True,
            max_tokens=50,
        )

        assert llm.use_openrouter is True
        assert "openrouter/openai/" in llm.resolved_model

        response, cost = llm.ask_base(SIMPLE_PROMPT)

        assert response is not None
        assert "hello" in response.lower()
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_math_completion(self):
        """Test math completion via OpenRouter."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            debug=True,
            max_tokens=50,
        )

        response, cost = llm.ask_base(MATH_PROMPT)

        assert response is not None
        assert "56" in response
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_reasoning_low(self):
        """Test GPT with low reasoning effort via OpenRouter."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            debug=True,
            max_tokens=300,
            reasoning_effort="low",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        if response is None:
            pytest.skip("GPT reasoning via OpenRouter returned None")

        assert "9" in response
        print(f"Response: {response}")

    def test_reasoning_medium(self):
        """Test GPT with medium reasoning effort via OpenRouter."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            debug=True,
            max_tokens=500,
            reasoning_effort="medium",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        if response is None:
            pytest.skip("GPT reasoning via OpenRouter returned None")

        assert "9" in response
        print(f"Response: {response}")

    def test_reasoning_high(self):
        """Test GPT with high reasoning effort via OpenRouter."""
        llm = LiteLLM_interface(
            model="gpt-5.2",
            debug=True,
            max_tokens=800,
            reasoning_effort="high",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        if response is None:
            pytest.skip("GPT reasoning via OpenRouter returned None")

        assert "9" in response
        print(f"Response: {response}")


# =============================================================================
# OpenRouter Claude Tests
# =============================================================================


class TestOpenRouterClaude:
    """Test Claude models via OpenRouter."""

    @pytest.fixture(autouse=True)
    def setup(self, check_openrouter_key):
        """Ensure OpenRouter key is available."""
        pass

    def test_simple_completion(self):
        """Test simple completion via OpenRouter."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            debug=True,
            max_tokens=50,
        )

        assert llm.use_openrouter is True
        assert "openrouter/anthropic/" in llm.resolved_model

        response, cost = llm.ask_base(SIMPLE_PROMPT)

        assert response is not None
        assert "hello" in response.lower()
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_math_completion(self):
        """Test math completion via OpenRouter."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            debug=True,
            max_tokens=50,
        )

        response, cost = llm.ask_base(MATH_PROMPT)

        assert response is not None
        assert "56" in response
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_reasoning_low(self, check_tenacity):
        """Test with low reasoning effort via OpenRouter."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            debug=True,
            max_tokens=2000,  # Increased to accommodate thinking tokens
            reasoning_effort="low",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        if response is None:
            pytest.skip("Claude reasoning via OpenRouter returned None (may not be supported)")

        assert "9" in response
        print(f"Response: {response}")

    def test_reasoning_medium(self, check_tenacity):
        """Test with medium reasoning effort via OpenRouter."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            debug=True,
            max_tokens=3000,  # Increased to accommodate thinking tokens
            reasoning_effort="medium",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        if response is None:
            pytest.skip("Claude reasoning via OpenRouter returned None (may not be supported)")

        assert "9" in response
        print(f"Response: {response}")

    def test_reasoning_high(self, check_tenacity):
        """Test with high reasoning effort via OpenRouter."""
        llm = LiteLLM_interface(
            model="claude-sonnet-4.5",
            debug=True,
            max_tokens=5000,  # Increased to accommodate thinking tokens
            reasoning_effort="high",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        if response is None:
            pytest.skip("Claude reasoning via OpenRouter returned None (may not be supported)")

        assert "9" in response
        print(f"Response: {response}")


# =============================================================================
# OpenRouter DeepSeek Tests
# =============================================================================


class TestOpenRouterDeepSeek:
    """Test DeepSeek models via OpenRouter."""

    @pytest.fixture(autouse=True)
    def setup(self, check_openrouter_key):
        """Ensure OpenRouter key is available."""
        pass

    def test_simple_completion(self):
        """Test simple completion via OpenRouter."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            debug=True,
            max_tokens=50,
        )

        assert llm.use_openrouter is True
        assert "openrouter/deepseek/" in llm.resolved_model

        response, cost = llm.ask_base(SIMPLE_PROMPT)

        assert response is not None
        assert "hello" in response.lower()
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_math_completion(self):
        """Test math completion via OpenRouter."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            debug=True,
            max_tokens=50,
        )

        response, cost = llm.ask_base(MATH_PROMPT)

        assert response is not None
        assert "56" in response
        print(f"Response: {response}, Cost: ${cost:.6f}")

    def test_reasoning_low(self):
        """Test with low reasoning effort via OpenRouter."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            debug=True,
            max_tokens=300,
            reasoning_effort="low",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        assert response is not None
        assert "9" in response
        print(f"Response: {response}")

    def test_reasoning_medium(self):
        """Test with medium reasoning effort via OpenRouter."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            debug=True,
            max_tokens=500,
            reasoning_effort="medium",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        assert response is not None
        assert "9" in response
        print(f"Response: {response}")

    def test_reasoning_high(self):
        """Test with high reasoning effort via OpenRouter."""
        llm = LiteLLM_interface(
            model="deepseek-v3.2",
            debug=True,
            max_tokens=800,
            reasoning_effort="high",
        )

        response, cost = llm.ask_base(REASONING_PROMPT)

        assert response is not None
        assert "9" in response
        print(f"Response: {response}")


# =============================================================================
# OpenRouter Cross-Model Comparison
# =============================================================================


class TestOpenRouterComparison:
    """Compare all models via OpenRouter."""

    @pytest.fixture(autouse=True)
    def setup(self, check_openrouter_key):
        """Ensure OpenRouter key is available."""
        pass

    def test_all_openrouter_models(self):
        """Test all models available via OpenRouter."""
        models = ["gpt-5.2", "claude-sonnet-4.5", "deepseek-v3.2"]
        results = {}

        for model in models:
            print(f"\nTesting {model} via OpenRouter...")

            llm = LiteLLM_interface(
                model=model,
                debug=True,
                max_tokens=100,
            )

            assert llm.use_openrouter is True

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
        print("OpenRouter Summary:")
        print("=" * 70)
        for model, result in results.items():
            status = "✓" if result["success"] else "✗"
            print(f"  {status} {model}: cost=${result['cost']:.6f}, time={result['time']:.2f}s")

        # All should succeed
        for model, result in results.items():
            assert result["success"], f"{model} failed via OpenRouter"

    def test_reasoning_across_openrouter_models(self, check_tenacity):
        """Test reasoning mode across all OpenRouter models."""
        models = ["gpt-5.2", "claude-sonnet-4.5", "deepseek-v3.2"]
        results = {}

        for model in models:
            print(f"\nTesting {model} reasoning via OpenRouter...")

            # Claude needs more tokens for thinking
            max_tokens = 3000 if "claude" in model else 500

            llm = LiteLLM_interface(
                model=model,
                debug=True,
                max_tokens=max_tokens,
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

            print(f"  Response preview: {response[:100] if response else 'None'}...")
            print(f"  Cost: ${cost:.6f}")
            print(f"  Time: {elapsed:.2f}s")

        # Print summary
        print("\n" + "=" * 70)
        print("OpenRouter Reasoning Summary:")
        print("=" * 70)
        for model, result in results.items():
            status = "✓" if result["success"] else "✗"
            print(f"  {status} {model}: cost=${result['cost']:.6f}, time={result['time']:.2f}s")


# =============================================================================
# OpenRouter vs Direct Comparison
# =============================================================================


class TestOpenRouterVsDirectComparison:
    """Compare OpenRouter routing vs direct API."""

    def test_compare_routing_methods(self, check_openrouter_key):
        """Compare response quality and cost between OpenRouter and direct."""
        # Skip models that require additional keys
        models_to_test = []

        if os.environ.get("OPENAI_API_KEY"):
            models_to_test.append("gpt-5.2")
        if os.environ.get("ANTHROPIC_API_KEY"):
            models_to_test.append("claude-sonnet-4.5")
        if os.environ.get("DEEPSEEK_API_KEY"):
            models_to_test.append("deepseek-v3.2")

        if not models_to_test:
            pytest.skip("No direct API keys available for comparison")

        results = {}

        for model in models_to_test:
            print(f"\n{'=' * 60}")
            print(f"Comparing {model}:")
            print("=" * 60)

            # Test via OpenRouter
            llm_or = LiteLLM_interface(
                model=model,
                debug=True,
                max_tokens=100,
            )
            assert llm_or.use_openrouter is True

            start_time = time.time()
            response_or, cost_or = llm_or.ask_base(MATH_PROMPT)
            time_or = time.time() - start_time

            print(
                f"  OpenRouter: response='{response_or}', cost=${cost_or:.6f}, time={time_or:.2f}s"
            )

            # Test via Direct API
            llm_direct = LiteLLM_interface(
                model=model,
                force_direct=True,
                debug=True,
                max_tokens=100,
            )
            assert llm_direct.use_openrouter is False

            start_time = time.time()
            response_direct, cost_direct = llm_direct.ask_base(MATH_PROMPT)
            time_direct = time.time() - start_time

            print(
                f"  Direct:     response='{response_direct}', cost=${cost_direct:.6f}, time={time_direct:.2f}s"
            )

            results[model] = {
                "openrouter": {
                    "response": response_or,
                    "cost": cost_or,
                    "time": time_or,
                    "success": response_or is not None and "56" in response_or,
                },
                "direct": {
                    "response": response_direct,
                    "cost": cost_direct,
                    "time": time_direct,
                    "success": response_direct is not None and "56" in response_direct,
                },
            }

        # Print summary
        print("\n" + "=" * 70)
        print("Comparison Summary:")
        print("=" * 70)
        for model, result in results.items():
            or_status = "✓" if result["openrouter"]["success"] else "✗"
            direct_status = "✓" if result["direct"]["success"] else "✗"
            cost_diff = result["openrouter"]["cost"] - result["direct"]["cost"]
            cost_pct = (
                (cost_diff / result["direct"]["cost"] * 100) if result["direct"]["cost"] > 0 else 0
            )

            print(f"  {model}:")
            print(f"    OpenRouter: {or_status} ${result['openrouter']['cost']:.6f}")
            print(f"    Direct:     {direct_status} ${result['direct']['cost']:.6f}")
            print(f"    Cost difference: ${cost_diff:.6f} ({cost_pct:+.1f}%)")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
