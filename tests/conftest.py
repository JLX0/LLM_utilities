"""
Pytest configuration and shared fixtures for LLM_utils tests.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark all tests in this directory as integration tests
        item.add_marker(pytest.mark.integration)

        # Mark tests that make multiple API calls as slow
        if "comprehensive" in item.nodeid.lower() or "comparison" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def api_keys_status():
    """Report status of all API keys at session start."""
    keys = {
        "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY")),
    }

    print("\n" + "=" * 60)
    print("API Keys Status:")
    print("=" * 60)
    for key, available in keys.items():
        status = "✓ Available" if available else "✗ Not set"
        print(f"  {key}: {status}")
    print("=" * 60 + "\n")

    return keys


@pytest.fixture(scope="session", autouse=True)
def session_setup(api_keys_status):
    """Setup that runs once at the start of the test session."""
    # Check if at least one key is available
    if not any(api_keys_status.values()):
        pytest.skip("No API keys available. Please set at least one API key.")

    yield

    print("\n" + "=" * 60)
    print("Test session completed.")
    print("=" * 60)
