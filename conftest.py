"""Pytest configuration for aclarai-ui tests."""

import os
import sys

import pytest


def pytest_configure(config: pytest.Config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring real services"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list):
    """Skip integration tests unless -m integration is specified."""
    if config.getoption("-m") == "integration":
        # Integration mode - run only integration tests
        return
    # Unit test mode - skip integration tests by default
    skip_integration = pytest.mark.skip(
        reason="integration tests require -m integration flag"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# Add the aclarai_ui module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
# Only configure Playwright fixtures if it's available
if PLAYWRIGHT_AVAILABLE:

    @pytest.fixture(scope="session")
    def browser():
        """Provide a browser instance for testing."""
        with sync_playwright() as p:
            # Use chromium with minimal setup
            browser = p.chromium.launch(headless=True)
            yield browser
            browser.close()

    @pytest.fixture
    def page(browser: None):
        """Provide a page instance for testing."""
        if browser is None:
            pytest.skip("Browser not available")
        context = browser.new_context()
        page = context.new_page()
        yield page
        context.close()
else:

    @pytest.fixture
    def browser():
        """Dummy browser fixture when Playwright is not available."""
        pytest.skip("Playwright not available")

    @pytest.fixture
    def page():
        """Dummy page fixture when Playwright is not available."""
        pytest.skip("Playwright not available")
