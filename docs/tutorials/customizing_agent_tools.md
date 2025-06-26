# Tutorial: Customizing the Web Search Tool

aclarai's agents can use "tools" to gather more information before making a decision. One of the most powerful is the web search tool, which allows agents to access up-to-date information from the internet.

This tutorial will show you how to replace the default web search provider (Tavily) with a custom one, using the [Brave Search API](https://brave.com/search/api/) as an example.

## What You'll Learn

- How to create a custom tool class compatible with aclarai's `ToolFactory`.
- How to register your new tool so the factory can find it.
- How to configure aclarai to use your custom tool.

## Prerequisites

- A working aclarai setup.
- Basic Python knowledge.
- Access to the `shared/aclarai_shared/tools/` directory in your aclarai installation.

---

## Step 1: Create Your Custom Provider File

First, create a Python file for your custom search provider. All web search providers should inherit from `WebSearchProvider`.

**Create a new file: `shared/aclarai_shared/tools/implementations/web_search/brave.py`**

```python
# shared/aclarai_shared/tools/implementations/web_search/brave.py

import logging
import os
from typing import Any, Dict, List, Optional

# Import the base classes from the web_search module
from .base import WebSearchProvider, WebSearchResult

logger = logging.getLogger(__name__)

# You would typically use the official SDK for the service. We'll simulate it here.
class BraveSearchAPI:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Brave Search API key is required.")
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list:
        """Simulates a call to the Brave Search API."""
        logger.info(f"--- Faking a Brave Search for: '{query}' ---")
        return [
            {"title": f"Brave Result 1 for {query}", "url": "https://example.com/brave1", "snippet": "This is the first Brave search result..."},
            {"title": f"Brave Result 2 for {query}", "url": "https://example.com/brave2", "snippet": "This is the second Brave search result..."},
        ]

class BraveSearchProvider(WebSearchProvider):
    """A provider for performing web searches using the Brave Search API."""

    def __init__(self, api_key: str):
        self._api = BraveSearchAPI(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """Executes a search and returns standardized results."""
        try:
            raw_results = self._api.search(query=query, max_results=max_results)
            return [
                WebSearchResult(
                    title=r.get("title", "Untitled"),
                    url=r.get("url", ""),
                    snippet=r.get("snippet", "")
                )
                for r in raw_results
            ]
        except Exception as e:
            logger.error(f"Brave Search failed: {e}", extra={"query": query})
            return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional["BraveSearchProvider"]:
        """Creates an instance from the system configuration."""
        api_key_var = config.get("api_key_env_var")
        if not api_key_var:
            logger.error("Brave provider requires 'api_key_env_var' in config.")
            return None

        api_key = os.getenv(api_key_var)
        if not api_key:
            logger.warning(
                f"Web search enabled with Brave provider, but environment variable '{api_key_var}' is not set."
            )
            return None

        return cls(api_key=api_key)

```

## Step 2: Register Your New Provider

The `ToolFactory` needs to know that the string `"brave"` maps to your new `BraveSearchProvider` class. You do this by registering it.

**Edit `shared/aclarai_shared/tools/implementations/web_search/provider.py`:**

```python
# shared/aclarai_shared/tools/implementations/web_search/provider.py

import logging
from typing import Any, Dict, Optional, Type

from .base import WebSearchProvider
from .tavily import TavilySearchProvider
# 1. Import your new provider class
from .brave import BraveSearchProvider

logger = logging.getLogger(__name__)

# 2. Add your provider to the registry dictionary
#    The key ("brave") is what you'll use in the config file.
PROVIDERS: Dict[str, Type[WebSearchProvider]] = {
    "tavily": TavilySearchProvider,
    "brave": BraveSearchProvider,
}

# ... (the rest of the file remains the same) ...
```

## Step 3: Add Your API Key to the Environment

Your tool needs an API key to authenticate. Add the new environment variable to your project's `.env` file.

**Edit your `.env` file:**

```bash
# .env

# ... other variables ...

# Add your new API key
BRAVE_SEARCH_API_KEY="your_brave_search_api_key_here"
```

## Step 4: Configure aclarai to Use Your New Tool

Finally, tell the `ToolFactory` to use your Brave provider by default.

**Edit `settings/aclarai.config.yaml`:**

Find the `tools` section (or add it) and update the `web_search` settings.

```yaml
# settings/aclarai.config.yaml

tools:
  web_search:
    enabled: true # Make sure to enable it!
    # This is the switch. Change "tavily" to "brave".
    provider: "brave"
    # Tell the factory which environment variable to look for.
    api_key_env_var: "BRAVE_SEARCH_API_KEY"
    max_results: 3
```

## Conclusion

That's it! When you restart aclarai, the `ToolFactory` will:
1.  Read the `tools.web_search` configuration.
2.  See the provider is set to `"brave"`.
3.  Look up `"brave"` in its registry and find your `BraveSearchProvider` class.
4.  Call `BraveSearchProvider.from_config()`, which reads the `BRAVE_SEARCH_API_KEY` from your environment.
5.  Initialize the tool and provide it to any agent that requests it.

This pluggable architecture allows you to tailor aclarai's data-gathering capabilities to your specific needs, whether you're connecting to public APIs or internal, proprietary data sources.