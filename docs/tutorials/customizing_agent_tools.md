# Tutorial: Customizing Agent Tools

aclarai's evaluation agents (like the ones that check claims for entailment and coverage) can use "tools" to gather more information before making a decision. One of the most powerful tools is a web search tool.

By default, aclarai comes with a basic web search tool. However, you can easily replace it with your own custom tool—for example, to use a specific search API like Brave Search, or to connect to a private, internal knowledge base.

This tutorial will show you how to create and plug in your own custom search tool.

## What You'll Learn

-   How to create a custom tool class that is compatible with aclarai's agents.
-   How to configure aclarai to use your custom tool instead of the default one.

## Prerequisites

-   A working aclarai setup.
-   Basic Python knowledge.
-   Access to the `shared/aclarai_tools/` directory in your aclarai installation.

---

## Step 1: Create Your Custom Tool File

First, you need to create a Python file for your custom tool. We recommend placing it in the `shared/aclarai_tools/` directory to make it accessible to all services.

Let's create a tool for the [Brave Search API](https://brave.com/search/api/).

**Create a new file: `shared/aclarai_tools/brave_search.py`**

```python
# shared/aclarai_tools/brave_search.py

import os
from llama_index.core.tools import BaseTool, ToolMetadata

# You would typically use the official SDK for the service you're integrating.
# For this example, we'll simulate it.
class BraveSearchAPI:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Brave Search API key is required.")
        self.api_key = api_key

    def search(self, query: str, count: int = 3) -> list:
        """Simulates a call to the Brave Search API."""
        print(f"--- Faking a Brave Search for: '{query}' ---")
        return [
            {"title": f"Result 1 for {query}", "snippet": "This is the first search result..."},
            {"title": f"Result 2 for {query}", "snippet": "This is the second search result..."},
            {"title": f"Result 3 for {query}", "snippet": "This is the third search result..."},
        ]

class BraveSearchTool(BaseTool):
    """A tool for performing web searches using the Brave Search API."""

    def __init__(self, api_key: str):
        self._api = BraveSearchAPI(api_key=api_key)

    @property
    def metadata(self) -> ToolMetadata:
        """
        Provides metadata for the agent.
        
        IMPORTANT: The 'name' should be generic like "web_search". This allows the
        agent's prompts (which are written to use a tool named "web_search") to
        work with any search provider you plug in.
        """
        return ToolMetadata(
            name="web_search",
            description="Performs a web search using the Brave Search API to find up-to-date information or verify facts."
        )

    def __call__(self, query: str) -> str:
        """The main entry point for the tool when called by an agent."""
        try:
            results = self._api.search(query)
            # Format the results into a single string for the LLM to read.
            formatted_results = "\n".join(
                [f"Title: {r['title']}\nSnippet: {r['snippet']}" for r in results]
            )
            return formatted_results if formatted_results else "No results found."
        except Exception as e:
            return f"Error performing search: {e}"

```

## Step 2: Add Your API Key

For your tool to work, it needs an API key. Add it to your `.env` file in the project root.

**Edit your `.env` file:**

```bash
# .env

# ... other variables ...

# Add your new API key
BRAVE_SEARCH_API_KEY="your_brave_search_api_key_here"
```

The aclarai configuration system will automatically load this variable.

## Step 3: Configure aclarai to Use Your New Tool

Now, you just need to tell aclarai to use your `BraveSearchTool` instead of the default one. You do this by editing your configuration file.

**Edit `settings/aclarai.config.yaml`:**

Find or add the `agents` section and specify your new provider.

```yaml
# settings/aclarai.config.yaml

agents:
  evaluation:
    # This is the switch. Change "default_web_search" to "brave_search".
    web_search_provider: "brave_search"
```

That's it! You don't need to change any other code.

## Step 4: How it Works

When an evaluation agent starts, it performs these steps:

1.  It reads the `settings/aclarai.config.yaml` file.
2.  It sees that `agents.evaluation.web_search_provider` is set to `"brave_search"`.
3.  It looks inside its internal logic for the `"brave_search"` key and finds your `BraveSearchTool` class.
4.  It instantiates your tool, passing the API key from your `.env` file.
5.  It provides this tool to the LLM agent.

Now, whenever the agent needs to perform a web search to evaluate a claim, it will automatically use your `BraveSearchTool`.

## Conclusion

By following this pattern, you can create and plug in any number of custom tools to enhance the reasoning capabilities of aclarai's agents. This pluggable architecture allows you to tailor the system to your specific needs and data sources, whether they are public APIs or internal, proprietary systems.