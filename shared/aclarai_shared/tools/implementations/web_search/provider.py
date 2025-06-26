"""Web search provider factory system."""

import logging
from typing import Any, Dict, Optional, Type

from .base import WebSearchProvider
from .tavily import TavilySearchProvider

logger = logging.getLogger(__name__)


# Registry of available providers
PROVIDERS: Dict[str, Type[WebSearchProvider]] = {"tavily": TavilySearchProvider}


def create_provider(
    provider_name: str, config: Dict[str, Any]
) -> Optional[WebSearchProvider]:
    """Create a web search provider instance.

    Args:
        provider_name: Name of the provider to create
        config: Provider-specific configuration

    Returns:
        Configured provider instance or None if creation fails
    """
    provider_cls = PROVIDERS.get(provider_name.lower())
    if not provider_cls:
        logger.error(
            f"Unknown web search provider: {provider_name}",
            extra={"provider": provider_name, "available": list(PROVIDERS.keys())},
        )
        return None

    try:
        return provider_cls.from_config(config)
    except Exception as e:
        logger.error(
            f"Failed to create provider {provider_name}: {str(e)}",
            extra={"provider": provider_name, "config": config},
        )
        return None


def register_provider(name: str, provider_cls: Type[WebSearchProvider]) -> None:
    """Register a new web search provider.

    Args:
        name: Name to register the provider under
        provider_cls: Provider class to register
    """
    PROVIDERS[name.lower()] = provider_cls
    logger.info(f"Registered web search provider: {name}")
