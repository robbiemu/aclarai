"""Base classes and interfaces for aclarai tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from llama_index.core.tools import BaseTool, ToolMetadata
from pydantic import BaseModel


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for tool providers that can be loaded by the factory."""
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional['ToolProvider']:
        """Create a new instance of the tool provider from config.
        
        Args:
            config: Tool-specific configuration dictionary
            
        Returns:
            An initialized provider instance or None if the provider
            cannot be initialized with the given config
        """
        ...


class AclaraiBaseTool(BaseTool, ABC):
    """Base class for all aclarai tools.
    
    Extends LlamaIndex BaseTool with common functionality needed across
    aclarai tools.
    """
    
    def __init__(self) -> None:
        """Initialize the tool with its metadata."""
        super().__init__(metadata=self.metadata)
    
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Get the tool's metadata.
        
        This property must be implemented by all concrete tools to provide
        their name, description, and other metadata needed by agents.
        """
        pass
    
    @abstractmethod
    def _call(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool's primary functionality.
        
        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool
            
        Returns:
            A string response containing the tool's output or error message
        """
        pass
