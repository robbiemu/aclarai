from abc import ABC, abstractmethod
from typing import Optional

# Re-export VectorStore from its canonical location. This provides a single
# point of import for components that need it and establishes the type
# contract for the manager.
from llama_index.core.vector_stores.types import VectorStore


class VectorStoreManager(ABC):
    """
    Abstract base class for a manager that provides access to named VectorStore instances.

    This class defines the interface that the ToolFactory expects for retrieving
    vector stores. This decouples the factory from the concrete implementation of
    how those stores are created or managed, which might live in a different
    service or package.
    """

    @abstractmethod
    def get_store(self, name: str) -> Optional[VectorStore]:
        """
        Retrieve a named VectorStore instance.

        Args:
            name: The name of the vector store collection (e.g., "utterances").

        Returns:
            An initialized VectorStore instance, or None if the store
            with the given name is not found or could not be initialized.
        """
        pass


__all__ = ["VectorStore", "VectorStoreManager"]
