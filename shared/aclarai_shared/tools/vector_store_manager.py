from abc import ABC, abstractmethod
import copy
from typing import Dict, Optional

# Re-export VectorStore from its canonical location. This provides a single
# point of import for components that need it and establishes the type
# contract for the manager.
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.embedding.storage import aclaraiVectorStore
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


class aclaraiVectorStoreManager(VectorStoreManager):
    """
    Concrete implementation of VectorStoreManager for the aclarai ecosystem.
    It creates and caches `aclaraiVectorStore` instances for different collections.
    """

    def __init__(self, config: aclaraiConfig):
        """
        Initializes the vector store manager.
        Args:
            config: The main aclarai configuration object.
        """
        self.config = config
        self._stores: Dict[str, VectorStore] = {}

    def get_store(self, name: str) -> Optional[VectorStore]:
        """
        Retrieves a vector store for a specific collection, creating it if necessary.
        Args:
            name: The name of the collection (e.g., "utterances", "concepts").
        Returns:
            An instance of `aclaraiVectorStore` configured for the given collection.
        """
        if name in self._stores:
            return self._stores[name]

        try:
            # Create a copy of the config to avoid modifying the original
            store_config = copy.deepcopy(self.config)
            # Set the collection name for this specific store instance
            store_config.embedding.collection_name = name

            # Instantiate the concrete vector store with the specific config
            store = aclaraiVectorStore(config=store_config)
            self._stores[name] = store
            return store
        except Exception as e:
            # Log the error appropriately
            print(f"Error creating vector store for collection '{name}': {e}")
            return None


__all__ = ["VectorStore", "VectorStoreManager", "aclaraiVectorStoreManager"]
