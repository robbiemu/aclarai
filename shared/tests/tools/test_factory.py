"""Tests for the tool factory implementation."""

import os
from unittest import mock

import pytest
from aclarai_shared.tools import ToolFactory
from aclarai_shared.tools.vector_store_manager import VectorStore, VectorStoreManager
from llama_index.core.vector_stores.types import VectorStoreQueryResult


@pytest.fixture
def mock_vector_store_manager():
    """Provide a mock VectorStoreManager for testing."""
    manager = mock.Mock(spec=VectorStoreManager)
    mock_vector_store = mock.Mock(spec=VectorStore)
    mock_vector_store.query.return_value = VectorStoreQueryResult(
        nodes=[], similarities=[]
    )
    manager.get_store.side_effect = lambda name: {
        "utterances": mock_vector_store,
        "concepts": mock_vector_store,
        "test_collection": mock_vector_store,
    }.get(name)
    return manager


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    return {
        "tools": {
            "neo4j": {
                "enabled": True,
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password",
            },
            "vector_search": {
                "enabled": True,
                "collections": ["test_collection"],
                "similarity_threshold": 0.7,
            },
            "web_search": {
                "enabled": True,
                "provider": "tavily",
                "api_key_env_var": "TAVILY_API_KEY",
                "max_results": 5,
            },
        }
    }


def test_factory_initialization(mock_config, mock_vector_store_manager):
    """Test that the factory can be initialized with config."""
    factory = ToolFactory(mock_config, mock_vector_store_manager)
    assert factory._config == mock_config["tools"]
    assert factory.vector_store_manager is mock_vector_store_manager


def test_get_tools_for_agent_with_all_enabled(mock_config, mock_vector_store_manager):
    """Test getting tools when all are enabled and configured."""
    # Setup
    os.environ["TAVILY_API_KEY"] = "test_key"
    factory = ToolFactory(mock_config, mock_vector_store_manager)

    # Execute
    tools = factory.get_tools_for_agent("test_agent")

    # Verify
    assert len(tools) == 3
    tool_names = {tool.metadata.name for tool in tools}
    assert tool_names == {"neo4j_query", "vector_search", "web_search"}


def test_get_tools_for_agent_with_web_search_disabled(
    mock_config, mock_vector_store_manager
):
    """Test getting tools when web search is disabled."""
    # Modify config
    config = mock_config.copy()
    config["tools"]["web_search"]["enabled"] = False

    # Setup and execute
    factory = ToolFactory(config, mock_vector_store_manager)
    tools = factory.get_tools_for_agent("test_agent")

    # Verify
    assert len(tools) == 2
    tool_names = {tool.metadata.name for tool in tools}
    assert tool_names == {"neo4j_query", "vector_search"}


def test_get_tools_for_agent_with_missing_api_key(
    mock_config, mock_vector_store_manager
):
    """Test that web search tool is not created when API key is missing."""
    # Setup (ensure env var is not set)
    os.environ.pop("TAVILY_API_KEY", None)
    factory = ToolFactory(mock_config, mock_vector_store_manager)

    # Execute
    tools = factory.get_tools_for_agent("test_agent")

    # Verify
    assert len(tools) == 2
    tool_names = {tool.metadata.name for tool in tools}
    assert "web_search" not in tool_names


def test_tool_caching(mock_config, mock_vector_store_manager):
    """Test that tools are properly cached."""
    factory = ToolFactory(mock_config, mock_vector_store_manager)

    # Get tools twice
    tools1 = factory.get_tools_for_agent("test_agent")
    tools2 = factory.get_tools_for_agent("test_agent")

    # Verify same instances are returned
    assert tools1 is tools2


def test_invalid_web_search_provider(mock_config, mock_vector_store_manager):
    """Test handling of invalid web search provider."""
    # Modify config with invalid provider
    config = mock_config.copy()
    config["tools"]["web_search"]["provider"] = "invalid_provider"

    # Setup and execute
    factory = ToolFactory(config, mock_vector_store_manager)
    tools = factory.get_tools_for_agent("test_agent")

    # Verify web search tool was not created
    tool_names = {tool.metadata.name for tool in tools}
    assert "web_search" not in tool_names
