"""Tests for the tool factory implementation."""

import os
from unittest import mock

import pytest
from llama_index.vector_stores.types import VectorStore

from aclarai_shared.tools import ToolFactory


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    return {
        'tools': {
            'neo4j': {
                'enabled': True,
                'uri': 'bolt://localhost:7687',
                'username': 'neo4j',
                'password': 'password'
            },
            'vector_search': {
                'enabled': True,
                'vector_stores': {
                    'test_collection': mock.Mock(spec=VectorStore)
                },
                'similarity_threshold': 0.7
            },
            'web_search': {
                'enabled': True,
                'provider': 'tavily',
                'api_key_env_var': 'TAVILY_API_KEY',
                'max_results': 5
            }
        }
    }


def test_factory_initialization(mock_config):
    """Test that the factory can be initialized with config."""
    factory = ToolFactory(mock_config)
    assert factory._config == mock_config['tools']


def test_get_tools_for_agent_with_all_enabled(mock_config):
    """Test getting tools when all are enabled and configured."""
    # Setup
    os.environ['TAVILY_API_KEY'] = 'test_key'
    factory = ToolFactory(mock_config)
    
    # Execute
    tools = factory.get_tools_for_agent('test_agent')
    
    # Verify
    assert len(tools) == 3
    tool_names = {tool.metadata.name for tool in tools}
    assert tool_names == {'neo4j_query', 'vector_search', 'web_search'}


def test_get_tools_for_agent_with_web_search_disabled(mock_config):
    """Test getting tools when web search is disabled."""
    # Modify config
    config = mock_config.copy()
    config['tools']['web_search']['enabled'] = False
    
    # Setup and execute
    factory = ToolFactory(config)
    tools = factory.get_tools_for_agent('test_agent')
    
    # Verify
    assert len(tools) == 2
    tool_names = {tool.metadata.name for tool in tools}
    assert tool_names == {'neo4j_query', 'vector_search'}


def test_get_tools_for_agent_with_missing_api_key(mock_config):
    """Test that web search tool is not created when API key is missing."""
    # Setup (ensure env var is not set)
    os.environ.pop('TAVILY_API_KEY', None)
    factory = ToolFactory(mock_config)
    
    # Execute
    tools = factory.get_tools_for_agent('test_agent')
    
    # Verify
    assert len(tools) == 2
    tool_names = {tool.metadata.name for tool in tools}
    assert 'web_search' not in tool_names


def test_tool_caching(mock_config):
    """Test that tools are properly cached."""
    factory = ToolFactory(mock_config)
    
    # Get tools twice
    tools1 = factory.get_tools_for_agent('test_agent')
    tools2 = factory.get_tools_for_agent('test_agent')
    
    # Verify same instances are returned
    assert tools1 is tools2


def test_invalid_web_search_provider(mock_config):
    """Test handling of invalid web search provider."""
    # Modify config with invalid provider
    config = mock_config.copy()
    config['tools']['web_search']['provider'] = 'invalid_provider'
    
    # Setup and execute
    factory = ToolFactory(config)
    tools = factory.get_tools_for_agent('test_agent')
    
    # Verify web search tool was not created
    tool_names = {tool.metadata.name for tool in tools}
    assert 'web_search' not in tool_names
