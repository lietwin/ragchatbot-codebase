import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

# Add backend directory to path for imports
backend_path = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.dirname(backend_path)  # Go up one level to backend/
sys.path.insert(0, backend_path)

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from tests.fixtures.sample_data import SAMPLE_COURSES, SAMPLE_CHUNKS, SAMPLE_SEARCH_RESULTS


@pytest.fixture
def sample_course():
    """Sample course object for testing"""
    return SAMPLE_COURSES[0]


@pytest.fixture
def sample_courses():
    """List of sample courses for testing"""
    return SAMPLE_COURSES


@pytest.fixture
def sample_chunks():
    """Sample course chunks for testing"""
    return SAMPLE_CHUNKS


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing"""
    mock_store = Mock()
    
    # Mock successful search results
    mock_store.search.return_value = SearchResults(
        documents=["Sample course content about MCP fundamentals"],
        metadata=[{"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 1}],
        distances=[0.1],
        error=None
    )
    
    # Mock course catalog query
    mock_store.course_catalog.query.return_value = {
        'documents': [['MCP: Build Rich-Context AI Apps']],
        'metadatas': [[{
            'title': 'MCP: Build Rich-Context AI Apps',
            'instructor': 'Elie Schoppik',
            'course_link': 'https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps/',
            'lessons_json': '[]',
            'lesson_count': 0
        }]]
    }
    
    return mock_store


@pytest.fixture
def mock_vector_store_empty():
    """Mock VectorStore that returns empty results"""
    mock_store = Mock()
    mock_store.search.return_value = SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )
    return mock_store


@pytest.fixture
def mock_vector_store_error():
    """Mock VectorStore that returns error results"""
    mock_store = Mock()
    mock_store.search.return_value = SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Database connection failed"
    )
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Mock successful response
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a sample response")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_anthropic_client_with_tools():
    """Mock Anthropic client that triggers tool use"""
    mock_client = Mock()
    
    # Mock initial tool use response
    mock_tool_content = Mock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.name = "search_course_content"
    mock_tool_content.input = {"query": "test query"}
    mock_tool_content.id = "tool_123"
    
    mock_initial_response = Mock()
    mock_initial_response.content = [mock_tool_content]
    mock_initial_response.stop_reason = "tool_use"
    
    # Mock final response after tool execution
    mock_final_response = Mock()
    mock_final_response.content = [Mock(text="Final response after tool execution")]
    
    mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
    
    return mock_client


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for testing"""
    mock_manager = Mock()
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    ]
    mock_manager.execute_tool.return_value = "Sample search results"
    mock_manager.get_last_sources.return_value = []
    return mock_manager


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Mock()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def temp_chroma_path(tmp_path):
    """Temporary ChromaDB path for testing"""
    return str(tmp_path / "test_chroma")