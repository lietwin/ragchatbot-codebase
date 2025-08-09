"""Unit tests for CourseSearchTool"""

import pytest
from unittest.mock import Mock
from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test class for CourseSearchTool functionality"""

    def test_get_tool_definition(self):
        """Test that tool definition is correctly structured"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]

    def test_execute_with_valid_results(self, mock_vector_store):
        """Test successful search execution with results"""
        tool = CourseSearchTool(mock_vector_store)

        # Mock search results
        mock_vector_store.search.return_value = SearchResults(
            documents=["Sample MCP content about protocols"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )

        result = tool.execute("What is MCP?")

        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name=None, lesson_number=None
        )

        # Verify result format
        assert isinstance(result, str)
        assert "MCP Course" in result
        assert "Lesson 1" in result
        assert "Sample MCP content" in result

        # Verify sources were tracked
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "MCP Course - Lesson 1"

    def test_execute_with_course_filter(self, mock_vector_store):
        """Test search execution with course name filter"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Course-specific content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": None}],
            distances=[0.1],
            error=None,
        )

        result = tool.execute("test query", course_name="Specific Course")

        # Verify search called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="Specific Course", lesson_number=None
        )

        assert "Specific Course" in result

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test search execution with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Lesson-specific content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.1],
            error=None,
        )

        result = tool.execute("test query", lesson_number=3)

        # Verify search called with lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=3
        )

        assert "Lesson 3" in result

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test search execution with both course and lesson filters"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 2}],
            distances=[0.1],
            error=None,
        )

        result = tool.execute("test query", course_name="MCP Course", lesson_number=2)

        # Verify search called with both filters
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="MCP Course", lesson_number=2
        )

        assert "MCP Course" in result
        assert "Lesson 2" in result

    def test_execute_with_empty_results(self, mock_vector_store_empty):
        """Test handling of empty search results"""
        tool = CourseSearchTool(mock_vector_store_empty)

        result = tool.execute("nonexistent query")

        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0

    def test_execute_with_empty_results_and_filters(self, mock_vector_store_empty):
        """Test empty results with filter information"""
        tool = CourseSearchTool(mock_vector_store_empty)

        result = tool.execute("test query", course_name="MCP", lesson_number=1)

        assert "No relevant content found in course 'MCP' in lesson 1" in result

    def test_execute_with_search_error(self, mock_vector_store_error):
        """Test handling of search errors from vector store"""
        tool = CourseSearchTool(mock_vector_store_error)

        result = tool.execute("test query")

        assert result == "Database connection failed"
        assert len(tool.last_sources) == 0

    def test_execute_with_multiple_results(self, mock_vector_store):
        """Test handling multiple search results"""
        tool = CourseSearchTool(mock_vector_store)

        # Mock multiple results
        mock_vector_store.search.return_value = SearchResults(
            documents=["First result content", "Second result content"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )

        result = tool.execute("test query")

        # Verify both results are included
        assert "Course A" in result
        assert "Course B" in result
        assert "First result content" in result
        assert "Second result content" in result

        # Verify sources tracked for both results
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert tool.last_sources[1]["text"] == "Course B - Lesson 2"

    def test_result_formatting_without_lesson(self, mock_vector_store):
        """Test result formatting when lesson_number is None"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Content without lesson"],
            metadata=[{"course_title": "Test Course", "lesson_number": None}],
            distances=[0.1],
            error=None,
        )

        result = tool.execute("test query")

        # Should not include "Lesson" text when lesson_number is None
        assert "[Test Course]" in result
        assert "Lesson" not in result

        # Source should not include lesson info
        assert tool.last_sources[0]["text"] == "Test Course"

    def test_source_tracking_reset_on_new_search(self, mock_vector_store):
        """Test that sources are properly reset between searches"""
        tool = CourseSearchTool(mock_vector_store)

        # First search
        mock_vector_store.search.return_value = SearchResults(
            documents=["First search result"],
            metadata=[{"course_title": "Course 1", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        tool.execute("first query")
        assert len(tool.last_sources) == 1

        # Second search - should replace previous sources
        mock_vector_store.search.return_value = SearchResults(
            documents=["Second search result"],
            metadata=[{"course_title": "Course 2", "lesson_number": 2}],
            distances=[0.1],
            error=None,
        )
        tool.execute("second query")

        # Should only have sources from second search
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Course 2 - Lesson 2"

    def test_lesson_link_retrieval(self, mock_vector_store):
        """Test lesson link retrieval when available"""
        tool = CourseSearchTool(mock_vector_store)

        # Mock get_lesson_link method
        mock_vector_store.get_lesson_link = Mock(
            return_value="https://example.com/lesson1"
        )

        mock_vector_store.search.return_value = SearchResults(
            documents=["Content with link"],
            metadata=[{"course_title": "Course with Links", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )

        tool.execute("test query")

        # Verify lesson link was retrieved and stored
        mock_vector_store.get_lesson_link.assert_called_once_with(
            "Course with Links", 1
        )
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"

    def test_metadata_with_missing_fields(self, mock_vector_store):
        """Test handling of metadata with missing fields"""
        tool = CourseSearchTool(mock_vector_store)

        # Mock results with incomplete metadata
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content with incomplete metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1],
            error=None,
        )

        result = tool.execute("test query")

        # Should handle missing fields gracefully
        assert "[unknown]" in result
        assert "Content with incomplete metadata" in result

        # Source should use default values
        assert tool.last_sources[0]["text"] == "unknown"
