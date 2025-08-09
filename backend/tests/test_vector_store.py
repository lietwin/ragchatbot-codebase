"""Unit tests for VectorStore"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from vector_store import VectorStore, SearchResults
from models import Course, CourseChunk, Lesson


class TestVectorStore:
    """Test class for VectorStore functionality"""

    def test_init_with_mock_chromadb(self):
        """Test VectorStore initialization"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ) as mock_embedding,
        ):

            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )

            store = VectorStore("./test_path", "test-model", max_results=3)

            # Verify initialization
            assert store.max_results == 3
            mock_client.assert_called_once()
            mock_embedding.assert_called_once_with(model_name="test-model")
            assert store.course_catalog == mock_collection
            assert store.course_content == mock_collection

    def test_search_without_filters(self):
        """Test search method without any filters"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )

            # Mock query results
            mock_collection.query.return_value = {
                "documents": [["Sample document content"]],
                "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
                "distances": [[0.1]],
            }

            store = VectorStore("./test_path", "test-model")
            result = store.search("test query")

            # Verify query was called correctly
            mock_collection.query.assert_called_once_with(
                query_texts=["test query"],
                n_results=5,  # default max_results
                where=None,  # no filters
            )

            # Verify results
            assert isinstance(result, SearchResults)
            assert len(result.documents) == 1
            assert result.documents[0] == "Sample document content"
            assert result.error is None

    def test_search_with_course_name_filter(self):
        """Test search with course name that needs resolution"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_catalog_collection = Mock()
            mock_content_collection = Mock()

            # Mock get_or_create_collection to return different collections
            def mock_get_collection(name, embedding_function):
                if name == "course_catalog":
                    return mock_catalog_collection
                return mock_content_collection

            mock_client.return_value.get_or_create_collection.side_effect = (
                mock_get_collection
            )

            store = VectorStore("./test_path", "test-model")

            # Mock course name resolution
            mock_catalog_collection.query.return_value = {
                "documents": [["MCP Course"]],
                "metadatas": [[{"title": "MCP: Build Rich-Context AI Apps"}]],
            }

            # Mock content search
            mock_content_collection.query.return_value = {
                "documents": [["MCP content"]],
                "metadatas": [
                    [
                        {
                            "course_title": "MCP: Build Rich-Context AI Apps",
                            "lesson_number": 1,
                        }
                    ]
                ],
                "distances": [[0.1]],
            }

            result = store.search("test query", course_name="MCP")

            # Verify course name was resolved
            mock_catalog_collection.query.assert_called_once_with(
                query_texts=["MCP"], n_results=1
            )

            # Verify content search with resolved title
            mock_content_collection.query.assert_called_once_with(
                query_texts=["test query"],
                n_results=5,
                where={"course_title": "MCP: Build Rich-Context AI Apps"},
            )

            assert result.error is None
            assert len(result.documents) == 1

    def test_search_with_course_name_not_found(self):
        """Test search when course name cannot be resolved"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_catalog_collection = Mock()
            mock_content_collection = Mock()

            def mock_get_collection(name, embedding_function):
                if name == "course_catalog":
                    return mock_catalog_collection
                return mock_content_collection

            mock_client.return_value.get_or_create_collection.side_effect = (
                mock_get_collection
            )

            store = VectorStore("./test_path", "test-model")

            # Mock empty course resolution
            mock_catalog_collection.query.return_value = {
                "documents": [[]],
                "metadatas": [[]],
            }

            result = store.search("test query", course_name="NonexistentCourse")

            # Should return error for course not found
            assert result.error == "No course found matching 'NonexistentCourse'"
            assert len(result.documents) == 0

    def test_search_with_lesson_number_filter(self):
        """Test search with lesson number filter"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )

            mock_collection.query.return_value = {
                "documents": [["Lesson 2 content"]],
                "metadatas": [[{"course_title": "Test Course", "lesson_number": 2}]],
                "distances": [[0.1]],
            }

            store = VectorStore("./test_path", "test-model")
            result = store.search("test query", lesson_number=2)

            # Verify query with lesson filter
            mock_collection.query.assert_called_once_with(
                query_texts=["test query"], n_results=5, where={"lesson_number": 2}
            )

            assert result.error is None

    def test_search_with_both_filters(self):
        """Test search with both course name and lesson number"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_catalog_collection = Mock()
            mock_content_collection = Mock()

            def mock_get_collection(name, embedding_function):
                if name == "course_catalog":
                    return mock_catalog_collection
                return mock_content_collection

            mock_client.return_value.get_or_create_collection.side_effect = (
                mock_get_collection
            )

            store = VectorStore("./test_path", "test-model")

            # Mock successful course resolution
            mock_catalog_collection.query.return_value = {
                "documents": [["Resolved Course"]],
                "metadatas": [[{"title": "Resolved Course Title"}]],
            }

            mock_content_collection.query.return_value = {
                "documents": [["Filtered content"]],
                "metadatas": [
                    [{"course_title": "Resolved Course Title", "lesson_number": 3}]
                ],
                "distances": [[0.1]],
            }

            result = store.search("test query", course_name="Course", lesson_number=3)

            # Verify combined filter
            mock_content_collection.query.assert_called_once_with(
                query_texts=["test query"],
                n_results=5,
                where={
                    "$and": [
                        {"course_title": "Resolved Course Title"},
                        {"lesson_number": 3},
                    ]
                },
            )

            assert result.error is None

    def test_search_with_custom_limit(self):
        """Test search with custom result limit"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )

            mock_collection.query.return_value = {
                "documents": [["Content 1", "Content 2"]],
                "metadatas": [
                    [
                        {"course_title": "Course", "lesson_number": 1},
                        {"course_title": "Course", "lesson_number": 2},
                    ]
                ],
                "distances": [[0.1, 0.2]],
            }

            store = VectorStore("./test_path", "test-model", max_results=5)
            result = store.search("test query", limit=2)

            # Verify custom limit was used
            mock_collection.query.assert_called_once_with(
                query_texts=["test query"],
                n_results=2,  # custom limit, not default max_results
                where=None,
            )

    def test_search_with_chromadb_exception(self):
        """Test search handling ChromaDB exceptions"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )

            # Mock ChromaDB exception
            mock_collection.query.side_effect = Exception("ChromaDB connection error")

            store = VectorStore("./test_path", "test-model")
            result = store.search("test query")

            # Should return error result
            assert result.error == "Search error: ChromaDB connection error"
            assert len(result.documents) == 0

    def test_resolve_course_name_success(self):
        """Test successful course name resolution"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_catalog_collection = Mock()
            mock_content_collection = Mock()

            def mock_get_collection(name, embedding_function):
                if name == "course_catalog":
                    return mock_catalog_collection
                return mock_content_collection

            mock_client.return_value.get_or_create_collection.side_effect = (
                mock_get_collection
            )

            store = VectorStore("./test_path", "test-model")

            mock_catalog_collection.query.return_value = {
                "documents": [["Course document"]],
                "metadatas": [[{"title": "Full Course Title"}]],
            }

            result = store._resolve_course_name("partial name")

            assert result == "Full Course Title"

    def test_resolve_course_name_not_found(self):
        """Test course name resolution when not found"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_catalog_collection = Mock()
            mock_content_collection = Mock()

            def mock_get_collection(name, embedding_function):
                if name == "course_catalog":
                    return mock_catalog_collection
                return mock_content_collection

            mock_client.return_value.get_or_create_collection.side_effect = (
                mock_get_collection
            )

            store = VectorStore("./test_path", "test-model")

            mock_catalog_collection.query.return_value = {
                "documents": [[]],
                "metadatas": [[]],
            }

            result = store._resolve_course_name("nonexistent")

            assert result is None

    def test_resolve_course_name_with_exception(self):
        """Test course name resolution with exception"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_catalog_collection = Mock()
            mock_content_collection = Mock()

            def mock_get_collection(name, embedding_function):
                if name == "course_catalog":
                    return mock_catalog_collection
                return mock_content_collection

            mock_client.return_value.get_or_create_collection.side_effect = (
                mock_get_collection
            )

            store = VectorStore("./test_path", "test-model")

            mock_catalog_collection.query.side_effect = Exception("Query failed")

            with patch("builtins.print") as mock_print:
                result = store._resolve_course_name("test")

                # Should print error and return None
                mock_print.assert_called_once_with(
                    "Error resolving course name: Query failed"
                )
                assert result is None

    def test_build_filter_no_parameters(self):
        """Test filter building with no parameters"""
        with (
            patch("chromadb.PersistentClient"),
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            store = VectorStore("./test_path", "test-model")
            result = store._build_filter(None, None)

            assert result is None

    def test_build_filter_course_only(self):
        """Test filter building with course title only"""
        with (
            patch("chromadb.PersistentClient"),
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            store = VectorStore("./test_path", "test-model")
            result = store._build_filter("Test Course", None)

            assert result == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self):
        """Test filter building with lesson number only"""
        with (
            patch("chromadb.PersistentClient"),
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            store = VectorStore("./test_path", "test-model")
            result = store._build_filter(None, 2)

            assert result == {"lesson_number": 2}

    def test_build_filter_both_parameters(self):
        """Test filter building with both parameters"""
        with (
            patch("chromadb.PersistentClient"),
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            store = VectorStore("./test_path", "test-model")
            result = store._build_filter("Test Course", 3)

            expected = {"$and": [{"course_title": "Test Course"}, {"lesson_number": 3}]}
            assert result == expected

    def test_add_course_metadata(self, sample_course):
        """Test adding course metadata to vector store"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_catalog_collection = Mock()
            mock_content_collection = Mock()

            def mock_get_collection(name, embedding_function):
                if name == "course_catalog":
                    return mock_catalog_collection
                return mock_content_collection

            mock_client.return_value.get_or_create_collection.side_effect = (
                mock_get_collection
            )

            store = VectorStore("./test_path", "test-model")
            store.add_course_metadata(sample_course)

            # Verify catalog collection was called
            mock_catalog_collection.add.assert_called_once()
            call_args = mock_catalog_collection.add.call_args[1]

            assert call_args["documents"] == [sample_course.title]
            assert call_args["ids"] == [sample_course.title]

            metadata = call_args["metadatas"][0]
            assert metadata["title"] == sample_course.title
            assert metadata["instructor"] == sample_course.instructor
            assert "lessons_json" in metadata

    def test_add_course_content(self, sample_chunks):
        """Test adding course content chunks to vector store"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_catalog_collection = Mock()
            mock_content_collection = Mock()

            def mock_get_collection(name, embedding_function):
                if name == "course_catalog":
                    return mock_catalog_collection
                return mock_content_collection

            mock_client.return_value.get_or_create_collection.side_effect = (
                mock_get_collection
            )

            store = VectorStore("./test_path", "test-model")
            store.add_course_content(sample_chunks)

            # Verify content collection was called
            mock_content_collection.add.assert_called_once()
            call_args = mock_content_collection.add.call_args[1]

            assert len(call_args["documents"]) == len(sample_chunks)
            assert len(call_args["metadatas"]) == len(sample_chunks)
            assert len(call_args["ids"]) == len(sample_chunks)

    def test_add_course_content_empty_list(self):
        """Test adding empty course content list"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )

            store = VectorStore("./test_path", "test-model")
            store.add_course_content([])

            # Should not call add when empty list
            mock_collection.add.assert_not_called()

    def test_get_existing_course_titles(self):
        """Test getting existing course titles"""
        with (
            patch("chromadb.PersistentClient") as mock_client,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_catalog_collection = Mock()
            mock_content_collection = Mock()

            def mock_get_collection(name, embedding_function):
                if name == "course_catalog":
                    return mock_catalog_collection
                return mock_content_collection

            mock_client.return_value.get_or_create_collection.side_effect = (
                mock_get_collection
            )

            store = VectorStore("./test_path", "test-model")

            mock_catalog_collection.get.return_value = {
                "ids": ["Course 1", "Course 2", "Course 3"]
            }

            result = store.get_existing_course_titles()

            assert result == ["Course 1", "Course 2", "Course 3"]

    def test_search_results_from_chroma(self):
        """Test SearchResults creation from ChromaDB results"""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key1": "value1"}, {"key2": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"key1": "value1"}, {"key2": "value2"}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_search_results_empty_chroma(self):
        """Test SearchResults creation from empty ChromaDB results"""
        chroma_results = {"documents": [], "metadatas": [], "distances": []}

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.is_empty() is True

    def test_search_results_error_creation(self):
        """Test SearchResults error creation"""
        results = SearchResults.empty("Test error message")

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"
        assert results.is_empty() is True
