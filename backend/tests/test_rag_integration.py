"""Integration tests for the complete RAG system"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGIntegration:
    """Test class for RAG system integration"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for RAG system"""
        config = Mock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "test-model"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-api-key"
        config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        config.MAX_HISTORY = 2
        return config

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.ToolManager')
    def test_rag_system_initialization(
        self, mock_tool_manager, mock_outline_tool, mock_search_tool,
        mock_session_manager, mock_ai_generator, mock_vector_store, 
        mock_document_processor, mock_config
    ):
        """Test RAG system initialization with all components"""
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Verify all components were initialized
        mock_document_processor.assert_called_once_with(
            mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP
        )
        mock_vector_store.assert_called_once_with(
            mock_config.CHROMA_PATH, mock_config.EMBEDDING_MODEL, mock_config.MAX_RESULTS
        )
        mock_ai_generator.assert_called_once_with(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )
        mock_session_manager.assert_called_once_with(mock_config.MAX_HISTORY)
        
        # Verify tool manager and tools
        mock_tool_manager.assert_called_once()
        mock_search_tool.assert_called_once()
        mock_outline_tool.assert_called_once()

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore') 
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.ToolManager')
    def test_successful_content_query(
        self, mock_tool_manager, mock_outline_tool, mock_search_tool,
        mock_session_manager, mock_ai_generator, mock_vector_store,
        mock_document_processor, mock_config
    ):
        """Test successful content query end-to-end"""
        
        # Setup mocks
        mock_tool_manager_instance = Mock()
        mock_tool_manager.return_value = mock_tool_manager_instance
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator.return_value = mock_ai_generator_instance
        mock_ai_generator_instance.generate_response.return_value = "AI generated response about MCP"
        
        mock_tool_manager_instance.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search content"}
        ]
        mock_tool_manager_instance.get_last_sources.return_value = [
            {"text": "MCP Course - Lesson 1", "link": "https://example.com"}
        ]
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Execute query
        response, sources = rag_system.query("What is MCP?")
        
        # Verify AI generator was called
        mock_ai_generator_instance.generate_response.assert_called_once()
        call_args = mock_ai_generator_instance.generate_response.call_args[1]
        
        assert "What is MCP?" in call_args["query"]
        assert call_args["tools"] == mock_tool_manager_instance.get_tool_definitions.return_value
        assert call_args["tool_manager"] == mock_tool_manager_instance
        
        # Verify results
        assert response == "AI generated response about MCP"
        assert len(sources) == 1
        assert sources[0]["text"] == "MCP Course - Lesson 1"
        
        # Verify sources were reset
        mock_tool_manager_instance.reset_sources.assert_called_once()

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.ToolManager')
    def test_query_with_conversation_history(
        self, mock_tool_manager, mock_outline_tool, mock_search_tool,
        mock_session_manager, mock_ai_generator, mock_vector_store,
        mock_document_processor, mock_config
    ):
        """Test query with conversation history"""
        
        # Setup mocks
        mock_session_manager_instance = Mock()
        mock_session_manager.return_value = mock_session_manager_instance
        mock_session_manager_instance.get_conversation_history.return_value = "Previous context"
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator.return_value = mock_ai_generator_instance
        mock_ai_generator_instance.generate_response.return_value = "Contextual response"
        
        mock_tool_manager_instance = Mock()
        mock_tool_manager.return_value = mock_tool_manager_instance
        mock_tool_manager_instance.get_last_sources.return_value = []
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Execute query with session ID
        response, sources = rag_system.query("Follow up question", session_id="session_123")
        
        # Verify conversation history was retrieved
        mock_session_manager_instance.get_conversation_history.assert_called_once_with("session_123")
        
        # Verify AI generator got history
        call_args = mock_ai_generator_instance.generate_response.call_args[1]
        assert call_args["conversation_history"] == "Previous context"
        
        # Verify conversation was updated
        mock_session_manager_instance.add_exchange.assert_called_once_with(
            "session_123", "Follow up question", "Contextual response"
        )

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.ToolManager')
    def test_query_without_session(
        self, mock_tool_manager, mock_outline_tool, mock_search_tool,
        mock_session_manager, mock_ai_generator, mock_vector_store,
        mock_document_processor, mock_config
    ):
        """Test query without session ID (no history)"""
        
        # Setup mocks
        mock_session_manager_instance = Mock()
        mock_session_manager.return_value = mock_session_manager_instance
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator.return_value = mock_ai_generator_instance
        mock_ai_generator_instance.generate_response.return_value = "Response without history"
        
        mock_tool_manager_instance = Mock()
        mock_tool_manager.return_value = mock_tool_manager_instance
        mock_tool_manager_instance.get_last_sources.return_value = []
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Execute query without session ID
        response, sources = rag_system.query("Standalone question")
        
        # Verify no history operations
        mock_session_manager_instance.get_conversation_history.assert_not_called()
        mock_session_manager_instance.add_exchange.assert_not_called()
        
        # Verify AI generator got None for history
        call_args = mock_ai_generator_instance.generate_response.call_args[1]
        assert call_args["conversation_history"] is None

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.ToolManager')
    def test_tools_registration(
        self, mock_tool_manager, mock_outline_tool, mock_search_tool,
        mock_session_manager, mock_ai_generator, mock_vector_store,
        mock_document_processor, mock_config
    ):
        """Test that both tools are properly registered"""
        
        # Setup mock instances
        mock_tool_manager_instance = Mock()
        mock_tool_manager.return_value = mock_tool_manager_instance
        
        mock_search_tool_instance = Mock()
        mock_search_tool.return_value = mock_search_tool_instance
        
        mock_outline_tool_instance = Mock()
        mock_outline_tool.return_value = mock_outline_tool_instance
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Verify both tools were registered
        assert mock_tool_manager_instance.register_tool.call_count == 2
        mock_tool_manager_instance.register_tool.assert_any_call(mock_search_tool_instance)
        mock_tool_manager_instance.register_tool.assert_any_call(mock_outline_tool_instance)

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.ToolManager')
    def test_ai_generator_exception_handling(
        self, mock_tool_manager, mock_outline_tool, mock_search_tool,
        mock_session_manager, mock_ai_generator, mock_vector_store,
        mock_document_processor, mock_config
    ):
        """Test handling of AI generator exceptions"""
        
        # Setup mocks
        mock_ai_generator_instance = Mock()
        mock_ai_generator.return_value = mock_ai_generator_instance
        mock_ai_generator_instance.generate_response.side_effect = Exception("AI API failed")
        
        mock_tool_manager_instance = Mock()
        mock_tool_manager.return_value = mock_tool_manager_instance
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Should propagate the exception
        with pytest.raises(Exception, match="AI API failed"):
            rag_system.query("Test question")

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.ToolManager')
    def test_get_course_analytics(
        self, mock_tool_manager, mock_outline_tool, mock_search_tool,
        mock_session_manager, mock_ai_generator, mock_vector_store,
        mock_document_processor, mock_config
    ):
        """Test course analytics functionality"""
        
        # Setup mocks
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        mock_vector_store_instance.get_course_count.return_value = 3
        mock_vector_store_instance.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3"
        ]
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Get analytics
        analytics = rag_system.get_course_analytics()
        
        # Verify results
        assert analytics["total_courses"] == 3
        assert analytics["course_titles"] == ["Course 1", "Course 2", "Course 3"]

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator') 
    @patch('rag_system.SessionManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.ToolManager')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_success(
        self, mock_listdir, mock_exists, mock_tool_manager, mock_outline_tool,
        mock_search_tool, mock_session_manager, mock_ai_generator,
        mock_vector_store, mock_document_processor, mock_config
    ):
        """Test successful course folder processing"""
        
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.txt", "readme.md"]
        
        mock_document_processor_instance = Mock()
        mock_document_processor.return_value = mock_document_processor_instance
        
        mock_course1 = Mock()
        mock_course1.title = "Course 1"
        mock_course2 = Mock()
        mock_course2.title = "Course 2"
        
        mock_document_processor_instance.process_course_document.side_effect = [
            (mock_course1, ["chunk1", "chunk2"]),
            (mock_course2, ["chunk3"])
        ]
        
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        mock_vector_store_instance.get_existing_course_titles.return_value = []
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Add course folder
        courses_added, chunks_added = rag_system.add_course_folder("./docs")
        
        # Verify results
        assert courses_added == 2
        assert chunks_added == 3
        
        # Verify courses were added to vector store
        assert mock_vector_store_instance.add_course_metadata.call_count == 2
        assert mock_vector_store_instance.add_course_content.call_count == 2

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.ToolManager')
    @patch('os.path.exists')
    def test_add_course_folder_not_exists(
        self, mock_exists, mock_tool_manager, mock_outline_tool,
        mock_search_tool, mock_session_manager, mock_ai_generator,
        mock_vector_store, mock_document_processor, mock_config
    ):
        """Test course folder processing when folder doesn't exist"""
        
        # Setup mocks
        mock_exists.return_value = False
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Add nonexistent course folder
        with patch('builtins.print') as mock_print:
            courses_added, chunks_added = rag_system.add_course_folder("./nonexistent")
            
            # Should print error and return 0, 0
            mock_print.assert_called_once_with("Folder ./nonexistent does not exist")
            assert courses_added == 0
            assert chunks_added == 0

    def test_real_system_initialization_with_temp_db(self, mock_config, temp_chroma_path):
        """Test RAG system with real components but temporary database"""
        
        # Use temporary database path
        mock_config.CHROMA_PATH = temp_chroma_path
        
        # This test would use real components but with temporary storage
        # Commented out because it requires actual dependencies
        
        # rag_system = RAGSystem(mock_config)
        # 
        # # Verify system can be created
        # assert rag_system is not None
        # assert rag_system.vector_store is not None
        # assert rag_system.tool_manager is not None
        pass

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.ToolManager')
    def test_tool_manager_error_handling(
        self, mock_tool_manager, mock_outline_tool, mock_search_tool,
        mock_session_manager, mock_ai_generator, mock_vector_store,
        mock_document_processor, mock_config
    ):
        """Test error handling in tool manager operations"""
        
        # Setup mocks
        mock_tool_manager_instance = Mock()
        mock_tool_manager.return_value = mock_tool_manager_instance
        mock_tool_manager_instance.get_last_sources.side_effect = Exception("Tool manager error")
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator.return_value = mock_ai_generator_instance
        mock_ai_generator_instance.generate_response.return_value = "Response"
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Should propagate tool manager exception
        with pytest.raises(Exception, match="Tool manager error"):
            rag_system.query("Test question")

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.CourseOutlineTool') 
    @patch('rag_system.ToolManager')
    def test_empty_sources_handling(
        self, mock_tool_manager, mock_outline_tool, mock_search_tool,
        mock_session_manager, mock_ai_generator, mock_vector_store,
        mock_document_processor, mock_config
    ):
        """Test handling when no sources are returned"""
        
        # Setup mocks
        mock_tool_manager_instance = Mock()
        mock_tool_manager.return_value = mock_tool_manager_instance
        mock_tool_manager_instance.get_last_sources.return_value = []
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator.return_value = mock_ai_generator_instance
        mock_ai_generator_instance.generate_response.return_value = "No sources found"
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Execute query
        response, sources = rag_system.query("Unknown topic")
        
        # Verify empty sources handled properly
        assert response == "No sources found"
        assert sources == []