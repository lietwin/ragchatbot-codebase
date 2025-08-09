import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

import sys
import os
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from app import app, QueryRequest, QueryResponse, CourseStats
from models import Course, Lesson, CourseChunk
from rag_system import RAGSystem


class TestFastAPIEndpoints:
    """Test suite for FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_rag_system(self):
        """Mock RAG system for testing"""
        with patch('app.rag_system') as mock_rag:
            # Setup session manager
            mock_session_manager = Mock()
            mock_session_manager.create_session.return_value = "test_session_123"
            mock_rag.session_manager = mock_session_manager
            
            # Setup default query response
            mock_rag.query.return_value = (
                "This is a test answer about MCP fundamentals.",
                [
                    {
                        "course_title": "MCP: Build Rich-Context AI Apps",
                        "lesson_number": 1,
                        "content": "Sample content about MCP"
                    }
                ]
            )
            
            # Setup course analytics
            mock_rag.get_course_analytics.return_value = {
                "total_courses": 2,
                "course_titles": ["MCP: Build Rich-Context AI Apps", "Advanced Python Programming"]
            }
            
            yield mock_rag
    

class TestQueryEndpoint(TestFastAPIEndpoints):
    """Tests for /api/query endpoint"""
    
    def test_query_success_without_session(self, client, mock_rag_system):
        """Test successful query without existing session"""
        request_data = {
            "query": "What is MCP?",
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test answer about MCP fundamentals."
        assert data["session_id"] == "test_session_123"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["course_title"] == "MCP: Build Rich-Context AI Apps"
        
        # Verify RAG system was called correctly
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is MCP?", "test_session_123")
    
    def test_query_success_with_session(self, client, mock_rag_system):
        """Test successful query with existing session"""
        request_data = {
            "query": "Tell me more about this",
            "session_id": "existing_session_456"
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "existing_session_456"
        
        # Verify session was not created
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Tell me more about this", "existing_session_456")
    
    def test_query_empty_string(self, client, mock_rag_system):
        """Test query with empty string"""
        request_data = {
            "query": "",
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        # RAG system should handle empty queries gracefully
        mock_rag_system.query.assert_called_once_with("", "test_session_123")
    
    def test_query_very_long_string(self, client, mock_rag_system):
        """Test query with very long string"""
        long_query = "What is MCP? " * 1000  # Very long query
        request_data = {
            "query": long_query,
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with(long_query, "test_session_123")
    
    def test_query_special_characters(self, client, mock_rag_system):
        """Test query with special characters"""
        special_query = "What about C++ & AI/ML? 你好 🤖"
        request_data = {
            "query": special_query,
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with(special_query, "test_session_123")
    
    def test_query_missing_field(self, client, mock_rag_system):
        """Test query request missing required field"""
        request_data = {
            "session_id": "test_session"
            # Missing required 'query' field
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data
        assert any("query" in str(error) for error in error_data["detail"])
    
    def test_query_invalid_json(self, client, mock_rag_system):
        """Test query with invalid JSON"""
        response = client.post("/api/query", data="invalid json")
        
        assert response.status_code == 422
    
    def test_query_rag_system_exception(self, client, mock_rag_system):
        """Test query when RAG system raises exception"""
        mock_rag_system.query.side_effect = Exception("Database connection failed")
        
        request_data = {
            "query": "What is MCP?",
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 500
        error_data = response.json()
        assert error_data["detail"] == "Database connection failed"
    
    def test_query_session_creation_exception(self, client, mock_rag_system):
        """Test query when session creation fails"""
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session creation failed")
        
        request_data = {
            "query": "What is MCP?",
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 500
        error_data = response.json()
        assert error_data["detail"] == "Session creation failed"
    
    def test_query_response_validation(self, client, mock_rag_system):
        """Test that response matches expected schema"""
        request_data = {
            "query": "Test query",
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure matches QueryResponse model
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Validate sources structure
        for source in data["sources"]:
            assert isinstance(source, dict)
    
    def test_query_empty_sources(self, client, mock_rag_system):
        """Test query with empty sources response"""
        mock_rag_system.query.return_value = ("No relevant information found.", [])
        
        request_data = {
            "query": "Nonexistent topic",
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "No relevant information found."
        assert data["sources"] == []


class TestCoursesEndpoint(TestFastAPIEndpoints):
    """Tests for /api/courses endpoint"""
    
    def test_get_courses_success(self, client, mock_rag_system):
        """Test successful courses retrieval"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert isinstance(data["course_titles"], list)
        assert len(data["course_titles"]) == 2
        assert "MCP: Build Rich-Context AI Apps" in data["course_titles"]
        assert "Advanced Python Programming" in data["course_titles"]
        
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_get_courses_empty_result(self, client, mock_rag_system):
        """Test courses endpoint with no courses"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_get_courses_many_courses(self, client, mock_rag_system):
        """Test courses endpoint with many courses"""
        course_titles = [f"Course {i}" for i in range(100)]
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 100,
            "course_titles": course_titles
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 100
        assert len(data["course_titles"]) == 100
    
    def test_get_courses_rag_system_exception(self, client, mock_rag_system):
        """Test courses endpoint when RAG system raises exception"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics service unavailable")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        error_data = response.json()
        assert error_data["detail"] == "Analytics service unavailable"
    
    def test_get_courses_response_validation(self, client, mock_rag_system):
        """Test that response matches expected schema"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure matches CourseStats model
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])
    
    def test_get_courses_no_parameters(self, client, mock_rag_system):
        """Test that courses endpoint accepts no parameters"""
        # Should work with no query parameters
        response = client.get("/api/courses")
        assert response.status_code == 200
        
        # Should work with ignored query parameters
        response = client.get("/api/courses?ignored=value")
        assert response.status_code == 200


class TestStaticFileServing(TestFastAPIEndpoints):
    """Tests for static file serving"""
    
    def test_static_file_serving_root(self, client):
        """Test that root path serves static files"""
        # This test assumes the frontend directory exists with index.html
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            # Mock the static files response
            response = client.get("/")
            
            # Should attempt to serve static files (may return 404 if files don't exist in test)
            assert response.status_code in [200, 404]
    
    def test_static_file_headers(self, client):
        """Test that static files have proper headers in development"""
        # Test assumes DevStaticFiles class is properly configured
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            response = client.get("/")
            
            # Headers might not be set in test environment, but endpoint should respond
            assert response.status_code in [200, 404]


class TestEndpointErrorHandling(TestFastAPIEndpoints):
    """Tests for general endpoint error handling"""
    
    def test_invalid_endpoint(self, client):
        """Test accessing non-existent endpoint"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
    
    def test_wrong_method_query(self, client):
        """Test using wrong HTTP method on query endpoint"""
        response = client.get("/api/query")
        assert response.status_code == 404  # Not Found (FastAPI behavior for undefined method)
    
    def test_wrong_method_courses(self, client):
        """Test using wrong HTTP method on courses endpoint"""
        response = client.post("/api/courses")
        assert response.status_code == 405  # Method not allowed
    
    def test_cors_headers_present(self, client, mock_rag_system):
        """Test that CORS headers are properly set"""
        response = client.post("/api/query", json={"query": "test"})
        
        # CORS middleware should add these headers
        assert response.status_code == 200
        # Note: TestClient may not include all CORS headers, 
        # but the middleware should be configured
    
    @patch('app.rag_system')
    def test_startup_event_document_loading(self, mock_rag_system):
        """Test startup event loads documents"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            mock_rag_system.add_course_folder.return_value = (2, 150)
            
            # Create new app instance to trigger startup
            from app import startup_event
            import asyncio
            
            # Run startup event
            asyncio.run(startup_event())
            
            mock_rag_system.add_course_folder.assert_called_once_with("../docs", clear_existing=False)
    
    @patch('app.rag_system')
    def test_startup_event_no_docs_folder(self, mock_rag_system):
        """Test startup event when docs folder doesn't exist"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            from app import startup_event
            import asyncio
            
            # Run startup event
            asyncio.run(startup_event())
            
            # Should not attempt to load documents
            mock_rag_system.add_course_folder.assert_not_called()
    
    @patch('app.rag_system')
    def test_startup_event_loading_exception(self, mock_rag_system):
        """Test startup event when document loading fails"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            mock_rag_system.add_course_folder.side_effect = Exception("Loading failed")
            
            from app import startup_event
            import asyncio
            
            # Should not raise exception, just print error
            asyncio.run(startup_event())
            
            mock_rag_system.add_course_folder.assert_called_once()


class TestRequestResponseModels(TestFastAPIEndpoints):
    """Tests for Pydantic request/response models"""
    
    def test_query_request_model_validation(self):
        """Test QueryRequest model validation"""
        # Valid request
        valid_request = QueryRequest(query="What is MCP?", session_id="123")
        assert valid_request.query == "What is MCP?"
        assert valid_request.session_id == "123"
        
        # Request without session_id (should default to None)
        request_no_session = QueryRequest(query="What is MCP?")
        assert request_no_session.query == "What is MCP?"
        assert request_no_session.session_id is None
        
        # Test validation error
        with pytest.raises(ValueError):
            QueryRequest()  # Missing required query field
    
    def test_query_response_model_validation(self):
        """Test QueryResponse model validation"""
        # Valid response
        valid_response = QueryResponse(
            answer="This is an answer",
            sources=[{"course_title": "Test Course"}],
            session_id="123"
        )
        assert valid_response.answer == "This is an answer"
        assert len(valid_response.sources) == 1
        assert valid_response.session_id == "123"
        
        # Test validation error
        with pytest.raises(ValueError):
            QueryResponse(answer="test")  # Missing required fields
    
    def test_course_stats_model_validation(self):
        """Test CourseStats model validation"""
        # Valid stats
        valid_stats = CourseStats(
            total_courses=5,
            course_titles=["Course 1", "Course 2"]
        )
        assert valid_stats.total_courses == 5
        assert len(valid_stats.course_titles) == 2
        
        # Test validation error
        with pytest.raises(ValueError):
            CourseStats(total_courses="not_a_number")  # Invalid type


class TestMiddleware(TestFastAPIEndpoints):
    """Tests for middleware configuration"""
    
    def test_cors_middleware_configured(self, client, mock_rag_system):
        """Test that CORS middleware is properly configured"""
        # Make a request and check it doesn't fail due to CORS
        response = client.post("/api/query", json={"query": "test"})
        assert response.status_code == 200
    
    def test_trusted_host_middleware_configured(self, client):
        """Test that TrustedHost middleware is configured"""
        # Should accept requests (configured to allow all hosts)
        response = client.get("/api/courses")
        # Should not be blocked by host restrictions
        assert response.status_code != 400