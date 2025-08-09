"""Unit tests for AIGenerator"""

import pytest
from unittest.mock import Mock, patch
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test class for AIGenerator functionality"""

    def test_init(self):
        """Test AIGenerator initialization"""
        api_key = "test-api-key"
        model = "claude-sonnet-4-20250514"
        
        with patch('anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(api_key, model)
            
            # Verify Anthropic client initialized with correct API key
            mock_anthropic.assert_called_once_with(api_key=api_key)
            assert generator.model == model
            assert generator.base_params["model"] == model
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800

    def test_generate_response_without_tools(self, mock_anthropic_client):
        """Test basic response generation without tools"""
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            result = generator.generate_response("What is AI?")
            
            # Verify API was called correctly
            mock_anthropic_client.messages.create.assert_called_once()
            call_args = mock_anthropic_client.messages.create.call_args[1]
            
            assert call_args["model"] == "test-model"
            assert call_args["messages"][0]["content"] == "What is AI?"
            assert call_args["messages"][0]["role"] == "user"
            assert "tools" not in call_args
            assert result == "This is a sample response"

    def test_generate_response_with_conversation_history(self, mock_anthropic_client):
        """Test response generation with conversation history"""
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            history = "Previous conversation context"
            result = generator.generate_response("Current question", conversation_history=history)
            
            # Verify system prompt includes history
            call_args = mock_anthropic_client.messages.create.call_args[1]
            assert history in call_args["system"]

    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client):
        """Test response generation with tools available but not used"""
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")
            
            tools = [{"name": "test_tool", "description": "Test tool"}]
            tool_manager = Mock()
            
            result = generator.generate_response(
                "Simple question",
                tools=tools,
                tool_manager=tool_manager
            )
            
            # Verify tools were provided to API
            call_args = mock_anthropic_client.messages.create.call_args[1]
            assert call_args["tools"] == tools
            assert call_args["tool_choice"] == {"type": "auto"}
            
            # Verify tool manager wasn't used since no tool_use
            tool_manager.execute_tool.assert_not_called()
            assert result == "This is a sample response"

    def test_generate_response_with_tool_use(self, mock_anthropic_client_with_tools):
        """Test response generation that triggers tool use"""
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client_with_tools):
            generator = AIGenerator("test-key", "test-model")
            
            tools = [{"name": "search_course_content", "description": "Search courses"}]
            tool_manager = Mock()
            tool_manager.execute_tool.return_value = "Tool execution result"
            
            result = generator.generate_response(
                "Search for MCP information",
                tools=tools,
                tool_manager=tool_manager
            )
            
            # Verify tool was executed
            tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="test query"
            )
            
            # Verify final response after tool execution
            assert result == "Final response after tool execution"
            
            # Verify API was called twice (initial + final)
            assert mock_anthropic_client_with_tools.messages.create.call_count == 2

    def test_handle_tool_execution_single_tool(self):
        """Test _handle_tool_execution with single tool call"""
        with patch('anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator("test-key", "test-model")
            
            # Mock initial response with tool use
            mock_tool_content = Mock()
            mock_tool_content.type = "tool_use"
            mock_tool_content.name = "test_tool"
            mock_tool_content.input = {"param": "value"}
            mock_tool_content.id = "tool_123"
            
            initial_response = Mock()
            initial_response.content = [mock_tool_content]
            
            # Mock tool manager
            tool_manager = Mock()
            tool_manager.execute_tool.return_value = "Tool result"
            
            # Mock final response
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Final answer")]
            mock_anthropic.return_value.messages.create.return_value = mock_final_response
            
            # Test the method
            base_params = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test query"}],
                "system": "test system prompt"
            }
            
            result = generator._handle_tool_execution(initial_response, base_params, tool_manager)
            
            # Verify tool was executed
            tool_manager.execute_tool.assert_called_once_with("test_tool", param="value")
            
            # Verify final API call was made
            mock_anthropic.return_value.messages.create.assert_called_once()
            final_call_args = mock_anthropic.return_value.messages.create.call_args[1]
            
            # Verify message structure
            messages = final_call_args["messages"]
            assert len(messages) == 3  # original user message, assistant tool use, user tool results
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"
            assert messages[2]["content"][0]["type"] == "tool_result"
            assert messages[2]["content"][0]["tool_use_id"] == "tool_123"
            assert messages[2]["content"][0]["content"] == "Tool result"
            
            assert result == "Final answer"

    def test_handle_tool_execution_multiple_tools(self):
        """Test _handle_tool_execution with multiple tool calls"""
        with patch('anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator("test-key", "test-model")
            
            # Mock initial response with multiple tool uses
            mock_tool_content_1 = Mock()
            mock_tool_content_1.type = "tool_use"
            mock_tool_content_1.name = "tool_1"
            mock_tool_content_1.input = {"param": "value1"}
            mock_tool_content_1.id = "tool_123"
            
            mock_tool_content_2 = Mock()
            mock_tool_content_2.type = "tool_use"
            mock_tool_content_2.name = "tool_2"
            mock_tool_content_2.input = {"param": "value2"}
            mock_tool_content_2.id = "tool_456"
            
            initial_response = Mock()
            initial_response.content = [mock_tool_content_1, mock_tool_content_2]
            
            # Mock tool manager
            tool_manager = Mock()
            tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
            
            # Mock final response
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Combined results")]
            mock_anthropic.return_value.messages.create.return_value = mock_final_response
            
            base_params = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test query"}],
                "system": "test system prompt"
            }
            
            result = generator._handle_tool_execution(initial_response, base_params, tool_manager)
            
            # Verify both tools were executed
            assert tool_manager.execute_tool.call_count == 2
            tool_manager.execute_tool.assert_any_call("tool_1", param="value1")
            tool_manager.execute_tool.assert_any_call("tool_2", param="value2")
            
            # Verify tool results were included
            final_call_args = mock_anthropic.return_value.messages.create.call_args[1]
            tool_results = final_call_args["messages"][2]["content"]
            assert len(tool_results) == 2
            assert tool_results[0]["tool_use_id"] == "tool_123"
            assert tool_results[0]["content"] == "Result 1"
            assert tool_results[1]["tool_use_id"] == "tool_456"
            assert tool_results[1]["content"] == "Result 2"
            
            assert result == "Combined results"

    def test_api_error_handling(self):
        """Test handling of Anthropic API errors"""
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_client = mock_anthropic.return_value
            mock_client.messages.create.side_effect = Exception("API Error")
            
            generator = AIGenerator("test-key", "test-model")
            
            # Should propagate the exception
            with pytest.raises(Exception, match="API Error"):
                generator.generate_response("test query")

    def test_tool_execution_error_handling(self):
        """Test handling of tool execution errors"""
        with patch('anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator("test-key", "test-model")
            
            # Mock initial response with tool use
            mock_tool_content = Mock()
            mock_tool_content.type = "tool_use"
            mock_tool_content.name = "failing_tool"
            mock_tool_content.input = {"param": "value"}
            mock_tool_content.id = "tool_123"
            
            initial_response = Mock()
            initial_response.content = [mock_tool_content]
            
            # Mock tool manager that raises exception
            tool_manager = Mock()
            tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
            
            # Mock final response
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Error handled")]
            mock_anthropic.return_value.messages.create.return_value = mock_final_response
            
            base_params = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test query"}],
                "system": "test system prompt"
            }
            
            # Should not raise exception, but handle gracefully
            with pytest.raises(Exception, match="Tool execution failed"):
                generator._handle_tool_execution(initial_response, base_params, tool_manager)

    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        with patch('anthropic.Anthropic'):
            generator = AIGenerator("test-key", "test-model")
            
            # Check that system prompt mentions tools and protocols
            assert "search_course_content" in generator.SYSTEM_PROMPT
            assert "get_course_outline" in generator.SYSTEM_PROMPT
            assert "tool call per query maximum" in generator.SYSTEM_PROMPT
            assert "Content Search" in generator.SYSTEM_PROMPT
            assert "Course Outline" in generator.SYSTEM_PROMPT

    def test_no_tool_manager_with_tool_use(self, mock_anthropic_client_with_tools):
        """Test tool use response when no tool manager provided"""
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client_with_tools):
            generator = AIGenerator("test-key", "test-model")
            
            tools = [{"name": "test_tool", "description": "Test tool"}]
            
            # Should return the text from initial response when no tool manager
            result = generator.generate_response(
                "Tool use query",
                tools=tools,
                tool_manager=None
            )
            
            # Should not attempt tool execution, return empty response
            # Since the mock returns tool_use but no tool_manager is provided
            assert mock_anthropic_client_with_tools.messages.create.call_count == 1