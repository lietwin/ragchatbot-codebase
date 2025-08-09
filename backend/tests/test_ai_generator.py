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

        with patch("anthropic.Anthropic") as mock_anthropic:
            generator = AIGenerator(api_key, model)

            # Verify Anthropic client initialized with correct API key
            mock_anthropic.assert_called_once_with(api_key=api_key)
            assert generator.model == model
            assert generator.base_params["model"] == model
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800

    def test_generate_response_without_tools(self, mock_anthropic_client):
        """Test basic response generation without tools"""
        with patch("anthropic.Anthropic", return_value=mock_anthropic_client):
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
        with patch("anthropic.Anthropic", return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")

            history = "Previous conversation context"
            result = generator.generate_response(
                "Current question", conversation_history=history
            )

            # Verify system prompt includes history
            call_args = mock_anthropic_client.messages.create.call_args[1]
            assert history in call_args["system"]

    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client):
        """Test response generation with tools available but not used"""
        with patch("anthropic.Anthropic", return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "test-model")

            tools = [{"name": "test_tool", "description": "Test tool"}]
            tool_manager = Mock()

            result = generator.generate_response(
                "Simple question", tools=tools, tool_manager=tool_manager
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
        with patch(
            "anthropic.Anthropic", return_value=mock_anthropic_client_with_tools
        ):
            generator = AIGenerator("test-key", "test-model")

            tools = [{"name": "search_course_content", "description": "Search courses"}]
            tool_manager = Mock()
            tool_manager.execute_tool.return_value = "Tool execution result"

            result = generator.generate_response(
                "Search for MCP information", tools=tools, tool_manager=tool_manager
            )

            # Verify tool was executed
            tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="test query"
            )

            # Verify final response after tool execution
            assert result == "Final response after tool execution"

            # Verify API was called twice (initial + final)
            assert mock_anthropic_client_with_tools.messages.create.call_count == 2

    def test_api_error_handling(self):
        """Test handling of Anthropic API errors"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = mock_anthropic.return_value
            mock_client.messages.create.side_effect = Exception("API Error")

            generator = AIGenerator("test-key", "test-model")

            # Should propagate the exception
            with pytest.raises(Exception, match="API Error"):
                generator.generate_response("test query")

    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        with patch("anthropic.Anthropic"):
            generator = AIGenerator("test-key", "test-model")

            # Check that system prompt mentions tools and protocols
            assert "search_course_content" in generator.SYSTEM_PROMPT
            assert "get_course_outline" in generator.SYSTEM_PROMPT
            assert (
                "two sequential tool calling rounds per query"
                in generator.SYSTEM_PROMPT
            )
            assert "Strategic tool usage" in generator.SYSTEM_PROMPT
            assert "Content Search" in generator.SYSTEM_PROMPT
            assert "Course Outline" in generator.SYSTEM_PROMPT

    def test_no_tool_manager_with_tool_use(self, mock_anthropic_client_with_tools):
        """Test tool use response when no tool manager provided"""
        with patch(
            "anthropic.Anthropic", return_value=mock_anthropic_client_with_tools
        ):
            generator = AIGenerator("test-key", "test-model")

            tools = [{"name": "test_tool", "description": "Test tool"}]

            # Should return the text from initial response when no tool manager
            result = generator.generate_response(
                "Tool use query", tools=tools, tool_manager=None
            )

            # Should not attempt tool execution, return empty response
            # Since the mock returns tool_use but no tool_manager is provided
            assert mock_anthropic_client_with_tools.messages.create.call_count == 1

    def test_two_round_tool_sequence(self, mock_anthropic_client_two_rounds):
        """Test complete 2-round tool calling sequence"""
        with patch(
            "anthropic.Anthropic", return_value=mock_anthropic_client_two_rounds
        ):
            generator = AIGenerator("test-key", "test-model")

            tools = [
                {"name": "get_course_outline", "description": "Get course outline"},
                {"name": "search_course_content", "description": "Search content"},
            ]

            tool_manager = Mock()
            tool_manager.execute_tool.side_effect = [
                "Course outline result",  # Round 1
                "Search content result",  # Round 2
            ]

            result = generator.generate_response(
                "Find a course similar to lesson 4 of MCP course",
                tools=tools,
                tool_manager=tool_manager,
            )

            # Verify both tools were executed in sequence
            assert tool_manager.execute_tool.call_count == 2
            tool_manager.execute_tool.assert_any_call(
                "get_course_outline", course_title="MCP"
            )
            tool_manager.execute_tool.assert_any_call(
                "search_course_content", query="MCP fundamentals"
            )

            # Verify 3 API calls: round1, round2, final
            assert mock_anthropic_client_two_rounds.messages.create.call_count == 3

            assert result == "Final response after two rounds"

    def test_single_round_termination(self, mock_anthropic_client_terminating):
        """Test early termination when no more tools needed after first round"""
        with patch(
            "anthropic.Anthropic", return_value=mock_anthropic_client_terminating
        ):
            generator = AIGenerator("test-key", "test-model")

            tools = [{"name": "search_course_content", "description": "Search content"}]
            tool_manager = Mock()
            tool_manager.execute_tool.return_value = "Search result"

            result = generator.generate_response(
                "Search for MCP information", tools=tools, tool_manager=tool_manager
            )

            # Verify only one tool execution
            tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="test query"
            )

            # Verify only 2 API calls: round1 with tools, round2 terminates
            assert mock_anthropic_client_terminating.messages.create.call_count == 2

            assert result == "Final response after one round"

    def test_max_rounds_termination(self, mock_anthropic_client_two_rounds):
        """Test termination after maximum 2 rounds"""
        with patch(
            "anthropic.Anthropic", return_value=mock_anthropic_client_two_rounds
        ):
            generator = AIGenerator("test-key", "test-model")

            tools = [
                {"name": "get_course_outline", "description": "Get course outline"},
                {"name": "search_course_content", "description": "Search content"},
            ]

            tool_manager = Mock()
            tool_manager.execute_tool.side_effect = [
                "Course outline result",
                "Search content result",
            ]

            result = generator.generate_response(
                "Complex multi-step query", tools=tools, tool_manager=tool_manager
            )

            # Should reach max 2 rounds and then make final call without tools
            assert tool_manager.execute_tool.call_count == 2
            assert mock_anthropic_client_two_rounds.messages.create.call_count == 3

            # Final call should be without tools
            final_call_args = (
                mock_anthropic_client_two_rounds.messages.create.call_args_list[2][1]
            )
            assert "tools" not in final_call_args

            assert result == "Final response after two rounds"

    def test_tool_execution_error_terminates_rounds(self):
        """Test termination when tool execution fails"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Mock response with tool use
            mock_tool_content = Mock()
            mock_tool_content.type = "tool_use"
            mock_tool_content.name = "failing_tool"
            mock_tool_content.input = {"param": "value"}
            mock_tool_content.id = "tool_123"

            mock_response = Mock()
            mock_response.content = [mock_tool_content]
            mock_response.stop_reason = "tool_use"

            mock_anthropic.return_value.messages.create.return_value = mock_response

            generator = AIGenerator("test-key", "test-model")

            tools = [{"name": "failing_tool", "description": "Tool that fails"}]
            tool_manager = Mock()
            tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

            result = generator.generate_response(
                "Query that triggers tool failure",
                tools=tools,
                tool_manager=tool_manager,
            )

            # Should only make one API call before terminating
            assert mock_anthropic.return_value.messages.create.call_count == 1

            # Should return current response when tool execution fails
            assert result == mock_response.content[0].text

    def test_conversation_context_preservation(self):
        """Test that conversation context is preserved across rounds"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            generator = AIGenerator("test-key", "test-model")

            # Mock responses for testing context preservation
            mock_responses = []

            # Round 1 response with tool use
            mock_tool_content = Mock()
            mock_tool_content.type = "tool_use"
            mock_tool_content.name = "search_tool"
            mock_tool_content.input = {"query": "test"}
            mock_tool_content.id = "tool_123"

            round1_response = Mock()
            round1_response.content = [mock_tool_content]
            round1_response.stop_reason = "tool_use"
            mock_responses.append(round1_response)

            # Round 2 response (terminating)
            round2_response = Mock()
            round2_response.content = [Mock(text="Final answer")]
            round2_response.stop_reason = "end_turn"
            mock_responses.append(round2_response)

            mock_anthropic.return_value.messages.create.side_effect = mock_responses

            tools = [{"name": "search_tool", "description": "Search tool"}]
            tool_manager = Mock()
            tool_manager.execute_tool.return_value = "Tool result"

            result = generator.generate_response(
                "Test query", tools=tools, tool_manager=tool_manager
            )

            # Verify that we made 2 API calls
            assert mock_anthropic.return_value.messages.create.call_count == 2

            # Verify conversation context grew properly
            call_args_list = mock_anthropic.return_value.messages.create.call_args_list

            # The test is showing that both calls are receiving accumulated messages
            # This is actually correct behavior - the messages list is shared and grows

            # Round 1: starts with just user message
            round1_messages = call_args_list[0][1]["messages"]
            assert round1_messages[0]["role"] == "user"
            assert round1_messages[0]["content"] == "Test query"

            # Round 2: should have grown to include assistant response and tool results
            round2_messages = call_args_list[1][1]["messages"]
            assert len(round2_messages) >= 3  # At least user + assistant + tool results
            assert round2_messages[0]["role"] == "user"  # Original query
            assert round2_messages[1]["role"] == "assistant"  # Tool use response
            assert round2_messages[2]["role"] == "user"  # Tool results
            assert round2_messages[2]["content"][0]["content"] == "Tool result"

            assert result == "Final answer"
