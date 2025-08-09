import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Tool Usage:
- **Content Search**: Use `search_course_content` for questions about specific course content or detailed educational materials
- **Course Outline**: Use `get_course_outline` for questions about course structure, lesson lists, or course overviews
- **Maximum two sequential tool calling rounds per query**
- **Strategic tool usage**: Use first round for initial research, second round for follow-up or refinement
- **Multiple tools per round**: You can call multiple tools within each round when needed
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use content search tool first, then answer
- **Course outline/structure questions**: Use outline tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

For outline queries, include:
- Course title and course link (if available)  
- Complete lesson structure with lesson numbers and titles
- Lesson links where available

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with up to 2 sequential tool calling rounds.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize conversation with user query
        messages = [{"role": "user", "content": query}]
        
        # Execute up to 2 rounds of conversation if tools are available
        if tools and tool_manager:
            return self._execute_conversation_rounds(messages, system_content, tools, tool_manager)
        else:
            # No tools - single API call
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            response = self.client.messages.create(**api_params)
            return response.content[0].text
    
    def _execute_conversation_rounds(self, messages: List[Dict], system_content: str, 
                                   tools: List, tool_manager) -> str:
        """
        Execute up to 2 sequential rounds of tool calling.
        
        Args:
            messages: Initial conversation messages
            system_content: System prompt content
            tools: Available tools
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after all rounds
        """
        MAX_ROUNDS = 2
        current_round = 1
        
        while current_round <= MAX_ROUNDS:
            # API call for current round
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
                "tools": tools,
                "tool_choice": {"type": "auto"}
            }
            
            try:
                response = self.client.messages.create(**api_params)
            except Exception as e:
                # API error - return error message or last response if available
                if len(messages) > 1:
                    return "An error occurred while processing your request."
                raise e
            
            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response.content})
            
            # Check if tools were used
            if response.stop_reason != "tool_use":
                # No tools used - return the response
                return response.content[0].text
            
            # Execute tools and add results to conversation
            tool_results = []
            tool_execution_failed = False
            
            for content_block in response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name, **content_block.input
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result
                        })
                    except Exception as e:
                        # Tool execution failed - terminate rounds
                        tool_execution_failed = True
                        break
            
            if tool_execution_failed or not tool_results:
                # Return current response if tool execution fails
                return response.content[0].text if response.content else "Tool execution failed"
            
            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})
            
            # Move to next round
            current_round += 1
        
        # Maximum rounds reached - make final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
            # No tools in final call
        }
        
        try:
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except Exception as e:
            # If final API call fails, return last available response
            return "An error occurred while generating the final response."
    
