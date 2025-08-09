"""Sample data fixtures for testing the RAG chatbot system"""

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults

# Sample lessons for testing
SAMPLE_LESSONS = [
    Lesson(
        lesson_number=0,
        title="Introduction",
        lesson_link="https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps/lesson/1/introduction",
    ),
    Lesson(
        lesson_number=1,
        title="Why MCP",
        lesson_link="https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps/lesson/2/why-mcp",
    ),
    Lesson(
        lesson_number=2,
        title="MCP Architecture",
        lesson_link="https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps/lesson/3/mcp-architecture",
    ),
]

# Sample courses for testing
SAMPLE_COURSES = [
    Course(
        title="MCP: Build Rich-Context AI Apps with Anthropic",
        course_link="https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps/",
        instructor="Elie Schoppik",
        lessons=SAMPLE_LESSONS,
    ),
    Course(
        title="Building Towards Computer Use with Anthropic",
        course_link="https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
        instructor="Colt Steele",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/1/introduction",
            ),
            Lesson(
                lesson_number=1,
                title="Multi-modal Requests",
                lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/2/multi-modal",
            ),
        ],
    ),
]

# Sample course chunks for testing
SAMPLE_CHUNKS = [
    CourseChunk(
        content="Welcome to MCP: Build Rich-Context AI Apps with Anthropic. The Model Context Protocol (MCP) is a standardized protocol that enables seamless integration between AI applications and external data sources.",
        course_title="MCP: Build Rich-Context AI Apps with Anthropic",
        lesson_number=0,
        chunk_index=0,
    ),
    CourseChunk(
        content="MCP solves the problem of AI applications being isolated from external data. By using MCP, your AI can access real-time data from databases, APIs, and other sources securely and efficiently.",
        course_title="MCP: Build Rich-Context AI Apps with Anthropic",
        lesson_number=1,
        chunk_index=1,
    ),
    CourseChunk(
        content="The MCP architecture consists of three main components: MCP servers that provide data access, MCP clients that consume data, and the MCP protocol that facilitates communication between them.",
        course_title="MCP: Build Rich-Context AI Apps with Anthropic",
        lesson_number=2,
        chunk_index=2,
    ),
    CourseChunk(
        content="Computer use capabilities enable AI models to interact directly with computer interfaces, taking screenshots and generating mouse clicks or keystrokes to complete tasks.",
        course_title="Building Towards Computer Use with Anthropic",
        lesson_number=0,
        chunk_index=0,
    ),
]

# Sample search results for testing
SAMPLE_SEARCH_RESULTS = SearchResults(
    documents=[
        "The Model Context Protocol (MCP) is a standardized protocol for AI applications.",
        "MCP enables seamless integration between AI and external data sources.",
        "Computer use capabilities allow AI to interact with computer interfaces.",
    ],
    metadata=[
        {
            "course_title": "MCP: Build Rich-Context AI Apps with Anthropic",
            "lesson_number": 0,
        },
        {
            "course_title": "MCP: Build Rich-Context AI Apps with Anthropic",
            "lesson_number": 1,
        },
        {
            "course_title": "Building Towards Computer Use with Anthropic",
            "lesson_number": 0,
        },
    ],
    distances=[0.1, 0.2, 0.3],
    error=None,
)

# Sample empty search results
EMPTY_SEARCH_RESULTS = SearchResults(
    documents=[], metadata=[], distances=[], error=None
)

# Sample error search results
ERROR_SEARCH_RESULTS = SearchResults(
    documents=[], metadata=[], distances=[], error="Database connection failed"
)

# Sample ChromaDB query results
SAMPLE_CHROMA_RESULTS = {
    "documents": [
        [
            "MCP: Build Rich-Context AI Apps with Anthropic",
            "Building Towards Computer Use with Anthropic",
        ]
    ],
    "metadatas": [
        [
            {
                "title": "MCP: Build Rich-Context AI Apps with Anthropic",
                "instructor": "Elie Schoppik",
                "course_link": "https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps/",
                "lessons_json": '[{"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": ""}]',
                "lesson_count": 1,
            },
            {
                "title": "Building Towards Computer Use with Anthropic",
                "instructor": "Colt Steele",
                "course_link": "https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
                "lessons_json": '[{"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": ""}]',
                "lesson_count": 1,
            },
        ]
    ],
    "distances": [[0.1, 0.2]],
}

# Sample Anthropic API responses
SAMPLE_ANTHROPIC_RESPONSE_TEXT = "Based on the course materials, MCP (Model Context Protocol) is a standardized protocol that enables AI applications to access external data sources securely."

SAMPLE_ANTHROPIC_TOOL_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "tool_use",
            "id": "tool_123",
            "name": "search_course_content",
            "input": {"query": "What is MCP?"},
        }
    ],
    "stop_reason": "tool_use",
}

SAMPLE_ANTHROPIC_FINAL_RESPONSE = {
    "id": "msg_456",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": SAMPLE_ANTHROPIC_RESPONSE_TEXT}],
    "stop_reason": "end_turn",
}
