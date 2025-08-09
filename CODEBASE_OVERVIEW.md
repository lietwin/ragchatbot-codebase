# Course Materials RAG System - Codebase Overview

## System Overview

This is a **Retrieval-Augmented Generation (RAG) system** designed for course materials that enables users to query educational content and receive intelligent, context-aware responses. The system combines semantic search capabilities with AI-powered response generation to provide accurate answers about course content.

### Core Purpose
- **Semantic Search**: Find relevant course content using vector embeddings
- **AI-Powered Responses**: Generate contextual answers using Anthropic's Claude
- **Course Management**: Organize and search across multiple courses and lessons
- **Conversation History**: Maintain session-based conversation context

### Architecture Style
The system follows a **modular, component-based architecture** with clear separation of concerns, making it maintainable and extensible.

## Technology Stack

### Backend Technologies
- **FastAPI**: Modern, fast web framework for building APIs
- **ChromaDB**: Vector database for semantic search and embeddings storage
- **Anthropic Claude**: AI model (claude-sonnet-4-20250514) for response generation
- **SentenceTransformers**: Embedding model (all-MiniLM-L6-v2) for text vectorization
- **Python 3.13+**: Core programming language
- **uv**: Modern Python package manager

### Frontend Technologies
- **Vanilla JavaScript**: Client-side functionality
- **HTML5/CSS3**: User interface with responsive design
- **Markdown Rendering**: Support for formatted AI responses

### Development Tools
- **pytest**: Testing framework with comprehensive fixtures
- **Black**: Code formatting
- **flake8**: Linting and style checking
- **isort**: Import statement organization
- **mypy**: Static type checking

## Key Components

### 1. RAGSystem (`backend/rag_system.py`)
**Central orchestrator** that coordinates all system components.

**Responsibilities:**
- Manages document processing workflow
- Coordinates vector storage operations
- Handles AI generation requests
- Integrates search tools and session management
- Provides unified interface for course operations

**Key Methods:**
- `add_course_document()`: Process and store individual course documents
- `add_course_folder()`: Batch process multiple course documents
- `query()`: Handle user queries with AI generation and search
- `get_course_analytics()`: Provide system statistics

### 2. VectorStore (`backend/vector_store.py`)
**ChromaDB-based vector storage** with dual collection architecture.

**Collections:**
- **`course_catalog`**: Stores course metadata and structure
  - Metadata: title, instructor, course_link, lesson_count, lessons_json
- **`course_content`**: Stores text chunks for semantic search
  - Metadata: course_title, lesson_number, chunk_index

**Key Features:**
- Semantic similarity search using embeddings
- Filtered search by course name and lesson number
- Intelligent course name matching with fuzzy search
- Persistent storage with ChromaDB

### 3. AIGenerator (`backend/ai_generator.py`)
**Anthropic Claude API integration** with advanced tool-calling capabilities.

**Features:**
- Tool-based search integration (up to 2 rounds of tool calls)
- Conversation history management
- Structured response generation
- Error handling and fallback mechanisms

**System Prompt Strategy:**
- Specialized for educational content
- Tool usage guidelines for content vs. outline queries
- Response formatting standards
- Direct, concise answer protocol

### 4. DocumentProcessor (`backend/document_processor.py`)
**Text processing and course metadata extraction** engine.

**Capabilities:**
- Multi-format document support (TXT, PDF, DOCX)
- Intelligent text chunking with sentence-based splitting
- Course structure extraction (titles, lessons, instructors)
- Configurable chunk size and overlap settings

**Processing Pipeline:**
1. File reading with encoding detection
2. Course metadata extraction using regex patterns
3. Text normalization and cleaning
4. Sentence-based chunking with overlap
5. CourseChunk object creation with metadata

### 5. SearchTools (`backend/search_tools.py`)
**Tool-based search system** for AI model integration.

**Available Tools:**
- **CourseSearchTool**: Semantic content search with course/lesson filtering
- **CourseOutlineTool**: Course structure and lesson information retrieval
- **ToolManager**: Tool registration and execution coordination

**Search Features:**
- Intelligent course name resolution
- Lesson-specific filtering
- Source tracking for response attribution
- Multi-round search capability

### 6. SessionManager (`backend/session_manager.py`)
**Conversation history and session management**.

**Features:**
- Session creation and tracking
- Message history with configurable limits
- Conversation context formatting
- Session clearing and cleanup

## Data Flow Architecture

### 1. Document Ingestion Pipeline
```
Course Documents (docs/) 
    ↓
DocumentProcessor (text chunking, metadata extraction)
    ↓
VectorStore (dual collections: catalog + content)
    ↓
ChromaDB (persistent vector storage)
```

### 2. Query Processing Pipeline
```
User Query
    ↓
SessionManager (conversation context)
    ↓
AIGenerator (Claude with tool access)
    ↓
SearchTools (semantic search)
    ↓
VectorStore (retrieve relevant content)
    ↓
AIGenerator (synthesize response)
    ↓
Response with sources
```

### 3. Tool-Based Search Flow
```
AI Model Query
    ↓
Tool Selection (content vs. outline)
    ↓
Course Name Resolution (fuzzy matching)
    ↓
Vector Search (filtered by course/lesson)
    ↓
Result Aggregation
    ↓
Source Attribution
```

## Directory Structure

```
ragchatbot-codebase/
├── backend/                    # Core application logic
│   ├── ai_generator.py        # Claude API integration
│   ├── app.py                 # FastAPI application and endpoints
│   ├── config.py              # Configuration management
│   ├── document_processor.py  # Document processing and chunking
│   ├── models.py              # Pydantic data models
│   ├── rag_system.py          # Main system orchestrator
│   ├── search_tools.py        # Tool-based search implementation
│   ├── session_manager.py     # Conversation session management
│   ├── vector_store.py        # ChromaDB vector storage
│   └── tests/                 # Comprehensive test suite
│       ├── conftest.py        # Test configuration and fixtures
│       ├── fixtures/          # Test data generators
│       ├── test_*.py          # Unit and integration tests
│       └── __init__.py
├── frontend/                   # Web interface
│   ├── index.html             # Main application page
│   ├── script.js              # Client-side functionality
│   └── style.css              # Responsive styling
├── docs/                       # Course materials storage
│   ├── course1_script.txt     # Sample course content
│   ├── course2_script.txt
│   ├── course3_script.txt
│   └── course4_script.txt
├── scripts/                    # Development utilities
│   ├── format.sh              # Code formatting (modifies files)
│   ├── lint.sh                # Code quality checks (read-only)
│   └── quality.sh             # Combined quality script
├── .env.example               # Environment variables template
├── .python-version            # Python version specification
├── CLAUDE.md                  # Development guidance for Claude
├── README.md                  # Project documentation
├── pyproject.toml             # Project configuration and dependencies
├── run.sh                     # Quick start script
└── uv.lock                    # Dependency lock file
```

## API Endpoints

### Core Endpoints (`backend/app.py`)

#### `POST /api/query`
**Process user queries and return AI-generated responses**

**Request:**
```json
{
  "query": "string",
  "session_id": "string (optional)"
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": ["string"],
  "source_links": ["string"],
  "session_id": "string"
}
```

#### `GET /api/courses`
**Retrieve course statistics and metadata**

**Response:**
```json
{
  "total_courses": "integer",
  "course_titles": ["string"]
}
```

#### `POST /api/clear-session`
**Clear conversation history for a session**

**Request:**
```json
{
  "session_id": "string"
}
```

### Static File Serving
- **`/`**: Serves frontend application (index.html)
- **Static assets**: CSS, JavaScript, and other frontend resources

### Middleware Configuration
- **CORS**: Configured for development with broad permissions
- **TrustedHost**: Allows all hosts for proxy compatibility
- **DevStaticFiles**: Custom static file handler with no-cache headers

## Configuration Management

### Configuration File (`backend/config.py`)

**Core Settings:**
```python
@dataclass
class Config:
    # AI Model Configuration
    ANTHROPIC_API_KEY: str          # Required API key
    ANTHROPIC_MODEL: str            # claude-sonnet-4-20250514
    
    # Embedding Configuration
    EMBEDDING_MODEL: str            # all-MiniLM-L6-v2
    
    # Document Processing
    CHUNK_SIZE: int = 800          # Text chunk size
    CHUNK_OVERLAP: int = 100       # Overlap between chunks
    MAX_RESULTS: int = 5           # Search result limit
    MAX_HISTORY: int = 2           # Conversation history limit
    
    # Storage
    CHROMA_PATH: str = "./chroma_db"  # Vector database location
```

### Environment Variables
**Required in `.env` file:**
```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Configurable Parameters
- **Chunk Processing**: Size and overlap for text segmentation
- **Search Behavior**: Maximum results and similarity thresholds
- **Conversation**: History length and session management
- **Storage**: Database paths and collection names

## Development Setup

### Prerequisites
- **Python 3.13+**: Required for modern language features
- **uv**: Modern Python package manager
- **Anthropic API Key**: For Claude AI integration

### Installation Steps

1. **Install uv package manager:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies:**
   ```bash
   uv sync                    # Production dependencies
   uv sync --group dev        # Include development tools
   ```

3. **Environment configuration:**
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

4. **Run the application:**
   ```bash
   # Quick start
   chmod +x run.sh
   ./run.sh
   
   # Manual start
   cd backend && uv run uvicorn app:app --reload --port 8000
   ```

### Development Commands

**Using uv for Python execution:**
```bash
uv run python script.py       # Run Python scripts
uv run uvicorn app:app         # Run web server
uv add package_name            # Add dependencies
```

**Application access:**
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Testing Approach

### Test Structure (`backend/tests/`)

**Test Categories:**
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **API Tests**: Endpoint functionality testing
- **System Tests**: End-to-end workflow testing

### Key Test Files

#### `conftest.py`
**Centralized test configuration and fixtures**
- Sample course and lesson data generators
- Mock component factories
- Test database setup and teardown
- Configuration overrides for testing

#### Test Coverage Areas
- **`test_rag_system.py`**: Core system orchestration
- **`test_vector_store.py`**: ChromaDB operations and search
- **`test_ai_generator.py`**: Claude API integration and tool calling
- **`test_search_tools.py`**: Tool-based search functionality
- **`test_app_endpoints.py`**: FastAPI endpoint testing
- **`test_config.py`**: Configuration management

### Testing Features
- **Fixtures**: Reusable test data and mock objects
- **Parametrized Tests**: Multiple scenario testing
- **Mock Integration**: External service mocking
- **Async Testing**: FastAPI endpoint testing
- **Type Safety**: Strict type checking in tests

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m api

# Run with coverage
uv run pytest --cov=backend
```

## Code Quality Tools

### Available Scripts (`scripts/`)

#### `format.sh` - Code Formatting (Modifies Files)
**Automatically fixes code style issues:**
```bash
./scripts/format.sh
```
**Actions:**
1. **isort**: Sorts and organizes import statements
2. **Black**: Formats code according to PEP 8 standards
3. **flake8**: Reports remaining linting issues
4. **mypy**: Performs static type checking

#### `lint.sh` - Quality Checks (Read-Only)
**Verifies code quality without modifications:**
```bash
./scripts/lint.sh
```
**Perfect for:**
- Pre-commit validation
- CI/CD pipeline integration
- Code review preparation

### Tool Configuration

#### Black Configuration (`pyproject.toml`)
```toml
[tool.black]
line-length = 88
target-version = ['py313']
```

#### isort Configuration
```toml
[tool.isort]
profile = "black"
multi_line_output = 3
line-length = 88
```

#### mypy Configuration
```toml
[tool.mypy]
python_version = "3.13"
disallow_untyped_defs = true
strict_equality = true
```

#### flake8 Configuration
```toml
[tool.flake8]
max-line-length = 88
extend-ignore = ["E203", "W503"]
```

## Unique System Features

### 1. Intelligent Course Name Matching
- Fuzzy search for course names in queries
- Partial name matching (e.g., "MCP" matches "MCP Server Implementation")
- Case-insensitive course resolution

### 2. Tool-Based AI Architecture
- Multiple search tools available to AI model
- Up to 2 rounds of tool calls for complex queries
- Automatic tool selection based on query type

### 3. Dual Collection Vector Storage
- Separate collections for course metadata and content
- Optimized search strategies for different query types
- Efficient filtering by course and lesson

### 4. Session-Based Conversation Management
- Persistent conversation history
- Configurable history limits
- Session isolation and cleanup

### 5. Comprehensive Source Attribution
- Detailed source tracking for all responses
- Lesson-level attribution with links
- Transparent information provenance

## Performance Considerations

### Vector Search Optimization
- **Embedding Caching**: Reuse embeddings for repeated queries
- **Collection Separation**: Optimized search paths for different query types
- **Result Limiting**: Configurable maximum results to control response time

### Memory Management
- **Conversation History Limits**: Prevents unbounded memory growth
- **Session Cleanup**: Automatic session management
- **Chunking Strategy**: Balanced chunk size for optimal retrieval

### Scalability Features
- **Persistent Storage**: ChromaDB for data persistence
- **Stateless Design**: Easy horizontal scaling
- **Configurable Parameters**: Tunable for different deployment sizes

## Deployment Considerations

### Development Mode
- **No-cache headers**: Ensures fresh static file serving
- **CORS permissive**: Allows cross-origin development
- **Auto-reload**: FastAPI development server with hot reloading

### Production Readiness
- **Environment Configuration**: Secure API key management
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging for monitoring
- **Health Checks**: Built-in endpoint monitoring

This comprehensive overview provides a complete understanding of the RAG system architecture, enabling developers to effectively work with, extend, and maintain the codebase.
