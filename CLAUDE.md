# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the application
```bash
./run.sh
```
Or manually:
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package management
- Install dependencies: `uv sync`
- Install dev dependencies: `uv sync --group dev`
- Python package manager: `uv` (not pip)

### Code Quality
- Format code: `./scripts/format.sh` or `uv run black backend/ main.py`
- Run quality checks: `./scripts/quality.sh`
- Individual tools:
  - Black formatting: `uv run black backend/ main.py`
  - Flake8 linting: `uv run flake8 backend/ main.py --max-line-length=88`
  - MyPy type checking: `uv run mypy backend/ main.py --ignore-missing-imports`

### Git operations
- Always use `gh` CLI for git operations (not direct git commands)
- No Claude attribution in commit messages per company rules

### Environment setup
- Required: Create `.env` file with `ANTHROPIC_API_KEY=your_key_here`
- Python version: 3.13+

## Architecture Overview

This is a full-stack RAG (Retrieval-Augmented Generation) chatbot system for querying course materials:

### Backend Structure (`/backend`)
- **FastAPI application** (`app.py`) - Main web server with API endpoints
- **RAG System** (`rag_system.py`) - Core orchestrator connecting all components
- **Vector Store** (`vector_store.py`) - ChromaDB-based vector storage with two collections:
  - `course_catalog`: Course metadata (titles, instructors)  
  - `course_content`: Chunked course content for semantic search
- **AI Generator** (`ai_generator.py`) - Anthropic Claude integration with tool support
- **Document Processor** (`document_processor.py`) - Processes course documents into structured data
- **Search Tools** (`search_tools.py`) - Tool-based search system for AI agent
- **Session Manager** (`session_manager.py`) - Conversation history management

### Frontend (`/frontend`)
- Simple HTML/CSS/JS interface served as static files
- Communicates with backend via `/api/query` and `/api/courses` endpoints

### Key Data Models (`models.py`)
- `Course`: Contains title, instructor, lessons list
- `CourseChunk`: Content chunks with course/lesson metadata
- `Lesson`: Individual lesson with title and links

### Tool-Based Architecture
The system uses a tool-based approach where the AI agent can call search functions rather than direct vector similarity search. This allows for more sophisticated query handling and course resolution.

### Vector Storage Strategy
- Dual collection system separates course metadata from content
- Course names are resolved via semantic search before content filtering
- Supports filtering by course title and lesson number

### API Endpoints
- `POST /api/query` - Main chat endpoint (returns answer + sources)
- `GET /api/courses` - Course analytics and statistics
- Static files served at root (`/`)

### Document Loading
Course documents are loaded from `/docs` folder on startup. Supports PDF, DOCX, and TXT files.