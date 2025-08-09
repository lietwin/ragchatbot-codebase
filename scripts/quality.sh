#!/bin/bash

# Quality check script for the RAG chatbot codebase
# This script runs code formatting, linting, and type checking

set -e  # Exit on any error

echo "🔍 Running code quality checks..."

# Change to project root
cd "$(dirname "$0")/.."

# Install dev dependencies if not already installed
echo "📦 Installing dev dependencies..."
uv sync --group dev

# Format code with black
echo "🎨 Formatting code with black..."
uv run black backend/ main.py --check --diff

# Run flake8 linting
echo "🔍 Running flake8 linting..."
uv run flake8 backend/ main.py --max-line-length=88 --extend-ignore=E203,W503

# Run mypy type checking
echo "🔬 Running mypy type checking..."
uv run mypy backend/ main.py --ignore-missing-imports

echo "✅ All quality checks passed!"