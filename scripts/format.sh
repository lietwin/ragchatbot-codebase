#!/bin/bash

# Auto-format code using black
# This script automatically formats all Python code in the project

set -e  # Exit on any error

echo "🎨 Auto-formatting code with black..."

# Change to project root
cd "$(dirname "$0")/.."

# Install dev dependencies if not already installed
echo "📦 Installing dev dependencies..."
uv sync --group dev

# Format all Python files
echo "🎨 Formatting Python files..."
uv run black backend/ main.py

echo "✅ Code formatting complete!"