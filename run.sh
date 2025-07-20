#!/bin/bash

# Pascal AI Assistant Startup Script

set -e

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found!"
    echo "Make sure you're in the pascal directory"
    exit 1
fi

# Display startup message
echo ""
echo "🤖 Starting Pascal AI Assistant..."
echo "=================================="
echo ""
echo "💡 Commands:"
echo "   'quit' or 'exit' - Stop Pascal"
echo "   'help' - Show available commands"
echo "   'status' - Show system status"
echo ""

# Start Pascal
python3 main.py

echo ""
echo "👋 Pascal has stopped. Goodbye!"
