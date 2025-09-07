#!/bin/bash

# Pascal AI Assistant Startup Script with Virtual Environment Management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found!"
    echo ""
    echo "Please run the installer first:"
    echo "  ./install.sh"
    echo ""
    echo "Or create virtual environment manually:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if virtual environment has the activation script
if [ ! -f "venv/bin/activate" ]; then
    print_error "Virtual environment appears corrupted!"
    echo "Recreate it with: ./install.sh"
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Verify we're in the virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_error "Failed to activate virtual environment"
    echo "Try manually:"
    echo "  source venv/bin/activate"
    echo "  python main.py"
    exit 1
fi

print_success "Virtual environment activated: $VIRTUAL_ENV"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    print_error "main.py not found!"
    echo "Make sure you're in the pascal directory"
    exit 1
fi

# Quick dependency check
print_status "Checking critical dependencies..."

# Check for aiohttp (critical for online functionality)
if ! python -c "import aiohttp" 2>/dev/null; then
    print_warning "aiohttp not found - online functionality may not work"
    echo "Install with: pip install aiohttp"
fi

# Check for Pascal modules
if ! python -c "from config.settings import settings" 2>/dev/null; then
    print_error "Pascal configuration not found"
    echo "Run the installer: ./install.sh"
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
echo "🔧 Troubleshooting:"
echo "   If Pascal doesn't work, run: python complete_diagnostic.py"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    print_warning ".env file not found"
    echo "For online functionality, copy .env.example to .env and add API keys"
    echo ""
fi

# Start Pascal
print_status "Starting Pascal..."
python main.py

# Cleanup message
echo ""
print_success "Pascal has stopped. Virtual environment remains active."
echo ""
echo "To deactivate virtual environment: deactivate"
echo "To restart Pascal: ./run.sh"
