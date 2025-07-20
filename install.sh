#!/bin/bash

# Pascal AI Assistant Installer
# Automated setup script for Raspberry Pi 5

set -e  # Exit on any error

echo "🤖 Installing Pascal AI Assistant..."
echo "=================================="

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️  Warning: This script is optimized for Raspberry Pi 5"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+' || echo "0.0")
REQUIRED_VERSION="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Python 3.8+ required. Found: $PYTHON_VERSION"
    echo "Please upgrade Python and try again."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    libasound2-dev \
    portaudio19-dev \
    python3-dev \
    libffi-dev \
    libssl-dev

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, removing old one..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/models
mkdir -p data/memory
mkdir -p data/personalities
mkdir -p data/cache

# Set permissions
echo "🔐 Setting permissions..."
chmod +x run.sh

# Create initial config files
echo "⚙️  Creating initial configuration..."
python3 utils/installer.py

echo ""
echo "🎉 Pascal installation complete!"
echo "================================"
echo ""
echo "To start Pascal:"
echo "  ./run.sh"
echo ""
echo "To activate the virtual environment manually:"
echo "  source venv/bin/activate"
echo ""
echo "Happy chatting with Pascal! 🤖"
