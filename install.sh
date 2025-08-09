#!/bin/bash

# Pascal AI Assistant Installer - Raspberry Pi 5 Optimized (Ollama Version)
# Automated setup script with ARM optimizations and Ollama integration

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🤖 Installing Pascal AI Assistant (Pi 5 Optimized with Ollama)"
echo "=============================================================="

# Check if running on Raspberry Pi
check_hardware() {
    print_status "Checking hardware compatibility..."
    
    if [ -f /proc/device-tree/model ]; then
        PI_MODEL=$(cat /proc/device-tree/model)
        if [[ $PI_MODEL == *"Raspberry Pi 5"* ]]; then
            print_success "Detected Raspberry Pi 5 - optimal compatibility"
            PI_VERSION="5"
        elif [[ $PI_MODEL == *"Raspberry Pi 4"* ]]; then
            print_warning "Detected Raspberry Pi 4 - good compatibility with Ollama"
            PI_VERSION="4"
        elif [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
            print_warning "Detected older Raspberry Pi - Ollama should still work"
            PI_VERSION="older"
            read -p "Continue installation? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            print_warning "Non-Raspberry Pi hardware detected - Ollama is cross-platform"
            PI_VERSION="unknown"
        fi
    else
        print_warning "Could not detect hardware type"
        PI_VERSION="unknown"
    fi
    
    # Check RAM
    TOTAL_RAM_MB=$(free -m | grep '^Mem:' | awk '{print $2}')
    TOTAL_RAM_GB=$((TOTAL_RAM_MB / 1024))
    print_status "System RAM: ${TOTAL_RAM_GB}GB"
    
    if [ $TOTAL_RAM_GB -lt 4 ]; then
        print_error "Insufficient RAM. Pascal with Ollama requires at least 4GB RAM."
        exit 1
    elif [ $TOTAL_RAM_GB -lt 8 ]; then
        print_warning "Limited RAM detected. Consider using smaller models."
    fi
    
    # Check storage space
    AVAILABLE_GB=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    print_status "Available storage: ${AVAILABLE_GB}GB"
    
    if [ $AVAILABLE_GB -lt 15 ]; then
        print_error "Insufficient storage space. Need at least 15GB free for Ollama and models."
        exit 1
    fi
}

# Check Python version
check_python() {
    print_status "Checking Python installation..."
    
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+' || echo "0.0")
    REQUIRED_VERSION="3.8"

    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        print_error "Python 3.8+ required. Found: $PYTHON_VERSION"
        print_status "Installing Python 3.8+..."
        sudo apt update
        sudo apt install -y python3 python3-pip python3-venv python3-dev
    else
        print_success "Python $PYTHON_VERSION detected"
    fi
}

# Update system packages
update_system() {
    print_status "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    
    print_status "Installing system dependencies..."
    sudo apt install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        git \
        curl \
        wget \
        build-essential \
        cmake \
        pkg-config \
        libasound2-dev \
        portaudio19-dev \
        libffi-dev \
        libssl-dev \
        htop \
        iotop \
        stress-ng \
        jq
    
    # Pi 5 specific optimizations
    if [ "$PI_VERSION" = "5" ]; then
        print_status "Applying Pi 5 specific optimizations..."
        
        # Check if config.txt modifications are needed
        if ! grep -q "gpu_mem=128" /boot/config.txt; then
            print_status "Optimizing GPU memory allocation..."
            echo "gpu_mem=128" | sudo tee -a /boot/config.txt
        fi
        
        if ! grep -q "arm_boost=1" /boot/config.txt; then
            print_status "Enabling ARM boost..."
            echo "arm_boost=1" | sudo tee -a /boot/config.txt
        fi
        
        print_warning "System optimizations applied. Reboot recommended after installation."
    fi
}

# Create virtual environment
setup_venv() {
    print_status "Creating Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists, removing old one..."
        rm -rf venv
    fi

    python3 -m venv venv
    source venv/bin/activate

    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip wheel setuptools
}

# Install Python dependencies (no longer need llama-cpp-python!)
install_python_deps() {
    print_status "Installing Python packages (Ollama-optimized)..."
    source venv/bin/activate
    
    # Install requirements (much faster without llama-cpp-python)
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    print_success "Python dependencies installed (no compilation needed with Ollama!)"
}

# Create directory structure
setup_directories() {
    print_status "Creating data directories..."
    
    mkdir -p data/models
    mkdir -p data/memory
    mkdir -p data/personalities
    mkdir -p data/cache
    mkdir -p logs
    
    # Create .gitkeep files
    touch data/cache/.gitkeep
    touch logs/.gitkeep
    
    print_success "Directory structure created"
}

# Set permissions
set_permissions() {
    print_status "Setting permissions..."
    
    chmod +x run.sh
    chmod +x download_models.sh
    chmod +x test_performance.py
    
    # Make logs directory writable
    chmod 755 logs
    
    print_success "Permissions set"
}

# Create initial configuration
create_config() {
    print_status "Creating initial configuration..."
    
    # Activate virtual environment for Python scripts
    source venv/bin/activate
    
    # Run installer utility
    python3 utils/installer.py
    
    # Create Ollama-optimized .env if it doesn't exist
    if [ ! -f ".env" ]; then
        print_status "Creating Ollama-optimized .env configuration..."
        cat > .env << EOF
# Pascal AI Assistant Environment Variables - Pi 5 Optimized with Ollama

# Performance settings (Ollama manages local models)
PERFORMANCE_MODE=balanced

# Debug settings
DEBUG=false
LOG_LEVEL=INFO
PERF_LOG=false

# API Keys (add your keys here for online fallback)
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here

# Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=30

# Advanced settings
MAX_CONCURRENT_REQUESTS=2
CACHE_EXPIRY=3600
EOF
        print_success "Created Ollama-optimized .env configuration"
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    source venv/bin/activate
    
    # Test Python imports
    python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import aiohttp
    print('✅ aiohttp installed successfully')
except ImportError as e:
    print('❌ aiohttp import failed:', e)
    sys.exit(1)

try:
    from config.settings import settings
    print('✅ Pascal configuration loaded')
    print(f'Hardware detected: {settings.get_hardware_info()}')
except ImportError as e:
    print('❌ Pascal configuration failed:', e)
    sys.exit(1)

print('✅ Installation test passed')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Offer Ollama installation
offer_ollama_installation() {
    echo ""
    print_status "Ollama Installation"
    echo "Pascal uses Ollama for local AI models. You can:"
    echo "1. Install Ollama and download models now (recommended)"
    echo "2. Install Ollama later with ./download_models.sh"
    echo "3. Skip Ollama (online-only mode)"
    echo ""
    
    read -p "Choose option (1-3): " ollama_choice
    
    case $ollama_choice in
        1)
            print_status "Installing Ollama and downloading models..."
            chmod +x download_models.sh
            ./download_models.sh
            ;;
        2)
            print_status "Ollama can be installed later with: ./download_models.sh"
            ;;
        3)
            print_warning "Skipping Ollama. Pascal will work in online-only mode."
            print_status "You can install Ollama later if needed."
            ;;
        *)
            print_warning "Invalid choice. Ollama can be installed later."
            ;;
    esac
}

# Display completion message
show_completion() {
    echo ""
    print_success "Pascal installation complete! 🎉"
    echo "========================================="
    echo ""
    
    # Show system info
    print_status "System Information:"
    echo "• Hardware: $PI_MODEL"
    echo "• RAM: ${TOTAL_RAM_GB}GB"
    echo "• Storage: ${AVAILABLE_GB}GB available"
    echo "• Python: $(python3 --version)"
    echo ""
    
    # Show next steps
    print_status "Next Steps:"
    echo "1. Start Pascal:"
    echo "   ./run.sh"
    echo ""
    echo "2. Install Ollama and models (if not done already):"
    echo "   ./download_models.sh"
    echo ""
    echo "3. Configure API keys in .env file (optional for online fallback)"
    echo ""
    
    # Show Ollama advantages
    print_status "Ollama Advantages:"
    echo "• ✅ No compilation needed (much faster installation)"
    echo "• ✅ Better ARM optimization for Pi 5"
    echo "• ✅ Automatic model management"
    echo "• ✅ Easy model switching"
    echo "• ✅ Built-in quantization"
    echo "• ✅ Reliable downloads"
    echo ""
    
    # Show performance tips
    print_status "Performance Tips:"
    echo "• Use phi3:mini for fastest responses on Pi 5"
    echo "• Use llama3.2:3b for balanced performance"
    echo "• Use gemma2:2b for minimal resource usage"
    echo "• Monitor temperature: vcgencmd measure_temp"
    echo "• Type 'status' in Pascal to see system information"
    echo ""
    
    if [ "$PI_VERSION" = "5" ]; then
        print_warning "Reboot recommended to apply Pi 5 optimizations"
    fi
    
    print_success "Happy chatting with Pascal! 🤖"
}

# Main installation flow
main() {
    # Perform installation steps
    check_hardware
    check_python
    update_system
    setup_venv
    install_python_deps  # Much faster without llama-cpp-python!
    setup_directories
    set_permissions
    create_config
    
    # Test installation
    if test_installation; then
        offer_ollama_installation
        show_completion
    else
        print_error "Installation failed during testing"
        exit 1
    fi
}

# Handle interruption
cleanup() {
    print_warning "Installation interrupted"
    exit 1
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Run main installation
main "$@"
