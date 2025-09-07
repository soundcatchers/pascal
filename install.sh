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
    
    # Ensure pip and venv are available
    if ! python3 -m pip --version >/dev/null 2>&1; then
        print_status "Installing pip..."
        sudo apt install -y python3-pip
    fi
    
    if ! python3 -m venv --help >/dev/null 2>&1; then
        print_status "Installing venv..."
        sudo apt install -y python3-venv
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

# Create and activate virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists, removing old one..."
        rm -rf venv
    fi

    # Create virtual environment
    python3 -m venv venv
    
    # Test activation
    if [ ! -f "venv/bin/activate" ]; then
        print_error "Virtual environment creation failed"
        exit 1
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Verify we're in the virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_success "Virtual environment created and activated"
        print_status "Virtual environment path: $VIRTUAL_ENV"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi

    # Upgrade pip in virtual environment
    print_status "Upgrading pip in virtual environment..."
    python -m pip install --upgrade pip wheel setuptools
    
    # Verify pip is working
    pip_version=$(pip --version)
    print_success "Pip version: $pip_version"
}

# Install Python dependencies (no longer need llama-cpp-python!)
install_python_deps() {
    print_status "Installing Python packages (Ollama-optimized)..."
    
    # Ensure we're in virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_status "Activating virtual environment..."
        source venv/bin/activate
    fi
    
    # Verify we're in virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_error "Not in virtual environment"
        exit 1
    fi
    
    # Install requirements (much faster without llama-cpp-python)
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Verify critical packages
    print_status "Verifying critical packages..."
    
    # Test aiohttp (critical for online LLM functionality)
    if python -c "import aiohttp; print(f'aiohttp {aiohttp.__version__} installed successfully')" 2>/dev/null; then
        print_success "aiohttp installed and working"
    else
        print_error "aiohttp installation failed - this will break online LLM functionality"
        exit 1
    fi
    
    # Test other critical packages
    critical_packages=("requests" "openai" "anthropic" "rich" "colorama")
    for package in "${critical_packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            print_success "$package installed"
        else
            print_warning "$package may not be installed correctly"
        fi
    done
    
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
    chmod +x diagnose_ollama.py
    chmod +x diagnose_online_apis.py
    chmod +x test_groq_fix.py
    
    # Make logs directory writable
    chmod 755 logs
    
    print_success "Permissions set"
}

# Create initial configuration
create_config() {
    print_status "Creating initial configuration..."
    
    # Activate virtual environment for Python scripts
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source venv/bin/activate
    fi
    
    # Run installer utility
    python utils/installer.py
    
    # Create Ollama-optimized .env if it doesn't exist
    if [ ! -f ".env" ]; then
        print_status "Creating Ollama-optimized .env configuration..."
        cat > .env << 'EOF'
# Pascal AI Assistant Environment Variables - Pi 5 Optimized with Ollama

# Performance settings (Ollama manages local models)
PERFORMANCE_MODE=balanced
STREAMING_ENABLED=true
KEEP_ALIVE_ENABLED=true
TARGET_RESPONSE_TIME=3.0
MAX_RESPONSE_TOKENS=200

# API Keys (add your keys here for online fallback)
# GROQ_API_KEY=gsk-your_groq_api_key_here
# GEMINI_API_KEY=your_gemini_api_key_here
# OPENAI_API_KEY=sk-your_openai_api_key_here

# Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=30
OLLAMA_KEEP_ALIVE=5m

# Debug settings
DEBUG=false
LOG_LEVEL=INFO
PERF_LOG=false

# Advanced settings
MAX_CONCURRENT_REQUESTS=2
CACHE_EXPIRY=3600
LLM_THREADS=4
LLM_CONTEXT=2048
EOF
        print_success "Created Ollama-optimized .env configuration"
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Ensure we're in virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source venv/bin/activate
    fi
    
    # Test Python imports
    python -c "
import sys
print(f'Python version: {sys.version}')
print(f'Virtual environment: {sys.prefix}')

try:
    import aiohttp
    print('✅ aiohttp installed successfully')
except ImportError as e:
    print('❌ aiohttp import failed:', e)
    sys.exit(1)

try:
    from config.settings import settings
    print('✅ Pascal configuration loaded')
    hw_info = settings.get_hardware_info()
    print(f'Hardware detected: {hw_info}')
except ImportError as e:
    print('❌ Pascal configuration failed:', e)
    sys.exit(1)

try:
    import requests
    import rich
    import colorama
    print('✅ Core dependencies imported successfully')
except ImportError as e:
    print('❌ Core dependency import failed:', e)
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

# Test virtual environment persistence
test_venv_persistence() {
    print_status "Testing virtual environment persistence..."
    
    # Test run.sh script
    if [ -f "run.sh" ]; then
        # Test that run.sh can activate the virtual environment
        bash -c "source venv/bin/activate && python -c 'import sys; print(f\"Virtual env test: {sys.prefix}\")'"
        if [ $? -eq 0 ]; then
            print_success "Virtual environment activation test passed"
        else
            print_warning "Virtual environment activation test failed"
        fi
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
            # Run in a subshell to preserve our virtual environment
            (./download_models.sh)
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
    echo "• Virtual Environment: $(pwd)/venv"
    echo ""
    
    # Show next steps
    print_status "Next Steps:"
    echo "1. Start Pascal:"
    echo "   ./run.sh"
    echo ""
    echo "2. Install Ollama and models (if not done already):"
    echo "   ./download_models.sh"
    echo ""
    echo "3. Configure API keys in .env file (optional for online fallback):"
    echo "   nano .env"
    echo "   Add your GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY"
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
    echo "• Use nemotron-mini:4b-instruct-q4_K_M for fastest responses"
    echo "• Use qwen2.5:3b for balanced performance"
    echo "• Use phi3:mini for minimal resources"
    echo "• Monitor temperature: vcgencmd measure_temp"
    echo "• Type 'status' in Pascal to see system information"
    echo ""
    
    # Show virtual environment info
    print_status "Virtual Environment:"
    echo "• Location: $(pwd)/venv"
    echo "• Activation: source venv/bin/activate (done automatically by run.sh)"
    echo "• Python: $(source venv/bin/activate && python --version)"
    echo ""
    
    if [ "$PI_VERSION" = "5" ]; then
        print_warning "Reboot recommended to apply Pi 5 optimizations"
    fi
    
    print_success "Happy chatting with Pascal! 🤖"
    echo ""
    print_status "To start Pascal: ./run.sh"
}

# Main installation flow
main() {
    # Perform installation steps
    check_hardware
    check_python
    update_system
    setup_venv
    install_python_deps
    setup_directories
    set_permissions
    create_config
    
    # Test installation
    if test_installation; then
        test_venv_persistence
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
    # Deactivate virtual environment if active
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        deactivate 2>/dev/null || true
    fi
    exit 1
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Run main installation
main "$@"
