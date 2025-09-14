#!/bin/bash

# Pascal AI Assistant FIXED Installer - Simplified for Nemotron + Groq
# Optimized for Raspberry Pi 5 with single offline/online LLM

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

echo "🤖 Installing Pascal AI Assistant - Simplified (Nemotron + Groq Only)"
echo "======================================================================"

# Check if running on Raspberry Pi
check_hardware() {
    print_status "Checking hardware compatibility..."
    
    if [ -f /proc/device-tree/model ]; then
        PI_MODEL=$(cat /proc/device-tree/model)
        if [[ $PI_MODEL == *"Raspberry Pi 5"* ]]; then
            print_success "Detected Raspberry Pi 5 - optimal compatibility"
            PI_VERSION="5"
        elif [[ $PI_MODEL == *"Raspberry Pi 4"* ]]; then
            print_warning "Detected Raspberry Pi 4 - good compatibility"
            PI_VERSION="4"
        elif [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
            print_warning "Detected older Raspberry Pi"
            PI_VERSION="older"
        else
            print_warning "Non-Raspberry Pi hardware detected"
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
        print_error "Insufficient RAM. Pascal requires at least 4GB RAM."
        exit 1
    elif [ $TOTAL_RAM_GB -lt 8 ]; then
        print_warning "Limited RAM detected. Consider using smaller models."
    fi
    
    # Check storage space
    AVAILABLE_GB=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    print_status "Available storage: ${AVAILABLE_GB}GB"
    
    if [ $AVAILABLE_GB -lt 10 ]; then
        print_error "Insufficient storage space. Need at least 10GB free."
        exit 1
    fi
}

# Check Python version
check_python() {
    print_status "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Installing..."
        sudo apt update
        sudo apt install -y python3 python3-pip python3-venv python3-dev
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+' || echo "0.0")
    
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        print_error "Python 3.8+ required. Found: $PYTHON_VERSION"
        print_status "Updating Python..."
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
        htop \
        jq
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

# Install Python dependencies (simplified)
install_python_deps() {
    print_status "Installing Python packages (simplified for Nemotron + Groq)..."
    
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
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Verify critical packages
    print_status "Verifying critical packages..."
    
    # Test aiohttp (critical for Groq API)
    if python -c "import aiohttp; print(f'✅ aiohttp {aiohttp.__version__} installed successfully')" 2>/dev/null; then
        print_success "aiohttp installed and working"
    else
        print_error "aiohttp installation failed - this will break Groq functionality"
        exit 1
    fi
    
    # Test other critical packages
    critical_packages=("requests" "rich" "colorama" "psutil")
    for package in "${critical_packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            print_success "$package installed"
        else
            print_warning "$package may not be installed correctly"
        fi
    done
    
    print_success "Python dependencies installed successfully!"
}

# Create directory structure
setup_directories() {
    print_status "Creating data directories..."
    
    mkdir -p data/models
    mkdir -p data/memory
    mkdir -p data/cache
    mkdir -p config/personalities
    mkdir -p logs
    
    # Create .gitkeep files
    touch data/cache/.gitkeep
    touch logs/.gitkeep
    
    print_success "Directory structure created"
}

# Set permissions
set_permissions() {
    print_status "Setting permissions..."
    
    # Make scripts executable
    chmod +x run.sh 2>/dev/null || echo "run.sh will be created"
    chmod +x download_models.sh 2>/dev/null || echo "download_models.sh exists"
    chmod +x *.py 2>/dev/null || echo "Python files found"
    
    # Make logs directory writable
    chmod 755 logs
    
    print_success "Permissions set"
}

# Create initial configuration (simplified)
create_config() {
    print_status "Creating initial configuration..."
    
    # Activate virtual environment for Python scripts
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source venv/bin/activate
    fi
    
    # Create personalities if they don't exist
    create_personality_files() {
        # Default personality
        cat > config/personalities/default.json << 'EOF'
{
  "name": "Pascal",
  "description": "A helpful AI assistant optimized for Raspberry Pi 5",
  "traits": {
    "helpfulness": 0.9,
    "curiosity": 0.8,
    "formality": 0.3,
    "humor": 0.6,
    "patience": 0.9
  },
  "speaking_style": {
    "tone": "friendly and efficient",
    "complexity": "adaptive to user level",
    "verbosity": "concise but helpful"
  },
  "system_prompt": "You are Pascal, a helpful AI assistant running on Raspberry Pi 5. You are knowledgeable, friendly, and efficient. You can work both offline (using Nemotron) and online (using Groq) depending on the query type."
}
EOF
    }
    
    create_personality_files
    print_success "Created personality configurations"
    
    # Create simplified .env if it doesn't exist
    if [ ! -f ".env" ]; then
        print_status "Creating .env configuration..."
        cat > .env << 'EOF'
# Pascal AI Assistant Environment Variables - Simplified

# 🚀 GROQ API (Primary Online Provider)
# Get from: https://console.groq.com/
GROQ_API_KEY=

# ⚡ PERFORMANCE SETTINGS
PERFORMANCE_MODE=balanced
STREAMING_ENABLED=true
TARGET_RESPONSE_TIME=2.0
MAX_RESPONSE_TOKENS=200

# 🦙 OLLAMA SETTINGS (for Nemotron offline model)
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=30
OLLAMA_KEEP_ALIVE=30m

# 🐛 DEBUG SETTINGS
DEBUG=false
LOG_LEVEL=INFO

# 📝 SETUP INSTRUCTIONS:
# 1. Get Groq API key from https://console.groq.com/
# 2. Replace the empty GROQ_API_KEY above with your actual key
# 3. Make sure key starts with gsk_
# 4. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh
# 5. Download Nemotron: ollama pull nemotron-mini:4b-instruct-q4_K_M
# 6. Run Pascal: ./run.sh
EOF
        print_success "Created .env configuration file"
        print_warning "Please add your Groq API key to .env file"
    fi
}

# Test installation (simplified)
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
    print(f'   Pascal version: {settings.version}')
    print(f'   Debug mode: {settings.debug_mode}')
except ImportError as e:
    print('❌ Pascal configuration failed:', e)
    sys.exit(1)

try:
    from modules.offline_llm import LightningOfflineLLM
    print('✅ LightningOfflineLLM imported successfully')
except ImportError as e:
    print('❌ LightningOfflineLLM import failed:', e)
    print('   This suggests a module naming issue')
    sys.exit(1)

try:
    from modules.online_llm import OnlineLLM
    print('✅ OnlineLLM imported successfully')
except ImportError as e:
    print('❌ OnlineLLM import failed:', e)
    sys.exit(1)

try:
    from modules.router import LightningRouter
    print('✅ LightningRouter imported successfully')
except ImportError as e:
    print('❌ LightningRouter import failed:', e)
    sys.exit(1)

print('✅ Installation test passed - All modules imported successfully!')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
        return 0
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Create run script (simplified)
create_run_script() {
    print_status "Creating run script..."
    
    cat > run.sh << 'EOF'
#!/bin/bash

# Pascal AI Assistant Startup Script - Simplified

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
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

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found!"
    echo "Please run the installer first: ./install.sh"
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Verify we're in the virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi

print_success "Virtual environment activated: $VIRTUAL_ENV"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    print_error "main.py not found!"
    echo "Make sure you're in the pascal directory"
    exit 1
fi

# Display startup message
echo ""
echo "🤖 Starting Pascal AI Assistant - Simplified (Nemotron + Groq)"
echo "============================================================="
echo ""
echo "💡 Commands:"
echo "   'quit' or 'exit' - Stop Pascal"
echo "   'help' - Show available commands"
echo "   'status' - Show system status"
echo ""
echo "🎯 Pascal Features:"
echo "   • Offline: Nemotron (fast local responses)"
echo "   • Online: Groq (current information)"
echo "   • Auto-routing based on query type"
echo ""

# Start Pascal
print_status "Starting Pascal..."
python main.py

# Cleanup message
echo ""
print_success "Pascal has stopped. Virtual environment remains active."
echo ""
echo "To deactivate virtual environment: deactivate"
echo "To restart Pascal: ./run.sh"
EOF
    
    chmod +x run.sh
    print_success "Created run.sh script"
}

# Offer simplified Ollama installation
offer_ollama_installation() {
    echo ""
    print_status "Ollama Installation (for Nemotron offline model)"
    echo "Pascal uses Ollama to run Nemotron locally. You can:"
    echo "1. Install Ollama and download Nemotron now (recommended)"
    echo "2. Install later with manual commands"
    echo "3. Skip Ollama (online-only mode with Groq)"
    echo ""
    
    read -p "Choose option (1-3): " ollama_choice
    
    case $ollama_choice in
        1)
            print_status "Installing Ollama and downloading Nemotron..."
            
            # Install Ollama
            if ! command -v ollama &> /dev/null; then
                print_status "Installing Ollama..."
                curl -fsSL https://ollama.ai/install.sh | sh
                sudo systemctl enable ollama
                sudo systemctl start ollama
                sleep 5
            else
                print_success "Ollama already installed"
                sudo systemctl start ollama
                sleep 3
            fi
            
            # Download Nemotron
            print_status "Downloading Nemotron model (this may take a few minutes)..."
            if ollama pull nemotron-mini:4b-instruct-q4_K_M; then
                print_success "Nemotron model downloaded successfully"
                
                # Test the model
                print_status "Testing Nemotron model..."
                if echo "Say hello" | ollama run nemotron-mini:4b-instruct-q4_K_M > /dev/null 2>&1; then
                    print_success "Nemotron model is working correctly"
                else
                    print_warning "Nemotron model test failed, but model is downloaded"
                fi
            else
                print_error "Failed to download Nemotron model"
                print_status "You can download it later with:"
                print_status "  ollama pull nemotron-mini:4b-instruct-q4_K_M"
            fi
            ;;
        2)
            print_status "Ollama can be installed later with these commands:"
            echo "  curl -fsSL https://ollama.ai/install.sh | sh"
            echo "  sudo systemctl start ollama"
            echo "  ollama pull nemotron-mini:4b-instruct-q4_K_M"
            ;;
        3)
            print_warning "Skipping Ollama. Pascal will work in online-only mode with Groq."
            print_status "Make sure to add your Groq API key to .env file"
            ;;
        *)
            print_warning "Invalid choice. Ollama can be installed later."
            ;;
    esac
}

# Display completion message (simplified)
show_completion() {
    echo ""
    print_success "Pascal AI Assistant installation complete! 🎉"
    echo "========================================================="
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
    echo "1. Configure Groq API key in .env file:"
    echo "   nano .env"
    echo "   Add your GROQ_API_KEY=gsk_your-actual-key"
    echo ""
    echo "2. Get Groq API key from: https://console.groq.com/"
    echo ""
    echo "3. Test Pascal:"
    echo "   ./run.sh"
    echo ""
    
    # Show Pascal features
    print_status "Pascal Features:"
    echo "• Offline LLM: Nemotron (fast, private, runs locally)"
    echo "• Online LLM: Groq (current information, research)"
    echo "• Smart routing: Current info → Groq, General → Nemotron"
    echo "• Streaming responses for instant feedback"
    echo "• Optimized for Raspberry Pi 5 performance"
    echo ""
    
    print_success "Ready to start Pascal! 🤖"
    echo ""
    print_status "To start: ./run.sh"
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
    create_run_script
    
    # Test installation
    if test_installation; then
        offer_ollama_installation
        show_completion
    else
        print_error "Installation failed during testing"
        print_status "Common fixes:"
        print_status "1. Check virtual environment: source venv/bin/activate"
        print_status "2. Install dependencies: pip install -r requirements.txt"
        print_status "3. Check Python version: python3 --version"
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
