#!/bin/bash

# Pascal AI Assistant FIXED Installer - Raspberry Pi 5 Optimized
# Updated for the simplified Groq + Ollama version

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

echo "🤖 Installing Pascal AI Assistant FIXED (Pi 5 Optimized with Groq + Ollama)"
echo "============================================================================"

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

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python packages (FIXED version)..."
    
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
    
    # Install requirements (simplified for fixed version)
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Verify critical packages
    print_status "Verifying critical packages..."
    
    # Test aiohttp (critical for online LLM functionality)
    if python -c "import aiohttp; print(f'✅ aiohttp {aiohttp.__version__} installed successfully')" 2>/dev/null; then
        print_success "aiohttp installed and working"
    else
        print_error "aiohttp installation failed - this will break online functionality"
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
    chmod +x run.sh 2>/dev/null || echo "run.sh not found, will be created"
    chmod +x download_models.sh 2>/dev/null || echo "download_models.sh exists"
    chmod +x *.py 2>/dev/null || echo "Python files found"
    
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
    
    # Create personalities if they don't exist
    create_personality_files() {
        # Default personality
        cat > config/personalities/default.json << 'EOF'
{
  "name": "Pascal",
  "description": "A helpful, intelligent AI assistant with a friendly personality",
  "traits": {
    "helpfulness": 0.9,
    "curiosity": 0.8,
    "formality": 0.3,
    "humor": 0.6,
    "patience": 0.9
  },
  "speaking_style": {
    "tone": "friendly and approachable",
    "complexity": "adaptive to user level",
    "verbosity": "concise but thorough",
    "examples": true
  },
  "knowledge_focus": [
    "programming and technology",
    "problem-solving",
    "learning and education",
    "creative projects"
  ],
  "conversation_style": {
    "greeting": "Hello! I'm Pascal. How can I help you today?",
    "thinking": "Let me think about that...",
    "clarification": "Could you help me understand what you mean by",
    "completion": "I hope that helps! Is there anything else you'd like to know?",
    "error": "I'm having trouble with that. Let me try a different approach."
  },
  "system_prompt": "You are Pascal, a helpful AI assistant. You are knowledgeable, friendly, and always eager to help. You explain things clearly and ask for clarification when needed. You maintain a consistent personality across all interactions."
}
EOF
        
        # Assistant personality
        cat > config/personalities/assistant.json << 'EOF'
{
  "name": "Pascal Assistant",
  "description": "A more formal, professional version of Pascal for business use",
  "traits": {
    "helpfulness": 0.95,
    "curiosity": 0.7,
    "formality": 0.8,
    "humor": 0.3,
    "patience": 0.95
  },
  "speaking_style": {
    "tone": "professional and courteous",
    "complexity": "technical when appropriate",
    "verbosity": "detailed and comprehensive",
    "examples": true
  },
  "system_prompt": "You are Pascal, a professional AI assistant. You are knowledgeable, efficient, and maintain a courteous, business-appropriate demeanor. You provide detailed, accurate information and maintain professionalism in all interactions."
}
EOF
    }
    
    create_personality_files
    print_success "Created personality configurations"
    
    # Create .env if it doesn't exist
    if [ ! -f ".env" ]; then
        print_status "Creating .env configuration..."
        cp .env.example .env 2>/dev/null || cat > .env << 'EOF'
# Pascal AI Assistant Environment Variables - FIXED VERSION

# 🚀 GROQ API (Primary and Only Online Provider)
# Get from: https://console.groq.com/
GROQ_API_KEY=

# ⚡ PERFORMANCE SETTINGS
PERFORMANCE_MODE=balanced
STREAMING_ENABLED=true
KEEP_ALIVE_ENABLED=true
TARGET_RESPONSE_TIME=2.0
MAX_RESPONSE_TOKENS=200

# 🦙 OLLAMA SETTINGS
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=30
OLLAMA_KEEP_ALIVE=30m

# 🐛 DEBUG SETTINGS
DEBUG=false
LOG_LEVEL=INFO
EOF
        print_success "Created .env configuration file"
        print_warning "Please add your Groq API key to .env file"
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
    print(f'   Pascal version: {settings.version}')
    print(f'   Debug mode: {settings.debug_mode}')
except ImportError as e:
    print('❌ Pascal configuration failed:', e)
    sys.exit(1)

try:
    from modules.offline_llm import LightningOfflineLLM
    print('✅ LightningOfflineLLM imported (FIXED CLASS NAME)')
except ImportError as e:
    print('❌ LightningOfflineLLM import failed:', e)
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
    else:
        print_error "Installation test failed"
        return 1
    fi
}

# Create run script
create_run_script() {
    print_status "Creating run script..."
    
    cat > run.sh << 'EOF'
#!/bin/bash

# Pascal AI Assistant Startup Script FIXED

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
    echo "Please run the installer first: ./install_fixed.sh"
    exit 1
fi

# Check if virtual environment has the activation script
if [ ! -f "venv/bin/activate" ]; then
    print_error "Virtual environment appears corrupted!"
    echo "Recreate it with: ./install_fixed.sh"
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
echo "🤖 Starting Pascal AI Assistant FIXED..."
echo "======================================"
echo ""
echo "💡 Commands:"
echo "   'quit' or 'exit' - Stop Pascal"
echo "   'help' - Show available commands"
echo "   'status' - Show system status"
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
            if [ -f "./download_models.sh" ]; then
                chmod +x download_models.sh
                ./download_models.sh
            else
                print_warning "download_models.sh not found, installing Ollama manually..."
                curl -fsSL https://ollama.ai/install.sh | sh
                sudo systemctl enable ollama
                sudo systemctl start ollama
                sleep 5
                ollama pull phi3:mini
                print_success "Ollama installed with phi3:mini model"
            fi
            ;;
        2)
            print_status "Ollama can be installed later with: ./download_models.sh"
            ;;
        3)
            print_warning "Skipping Ollama. Pascal will work in online-only mode."
            ;;
        *)
            print_warning "Invalid choice. Ollama can be installed later."
            ;;
    esac
}

# Display completion message
show_completion() {
    echo ""
    print_success "Pascal FIXED installation complete! 🎉"
    echo "================================================"
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
    echo "1. Configure API key in .env file:"
    echo "   nano .env"
    echo "   Add your GROQ_API_KEY=gsk_your-actual-key"
    echo ""
    echo "2. Test the fixed installation:"
    echo "   python test_quick_fix.py"
    echo ""
    echo "3. Start Pascal:"
    echo "   ./run.sh"
    echo ""
    
    # Show API key instructions
    print_status "API Key Setup:"
    echo "• Get Groq API key from: https://console.groq.com/"
    echo "• Add to .env file: GROQ_API_KEY=gsk_your-actual-key"
    echo "• Make sure key starts with gsk_ (underscore format)"
    echo ""
    
    print_success "Happy chatting with Pascal! 🤖"
    echo ""
    print_status "To test: python test_quick_fix.py"
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
