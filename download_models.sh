#!/bin/bash

# Pascal AI Assistant - Model Download Script for Raspberry Pi 5 (Ollama Version)
# Downloads and manages models using Ollama for optimal Pi 5 performance

set -e

echo "🤖 Pascal AI Assistant - Ollama Model Manager for Raspberry Pi 5"
echo "================================================================="

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

# Check if we're on Raspberry Pi
check_pi() {
    if [ -f /proc/device-tree/model ]; then
        PI_MODEL=$(cat /proc/device-tree/model)
        if [[ $PI_MODEL == *"Raspberry Pi 5"* ]]; then
            print_success "Detected Raspberry Pi 5 - proceeding with optimized setup"
        elif [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
            print_warning "Detected $PI_MODEL - Ollama optimized for Pi 5"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            print_warning "Not running on Raspberry Pi - Ollama works on most systems"
        fi
    fi
}

# Check available space
check_space() {
    AVAILABLE_GB=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    print_status "Available disk space: ${AVAILABLE_GB}GB"
    
    if [ $AVAILABLE_GB -lt 15 ]; then
        print_error "Insufficient disk space. Need at least 15GB free for Ollama and models."
        exit 1
    fi
}

# Check RAM
check_ram() {
    TOTAL_RAM_MB=$(free -m | grep '^Mem:' | awk '{print $2}')
    TOTAL_RAM_GB=$((TOTAL_RAM_MB / 1024))
    print_status "System RAM: ${TOTAL_RAM_GB}GB"
    
    if [ $TOTAL_RAM_GB -lt 8 ]; then
        print_warning "Less than 8GB RAM detected. Some models may not run optimally."
    fi
}

# Install Ollama
install_ollama() {
    print_status "Checking Ollama installation..."
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama is already installed"
        ollama --version
        return 0
    fi
    
    print_status "Installing Ollama..."
    
    # Download and install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama installed successfully"
        
        # Start Ollama service
        print_status "Starting Ollama service..."
        sudo systemctl enable ollama
        sudo systemctl start ollama
        
        # Wait for service to start
        sleep 5
        
        print_success "Ollama service started"
        return 0
    else
        print_error "Ollama installation failed"
        return 1
    fi
}

# Configure Ollama for Pi 5
configure_ollama() {
    print_status "Configuring Ollama for Raspberry Pi 5..."
    
    # Create Ollama service override for Pi 5 optimization
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    
    cat << EOF | sudo tee /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_MAX_QUEUE=4"
Environment="OLLAMA_FLASH_ATTENTION=0"
Environment="OLLAMA_KV_CACHE_TYPE=f16"
Environment="OLLAMA_NUM_THREAD=4"
Environment="OLLAMA_TMPDIR=/tmp"
EOF

    # Reload systemd and restart Ollama
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    
    # Wait for restart
    sleep 5
    
    print_success "Ollama configured for Pi 5 optimization"
}

# Test Ollama connection
test_ollama() {
    print_status "Testing Ollama connection..."
    
    # Try to connect to Ollama
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/version > /dev/null; then
            print_success "Ollama is running and accessible"
            return 0
        fi
        
        print_status "Waiting for Ollama to start... (attempt $i/10)"
        sleep 2
    done
    
    print_error "Could not connect to Ollama after 20 seconds"
    return 1
}

# Download models using Ollama
download_models() {
    print_status "Available models optimized for Raspberry Pi 5:"
    echo ""
    echo "1. Phi-3 Mini (4K) - 2.3GB - Fastest responses, best for Pi 5"
    echo "2. Llama 3.2 3B - 2.0GB - Excellent efficiency and speed"
    echo "3. Gemma 2 2B - 1.6GB - Very fast, good for quick tasks"
    echo "4. Qwen2.5 7B - 4.4GB - Best quality (requires good cooling)"
    echo "5. Download recommended set (Phi-3 Mini + Llama 3.2 3B)"
    echo "6. Download all models"
    echo "7. Custom selection"
    echo ""
    
    read -p "Choose an option (1-7): " choice
    
    case $choice in
        1)
            download_single_model "phi3:mini" "Phi-3 Mini (4K)" "2.3GB"
            ;;
        2)
            download_single_model "llama3.2:3b" "Llama 3.2 3B" "2.0GB"
            ;;
        3)
            download_single_model "gemma2:2b" "Gemma 2 2B" "1.6GB"
            ;;
        4)
            download_single_model "qwen2.5:7b" "Qwen2.5 7B" "4.4GB"
            ;;
        5)
            download_single_model "phi3:mini" "Phi-3 Mini (4K)" "2.3GB"
            download_single_model "llama3.2:3b" "Llama 3.2 3B" "2.0GB"
            ;;
        6)
            download_single_model "phi3:mini" "Phi-3 Mini (4K)" "2.3GB"
            download_single_model "llama3.2:3b" "Llama 3.2 3B" "2.0GB"
            download_single_model "gemma2:2b" "Gemma 2 2B" "1.6GB"
            download_single_model "qwen2.5:7b" "Qwen2.5 7B" "4.4GB"
            ;;
        7)
            echo "Available models: phi3:mini, llama3.2:3b, gemma2:2b, qwen2.5:7b"
            echo "Enter model names separated by spaces:"
            read -p "Models: " selected_models
            for model in $selected_models; do
                case $model in
                    "phi3:mini") download_single_model "phi3:mini" "Phi-3 Mini (4K)" "2.3GB" ;;
                    "llama3.2:3b") download_single_model "llama3.2:3b" "Llama 3.2 3B" "2.0GB" ;;
                    "gemma2:2b") download_single_model "gemma2:2b" "Gemma 2 2B" "1.6GB" ;;
                    "qwen2.5:7b") download_single_model "qwen2.5:7b" "Qwen2.5 7B" "4.4GB" ;;
                    *) print_warning "Unknown model: $model" ;;
                esac
            done
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Download a single model
download_single_model() {
    local model_name="$1"
    local display_name="$2"
    local size="$3"
    
    print_status "Downloading $display_name ($size)..."
    
    # Check if model is already downloaded
    if ollama list | grep -q "$model_name"; then
        print_warning "$display_name is already downloaded"
        return 0
    fi
    
    # Download with progress
    if ollama pull "$model_name"; then
        print_success "Downloaded $display_name successfully"
        return 0
    else
        print_error "Failed to download $display_name"
        return 1
    fi
}

# Verify downloaded models
verify_models() {
    print_status "Verifying downloaded models..."
    
    MODEL_LIST=$(ollama list 2>/dev/null || echo "")
    
    if [ -z "$MODEL_LIST" ] || [ "$(echo "$MODEL_LIST" | wc -l)" -le 1 ]; then
        print_error "No models found!"
        return 1
    fi
    
    print_success "Available models:"
    echo "$MODEL_LIST"
    
    return 0
}

# Test model inference
test_models() {
    print_status "Testing model performance..."
    
    # Get list of downloaded models
    MODELS=$(ollama list | tail -n +2 | awk '{print $1}' | grep -v "^$")
    
    if [ -z "$MODELS" ]; then
        print_warning "No models available for testing"
        return
    fi
    
    read -p "Test model performance? This will take 1-2 minutes (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi
    
    for model in $MODELS; do
        print_status "Testing $model..."
        
        start_time=$(date +%s)
        
        # Simple test prompt
        response=$(echo "Hello! Please respond with exactly: Test successful" | ollama run "$model" --verbose 2>/dev/null || echo "Error")
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        if [[ "$response" == *"test"* ]] && [[ "$response" == *"successful"* ]]; then
            print_success "$model: Response time ${duration}s ✅"
        else
            print_warning "$model: Test failed or timed out (${duration}s) ❌"
        fi
        
        echo ""
    done
}

# Create model configuration for Pascal
create_model_config() {
    print_status "Creating Pascal model configuration..."
    
    # Get script directory and create config
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    CONFIG_FILE="$SCRIPT_DIR/data/models/ollama_config.json"
    
    # Ensure directory exists
    mkdir -p "$(dirname "$CONFIG_FILE")"
    
    # Get available models
    MODELS=$(ollama list | tail -n +2 | awk '{print $1}' | grep -v "^$" | head -10)
    
    # Create JSON config
    cat > "$CONFIG_FILE" << EOF
{
    "ollama_enabled": true,
    "ollama_host": "http://localhost:11434",
    "download_date": "$(date -Iseconds)",
    "pi_model": "$(cat /proc/device-tree/model 2>/dev/null || echo 'Unknown')",
    "available_models": [
EOF

    # Add models to config
    FIRST=true
    for model in $MODELS; do
        if [ "$FIRST" = false ]; then
            echo "        ," >> "$CONFIG_FILE"
        fi
        
        # Get model info
        MODEL_INFO=$(ollama show "$model" 2>/dev/null || echo "")
        PARAMS=$(echo "$MODEL_INFO" | grep -i "parameters" | head -1 || echo "Unknown parameters")
        SIZE=$(ollama list | grep "$model" | awk '{print $2}' || echo "Unknown")
        
        echo "        {" >> "$CONFIG_FILE"
        echo "            \"name\": \"$model\"," >> "$CONFIG_FILE"
        echo "            \"size\": \"$SIZE\"," >> "$CONFIG_FILE"
        echo "            \"parameters\": \"$PARAMS\"" >> "$CONFIG_FILE"
        echo -n "        }" >> "$CONFIG_FILE"
        FIRST=false
    done

    cat >> "$CONFIG_FILE" << EOF

    ],
    "model_recommendations": {
        "fastest": "phi3:mini - Optimized for Pi 5, fastest responses",
        "balanced": "llama3.2:3b - Good balance of speed and quality",
        "quality": "qwen2.5:7b - Best quality (needs good cooling)",
        "lightweight": "gemma2:2b - Minimal resource usage"
    },
    "performance_profiles": {
        "speed": {
            "preferred_models": ["phi3:mini", "gemma2:2b"],
            "temperature": 0.3,
            "max_tokens": 100
        },
        "balanced": {
            "preferred_models": ["llama3.2:3b", "phi3:mini"],
            "temperature": 0.7,
            "max_tokens": 200
        },
        "quality": {
            "preferred_models": ["qwen2.5:7b", "llama3.2:3b"],
            "temperature": 0.8,
            "max_tokens": 300
        }
    }
}
EOF

    print_success "Pascal configuration saved to $CONFIG_FILE"
}

# Show usage instructions
show_completion() {
    print_success "Ollama setup complete! 🎉"
    echo "================================"
    echo ""
    
    print_status "Available Commands:"
    echo "• ollama list                 - Show downloaded models"
    echo "• ollama run [model]          - Chat with a model"
    echo "• ollama pull [model]         - Download a new model"
    echo "• ollama rm [model]           - Remove a model"
    echo "• systemctl status ollama     - Check Ollama service"
    echo ""
    
    print_status "Pascal Integration:"
    echo "• Models are automatically detected by Pascal"
    echo "• Start Pascal with: ./run.sh"
    echo "• Use 'status' command in Pascal to see available models"
    echo "• Switch models with 'model [name]' command"
    echo ""
    
    print_status "Performance Tips:"
    echo "• phi3:mini - Best for quick responses (recommended)"
    echo "• llama3.2:3b - Good balance of speed and quality"
    echo "• gemma2:2b - Fastest, minimal resources"
    echo "• qwen2.5:7b - Highest quality (ensure good cooling)"
    echo ""
    
    print_status "Monitoring:"
    echo "• Monitor temperature: vcgencmd measure_temp"
    echo "• Monitor memory: free -h"
    echo "• Monitor CPU: htop"
    echo ""
    
    print_success "Ready to start Pascal! Run: ./run.sh"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    # Ollama manages its own cleanup
}

# Main execution
main() {
    echo "Starting Pascal Ollama setup process..."
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    # Run checks and setup
    check_pi
    check_space
    check_ram
    
    # Install and configure Ollama
    if ! install_ollama; then
        print_error "Ollama installation failed"
        exit 1
    fi
    
    configure_ollama
    
    if ! test_ollama; then
        print_error "Ollama connection test failed"
        exit 1
    fi
    
    # Download models
    download_models
    
    # Verify and test
    if verify_models; then
        test_models
        create_model_config
        show_completion
    else
        print_error "Model verification failed"
        exit 1
    fi
}

# Handle interruption
cleanup_interrupt() {
    print_warning "Setup interrupted"
    cleanup
    exit 1
}

# Set trap for cleanup
trap cleanup_interrupt SIGINT SIGTERM

# Run main function
main "$@"
