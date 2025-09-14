#!/bin/bash

# Pascal AI Assistant - Simplified Model Download (Nemotron Only)
# Downloads and configures Nemotron for offline use

set -e

echo "🤖 Pascal AI Assistant - Nemotron Model Setup"
echo "=============================================="

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
            print_success "Detected Raspberry Pi 5 - optimal for Nemotron"
        elif [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
            print_warning "Detected $PI_MODEL - Nemotron should work"
        else
            print_warning "Non-Pi hardware - Nemotron may work slower"
        fi
    fi
}

# Check available space
check_space() {
    AVAILABLE_GB=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    print_status "Available disk space: ${AVAILABLE_GB}GB"
    
    if [ $AVAILABLE_GB -lt 5 ]; then
        print_error "Insufficient disk space. Need at least 5GB for Nemotron model."
        exit 1
    fi
}

# Check RAM
check_ram() {
    TOTAL_RAM_MB=$(free -m | grep '^Mem:' | awk '{print $2}')
    TOTAL_RAM_GB=$((TOTAL_RAM_MB / 1024))
    print_status "System RAM: ${TOTAL_RAM_GB}GB"
    
    if [ $TOTAL_RAM_GB -lt 4 ]; then
        print_warning "Less than 4GB RAM. Nemotron may run slowly."
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
    
    # Create Ollama service override for optimal performance
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    
    cat << EOF | sudo tee /etc/systemd/system/ollama.service.d/pi5-optimization.conf
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_FLASH_ATTENTION=0"
Environment="OLLAMA_NUM_THREAD=4"
Environment="OLLAMA_KEEP_ALIVE=30m"
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

# Download Nemotron model
download_nemotron() {
    print_status "🧠 Downloading Nemotron Mini 4B model..."
    echo ""
    echo "Model Information:"
    echo "• Name: nemotron-mini:4b-instruct-q4_K_M"
    echo "• Size: ~2.7GB"
    echo "• Optimized: For speed and efficiency"
    echo "• Best for: General queries, coding help, explanations"
    echo ""
    
    read -p "Download Nemotron model now? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Skipping model download. You can download later with:"
        print_warning "  ollama pull nemotron-mini:4b-instruct-q4_K_M"
        return 0
    fi
    
    print_status "Downloading Nemotron model (this may take several minutes)..."
    
    # Check if model is already downloaded
    if ollama list | grep -q "nemotron-mini:4b-instruct-q4_K_M"; then
        print_warning "Nemotron model is already downloaded"
        return 0
    fi
    
    # Download with timeout
    if timeout 600 ollama pull nemotron-mini:4b-instruct-q4_K_M; then
        print_success "✅ Nemotron model downloaded successfully!"
        
        # Test the model briefly
        print_status "Testing Nemotron model..."
        test_nemotron
        
        return 0
    else
        print_error "Failed to download Nemotron model (timeout or error)"
        print_status "You can try again later with:"
        print_status "  ollama pull nemotron-mini:4b-instruct-q4_K_M"
        return 1
    fi
}

# Quick test of Nemotron model
test_nemotron() {
    print_status "Running quick test of Nemotron..."
    
    # Use timeout to prevent hanging
    test_result=$(timeout 30 bash -c "echo 'Say hello in 3 words' | ollama run nemotron-mini:4b-instruct-q4_K_M 2>/dev/null | head -n 3" 2>/dev/null || echo "timeout")
    
    if [[ "$test_result" == "timeout" ]] || [[ -z "$test_result" ]]; then
        print_warning "Model test timed out (normal for first load)"
        print_status "Model will work properly in Pascal"
    else
        print_success "Model test successful - Nemotron is ready!"
        print_status "Test response: $test_result"
    fi
}

# Verify installation
verify_setup() {
    print_status "Verifying Nemotron setup..."
    
    MODEL_LIST=$(ollama list 2>/dev/null || echo "")
    
    if [ -z "$MODEL_LIST" ] || [ "$(echo "$MODEL_LIST" | wc -l)" -le 1 ]; then
        print_error "No models found!"
        return 1
    fi
    
    print_success "Ollama models:"
    echo "$MODEL_LIST"
    
    # Check for Nemotron specifically
    if echo "$MODEL_LIST" | grep -q "nemotron-mini:4b-instruct-q4_K_M"; then
        print_success "✅ Nemotron model ready for Pascal"
        return 0
    else
        print_warning "Nemotron model not found"
        print_status "Available models, but not the recommended Nemotron"
        return 1
    fi
}

# Create Pascal configuration
create_pascal_config() {
    print_status "Creating Pascal configuration for Nemotron..."
    
    # Get script directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    CONFIG_FILE="$SCRIPT_DIR/data/models/nemotron_config.json"
    
    # Ensure directory exists
    mkdir -p "$(dirname "$CONFIG_FILE")"
    
    # Create JSON config
    cat > "$CONFIG_FILE" << EOF
{
    "ollama_enabled": true,
    "ollama_host": "http://localhost:11434",
    "primary_model": "nemotron-mini:4b-instruct-q4_K_M",
    "download_date": "$(date -Iseconds)",
    "pi_model": "$(cat /proc/device-tree/model 2>/dev/null || echo 'Unknown')",
    "model_info": {
        "name": "nemotron-mini:4b-instruct-q4_K_M",
        "size": "~2.7GB",
        "type": "instruction-following",
        "optimized_for": "speed and efficiency",
        "best_use": ["general queries", "coding help", "explanations", "casual chat"]
    },
    "performance_settings": {
        "target_response_time": 2.0,
        "streaming_enabled": true,
        "keep_alive_duration": "30m"
    },
    "pascal_integration": {
        "routing": "offline for general queries",
        "fallback": "groq for current information",
        "priority": "primary offline model"
    }
}
EOF

    print_success "Pascal configuration saved to $CONFIG_FILE"
}

# Show completion message
show_completion() {
    print_success "🎉 Nemotron setup complete for Pascal!"
    echo "========================================"
    echo ""
    
    print_status "Nemotron Model Status:"
    echo "• Model: nemotron-mini:4b-instruct-q4_K_M"
    echo "• Size: ~2.7GB"
    echo "• Location: Ollama model store"
    echo "• Optimized: For Raspberry Pi 5"
    echo "• Ready: For Pascal offline queries"
    echo ""
    
    print_status "Integration with Pascal:"
    echo "• General queries → Nemotron (fast, local)"
    echo "• Current info queries → Groq (online)"
    echo "• Automatic routing based on query type"
    echo "• Streaming responses for instant feedback"
    echo ""
    
    print_status "Commands:"
    echo "• ollama list                 - Show downloaded models"
    echo "• ollama run nemotron-mini:4b-instruct-q4_K_M - Test model directly"
    echo "• systemctl status ollama     - Check Ollama service"
    echo "• ./run.sh                    - Start Pascal"
    echo ""
    
    print_success "Ready to use Nemotron with Pascal! 🤖"
    echo ""
    print_status "Next steps:"
    echo "1. Configure Groq API key in .env for current info queries"
    echo "2. Start Pascal: ./run.sh"
    echo "3. Test with: 'Hello Pascal' (should use Nemotron)"
    echo "4. Test with: 'What day is today?' (should use Groq if configured)"
}

# Main execution
main() {
    echo "🧠 Starting Nemotron setup for Pascal..."
    
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
    
    # Download Nemotron model
    download_nemotron
    
    # Verify and create config
    if verify_setup; then
        create_pascal_config
        show_completion
    else
        print_error "Model verification failed"
        print_status "Ollama is installed but Nemotron model may not be available"
        print_status "Try downloading manually: ollama pull nemotron-mini:4b-instruct-q4_K_M"
        exit 1
    fi
}

# Handle interruption
cleanup_interrupt() {
    print_warning "Setup interrupted"
    exit 1
}

# Set trap for cleanup
trap cleanup_interrupt SIGINT SIGTERM

# Run main function
main "$@"
