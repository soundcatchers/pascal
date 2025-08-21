#!/bin/bash

# Pascal AI Assistant - Lightning Model Download Script for Raspberry Pi 5
# Downloads optimized models for sub-3-second responses

set -e

echo "⚡ Pascal AI Assistant - Lightning Model Manager for Raspberry Pi 5"
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
            print_success "Detected Raspberry Pi 5 - optimized for lightning speed ⚡"
        elif [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
            print_warning "Detected $PI_MODEL - Lightning models optimized for Pi 5"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else:
            print_warning "Not running on Raspberry Pi - Lightning models work on most systems"
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
        print_warning "Less than 8GB RAM detected. Lightning models are optimized for 8GB+."
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

# Configure Ollama for Lightning speed
configure_ollama_lightning() {
    print_status "Configuring Ollama for Lightning speed on Pi 5..."
    
    # Create Ollama service override for lightning performance
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    
    cat << EOF | sudo tee /etc/systemd/system/ollama.service.d/lightning.conf
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_MAX_QUEUE=2"
Environment="OLLAMA_FLASH_ATTENTION=0"
Environment="OLLAMA_KV_CACHE_TYPE=f16"
Environment="OLLAMA_NUM_THREAD=4"
Environment="OLLAMA_TMPDIR=/tmp"
Environment="OLLAMA_KEEP_ALIVE=5m"
EOF

    # Reload systemd and restart Ollama
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    
    # Wait for restart
    sleep 5
    
    print_success "Ollama configured for Lightning performance ⚡"
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

# Download Lightning models
download_lightning_models() {
    print_status "⚡ Lightning Model Selection:"
    echo ""
    echo "Primary Models (Optimized for 1-3 second responses):"
    echo "1. nemotron-mini:4b-instruct-q4_K_M - PRIMARY (Fastest, 2.5GB)"
    echo "2. qwen3:4b-instruct - FALLBACK 1 (Fast, 2.3GB)"
    echo "3. gemma3:4b-it-q4_K_M - FALLBACK 2 (Reliable, 2.4GB)"
    echo ""
    echo "4. Download all Lightning models (recommended)"
    echo "5. Download primary model only (nemotron-mini)"
    echo "6. Custom selection"
    echo "7. Skip model download"
    echo ""
    
    read -p "Choose an option (1-7): " choice
    
    case $choice in
        1)
            download_single_model "nemotron-mini:4b-instruct-q4_K_M" "Nemotron Mini 4B (PRIMARY)" "2.5GB"
            ;;
        2)
            download_single_model "qwen3:4b-instruct" "Qwen3 4B (FALLBACK 1)" "2.3GB"
            ;;
        3)
            download_single_model "gemma3:4b-it-q4_K_M" "Gemma3 4B (FALLBACK 2)" "2.4GB"
            ;;
        4)
            print_status "Downloading all Lightning models for maximum reliability..."
            download_single_model "nemotron-mini:4b-instruct-q4_K_M" "Nemotron Mini 4B (PRIMARY)" "2.5GB"
            download_single_model "qwen3:4b-instruct" "Qwen3 4B (FALLBACK 1)" "2.3GB"
            download_single_model "gemma3:4b-it-q4_K_M" "Gemma3 4B (FALLBACK 2)" "2.4GB"
            ;;
        5)
            download_single_model "nemotron-mini:4b-instruct-q4_K_M" "Nemotron Mini 4B (PRIMARY)" "2.5GB"
            ;;
        6)
            echo "Available Lightning models:"
            echo "  nemotron-mini:4b-instruct-q4_K_M (primary)"
            echo "  qwen3:4b-instruct (fallback 1)"
            echo "  gemma3:4b-it-q4_K_M (fallback 2)"
            echo ""
            echo "Enter model names separated by spaces:"
            read -p "Models: " selected_models
            for model in $selected_models; do
                case $model in
                    "nemotron-mini:4b-instruct-q4_K_M") 
                        download_single_model "$model" "Nemotron Mini 4B" "2.5GB" ;;
                    "qwen3:4b-instruct") 
                        download_single_model "$model" "Qwen3 4B" "2.3GB" ;;
                    "gemma3:4b-it-q4_K_M") 
                        download_single_model "$model" "Gemma3 4B" "2.4GB" ;;
                    *) 
                        print_warning "Unknown model: $model" ;;
                esac
            done
            ;;
        7)
            print_warning "Skipping model download. You can download models later from Pascal."
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
    
    print_status "⚡ Downloading $display_name ($size)..."
    
    # Check if model is already downloaded
    if ollama list | grep -q "$model_name"; then
        print_warning "$display_name is already downloaded"
        
        # Ensure it's loaded with keep-alive
        print_status "Loading model with keep-alive..."
        ollama run "$model_name" --keepalive 5m <<< ""
        return 0
    fi
    
    # Download with progress
    if ollama pull "$model_name"; then
        print_success "Downloaded $display_name successfully ⚡"
        
        # Load model with keep-alive
        print_status "Loading model with keep-alive for instant responses..."
        ollama run "$model_name" --keepalive 5m <<< ""
        
        return 0
    else
        print_error "Failed to download $display_name"
        return 1
    fi
}

# Verify downloaded models
verify_models() {
    print_status "Verifying Lightning models..."
    
    MODEL_LIST=$(ollama list 2>/dev/null || echo "")
    
    if [ -z "$MODEL_LIST" ] || [ "$(echo "$MODEL_LIST" | wc -l)" -le 1 ]; then
        print_error "No models found!"
        return 1
    fi
    
    print_success "Available models:"
    echo "$MODEL_LIST"
    
    # Check for Lightning models
    LIGHTNING_MODELS=0
    if echo "$MODEL_LIST" | grep -q "nemotron-mini:4b-instruct-q4_K_M"; then
        print_success "✅ Primary model ready: nemotron-mini:4b-instruct-q4_K_M"
        LIGHTNING_MODELS=$((LIGHTNING_MODELS + 1))
    fi
    if echo "$MODEL_LIST" | grep -q "qwen3:4b-instruct"; then
        print_success "✅ Fallback 1 ready: qwen3:4b-instruct"
        LIGHTNING_MODELS=$((LIGHTNING_MODELS + 1))
    fi
    if echo "$MODEL_LIST" | grep -q "gemma3:4b-it-q4_K_M"; then
        print_success "✅ Fallback 2 ready: gemma3:4b-it-q4_K_M"
        LIGHTNING_MODELS=$((LIGHTNING_MODELS + 1))
    fi
    
    if [ $LIGHTNING_MODELS -eq 0 ]; then
        print_warning "No Lightning models found. Pascal will use any available model."
    else
        print_success "⚡ $LIGHTNING_MODELS Lightning model(s) ready for sub-3-second responses!"
    fi
    
    return 0
}

# Test model performance
test_lightning_performance() {
    print_status "Testing Lightning performance..."
    
    # Get list of downloaded models
    MODELS=$(ollama list | tail -n +2 | awk '{print $1}' | grep -v "^$")
    
    if [ -z "$MODELS" ]; then
        print_warning "No models available for testing"
        return
    fi
    
    read -p "Test Lightning model performance? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi
    
    # Test primary model if available
    if echo "$MODELS" | grep -q "nemotron-mini:4b-instruct-q4_K_M"; then
        print_status "⚡ Testing nemotron-mini (PRIMARY)..."
        
        start_time=$(date +%s.%N)
        
        # Simple test prompt
        response=$(echo "Say 'Lightning fast!' in 5 words or less" | ollama run "nemotron-mini:4b-instruct-q4_K_M" --verbose 2>/dev/null || echo "Error")
        
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        
        print_success "Response time: ${duration}s"
        
        if (( $(echo "$duration < 3" | bc -l) )); then
            print_success "⚡ LIGHTNING FAST! Under 3 seconds!"
        elif (( $(echo "$duration < 5" | bc -l) )); then
            print_warning "Good speed, but not quite lightning (3-5s)"
        else
            print_warning "Slower than target (>5s). Check cooling and resources."
        fi
        
        echo ""
    fi
}

# Create model configuration for Pascal
create_lightning_config() {
    print_status "Creating Lightning configuration for Pascal..."
    
    # Get script directory and create config
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    CONFIG_FILE="$SCRIPT_DIR/data/models/lightning_config.json"
    
    # Ensure directory exists
    mkdir -p "$(dirname "$CONFIG_FILE")"
    
    # Get available models
    MODELS=$(ollama list | tail -n +2 | awk '{print $1}' | grep -v "^$" | head -10)
    
    # Create JSON config
    cat > "$CONFIG_FILE" << EOF
{
    "ollama_enabled": true,
    "ollama_host": "http://localhost:11434",
    "lightning_mode": true,
    "download_date": "$(date -Iseconds)",
    "pi_model": "$(cat /proc/device-tree/model 2>/dev/null || echo 'Unknown')",
    "preferred_models": [
        "nemotron-mini:4b-instruct-q4_K_M",
        "qwen3:4b-instruct",
        "gemma3:4b-it-q4_K_M"
    ],
    "available_models": [
EOF

    # Add models to config
    FIRST=true
    for model in $MODELS; do
        if [ "$FIRST" = false ]; then
            echo "        ," >> "$CONFIG_FILE"
        fi
        
        SIZE=$(ollama list | grep "$model" | awk '{print $2}' || echo "Unknown")
        
        # Determine priority
        PRIORITY=99
        if [[ "$model" == "nemotron-mini:4b-instruct-q4_K_M" ]]; then
            PRIORITY=1
        elif [[ "$model" == "qwen3:4b-instruct" ]]; then
            PRIORITY=2
        elif [[ "$model" == "gemma3:4b-it-q4_K_M" ]]; then
            PRIORITY=3
        fi
        
        echo "        {" >> "$CONFIG_FILE"
        echo "            \"name\": \"$model\"," >> "$CONFIG_FILE"
        echo "            \"size\": \"$SIZE\"," >> "$CONFIG_FILE"
        echo "            \"priority\": $PRIORITY" >> "$CONFIG_FILE"
        echo -n "        }" >> "$CONFIG_FILE"
        FIRST=false
    done

    cat >> "$CONFIG_FILE" << EOF

    ],
    "performance_settings": {
        "target_response_time": 3.0,
        "streaming_enabled": true,
        "keep_alive_duration": "5m",
        "first_token_target": 1.0
    },
    "lightning_tips": {
        "primary": "nemotron-mini:4b-instruct-q4_K_M - Fastest responses",
        "fallback1": "qwen3:4b-instruct - Good balance",
        "fallback2": "gemma3:4b-it-q4_K_M - Most reliable"
    }
}
EOF

    print_success "Lightning configuration saved to $CONFIG_FILE"
}

# Show completion message
show_lightning_completion() {
    print_success "⚡ Lightning setup complete! 🎉"
    echo "================================="
    echo ""
    
    print_status "Lightning Performance Achieved:"
    echo "• Target: 1-3 second responses ⚡"
    echo "• Streaming: Instant feedback"
    echo "• Keep-alive: Models stay loaded"
    echo "• Optimized: For Raspberry Pi 5"
    echo ""
    
    print_status "Available Commands:"
    echo "• ollama list                 - Show downloaded models"
    echo "• ollama run [model]          - Test a model"
    echo "• systemctl status ollama     - Check Ollama service"
    echo ""
    
    print_status "Pascal Integration:"
    echo "• Models are automatically detected"
    echo "• Start Pascal with: ./run.sh"
    echo "• Primary model loads on startup"
    echo "• Automatic fallback if needed"
    echo ""
    
    print_status "⚡ Lightning Tips:"
    echo "• Keep Pi 5 cool for sustained performance"
    echo "• Use 'profile speed' in Pascal for fastest responses"
    echo "• Streaming provides instant feedback"
    echo "• Models stay loaded for 5 minutes (keep-alive)"
    echo ""
    
    print_status "Monitoring:"
    echo "• Temperature: vcgencmd measure_temp"
    echo "• Memory: free -h"
    echo "• CPU: htop"
    echo ""
    
    print_success "Ready for Lightning-fast Pascal! Run: ./run.sh"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
}

# Main execution
main() {
    echo "⚡ Starting Pascal Lightning setup..."
    
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
    
    configure_ollama_lightning
    
    if ! test_ollama; then
        print_error "Ollama connection test failed"
        exit 1
    fi
    
    # Download Lightning models
    download_lightning_models
    
    # Verify and test
    if verify_models; then
        test_lightning_performance
        create_lightning_config
        show_lightning_completion
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
