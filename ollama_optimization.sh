#!/bin/bash

# Ollama Optimization Script for Raspberry Pi 5
# Fixes performance issues and optimizes for 2-4 second responses

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

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🚀 Ollama Pi 5 Performance Optimization"
echo "======================================"

# Check if running as root for systemd changes
if [[ $EUID -eq 0 ]]; then
    print_warning "Running as root - can modify system settings"
    CAN_MODIFY_SYSTEM=true
else
    print_warning "Not running as root - will suggest system commands"
    CAN_MODIFY_SYSTEM=false
fi

# Step 1: Stop Ollama service
print_status "Stopping Ollama service..."
if $CAN_MODIFY_SYSTEM; then
    systemctl stop ollama || true
else
    echo "Run: sudo systemctl stop ollama"
fi

# Step 2: Create optimized Ollama configuration
print_status "Creating optimized Ollama configuration..."

OLLAMA_SERVICE_DIR="/etc/systemd/system/ollama.service.d"
OLLAMA_CONFIG_FILE="$OLLAMA_SERVICE_DIR/pi5-optimization.conf"

if $CAN_MODIFY_SYSTEM; then
    mkdir -p "$OLLAMA_SERVICE_DIR"
    
    cat > "$OLLAMA_CONFIG_FILE" << 'EOF'
[Service]
# Pi 5 Optimized Ollama Configuration
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_ORIGINS=*"

# Performance Settings for Pi 5
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_MAX_QUEUE=1"

# Memory Management
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_LOAD_TIMEOUT=60s"

# CPU Optimization for ARM64
Environment="OLLAMA_NUM_THREAD=4"
Environment="OLLAMA_FLASH_ATTENTION=0"

# Network and Connection Settings
Environment="OLLAMA_REQUEST_TIMEOUT=30s"
Environment="OLLAMA_MAX_REQUEST_SIZE=32MB"

# Debug settings (can be disabled in production)
Environment="OLLAMA_DEBUG=0"
Environment="OLLAMA_VERBOSE=0"

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096
EOF

    print_success "Created optimized Ollama service configuration"
else
    print_warning "Create this file as root: $OLLAMA_CONFIG_FILE"
    echo "Contents should be:"
    cat << 'EOF'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_NUM_THREAD=4"
Environment="OLLAMA_FLASH_ATTENTION=0"
LimitNOFILE=65536
EOF
fi

# Step 3: Create Ollama modelfile for optimized Nemotron
print_status "Creating optimized Nemotron modelfile..."

MODELFILE_PATH="/tmp/Modelfile.nemotron-fast"

cat > "$MODELFILE_PATH" << 'EOF'
FROM nemotron-mini:4b-instruct-q4_K_M

# Optimized parameters for Pi 5 speed
PARAMETER num_ctx 512
PARAMETER num_predict 150
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.05
PARAMETER num_thread 4
PARAMETER num_gpu 0

# Stop tokens to prevent runaway generation
PARAMETER stop "</s>"
PARAMETER stop "<|end|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "Human:"
PARAMETER stop "User:"

# System message optimized for speed
SYSTEM """You are Pascal, a helpful AI assistant. Give concise, accurate responses. Be direct and helpful."""

# Template for fast inference
TEMPLATE """{{ if .System }}System: {{ .System }}

{{ end }}User: {{ .Prompt }}
Assistant: """
EOF

print_success "Created optimized Nemotron modelfile"

# Step 4: Reload systemd and restart Ollama
print_status "Restarting Ollama with optimizations..."

if $CAN_MODIFY_SYSTEM; then
    systemctl daemon-reload
    systemctl restart ollama
    sleep 5
    
    # Check if service started successfully
    if systemctl is-active --quiet ollama; then
        print_success "Ollama service restarted successfully"
    else
        print_error "Ollama service failed to start"
        systemctl status ollama
        exit 1
    fi
else
    echo "Run these commands as root:"
    echo "sudo systemctl daemon-reload"
    echo "sudo systemctl restart ollama"
fi

# Step 5: Wait for Ollama to be ready
print_status "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        print_success "Ollama is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "Ollama did not start within 30 seconds"
        exit 1
    fi
    sleep 1
done

# Step 6: Create optimized model
print_status "Creating optimized Nemotron model..."

if ollama list | grep -q "nemotron-mini:4b-instruct-q4_K_M"; then
    print_status "Creating fast-optimized version..."
    
    if ollama create nemotron-fast -f "$MODELFILE_PATH"; then
        print_success "Created optimized model: nemotron-fast"
    else
        print_warning "Failed to create optimized model, will use original"
    fi
else
    print_error "Base nemotron-mini:4b-instruct-q4_K_M model not found"
    print_status "Downloading base model..."
    if ollama pull nemotron-mini:4b-instruct-q4_K_M; then
        print_success "Downloaded base model"
        if ollama create nemotron-fast -f "$MODELFILE_PATH"; then
            print_success "Created optimized model: nemotron-fast"
        fi
    else
        print_error "Failed to download base model"
        exit 1
    fi
fi

# Step 7: Test performance
print_status "Testing optimized performance..."

# Function to test model speed
test_model_speed() {
    local model=$1
    local prompt="Hello! Respond with just 'Hi' and nothing else."
    
    print_status "Testing $model..."
    
    local start_time=$(date +%s.%N)
    local response=$(echo "$prompt" | ollama run "$model" 2>/dev/null | head -n 1)
    local end_time=$(date +%s.%N)
    
    local duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "unknown")
    
    if [ -n "$response" ]; then
        print_success "$model: $duration seconds - Response: '$response'"
        return 0
    else
        print_error "$model: Failed to respond"
        return 1
    fi
}

# Test available models
if ollama list | grep -q "nemotron-fast"; then
    test_model_speed "nemotron-fast"
fi

if ollama list | grep -q "nemotron-mini:4b-instruct-q4_K_M"; then
    test_model_speed "nemotron-mini:4b-instruct-q4_K_M"
fi

# Step 8: Configure Pascal to use optimized model
print_status "Updating Pascal configuration..."

PASCAL_CONFIG_FILE="data/models/nemotron_config.json"

if [ -f "$PASCAL_CONFIG_FILE" ]; then
    # Backup original
    cp "$PASCAL_CONFIG_FILE" "${PASCAL_CONFIG_FILE}.backup"
fi

# Create directory if it doesn't exist
mkdir -p "$(dirname "$PASCAL_CONFIG_FILE")"

cat > "$PASCAL_CONFIG_FILE" << 'EOF'
{
    "ollama_enabled": true,
    "ollama_host": "http://localhost:11434",
    "primary_model": "nemotron-fast",
    "fallback_model": "nemotron-mini:4b-instruct-q4_K_M",
    "optimization_date": "'$(date -Iseconds)'",
    "pi_model": "'$(cat /proc/device-tree/model 2>/dev/null || echo 'Unknown')'",
    "model_info": {
        "name": "nemotron-fast",
        "base_model": "nemotron-mini:4b-instruct-q4_K_M",
        "size": "~2.7GB",
        "type": "instruction-following",
        "optimized_for": "Pi 5 speed and efficiency",
        "target_response_time": "2-4 seconds",
        "best_use": ["general queries", "coding help", "explanations", "casual chat"]
    },
    "performance_settings": {
        "target_response_time": 2.0,
        "max_response_time": 8.0,
        "streaming_enabled": true,
        "keep_alive_duration": "30m",
        "num_ctx": 512,
        "num_predict": 150,
        "temperature": 0.7,
        "timeout": 15
    },
    "pascal_integration": {
        "routing": "offline for general queries",
        "fallback": "groq for current information",
        "priority": "primary offline model",
        "optimization_level": "high_speed"
    }
}
EOF

print_success "Updated Pascal configuration with optimized settings"

# Step 9: Cleanup
rm -f "$MODELFILE_PATH"

# Final summary
echo ""
print_success "🎉 Ollama Pi 5 Optimization Complete!"
echo "============================================"
echo ""
print_status "Optimizations Applied:"
echo "• Ollama service configuration optimized for Pi 5"
echo "• Model parameters tuned for 2-4 second responses"
echo "• Memory management improved with 30-minute keep-alive"
echo "• Context window reduced to 512 tokens for speed"
echo "• Connection timeouts optimized"
echo "• CPU threading set to 4 cores (Pi 5 maximum)"
echo ""
print_status "Available Models:"
ollama list | grep -E "(nemotron|NAME)" || echo "No models found"
echo ""
print_status "Next Steps:"
echo "1. Test Pascal: ./run.sh"
echo "2. Test with: 'Hello Pascal'"
echo "3. Check performance with: 'status' command"
echo "4. Monitor response times - should be 2-4 seconds"
echo ""
print_status "Performance Testing:"
echo "• Run: python performance_test.py"
echo "• Run: python ollama_diagnostic.py"
echo ""
if [ ! $CAN_MODIFY_SYSTEM ]; then
    print_warning "Manual steps required (run as sudo):"
    echo "• sudo systemctl daemon-reload"
    echo "• sudo systemctl restart ollama"
fi

# Show current Ollama status
print_status "Current Ollama Status:"
if systemctl is-active --quiet ollama; then
    print_success "✅ Ollama service is running"
    if curl -s http://localhost:11434/api/version > /dev/null; then
        print_success "✅ Ollama API is responding"
    else
        print_warning "⚠️ Ollama API not responding"
    fi
else
    print_error "❌ Ollama service is not running"
fi
