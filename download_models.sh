#!/bin/bash

# Pascal AI Assistant - Model Download Script for Raspberry Pi 5
# Downloads optimized GGUF models for best Pi 5 performance

set -e

echo "🤖 Pascal AI Assistant - Model Downloader for Raspberry Pi 5"
echo "=============================================================="

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
            print_success "Detected Raspberry Pi 5 - proceeding with optimized downloads"
        elif [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
            print_warning "Detected $PI_MODEL - these models are optimized for Pi 5"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            print_warning "Not running on Raspberry Pi - models are optimized for Pi 5"
        fi
    fi
}

# Check available space
check_space() {
    AVAILABLE_GB=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    print_status "Available disk space: ${AVAILABLE_GB}GB"
    
    if [ $AVAILABLE_GB -lt 20 ]; then
        print_error "Insufficient disk space. Need at least 20GB free."
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

# Create models directory
setup_directory() {
    print_status "Setting up models directory..."
    
    # Get script directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    MODELS_DIR="$SCRIPT_DIR/data/models"
    
    mkdir -p "$MODELS_DIR"
    cd "$MODELS_DIR"
    
    print_success "Models directory: $MODELS_DIR"
}

# Download function with progress and retry
download_model() {
    local name="$1"
    local url="$2"
    local filename="$3"
    local size="$4"
    
    print_status "Downloading $name ($size)..."
    print_status "URL: $url"
    
    # Check if file already exists
    if [ -f "$filename" ]; then
        print_warning "$filename already exists. Skipping download."
        return 0
    fi
    
    # Download with wget, showing progress
    if wget --progress=bar:force:noscroll -O "$filename.tmp" "$url"; then
        mv "$filename.tmp" "$filename"
        print_success "Downloaded $name successfully"
        
        # Verify file size (basic check)
        ACTUAL_SIZE=$(du -h "$filename" | cut -f1)
        print_status "File size: $ACTUAL_SIZE"
        
        return 0
    else
        print_error "Failed to download $name"
        rm -f "$filename.tmp"
        return 1
    fi
}

# Main download function
download_models() {
    print_status "Starting model downloads..."
    
    # Model definitions: name, URL, filename, approximate size
    declare -A MODELS
    
    # Gemma 2-9B (Q4_K_M) - Best overall quality for Pi 5
    MODELS["gemma2-9b"]="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf|gemma-2-9b-it-Q4_K_M.gguf|5.4GB"
    
    # Qwen2.5-7B (Q4_K_M) - Excellent efficiency
    MODELS["qwen2.5-7b"]="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf|qwen2.5-7b-instruct-q4_k_m.gguf|4.4GB"
    
    # Phi-3-Mini (Q5_K_M) - Fastest responses
    MODELS["phi3-mini"]="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q5_k_m.gguf|Phi-3-mini-4k-instruct-q5_k_m.gguf|2.8GB"
    
    # Interactive selection
    echo ""
    echo "Available models for Raspberry Pi 5:"
    echo "1. Gemma 2-9B (Q4_K_M) - 5.4GB - Best overall quality and reasoning"
    echo "2. Qwen2.5-7B (Q4_K_M) - 4.4GB - Excellent balance of speed and quality"
    echo "3. Phi-3-Mini (Q5_K_M) - 2.8GB - Fastest responses, good for quick tasks"
    echo "4. Download all models (12.6GB total)"
    echo "5. Custom selection"
    echo ""
    
    read -p "Choose an option (1-5): " choice
    
    case $choice in
        1)
            download_model "Gemma 2-9B" $(echo ${MODELS["gemma2-9b"]} | cut -d'|' -f1) $(echo ${MODELS["gemma2-9b"]} | cut -d'|' -f2) $(echo ${MODELS["gemma2-9b"]} | cut -d'|' -f3)
            ;;
        2)
            download_model "Qwen2.5-7B" $(echo ${MODELS["qwen2.5-7b"]} | cut -d'|' -f1) $(echo ${MODELS["qwen2.5-7b"]} | cut -d'|' -f2) $(echo ${MODELS["qwen2.5-7b"]} | cut -d'|' -f3)
            ;;
        3)
            download_model "Phi-3-Mini" $(echo ${MODELS["phi3-mini"]} | cut -d'|' -f1) $(echo ${MODELS["phi3-mini"]} | cut -d'|' -f2) $(echo ${MODELS["phi3-mini"]} | cut -d'|' -f3)
            ;;
        4)
            for model_key in "${!MODELS[@]}"; do
                model_data=${MODELS[$model_key]}
                url=$(echo $model_data | cut -d'|' -f1)
                filename=$(echo $model_data | cut -d'|' -f2)
                size=$(echo $model_data | cut -d'|' -f3)
                download_model "$model_key" "$url" "$filename" "$size"
            done
            ;;
        5)
            echo "Select models to download (space-separated numbers, e.g., '1 3'):"
            read -p "Models: " selected
            for num in $selected; do
                case $num in
                    1) download_model "Gemma 2-9B" $(echo ${MODELS["gemma2-9b"]} | cut -d'|' -f1) $(echo ${MODELS["gemma2-9b"]} | cut -d'|' -f2) $(echo ${MODELS["gemma2-9b"]} | cut -d'|' -f3) ;;
                    2) download_model "Qwen2.5-7B" $(echo ${MODELS["qwen2.5-7b"]} | cut -d'|' -f1) $(echo ${MODELS["qwen2.5-7b"]} | cut -d'|' -f2) $(echo ${MODELS["qwen2.5-7b"]} | cut -d'|' -f3) ;;
                    3) download_model "Phi-3-Mini" $(echo ${MODELS["phi3-mini"]} | cut -d'|' -f1) $(echo ${MODELS["phi3-mini"]} | cut -d'|' -f2) $(echo ${MODELS["phi3-mini"]} | cut -d'|' -f3) ;;
                    *) print_warning "Invalid selection: $num" ;;
                esac
            done
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Verify downloads
verify_models() {
    print_status "Verifying downloaded models..."
    
    MODEL_COUNT=0
    TOTAL_SIZE=0
    
    for gguf_file in *.gguf; do
        if [ -f "$gguf_file" ]; then
            SIZE=$(du -m "$gguf_file" | cut -f1)
            TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
            MODEL_COUNT=$((MODEL_COUNT + 1))
            print_success "✓ $gguf_file (${SIZE}MB)"
        fi
    done
    
    if [ $MODEL_COUNT -eq 0 ]; then
        print_error "No GGUF models found!"
        exit 1
    fi
    
    TOTAL_SIZE_GB=$((TOTAL_SIZE / 1024))
    print_success "Found $MODEL_COUNT model(s), total size: ${TOTAL_SIZE_GB}GB"
}

# Test model loading (optional)
test_model() {
    print_status "Testing model compatibility..."
    
    # Check if Python and required packages are available
    if command -v python3 &> /dev/null; then
        if python3 -c "import llama_cpp" 2> /dev/null; then
            print_success "llama-cpp-python is installed and ready"
            
            # Quick test of model loading
            read -p "Test model loading? This will take 30-60 seconds (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                python3 -c "
import sys
sys.path.append('../../')
from modules.offline_llm import OptimizedOfflineLLM
import asyncio

async def test():
    print('Testing model loading...')
    llm = OptimizedOfflineLLM()
    if await llm.initialize():
        print('✅ Model loading test successful!')
        stats = llm.get_performance_stats()
        print(f'Loaded model: {stats[\"model_name\"]}')
        print(f'RAM usage: {stats[\"model_ram_usage\"]}')
        await llm.close()
    else:
        print('❌ Model loading test failed')

asyncio.run(test())
"
            fi
        else
            print_warning "llama-cpp-python not installed. Run: pip install llama-cpp-python"
        fi
    else
        print_warning "Python3 not found. Please install Python 3.8+"
    fi
}

# Create model info file
create_model_info() {
    print_status "Creating model information file..."
    
    cat > model_info.json << EOF
{
    "download_date": "$(date -Iseconds)",
    "pi_model": "$(cat /proc/device-tree/model 2>/dev/null || echo 'Unknown')",
    "total_models": $(ls -1 *.gguf 2>/dev/null | wc -l),
    "models": [
EOF

    FIRST=true
    for gguf_file in *.gguf; do
        if [ -f "$gguf_file" ]; then
            if [ "$FIRST" = false ]; then
                echo "        ," >> model_info.json
            fi
            SIZE=$(du -h "$gguf_file" | cut -f1)
            echo "        {" >> model_info.json
            echo "            \"filename\": \"$gguf_file\"," >> model_info.json
            echo "            \"size\": \"$SIZE\"" >> model_info.json
            echo -n "        }" >> model_info.json
            FIRST=false
        fi
    done

    cat >> model_info.json << EOF

    ],
    "recommendations": {
        "fastest": "Phi-3-mini for quick responses",
        "balanced": "Qwen2.5-7B for general use",
        "highest_quality": "Gemma 2-9B for complex tasks"
    }
}
EOF

    print_success "Model info saved to model_info.json"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up temporary files..."
    rm -f *.tmp
}

# Main execution
main() {
    echo "Starting Pascal model download process..."
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    # Run checks
    check_pi
    check_space
    check_ram
    
    # Setup
    setup_directory
    
    # Download models
    download_models
    
    # Verify
    verify_models
    
    # Create info file
    create_model_info
    
    # Optional testing
    test_model
    
    echo ""
    print_success "Model download complete!"
    print_status "Models location: $(pwd)"
    print_status "You can now run Pascal with: ./run.sh"
    
    echo ""
    echo "Model Performance Guide:"
    echo "• Phi-3-Mini: Fastest responses (1-2s), good for chat"
    echo "• Qwen2.5-7B: Balanced performance (2-4s), versatile"  
    echo "• Gemma 2-9B: Best quality (3-6s), complex reasoning"
    echo ""
    echo "Use 'personality speed/balanced/quality' in Pascal to optimize!"
}

# Run main function
main "$@"
