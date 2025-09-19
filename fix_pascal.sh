#!/bin/bash

# Pascal AI Assistant - System Fix Script
# Addresses compatibility issues and fixes common problems

set -e

# Colors
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

echo "🔧 Pascal AI Assistant - System Fix Script"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    print_error "main.py not found. Please run this script from the Pascal directory."
    exit 1
fi

# Step 1: Fix Python dependencies
print_status "Fixing Python dependencies..."

if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install fixed requirements with compatible versions
print_status "Installing compatible dependencies..."

# Remove problematic packages first
pip uninstall -y aiohttp aiofiles || true

# Install specific compatible versions
pip install aiohttp==3.8.6
pip install aiofiles==23.2.0
pip install requests==2.31.0
pip install python-dotenv==1.0.0
pip install rich==13.7.0
pip install colorama==0.4.6
pip install psutil==5.9.6
pip install json5==0.9.14
pip install typing-extensions==4.8.0

print_success "Dependencies fixed with compatible versions"

# Step 2: Check Ollama service
print_status "Checking Ollama service..."

if command -v ollama &> /dev/null; then
    print_success "Ollama command found"
    
    # Check if service is running
    if systemctl is-active --quiet ollama; then
        print_success "Ollama service is running"
    else
        print_warning "Ollama service not running - attempting to start..."
        if sudo systemctl start ollama; then
            print_success "Ollama service started"
            sleep 3
        else
            print_error "Failed to start Ollama service"
        fi
    fi
    
    # Check for models
    if ollama list | grep -q "nemotron"; then
        print_success "Nemotron model found"
    else
        print_warning "Nemotron model not found"
        read -p "Download Nemotron model now? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Downloading Nemotron model..."
            ollama pull nemotron-mini:4b-instruct-q4_K_M || print_warning "Model download failed"
        fi
    fi
else
    print_error "Ollama not found. Install with: curl -fsSL https://ollama.ai/install.sh | sh"
fi

# Step 3: Check .env configuration
print_status "Checking configuration..."

if [ ! -f ".env" ]; then
    print_warning ".env file not found - creating from template..."
    
    cat > .env << 'EOF'
# Pascal AI Assistant Environment Variables

# GROQ API (for online features)
# Get from: https://console.groq.com/
GROQ_API_KEY=

# Optional API Keys
OPENWEATHER_API_KEY=
NEWS_API_KEY=

# Performance Settings
PERFORMANCE_MODE=balanced
STREAMING_ENABLED=true
TARGET_RESPONSE_TIME=2.0
MAX_RESPONSE_TOKENS=200

# Ollama Settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=30
OLLAMA_KEEP_ALIVE=30m

# Debug Settings
DEBUG=false
LOG_LEVEL=INFO
EOF
    
    print_success "Created .env template file"
    print_warning "Please add your Groq API key to .env for online features"
fi

# Step 4: Test the fixes
print_status "Testing fixes..."

# Test Python imports
python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import aiohttp
    print(f'✅ aiohttp {aiohttp.__version__} imported successfully')
except ImportError as e:
    print(f'❌ aiohttp import failed: {e}')
    sys.exit(1)

try:
    from config.settings import settings
    print(f'✅ Pascal settings loaded (v{settings.version})')
except ImportError as e:
    print(f'❌ Pascal settings import failed: {e}')
    sys.exit(1)

try:
    from modules.offline_llm import LightningOfflineLLM
    print('✅ LightningOfflineLLM imported successfully')
except ImportError as e:
    print(f'❌ LightningOfflineLLM import failed: {e}')
    sys.exit(1)

try:
    from modules.router import LightningRouter
    print('✅ LightningRouter imported successfully')
except ImportError as e:
    print(f'❌ LightningRouter import failed: {e}')
    sys.exit(1)

print('✅ All critical imports working!')
"

if [ $? -eq 0 ]; then
    print_success "Import tests passed"
else
    print_error "Import tests failed"
    exit 1
fi

# Step 5: Create diagnostic script shortcut
print_status "Creating diagnostic shortcuts..."

# Make diagnostic script executable
chmod +x complete_diagnostic.py || true

# Create quick test script
cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""Quick Pascal system test"""
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

async def quick_test():
    try:
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        
        print("🧪 Quick Pascal Test")
        print("===================")
        
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        router = LightningRouter(personality_manager, memory_manager)
        
        await router._check_llm_availability()
        
        print(f"Offline available: {router.offline_available}")
        print(f"Online available: {router.online_available}")
        print(f"Skills available: {router.skills_available}")
        
        if router.offline_available or router.online_available:
            print("\n🧪 Testing response...")
            response = await router.get_response("Hello Pascal")
            print(f"Response: {response[:100]}...")
            print("\n✅ Quick test passed - Pascal should work!")
        else:
            print("\n⚠️ No systems available - check configuration")
        
        await router.close()
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(quick_test())
    sys.exit(0 if result else 1)
EOF

chmod +x quick_test.py

print_success "Created quick_test.py for rapid testing"

# Step 6: Summary and next steps
echo ""
print_success "🎉 Pascal system fixes applied!"
echo "====================================="
echo ""

print_status "What was fixed:"
echo "• Updated aiohttp to compatible version (3.8.6)"
echo "• Fixed TCP connector compatibility issues"
echo "• Created/updated .env configuration"
echo "• Verified Python imports"
echo "• Created diagnostic tools"
echo ""

print_status "Next steps:"
echo "1. Run full diagnostic: python complete_diagnostic.py"
echo "2. Run quick test: python quick_test.py"
echo "3. If tests pass, start Pascal: ./run.sh"
echo ""

print_status "Troubleshooting:"
echo "• If Ollama issues: sudo systemctl start ollama"
echo "• If model missing: ollama pull nemotron-mini:4b-instruct-q4_K_M"
echo "• If online features needed: Add GROQ_API_KEY to .env"
echo "• For performance: Run ./ollama_optimization.sh"
echo ""

print_status "Test commands:"
echo "• python complete_diagnostic.py - Full system test"
echo "• python quick_test.py - Quick functionality test"
echo "• python ollama_diagnostic.py - Ollama-specific test"
echo ""

print_success "Pascal fix script completed!"
