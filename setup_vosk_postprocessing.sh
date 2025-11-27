#!/bin/bash
#
# setup_vosk_postprocessing.sh
# Sets up Vosk post-processing dependencies:
# - Python packages (symspellpy, recasepunc)
# - SymSpell dictionary for spell checking
# - Vosk Recasepunc model for punctuation/case restoration

set -e

echo "🔧 Pascal Voice Post-Processing Setup"
echo "====================================="
echo ""

# Create config directory
mkdir -p config

#######################################
# Python Dependencies FIRST
#######################################

echo "📦 Step 1: Python Dependencies"
echo "-------------------------------"

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Not in a virtual environment"
    echo "Recommended: Activate your venv first"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled"
        exit 1
    fi
fi

echo "Installing Python packages..."
pip install symspellpy

# Note: recasepunc has issues with newer Python/torch
# We'll use Vosk's recasepunc model instead which works better
echo "📝 Note: Using Vosk recasepunc model for better compatibility"
echo ""

#######################################
# SymSpell Dictionary Setup
#######################################

DICT_FILE="config/frequency_dictionary_en_82_765.txt"
DICT_URL="https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt"

echo "📥 Step 2: SymSpell Dictionary"
echo "------------------------------"

if [ -f "$DICT_FILE" ]; then
    echo "✅ Dictionary already exists: $DICT_FILE"
else
    echo "Downloading SymSpell dictionary (~1.3MB)..."
    
    if command -v wget > /dev/null; then
        wget -O "$DICT_FILE" "$DICT_URL"
    elif command -v curl > /dev/null; then
        curl -L -o "$DICT_FILE" "$DICT_URL"
    else
        echo "❌ Error: Neither wget nor curl is available"
        exit 1
    fi
    
    if [ -f "$DICT_FILE" ]; then
        echo "✅ Dictionary downloaded successfully"
    else
        echo "❌ Dictionary download failed"
        exit 1
    fi
fi

echo ""

#######################################
# Vosk Recasepunc Model Setup
#######################################

RECASEPUNC_DIR="config/vosk-recasepunc-en-0.22"
CHECKPOINT_DIR="$RECASEPUNC_DIR/checkpoint"
MODEL_URL="https://alphacephei.com/vosk/models/vosk-recasepunc-en-0.22.zip"
MODEL_ZIP="config/vosk-recasepunc-en-0.22.zip"

echo "📥 Step 3: Vosk Recasepunc Model"
echo "--------------------------------"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "✅ Recasepunc model already exists: $CHECKPOINT_DIR"
else
    echo "Downloading Vosk Recasepunc model (~50MB)..."
    echo "URL: $MODEL_URL"
    echo ""
    
    # Download model
    if command -v wget > /dev/null; then
        wget -O "$MODEL_ZIP" "$MODEL_URL"
    elif command -v curl > /dev/null; then
        curl -L -o "$MODEL_ZIP" "$MODEL_URL"
    else
        echo "❌ Error: Neither wget nor curl is available"
        exit 1
    fi
    
    # Check if download succeeded
    if [ ! -f "$MODEL_ZIP" ]; then
        echo "❌ Download failed"
        exit 1
    fi
    
    # Verify download size (should be ~50MB)
    DOWNLOAD_SIZE=$(stat -c%s "$MODEL_ZIP" 2>/dev/null || stat -f%z "$MODEL_ZIP" 2>/dev/null || echo "0")
    MIN_SIZE=$((40 * 1024 * 1024))  # 40MB minimum
    
    if [ "$DOWNLOAD_SIZE" -lt "$MIN_SIZE" ]; then
        echo "⚠️  Warning: Download is smaller than expected (${DOWNLOAD_SIZE} bytes)"
        rm -f "$MODEL_ZIP"
        echo "❌ Download appears incomplete. Please try again."
        exit 1
    fi
    
    # Extract model
    if command -v unzip > /dev/null; then
        echo "📦 Extracting model..."
        unzip -q "$MODEL_ZIP" -d config/
        rm "$MODEL_ZIP"
        
        # Verify extraction
        if [ -d "$CHECKPOINT_DIR" ]; then
            echo "✅ Model extracted successfully"
        else
            # Model might be in a different structure - check and fix
            if [ -d "$RECASEPUNC_DIR" ]; then
                echo "✅ Model directory exists: $RECASEPUNC_DIR"
            else
                echo "❌ Extraction failed - checkpoint directory not found"
                exit 1
            fi
        fi
    else
        echo "❌ Error: unzip is not available"
        echo "Install unzip: sudo apt-get install unzip"
        exit 1
    fi
fi

echo ""

#######################################
# Verification
#######################################

echo "✅ Setup Complete!"
echo "=================="
echo ""
echo "Installed components:"
echo "  ✓ Python package: symspellpy"
echo "  ✓ SymSpell dictionary: $DICT_FILE"
echo "  ✓ Vosk Recasepunc model: $RECASEPUNC_DIR"
echo ""
echo "Configuration (in config/settings.py or .env):"
echo "  VOICE_ENABLE_SPELL_CHECK=true"
echo "  VOICE_ENABLE_CONFIDENCE_FILTER=true"
echo "  VOICE_ENABLE_PUNCTUATION=true"
echo "  VOICE_CONFIDENCE_THRESHOLD=0.80"
echo ""
echo "Test post-processing:"
echo "  python modules/vosk_postprocessor.py"
echo ""
echo "Run Pascal with voice + post-processing:"
echo "  ./run.sh --voice"
echo ""
