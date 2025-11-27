#!/bin/bash
#
# setup_vosk_postprocessing.sh
# Sets up Vosk post-processing dependencies:
# - SymSpell dictionary for spell checking
# - Recasepunc checkpoint for punctuation/case restoration

set -e

echo "🔧 Pascal Voice Post-Processing Setup"
echo "====================================="
echo ""

# Create config directory
mkdir -p config

#######################################
# SymSpell Dictionary Setup
#######################################

DICT_FILE="config/frequency_dictionary_en_82_765.txt"
DICT_URL="https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt"

echo "📥 Step 1: SymSpell Dictionary"
echo "------------------------------"

if [ -f "$DICT_FILE" ]; then
    echo "✅ Dictionary already exists: $DICT_FILE"
else
    echo "Downloading SymSpell dictionary (~3MB)..."
    echo "URL: $DICT_URL"
    echo ""
    
    if command -v wget > /dev/null; then
        wget -O "$DICT_FILE" "$DICT_URL"
    elif command -v curl > /dev/null; then
        curl -L -o "$DICT_FILE" "$DICT_URL"
    else
        echo "❌ Error: Neither wget nor curl is available"
        echo "Please install wget or curl, then run this script again"
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
# Recasepunc Checkpoint Setup
#######################################

RECASEPUNC_DIR="config/recasepunc"
CHECKPOINT_DIR="$RECASEPUNC_DIR/checkpoint"
RELEASE_URL="https://github.com/benob/recasepunc/releases/download/v0.4/checkpoint.zip"

echo "📥 Step 2: Recasepunc Checkpoint"
echo "--------------------------------"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "✅ Checkpoint already exists: $CHECKPOINT_DIR"
else
    echo "Downloading Recasepunc checkpoint (~250MB)..."
    echo "URL: $RELEASE_URL"
    echo ""
    echo "⏳ This may take a few minutes on slower connections..."
    echo ""
    
    mkdir -p "$RECASEPUNC_DIR"
    
    # Download checkpoint
    if command -v wget > /dev/null; then
        wget -O "$RECASEPUNC_DIR/checkpoint.zip" "$RELEASE_URL"
    elif command -v curl > /dev/null; then
        curl -L -o "$RECASEPUNC_DIR/checkpoint.zip" "$RELEASE_URL"
    else
        echo "❌ Error: Neither wget nor curl is available"
        exit 1
    fi
    
    # Verify download size (checkpoint should be ~250MB)
    CHECKPOINT_SIZE=$(stat -f%z "$RECASEPUNC_DIR/checkpoint.zip" 2>/dev/null || stat -c%s "$RECASEPUNC_DIR/checkpoint.zip" 2>/dev/null || echo "0")
    MIN_SIZE=$((200 * 1024 * 1024))  # 200MB minimum
    
    if [ "$CHECKPOINT_SIZE" -lt "$MIN_SIZE" ]; then
        echo "⚠️  Warning: Downloaded checkpoint is smaller than expected ($CHECKPOINT_SIZE bytes)"
        echo "Expected: ~250MB minimum"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            rm "$RECASEPUNC_DIR/checkpoint.zip"
            echo "Setup cancelled. Please try downloading manually."
            exit 1
        fi
    fi
    
    # Extract checkpoint
    if command -v unzip > /dev/null; then
        echo "📦 Extracting checkpoint..."
        cd "$RECASEPUNC_DIR"
        unzip -q checkpoint.zip
        cd ../..
        
        # Remove zip file
        rm "$RECASEPUNC_DIR/checkpoint.zip"
        
        echo "✅ Checkpoint extracted successfully"
    else
        echo "❌ Error: unzip is not available"
        echo "Install unzip: sudo apt-get install unzip"
        exit 1
    fi
fi

echo ""

#######################################
# Python Dependencies
#######################################

echo "📦 Step 3: Python Dependencies"
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
pip install symspellpy recasepunc

echo ""

#######################################
# Verification
#######################################

echo "✅ Setup Complete!"
echo "=================="
echo ""
echo "Installed components:"
echo "  ✓ SymSpell dictionary: $DICT_FILE"
echo "  ✓ Recasepunc checkpoint: $CHECKPOINT_DIR"
echo "  ✓ Python packages: symspellpy, recasepunc"
echo ""
echo "Configuration (in config/settings.py or .env):"
echo "  VOICE_ENABLE_SPELL_CHECK=true"
echo "  VOICE_ENABLE_CONFIDENCE_FILTER=true"
echo "  VOICE_ENABLE_PUNCTUATION=true"
echo "  VOICE_CONFIDENCE_THRESHOLD=0.80"
echo ""
echo "Test post-processing installation:"
echo "  python modules/vosk_postprocessor.py"
echo ""
echo "Run Pascal with voice + post-processing:"
echo "  ./run.sh --voice"
echo ""
