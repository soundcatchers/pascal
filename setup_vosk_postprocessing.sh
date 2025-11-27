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
pip install symspellpy deepmultilingualpunctuation

echo "📝 Note: First run will download punctuation model (~1.5GB)"
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

echo ""

#######################################
# Verification
#######################################

echo "✅ Setup Complete!"
echo "=================="
echo ""
echo "Installed components:"
echo "  ✓ Python packages: symspellpy, deepmultilingualpunctuation"
echo "  ✓ SymSpell dictionary: $DICT_FILE"
echo ""
echo "📝 Note: Punctuation model (~1.5GB) downloads on first use"
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
