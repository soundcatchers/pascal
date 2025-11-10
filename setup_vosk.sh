#!/bin/bash
# Vosk Model Setup Script for Pascal Voice Input
# Downloads and extracts the Vosk English model for speech recognition

set -e

MODEL_NAME="vosk-model-small-en-us-0.15"
MODEL_URL="https://alphacephei.com/vosk/models/${MODEL_NAME}.zip"
MODEL_DIR="config/vosk_models"

echo "🎙️  Pascal Voice Input - Vosk Model Setup"
echo "=========================================="
echo ""

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check if model already exists
if [ -d "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "✅ Vosk model already exists: $MODEL_DIR/$MODEL_NAME"
    echo ""
    echo "To re-download, remove the directory first:"
    echo "  rm -rf $MODEL_DIR/$MODEL_NAME"
    exit 0
fi

echo "📥 Downloading Vosk model..."
echo "Model: $MODEL_NAME (~50MB)"
echo "URL: $MODEL_URL"
echo ""

# Download model
if command -v wget > /dev/null; then
    wget -O "$MODEL_DIR/${MODEL_NAME}.zip" "$MODEL_URL"
elif command -v curl > /dev/null; then
    curl -L -o "$MODEL_DIR/${MODEL_NAME}.zip" "$MODEL_URL"
else
    echo "❌ Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

echo ""
echo "📦 Extracting model..."

# Extract model
cd "$MODEL_DIR"
unzip -q "${MODEL_NAME}.zip"
rm "${MODEL_NAME}.zip"

echo ""
echo "✅ Vosk model installed successfully!"
echo ""
echo "Model location: $MODEL_DIR/$MODEL_NAME"
echo ""
echo "🎤 Next steps:"
echo "1. Install Python dependencies: pip install vosk pyaudio"
echo "2. Test voice input: python main.py --voice"
echo ""
