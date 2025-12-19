#!/usr/bin/env python3
"""
Setup script for downloading Piper TTS voice models for Pascal AI Assistant.

Downloads voice models for each personality:
- pascal: en_US-amy-medium (friendly, warm female voice)
- jarvis: en_GB-alan-medium (formal British male voice)  
- rick: en_US-lessac-medium (expressive male voice)
- default: en_US-lessac-medium (neutral fallback)

Usage:
    python setup_tts_voices.py           # Download all voices
    python setup_tts_voices.py pascal    # Download specific voice
    python setup_tts_voices.py --list    # List available voices
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# Voice definitions matching tts_voices.py
VOICES = {
    "pascal": {
        "model_name": "en_US-amy-medium",
        "description": "Friendly, warm female voice - matches Pascal's helpful personality",
        "base_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium",
        "size_mb": 75
    },
    "jarvis": {
        "model_name": "en_GB-alan-medium",
        "description": "Formal British male voice - matches JARVIS's butler-like demeanor",
        "base_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium",
        "size_mb": 75
    },
    "rick": {
        "model_name": "en_US-lessac-medium",
        "description": "Expressive male voice - closest match for Rick's sardonic tone",
        "base_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium",
        "size_mb": 75
    },
    "default": {
        "model_name": "en_US-lessac-medium",
        "description": "Default fallback voice - neutral and clear",
        "base_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium",
        "size_mb": 75
    }
}

def get_voices_dir() -> Path:
    """Get the TTS voices directory"""
    return Path(__file__).parent / "config" / "tts_voices"

def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """Download a file with progress indication"""
    try:
        print(f"  Downloading: {description or url}")
        
        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                print(f"\r  Progress: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\r  Progress: 100% - Done!")
        return True
        
    except urllib.error.HTTPError as e:
        print(f"\n  Error: HTTP {e.code} - {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n  Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n  Error: {e}")
        return False

def download_voice(personality: str, voices_dir: Path, force: bool = False) -> bool:
    """Download voice model for a personality"""
    if personality not in VOICES:
        print(f"Unknown personality: {personality}")
        print(f"Available: {', '.join(VOICES.keys())}")
        return False
    
    voice = VOICES[personality]
    model_name = voice["model_name"]
    base_url = voice["base_url"]
    
    # Create personality directory
    personality_dir = voices_dir / personality
    personality_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to download
    onnx_file = personality_dir / f"{model_name}.onnx"
    json_file = personality_dir / f"{model_name}.onnx.json"
    
    # Check if already exists
    if onnx_file.exists() and json_file.exists() and not force:
        print(f"  {personality}: Already downloaded (use --force to re-download)")
        return True
    
    print(f"\n[{personality.upper()}] {voice['description']}")
    print(f"  Model: {model_name} (~{voice['size_mb']}MB)")
    
    # Download ONNX model
    onnx_url = f"{base_url}/{model_name}.onnx"
    if not download_file(onnx_url, onnx_file, f"{model_name}.onnx"):
        return False
    
    # Download JSON config
    json_url = f"{base_url}/{model_name}.onnx.json"
    if not download_file(json_url, json_file, f"{model_name}.onnx.json"):
        # Clean up partial download
        if onnx_file.exists():
            onnx_file.unlink()
        return False
    
    print(f"  Installed to: {personality_dir}")
    return True

def list_voices(voices_dir: Path):
    """List all configured voices and their status"""
    print("\n" + "=" * 60)
    print("PIPER TTS VOICES FOR PASCAL")
    print("=" * 60)
    
    for personality, voice in VOICES.items():
        model_name = voice["model_name"]
        personality_dir = voices_dir / personality
        onnx_file = personality_dir / f"{model_name}.onnx"
        json_file = personality_dir / f"{model_name}.onnx.json"
        
        installed = onnx_file.exists() and json_file.exists()
        status = "✅ Installed" if installed else "❌ Not installed"
        
        print(f"\n[{personality.upper()}] {status}")
        print(f"  Model: {model_name}")
        print(f"  Description: {voice['description']}")
        print(f"  Size: ~{voice['size_mb']}MB")
        if installed:
            print(f"  Path: {personality_dir}")
    
    print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Download Piper TTS voice models for Pascal AI Assistant"
    )
    parser.add_argument(
        "personalities",
        nargs="*",
        help="Personalities to download voices for (default: all)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available voices and their status"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if voice exists"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="Custom directory for voice models"
    )
    
    args = parser.parse_args()
    
    # Get voices directory
    if args.dir:
        voices_dir = Path(args.dir)
    else:
        voices_dir = get_voices_dir()
    
    # List mode
    if args.list:
        list_voices(voices_dir)
        return
    
    # Download mode
    print("\n" + "=" * 60)
    print("PIPER TTS VOICE SETUP FOR PASCAL")
    print("=" * 60)
    print(f"\nVoices directory: {voices_dir}")
    
    # Determine which voices to download
    personalities = args.personalities if args.personalities else list(VOICES.keys())
    
    # Filter duplicates (rick and default use same model)
    seen_models = set()
    unique_personalities = []
    for p in personalities:
        if p in VOICES:
            model = VOICES[p]["model_name"]
            if model not in seen_models or args.force:
                unique_personalities.append(p)
                seen_models.add(model)
    
    # Download each voice
    success_count = 0
    fail_count = 0
    
    for personality in unique_personalities:
        if download_voice(personality, voices_dir, args.force):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Downloaded: {success_count}")
    print(f"  Failed: {fail_count}")
    
    if fail_count > 0:
        print("\nSome downloads failed. Check your internet connection and try again.")
        sys.exit(1)
    else:
        print("\nAll voices ready! Run Pascal with --speak to enable TTS:")
        print("  python main.py --voice --speak")

if __name__ == "__main__":
    main()
