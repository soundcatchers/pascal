#!/usr/bin/env python3
"""
TTS Test Script - Debug audio output issues

Run this to test if Piper TTS works independently:
    python test_tts.py
"""

import sys

def test_sounddevice():
    """Test sounddevice is working"""
    print("\n=== Testing sounddevice ===")
    try:
        import sounddevice as sd
        import numpy as np
        
        print(f"sounddevice version: {sd.__version__}")
        print(f"\nDefault output device: {sd.default.device[1]}")
        print(f"\nAvailable output devices:")
        devices = sd.query_devices()
        usb_speaker = None
        for i, d in enumerate(devices):
            if d['max_output_channels'] > 0:
                name_lower = d['name'].lower()
                marker = ""
                # Skip ReSpeaker (it's a mic array, not a speaker)
                is_mic = 'respeaker' in name_lower or 'mic' in name_lower
                if 'usb' in name_lower:
                    if is_mic:
                        marker = " <-- USB MIC (skip)"
                    else:
                        marker = " <-- USB SPEAKER"
                        if usb_speaker is None:
                            usb_speaker = i
                print(f"  [{i}] {d['name']} (channels: {d['max_output_channels']}){marker}")
        
        # Use USB speaker if found, otherwise use default (None)
        output_device = usb_speaker
        if usb_speaker is not None:
            print(f"\n[TEST] Using USB speaker: device {usb_speaker}")
        else:
            print(f"\n[TEST] Using system default audio")
        
        # Test beep at 48000Hz (most compatible sample rate)
        print("\n[TEST] Playing test tone (440Hz for 0.5s) at 48000Hz...")
        sample_rate = 48000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        tone = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
        
        sd.play(tone, sample_rate, device=output_device)
        sd.wait()
        print("[TEST] ✅ Test tone complete - did you hear it?")
        return True
    except Exception as e:
        print(f"[TEST] ❌ sounddevice error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_piper():
    """Test Piper TTS using subprocess mode (more reliable on Pi)"""
    print("\n=== Testing Piper TTS ===")
    try:
        from pathlib import Path
        import json
        
        # Find a model (models are in subdirectories: config/tts_voices/<personality>/*.onnx)
        voices_dir = Path("config/tts_voices")
        models = list(voices_dir.glob("**/*.onnx"))
        
        if not models:
            print(f"[TEST] ❌ No voice models found in {voices_dir}")
            print("Run: python setup_tts_voices.py")
            return False
        
        model_path = str(models[0])
        print(f"[TEST] Using model: {model_path}")
        
        # Get sample rate from JSON config
        config_path = model_path + ".json"
        sample_rate = 22050  # default
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                sample_rate = config.get('audio', {}).get('sample_rate', 22050)
        print(f"[TEST] ✅ Model config loaded (sample rate: {sample_rate})")
        
        # Synthesize to a temp WAV file (more reliable than BytesIO)
        import tempfile
        import wave
        import os
        
        test_text = "Hello, this is a test of Piper text to speech."
        print(f"[TEST] Synthesizing: {test_text}")
        
        # Create temp file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_wav.name
        temp_wav.close()
        
        try:
            # Use subprocess mode - more reliable on Pi
            import subprocess
            result = subprocess.run(
                ['piper', '--model', model_path, '--output_file', temp_path],
                input=test_text,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                print(f"[TEST] ❌ Piper CLI failed: {result.stderr}")
                return False
            
            # Read back the audio data
            with wave.open(temp_path, 'rb') as wav_file:
                n_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(n_frames)
            
            import numpy as np
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            print(f"[TEST] ✅ Synthesized {len(audio_data)} samples")
            
            if len(audio_data) == 0:
                print("[TEST] ❌ No audio data generated!")
                return False
            
            # Resample to 48000Hz and convert to float32
            audio_float = audio_data.astype(np.float32) / 32768.0
            original_rate = sample_rate
            playback_rate = 48000
            
            if original_rate != playback_rate:
                original_length = len(audio_float)
                new_length = int(original_length * playback_rate / original_rate)
                indices = np.linspace(0, original_length - 1, new_length)
                audio_float = np.interp(indices, np.arange(original_length), audio_float)
                print(f"[TEST] Resampled from {original_rate}Hz to {playback_rate}Hz")
            
            # Play it through USB speaker
            import sounddevice as sd
            
            # Find USB speaker
            usb_device = None
            for i, d in enumerate(sd.query_devices()):
                if d['max_output_channels'] > 0:
                    name = d['name'].lower()
                    if 'usb' in name and 'respeaker' not in name and 'mic' not in name:
                        usb_device = i
                        break
            
            print(f"[TEST] Playing synthesized speech on device {usb_device}...")
            sd.play(audio_float, playback_rate, device=usb_device)
            sd.wait()
            print("[TEST] ✅ Playback complete - did you hear the speech?")
            return True
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"[TEST] ❌ Piper error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_tts_manager():
    """Test the full TTS manager"""
    print("\n=== Testing TTSManager ===")
    try:
        import asyncio
        from modules.tts_manager import TTSManager
        
        async def run_test():
            tts = TTSManager(debug=True)
            await tts.initialize()
            await tts.speak("Hello Andy! Testing the complete TTS system.")
            await tts.close()
        
        asyncio.run(run_test())
        print("[TEST] ✅ TTSManager test complete")
        return True
    except Exception as e:
        print(f"[TEST] ❌ TTSManager error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("  Pascal TTS Debug Test Suite")
    print("=" * 50)
    
    # Run tests
    sd_ok = test_sounddevice()
    
    if not sd_ok:
        print("\n⚠️  Fix sounddevice issues first!")
        print("Try: pip install sounddevice numpy")
        print("Also check: sudo apt install libportaudio2")
        sys.exit(1)
    
    piper_ok = test_piper()
    
    if piper_ok:
        test_full_tts_manager()
    
    print("\n" + "=" * 50)
    if sd_ok and piper_ok:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed - check output above")
    print("=" * 50)
