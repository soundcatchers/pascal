"""
Piper TTS Implementation for Pascal AI Assistant

Handles the Piper text-to-speech engine with streaming audio output.
Supports interruption and multiple voice models.
"""

import asyncio
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Generator, Callable
from dataclasses import dataclass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


@dataclass
class PiperConfig:
    """Configuration for Piper TTS"""
    model_path: str
    config_path: Optional[str] = None
    speed: float = 1.0
    speaker_id: Optional[int] = None
    noise_scale: float = 0.667
    length_scale: float = 1.0
    noise_w: float = 0.8
    sentence_silence: float = 0.2


class PiperTTS:
    """
    Piper TTS engine wrapper with streaming support.
    
    Supports two modes:
    1. Python library mode (piper-tts package) - preferred
    2. Subprocess mode (piper binary) - fallback
    """
    
    def __init__(self, debug: bool = False, output_device: Optional[int] = None):
        self.debug = debug
        self.model = None
        self.model_path: Optional[str] = None
        self.config: Optional[PiperConfig] = None
        self.sample_rate: int = 22050
        self.available = False
        self.mode = None  # 'library' or 'subprocess'
        self.output_device = output_device  # Audio output device index
        
        # Streaming state
        self._stop_flag = False
        self._is_speaking = False
        self._stream: Optional[sd.OutputStream] = None
        self._speak_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_start: Optional[Callable] = None
        self._on_stop: Optional[Callable] = None
        self._on_word: Optional[Callable] = None
        
        self._detect_mode()
        self._detect_output_device()
    
    def _detect_mode(self):
        """Detect which mode to use (subprocess preferred - more reliable on Pi)"""
        # Try subprocess mode first (piper binary) - more reliable
        try:
            result = subprocess.run(
                ['piper', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.mode = 'subprocess'
                self.available = True
                if self.debug:
                    print(f"[TTS] ✅ Piper subprocess mode available: {result.stdout.strip()}")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Fallback to Python library (has issues with wave file handling on Pi)
        try:
            from piper import PiperVoice
            self.mode = 'library'
            self.available = True
            if self.debug:
                print("[TTS] ✅ Piper library mode available (fallback)")
            return
        except ImportError:
            pass
        
        if self.debug:
            print("[TTS] ❌ Piper not available (install piper-tts or piper binary)")
        self.available = False
    
    def _detect_output_device(self):
        """Auto-detect the best audio output device (prefer USB speakers, exclude mic arrays)"""
        if not SOUNDDEVICE_AVAILABLE:
            return
        
        if self.output_device is not None:
            return  # Already set explicitly
        
        try:
            devices = sd.query_devices()
            usb_speaker = None
            
            # Look for USB audio output devices (exclude mic arrays)
            for i, d in enumerate(devices):
                if d['max_output_channels'] > 0:
                    name_lower = d['name'].lower()
                    # Skip microphone arrays (ReSpeaker is a mic, not a speaker)
                    if 'respeaker' in name_lower or 'mic' in name_lower:
                        continue
                    # Prefer USB speakers
                    if 'usb' in name_lower and usb_speaker is None:
                        usb_speaker = i
                        if self.debug:
                            print(f"[TTS] 🔊 Found USB speaker: [{i}] {d['name']}")
            
            if usb_speaker is not None:
                self.output_device = usb_speaker
                if self.debug:
                    print(f"[TTS] ✅ Using USB speaker: device {usb_speaker}")
            else:
                # Use default device (pulse/default usually works best)
                self.output_device = None  # None = use system default
                if self.debug:
                    print(f"[TTS] 🔊 Using system default audio output")
                
        except Exception as e:
            if self.debug:
                print(f"[TTS] ⚠️  Could not detect audio devices: {e}")
    
    def load_model(self, model_path: str, config_path: Optional[str] = None) -> bool:
        """Load a voice model"""
        if not self.available:
            return False
        
        model_file = Path(model_path)
        if not model_file.exists():
            if self.debug:
                print(f"[TTS] ❌ Model not found: {model_path}")
            return False
        
        # Auto-detect config path
        if config_path is None:
            config_path = str(model_file) + ".json"
        
        if not Path(config_path).exists():
            if self.debug:
                print(f"[TTS] ❌ Config not found: {config_path}")
            return False
        
        self.model_path = str(model_path)
        self.config = PiperConfig(model_path=self.model_path, config_path=config_path)
        
        if self.mode == 'library':
            try:
                from piper import PiperVoice
                self.model = PiperVoice.load(self.model_path, config_path=config_path)
                self.sample_rate = self.model.config.sample_rate
                if self.debug:
                    print(f"[TTS] ✅ Loaded model: {model_file.name} (sample rate: {self.sample_rate})")
                return True
            except Exception as e:
                if self.debug:
                    print(f"[TTS] ❌ Failed to load model: {e}")
                return False
        else:
            # Subprocess mode - just verify files exist
            if self.debug:
                print(f"[TTS] ✅ Configured model: {model_file.name} (subprocess mode)")
            return True
    
    def unload_model(self):
        """Unload current model to free memory"""
        self.model = None
        self.model_path = None
        self.config = None
    
    def synthesize_to_file(self, text: str, output_path: str) -> bool:
        """Synthesize text to a WAV file"""
        if not self.available or not self.model_path:
            return False
        
        try:
            if self.mode == 'library' and self.model:
                import wave
                # Piper handles wave parameters internally
                wav_file = wave.open(output_path, 'w')
                self.model.synthesize(text, wav_file)
                wav_file.close()
                return True
            else:
                # Subprocess mode
                cmd = [
                    'piper',
                    '--model', self.model_path,
                    '--output_file', output_path
                ]
                if self.config and self.config.config_path:
                    cmd.extend(['--config', self.config.config_path])
                
                result = subprocess.run(
                    cmd,
                    input=text,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                return result.returncode == 0
        except Exception as e:
            if self.debug:
                print(f"[TTS] ❌ Synthesis error: {e}")
            return False
    
    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        """Generate audio stream (yields raw audio bytes)"""
        if not self.available or not self.model_path:
            return
        
        if self.mode == 'library' and self.model:
            try:
                # Try streaming API first (newer versions)
                if hasattr(self.model, 'synthesize_stream_raw'):
                    for audio_bytes in self.model.synthesize_stream_raw(text):
                        if self._stop_flag:
                            break
                        yield audio_bytes
                else:
                    # Fallback: synthesize to temp file and read back
                    import tempfile
                    import wave
                    import os
                    
                    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_path = temp_wav.name
                    temp_wav.close()
                    
                    try:
                        # Synthesize to temp file - Piper handles wave params internally
                        wav_file = wave.open(temp_path, 'w')
                        self.model.synthesize(text, wav_file)
                        wav_file.close()
                        
                        # Read back raw audio frames
                        with wave.open(temp_path, 'rb') as wav_file:
                            chunk_size = 4096
                            while True:
                                if self._stop_flag:
                                    break
                                chunk = wav_file.readframes(chunk_size // 2)  # 2 bytes per frame
                                if not chunk:
                                    break
                                yield chunk
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            except Exception as e:
                if self.debug:
                    print(f"[TTS] ❌ Stream error: {e}")
        else:
            # Subprocess mode - pipe to raw output
            try:
                cmd = [
                    'piper',
                    '--model', self.model_path,
                    '--output-raw'
                ]
                if self.config and self.config.config_path:
                    cmd.extend(['--config', self.config.config_path])
                
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Write text and close stdin
                process.stdin.write(text.encode('utf-8'))
                process.stdin.close()
                
                # Read audio chunks
                chunk_size = 4096
                while True:
                    if self._stop_flag:
                        process.terminate()
                        break
                    chunk = process.stdout.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
                
                process.wait()
            except Exception as e:
                if self.debug:
                    print(f"[TTS] ❌ Subprocess stream error: {e}")
    
    def speak(self, text: str, blocking: bool = True) -> bool:
        """
        Speak text through audio output.
        
        Args:
            text: Text to speak
            blocking: If True, wait until speech completes
        
        Returns:
            True if speech started successfully
        """
        if self.debug:
            print(f"[TTS] 🔊 Speaking: {text[:50]}...")
        
        if not self.available or not self.model_path:
            if self.debug:
                print("[TTS] ❌ Cannot speak - model not loaded")
            return False
        
        if not SOUNDDEVICE_AVAILABLE or not NUMPY_AVAILABLE:
            if self.debug:
                print("[TTS] ❌ Cannot speak - sounddevice/numpy not available")
            return False
        
        if self._is_speaking:
            self.stop()
            time.sleep(0.1)
        
        self._stop_flag = False
        
        if blocking:
            self._speak_internal(text)
        else:
            self._speak_thread = threading.Thread(target=self._speak_internal, args=(text,))
            self._speak_thread.start()
        
        return True
    
    def _speak_internal(self, text: str):
        """Internal speaking method"""
        self._is_speaking = True
        
        if self._on_start:
            self._on_start()
        
        try:
            if self.debug:
                print(f"[TTS] 🎧 Synthesizing audio...")
            
            # Collect all audio first (more reliable than streaming for some setups)
            audio_chunks = []
            for audio_bytes in self.synthesize_stream(text):
                if self._stop_flag:
                    break
                audio_chunks.append(audio_bytes)
            
            if not audio_chunks:
                if self.debug:
                    print("[TTS] ⚠️  No audio generated")
                return
            
            # Combine all audio
            all_audio = b''.join(audio_chunks)
            audio_data = np.frombuffer(all_audio, dtype=np.int16)
            
            # Convert to float32 for playback (required by sounddevice)
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Resample to 48000Hz if needed (most devices support this)
            playback_rate = 48000
            if self.sample_rate != playback_rate:
                # Simple linear resampling
                original_length = len(audio_float)
                new_length = int(original_length * playback_rate / self.sample_rate)
                indices = np.linspace(0, original_length - 1, new_length)
                audio_float = np.interp(indices, np.arange(original_length), audio_float)
                if self.debug:
                    print(f"[TTS] 🔄 Resampled from {self.sample_rate}Hz to {playback_rate}Hz")
            
            if self.debug:
                print(f"[TTS] ▶️  Playing {len(audio_float)} samples at {playback_rate}Hz (device: {self.output_device})")
            
            # Play audio using blocking call (more reliable)
            sd.play(audio_float, playback_rate, device=self.output_device)
            sd.wait()  # Wait until audio finishes
            
            if self.debug:
                print("[TTS] ✅ Playback complete")
            
        except Exception as e:
            if self.debug:
                print(f"[TTS] ❌ Playback error: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self._is_speaking = False
            
            if self._on_stop:
                self._on_stop()
    
    def speak_async(self, text: str) -> bool:
        """Speak text asynchronously (non-blocking)"""
        return self.speak(text, blocking=False)
    
    def stop(self):
        """Stop current speech immediately"""
        self._stop_flag = True
        
        if self._stream:
            try:
                self._stream.abort()
            except:
                pass
        
        # Wait for thread to finish
        if self._speak_thread and self._speak_thread.is_alive():
            self._speak_thread.join(timeout=0.5)
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self._is_speaking
    
    def set_callbacks(self, on_start: Optional[Callable] = None, 
                     on_stop: Optional[Callable] = None,
                     on_word: Optional[Callable] = None):
        """Set callback functions for speech events"""
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_word = on_word
    
    def get_info(self) -> dict:
        """Get TTS engine info"""
        return {
            'available': self.available,
            'mode': self.mode,
            'model_loaded': self.model_path is not None,
            'model_path': self.model_path,
            'sample_rate': self.sample_rate,
            'is_speaking': self._is_speaking
        }
