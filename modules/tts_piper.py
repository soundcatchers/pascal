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
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.model = None
        self.model_path: Optional[str] = None
        self.config: Optional[PiperConfig] = None
        self.sample_rate: int = 22050
        self.available = False
        self.mode = None  # 'library' or 'subprocess'
        
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
    
    def _detect_mode(self):
        """Detect which mode to use (library or subprocess)"""
        # Try Python library first
        try:
            from piper import PiperVoice
            self.mode = 'library'
            self.available = True
            if self.debug:
                print("[TTS] ✅ Piper library mode available")
            return
        except ImportError:
            pass
        
        # Try subprocess mode (piper binary)
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
        
        if self.debug:
            print("[TTS] ❌ Piper not available (install piper-tts or piper binary)")
        self.available = False
    
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
                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    
                    for audio_bytes in self.model.synthesize_stream_raw(text):
                        wav_file.writeframes(audio_bytes)
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
                for audio_bytes in self.model.synthesize_stream_raw(text):
                    if self._stop_flag:
                        break
                    yield audio_bytes
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
            # Open audio stream
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16'
            )
            self._stream.start()
            
            # Stream audio chunks
            for audio_bytes in self.synthesize_stream(text):
                if self._stop_flag:
                    break
                
                # Convert bytes to numpy array
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                self._stream.write(audio_data)
            
        except Exception as e:
            if self.debug:
                print(f"[TTS] ❌ Playback error: {e}")
        finally:
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except:
                    pass
                self._stream = None
            
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
