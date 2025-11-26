"""
Speech Input Manager - Vosk STT Engine Integration
Handles continuous speech recognition with Vosk for offline, real-time transcription
"""

import os
import json
import queue
import threading
from typing import Optional, Callable
from pathlib import Path
from modules.audio_device import AudioDeviceManager

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("[STT] ⚠️  Vosk not installed. Install with: pip install vosk")

class SpeechInputManager:
    """Manages continuous speech recognition using Vosk"""
    
    def __init__(self, model_path: Optional[str] = None, debug_audio: bool = False):
        self.audio_manager = AudioDeviceManager(debug_audio=debug_audio)
        self.model_path = model_path or self._find_model_path()
        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None
        self.stream = None
        
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.result_callback: Optional[Callable[[str, bool], None]] = None
        
        self.listen_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        
    def _find_model_path(self) -> Optional[str]:
        """Find Vosk model in common locations"""
        possible_paths = [
            'config/vosk_models/vosk-model-en-us-0.22',  # Primary: 1.8GB model (85% accuracy)
            'config/vosk_models/vosk-model-small-en-us-0.15',  # Fallback: old 50MB model
            'vosk-model-en-us-0.22',
            'vosk-model-small-en-us-0.15',
            '/usr/share/vosk/models/vosk-model-en-us-0.22',
            os.path.expanduser('~/vosk-model-en-us-0.22'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def initialize(self) -> bool:
        """Initialize audio device and Vosk model"""
        if not VOSK_AVAILABLE:
            print("[STT] ❌ Vosk not available. Please install: pip install vosk")
            return False
        
        print("[STT] Initializing speech recognition...")
        
        if not self.audio_manager.initialize():
            print("[STT] ❌ Failed to initialize audio device")
            return False
        
        device = self.audio_manager.get_input_device()
        if not device:
            print("[STT] ❌ No audio input device found")
            return False
        
        print(f"[STT] Using audio device: {device.name}")
        
        if not self.model_path:
            print("[STT] ❌ Vosk model not found!")
            print("[STT] 💡 Download with: ./setup_vosk.sh")
            print("[STT] 💡 Or manually:")
            print("[STT]    wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip")
            print("[STT]    unzip vosk-model-en-us-0.22.zip")
            print("[STT]    mkdir -p config/vosk_models")
            print("[STT]    mv vosk-model-en-us-0.22 config/vosk_models/")
            return False
        
        if not os.path.exists(self.model_path):
            print(f"[STT] ❌ Model path does not exist: {self.model_path}")
            return False
        
        print(f"[STT] Loading Vosk model from: {self.model_path}")
        
        try:
            self.model = Model(self.model_path)
            
            settings = self.audio_manager.get_recommended_settings()
            sample_rate = settings['rate']
            
            self.recognizer = KaldiRecognizer(self.model, sample_rate)
            self.recognizer.SetWords(True)
            
            print(f"[STT] ✅ Vosk initialized (sample rate: {sample_rate}Hz)")
            return True
            
        except Exception as e:
            print(f"[STT] ❌ Failed to load Vosk model: {e}")
            return False
    
    def start_listening(self, callback: Callable[[str, bool], None]):
        """Start continuous speech recognition
        
        Args:
            callback: Function called with (text, is_final) when speech is recognized
                     is_final=True for complete utterances, False for partial results
        """
        if self.is_listening:
            print("[STT] Already listening")
            return
        
        if not self.recognizer:
            print("[STT] ❌ STT not initialized. Call initialize() first.")
            return
        
        self.result_callback = callback
        self.is_listening = True
        
        self.stream = self.audio_manager.open_stream()
        if not self.stream:
            print("[STT] ❌ Failed to open audio stream")
            self.is_listening = False
            return
        
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        
        self.listen_thread.start()
        self.process_thread.start()
        
        print("[STT] 🎙️  Listening started (continuous mode)...")
    
    def _listen_loop(self):
        """Continuous audio capture loop (runs in separate thread)"""
        settings = self.audio_manager.get_recommended_settings()
        chunk_size = settings['chunk']
        
        while self.is_listening and self.stream:
            try:
                data = self.stream.read(chunk_size, exception_on_overflow=False)
                self.audio_queue.put(data)
            except Exception as e:
                if self.is_listening:
                    print(f"[STT] Audio read error: {e}")
                break
    
    def _process_loop(self):
        """Process audio data with Vosk (runs in separate thread)"""
        while self.is_listening:
            try:
                data = self.audio_queue.get(timeout=1)
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    
                    if text and self.result_callback:
                        self.result_callback(text, is_final=True)
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    text = partial.get('partial', '').strip()
                    
                    if text and self.result_callback:
                        self.result_callback(text, is_final=False)
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_listening:
                    print(f"[STT] Processing error: {e}")
                break
    
    def stop_listening(self):
        """Stop continuous speech recognition"""
        if not self.is_listening:
            return
        
        print("[STT] 🔇 Stopping listening...")
        self.is_listening = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"[STT] Error closing stream: {e}")
            finally:
                self.stream = None
        
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2)
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        print("[STT] ✅ Listening stopped")
    
    def get_device_info(self) -> dict:
        """Get information about the current audio device"""
        device = self.audio_manager.get_input_device()
        
        if not device:
            return {'available': False}
        
        return {
            'available': True,
            'name': device.name,
            'is_respeaker': device.is_respeaker,
            'channels': device.channels,
            'sample_rate': device.sample_rate,
        }
    
    def list_devices(self):
        """List all available audio input devices"""
        devices = self.audio_manager.list_devices()
        
        print("\n🎤 Available Audio Input Devices:")
        print("-" * 60)
        
        for device in devices:
            marker = "✅ ReSpeaker" if device.is_respeaker else "  "
            print(f"{marker} [{device.index}] {device.name}")
            print(f"     Channels: {device.channels}, Sample Rate: {device.sample_rate}Hz")
        
        print("-" * 60)
        
        if not devices:
            print("No audio input devices found.")
        
        return devices
    
    def close(self):
        """Clean up resources"""
        self.stop_listening()
        
        self.model = None
        self.recognizer = None
        
        if self.audio_manager:
            self.audio_manager.close()
        
        print("[STT] ✅ Speech input closed")
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.close()
