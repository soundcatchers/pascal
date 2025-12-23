"""
Speech Input Manager - Vosk STT Engine Integration
Handles continuous speech recognition with Vosk for offline, real-time transcription
"""

import os
import json
import queue
import re
import threading
from typing import Optional, Callable
from pathlib import Path
from modules.audio_device import AudioDeviceManager

PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("[STT] ⚠️  Vosk not installed. Install with: pip install vosk")

try:
    from modules.vosk_postprocessor import VoskPostProcessor
    POSTPROCESSOR_AVAILABLE = True
except ImportError:
    POSTPROCESSOR_AVAILABLE = False
    print("[STT] ⚠️  Post-processor not available (missing dependencies)")

try:
    from modules.voice_ai_corrector import VoiceAICorrector
    AI_CORRECTOR_AVAILABLE = True
except ImportError:
    AI_CORRECTOR_AVAILABLE = False

try:
    from modules.homophone_fixer import HomophoneFixer
    HOMOPHONE_FIXER_AVAILABLE = True
except ImportError:
    HOMOPHONE_FIXER_AVAILABLE = False

try:
    from modules.led_controller import get_led_controller, LEDController
    LED_AVAILABLE = True
except ImportError:
    LED_AVAILABLE = False

class SpeechInputManager:
    """Manages continuous speech recognition using Vosk"""
    
    NOISE_WORDS = {'the', 'a', 'an', 'i', 'uh', 'um', 'eh', 'ah', 'oh', 'huh', 'hmm', 'and', 'but', 'or', 'it', 'is', 'to', 'in', 'of'}
    MIN_WORD_COUNT = 1
    MIN_CHAR_COUNT = 3
    
    def __init__(self, model_path: Optional[str] = None, debug_audio: bool = False, enable_postprocessing: bool = True, led_controller: Optional['LEDController'] = None):
        self.audio_manager = AudioDeviceManager(debug_audio=debug_audio)
        self.model_path = model_path or self._find_model_path()
        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None
        self.stream = None
        
        self.is_listening = False
        self._stop_requested = False
        self._paused = False  # Pause STT during TTS playback to prevent feedback loop
        self._pause_until = 0  # Timestamp when pause expires (for cooldown)
        self.audio_queue = queue.Queue()
        self.result_callback: Optional[Callable[[str, bool], None]] = None
        
        self.listen_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        
        self.enable_postprocessing = enable_postprocessing and POSTPROCESSOR_AVAILABLE
        self.postprocessor: Optional[VoskPostProcessor] = None
        
        self.enable_ai_correction = False
        self.ai_corrector = None
        
        self.enable_homophone_fixer = HOMOPHONE_FIXER_AVAILABLE
        self.homophone_fixer = None
        
        self.led_controller = led_controller
        
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
            
            if self.enable_postprocessing:
                self._init_postprocessor()
            
            if self.enable_homophone_fixer:
                self._init_homophone_fixer()
            
            if self.enable_ai_correction:
                self._init_ai_corrector()
            
            return True
            
        except Exception as e:
            print(f"[STT] ❌ Failed to load Vosk model: {e}")
            return False
    
    def _init_postprocessor(self):
        """Initialize post-processor with settings"""
        try:
            from config.settings import settings
            config = settings.get_voice_postprocessing_config()
            
            self.postprocessor = VoskPostProcessor(
                enable_spell_check=config['spell_check'],
                enable_confidence_filter=config['confidence_filter'],
                enable_punctuation=config['punctuation'],
                confidence_threshold=config['confidence_threshold'],
                spell_check_max_distance=config['spell_check_max_distance']
            )
            
            status = self.postprocessor.get_status()
            
            # Check for fast mode
            if settings.voice_fast_mode:
                print(f"[STT] ⚡ FAST MODE: Heavy post-processing disabled (~500ms saved)")
            
            print(f"[STT] Post-processing status:")
            
            if status['spell_check']['enabled']:
                if status['spell_check']['initialized']:
                    print(f"[STT]   ✅ Spell check (confidence < {config['confidence_threshold']})")
                else:
                    print(f"[STT]   ⚠️  Spell check enabled but not initialized")
            
            if status['confidence_filter']['enabled']:
                print(f"[STT]   ✅ Confidence filtering (threshold: {config['confidence_threshold']})")
            
            if status['punctuation']['enabled']:
                if status['punctuation']['initialized']:
                    print(f"[STT]   ✅ Punctuation & case restoration")
                else:
                    print(f"[STT]   ⚠️  Punctuation enabled but not initialized")
            elif settings.voice_fast_mode:
                print(f"[STT]   ⏩ Punctuation skipped (fast mode)")
            
        except Exception as e:
            print(f"[STT] ⚠️  Post-processor initialization failed: {e}")
            self.enable_postprocessing = False
            self.postprocessor = None
    
    def _init_homophone_fixer(self):
        """Initialize the fast rule-based homophone fixer"""
        try:
            from config.settings import settings
            config = settings.get_voice_postprocessing_config()
            
            if not config.get('homophone_fixer', True):
                self.enable_homophone_fixer = False
                return
            
            self.homophone_fixer = HomophoneFixer(enabled=True)
            print(f"[STT]   ✅ Homophone fixer (instant, rule-based)")
            
        except Exception as e:
            print(f"[STT] ⚠️  Homophone fixer initialization failed: {e}")
            self.enable_homophone_fixer = False
            self.homophone_fixer = None
    
    def _apply_homophone_fix(self, text: str) -> str:
        """Apply homophone fixes - instant, no AI"""
        if not self.enable_homophone_fixer or not self.homophone_fixer or not text:
            return text
        
        try:
            fixed = self.homophone_fixer.fix(text)
            if fixed != text:
                print(f"[FIX] '{text}' → '{fixed}'")
            return fixed
        except Exception as e:
            return text
    
    def _is_noise(self, text: str) -> bool:
        """Filter out noise words from Vosk (common false positives during silence)"""
        if not text:
            return True
        
        text_clean = PUNCTUATION_PATTERN.sub('', text.lower()).strip()
        words = text_clean.split()
        
        if not words:
            return True
        
        if len(words) == 1 and words[0] in self.NOISE_WORDS:
            return True
        
        if len(text_clean) < self.MIN_CHAR_COUNT:
            return True
        
        return False
    
    def _strip_noise_prefix(self, text: str) -> str:
        """Strip leading noise words like 'the' that Vosk adds from background noise"""
        if not text:
            return text
        
        words = text.split()
        if len(words) >= 2 and words[0].lower().strip('.,!?') in self.NOISE_WORDS:
            return ' '.join(words[1:])
        
        return text
    
    def _init_ai_corrector(self):
        """Initialize AI corrector for context-aware word fixing"""
        try:
            from config.settings import settings
            config = settings.get_voice_postprocessing_config()
            
            if not config.get('ai_correction', False):
                self.enable_ai_correction = False
                return
            
            model = config.get('ai_correction_model', 'gemma2:2b')
            timeout = config.get('ai_correction_timeout', 5.0)
            
            self.ai_corrector = VoiceAICorrector(
                enabled=True,
                model=model,
                timeout=timeout
            )
            
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._warmup_ai_corrector())
                else:
                    loop.run_until_complete(self._warmup_ai_corrector())
            except RuntimeError:
                asyncio.run(self._warmup_ai_corrector())
            
        except Exception as e:
            print(f"[STT] ⚠️  AI corrector initialization failed: {e}")
            self.enable_ai_correction = False
            self.ai_corrector = None
    
    async def _warmup_ai_corrector(self):
        """Warm up the AI corrector model"""
        if self.ai_corrector:
            available = await self.ai_corrector.check_available()
            if available:
                await self.ai_corrector.warmup()
                print(f"[STT]   ✅ AI correction ready ({self.ai_corrector.model})")
            else:
                print(f"[STT]   ⚠️  AI correction model not available")
                print(f"[STT]   💡 Install with: ollama pull {self.ai_corrector.model}")
                self.enable_ai_correction = False
    
    def _apply_ai_correction(self, text: str) -> str:
        """Apply AI correction synchronously"""
        if not self.enable_ai_correction or not self.ai_corrector or not text:
            return text
        
        if len(text.split()) < 3:
            return text
        
        try:
            corrected = self.ai_corrector.correct_sync(text)
            if corrected and corrected != text:
                print(f"[AI] ✅ '{text}' → '{corrected}'")
            return corrected if corrected else text
        except Exception as e:
            print(f"[AI] ⚠️ Error: {e}")
            return text
    
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
        self._stop_requested = False
        
        self.stream = self.audio_manager.open_stream()
        if not self.stream:
            print("[STT] ❌ Failed to open audio stream")
            self.is_listening = False
            return
        
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        
        self.listen_thread.start()
        self.process_thread.start()
        
        if self.led_controller:
            self.led_controller.listening()
        
        print("[STT] 🎙️  Listening started (continuous mode)...")
    
    def _listen_loop(self):
        """Continuous audio capture loop (runs in separate thread)"""
        settings = self.audio_manager.get_recommended_settings()
        chunk_size = settings['chunk']
        
        while self.is_listening and self.stream and not self._stop_requested:
            try:
                data = self.stream.read(chunk_size, exception_on_overflow=False)
                if not self._stop_requested:
                    self.audio_queue.put(data)
            except Exception as e:
                if self.is_listening and not self._stop_requested:
                    print(f"[STT] Audio read error: {e}")
                break
    
    def _process_loop(self):
        """Process audio data with Vosk (runs in separate thread)"""
        import time
        while self.is_listening and not self._stop_requested:
            try:
                data = self.audio_queue.get(timeout=0.5)
                
                if self._stop_requested:
                    break
                
                # Skip processing while paused (during TTS playback) or in cooldown
                if self._paused or (self._pause_until > 0 and time.time() < self._pause_until):
                    continue
                
                if self.recognizer.AcceptWaveform(data):
                    result_json = self.recognizer.Result()
                    
                    if self.enable_postprocessing and self.postprocessor:
                        text = self.postprocessor.process(result_json).strip()
                    else:
                        result = json.loads(result_json)
                        text = result.get('text', '').strip()
                    
                    text = self._strip_noise_prefix(text)
                    
                    if self._is_noise(text):
                        continue
                    
                    if text:
                        text = self._apply_homophone_fix(text)
                        if self.enable_ai_correction:
                            text = self._apply_ai_correction(text)
                    
                    if text and self.result_callback and not self._stop_requested:
                        self.result_callback(text, is_final=True)
                else:
                    partial_json = self.recognizer.PartialResult()
                    partial = json.loads(partial_json)
                    text = partial.get('partial', '').strip()
                    
                    text = self._strip_noise_prefix(text)
                    
                    if self._is_noise(text):
                        continue
                    
                    if self.enable_postprocessing and self.postprocessor and text:
                        text = self.postprocessor.process_simple(text).strip()
                    
                    if text and self.result_callback and not self._stop_requested:
                        self.result_callback(text, is_final=False)
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_listening and not self._stop_requested:
                    print(f"[STT] Processing error: {e}")
                    # Don't break - try to continue listening after recoverable errors
                    continue
    
    def pause(self, cooldown_ms: int = 0):
        """Pause STT processing (use during TTS playback to prevent feedback loop)
        
        Args:
            cooldown_ms: Additional cooldown in milliseconds after resume() is called
        """
        self._paused = True
        self._cooldown_duration = cooldown_ms / 1000.0  # Convert to seconds
        # Do NOT call recognizer.Reset() here - it causes Vosk state errors
        # Just drain the audio queue to discard buffered audio
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def resume(self):
        """Resume STT processing after TTS playback completes"""
        import time
        if hasattr(self, '_cooldown_duration') and self._cooldown_duration > 0:
            self._pause_until = time.time() + self._cooldown_duration
        else:
            self._pause_until = time.time() + 0.5  # Default 500ms cooldown
        # Drain any audio captured during pause
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self._paused = False
    
    def is_paused(self) -> bool:
        """Check if STT is currently paused"""
        import time
        # Check explicit pause OR cooldown period
        if self._paused:
            return True
        if self._pause_until > 0 and time.time() < self._pause_until:
            return True
        return False
    
    def stop_listening(self):
        """Stop continuous speech recognition"""
        if not self.is_listening:
            return
        
        print("[STT] 🔇 Stopping listening...")
        
        # Signal threads to stop FIRST
        self._stop_requested = True
        self.is_listening = False
        
        # Give threads time to see the stop flag and exit their loops
        import time
        time.sleep(0.1)
        
        # Wait for threads to finish BEFORE closing stream
        # This prevents ALSA crashes from closing stream while threads are using it
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2)
        
        # NOW safe to close the stream (threads are done)
        if self.stream:
            try:
                self.stream.stop_stream()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            finally:
                self.stream = None
        
        # Clear any remaining audio data
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.led_controller:
            self.led_controller.idle()
        
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
        
        if self.ai_corrector:
            import asyncio
            try:
                asyncio.run(self.ai_corrector.close())
            except Exception:
                pass
            self.ai_corrector = None
        
        if self.audio_manager:
            self.audio_manager.close()
        
        print("[STT] ✅ Speech input closed")
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.close()
