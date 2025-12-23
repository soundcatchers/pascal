"""
Piper TTS Implementation for Pascal AI Assistant

Handles the Piper text-to-speech engine with streaming audio output.
Supports interruption, multiple voice models, and real-time streaming playback.
"""

import asyncio
import subprocess
import threading
import queue
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
                ['piper', '--help'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # --help returns 0, piper exists if we get here without FileNotFoundError
            self.mode = 'subprocess'
            self.available = True
            if self.debug:
                print("[TTS] ✅ Piper subprocess mode available")
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
    
    def speak_streaming(self, text: str, blocking: bool = True) -> bool:
        """
        Speak text with true streaming - start playback as synthesis begins.
        
        This reduces perceived latency by starting audio output immediately
        instead of waiting for full synthesis to complete.
        
        Args:
            text: Text to speak
            blocking: If True, wait until speech completes
        
        Returns:
            True if speech started successfully
        """
        if self.debug:
            print(f"[TTS] 🔊 Streaming: {text[:50]}...")
        
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
            self._speak_streaming_internal(text)
        else:
            self._speak_thread = threading.Thread(target=self._speak_streaming_internal, args=(text,))
            self._speak_thread.start()
        
        return True
    
    def _speak_streaming_internal(self, text: str):
        """
        Internal streaming speak method using queue-based audio playback.
        
        Starts audio output as soon as first chunk is ready, continues
        synthesizing and playing in parallel.
        """
        self._is_speaking = True
        
        if self._on_start:
            self._on_start()
        
        playback_rate = 48000
        resample_ratio = playback_rate / self.sample_rate
        
        # Audio buffer queue for producer-consumer pattern
        audio_queue = queue.Queue(maxsize=50)  # Buffer up to 50 chunks
        synthesis_done = threading.Event()
        playback_started = threading.Event()
        first_chunk_time = [None]  # Track latency
        
        def synthesize_producer():
            """Producer: synthesize audio and push to queue"""
            try:
                for audio_bytes in self.synthesize_stream(text):
                    if self._stop_flag:
                        break
                    
                    if first_chunk_time[0] is None:
                        first_chunk_time[0] = time.time()
                        if self.debug:
                            print(f"[TTS] ⚡ First audio chunk ready")
                    
                    # Convert and resample this chunk immediately
                    chunk_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    chunk_float = chunk_data.astype(np.float32) / 32768.0
                    
                    # Resample chunk
                    if self.sample_rate != playback_rate:
                        original_len = len(chunk_float)
                        new_len = int(original_len * resample_ratio)
                        if new_len > 0:
                            indices = np.linspace(0, original_len - 1, new_len)
                            chunk_float = np.interp(indices, np.arange(original_len), chunk_float)
                    
                    # Put chunk in queue (blocks if queue is full)
                    try:
                        audio_queue.put(chunk_float, timeout=5.0)
                    except queue.Full:
                        if self.debug:
                            print("[TTS] ⚠️  Audio queue full, dropping chunk")
            except Exception as e:
                if self.debug:
                    print(f"[TTS] ❌ Synthesis error: {e}")
            finally:
                synthesis_done.set()
        
        def audio_consumer():
            """Consumer: play audio chunks as they arrive"""
            try:
                # Wait for first chunk with timeout
                first_chunk = None
                try:
                    first_chunk = audio_queue.get(timeout=10.0)
                except queue.Empty:
                    if self.debug:
                        print("[TTS] ⚠️  No audio chunks received")
                    return
                
                if first_chunk is None or len(first_chunk) == 0:
                    return
                
                # Collect initial buffer for smooth playback (about 100ms worth)
                buffer_chunks = [first_chunk]
                buffer_samples = len(first_chunk)
                min_buffer = int(playback_rate * 0.1)  # 100ms buffer
                
                while buffer_samples < min_buffer and not synthesis_done.is_set():
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                        buffer_chunks.append(chunk)
                        buffer_samples += len(chunk)
                    except queue.Empty:
                        break
                
                playback_started.set()
                
                if self.debug:
                    print(f"[TTS] ▶️  Starting playback (buffered {buffer_samples} samples)")
                
                # Start streaming playback
                stream = sd.OutputStream(
                    samplerate=playback_rate,
                    channels=1,
                    dtype=np.float32,
                    device=self.output_device,
                    blocksize=2048
                )
                
                with stream:
                    # Play initial buffer
                    for chunk in buffer_chunks:
                        if self._stop_flag:
                            return
                        stream.write(chunk.reshape(-1, 1))
                    
                    # Continue playing as chunks arrive
                    while not self._stop_flag:
                        try:
                            chunk = audio_queue.get(timeout=0.2)
                            stream.write(chunk.reshape(-1, 1))
                        except queue.Empty:
                            if synthesis_done.is_set():
                                break
                    
                    # Drain any remaining chunks
                    while not audio_queue.empty():
                        try:
                            chunk = audio_queue.get_nowait()
                            stream.write(chunk.reshape(-1, 1))
                        except queue.Empty:
                            break
                
                if self.debug:
                    print("[TTS] ✅ Streaming playback complete")
                    
            except Exception as e:
                if self.debug:
                    print(f"[TTS] ❌ Playback error: {e}")
                    import traceback
                    traceback.print_exc()
        
        try:
            if self.debug:
                start_time = time.time()
                print(f"[TTS] 🎧 Starting streaming synthesis...")
            
            # Start producer and consumer threads
            producer_thread = threading.Thread(target=synthesize_producer)
            consumer_thread = threading.Thread(target=audio_consumer)
            
            producer_thread.start()
            consumer_thread.start()
            
            # Wait for both to complete
            producer_thread.join()
            consumer_thread.join()
            
            if self.debug and first_chunk_time[0]:
                latency = first_chunk_time[0] - start_time
                print(f"[TTS] ⏱️  First audio latency: {latency*1000:.0f}ms")
                
        except Exception as e:
            if self.debug:
                print(f"[TTS] ❌ Streaming error: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self._is_speaking = False
            
            if self._on_stop:
                self._on_stop()
    
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
