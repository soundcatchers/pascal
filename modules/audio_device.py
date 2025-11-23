"""
Audio Device Manager - ReSpeaker USB Mic Array Detection and Configuration
Handles PyAudio setup, device detection, and audio stream management for Pascal voice input
"""

import pyaudio
import os
import sys
from contextlib import contextmanager
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

try:
    from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    
    def py_error_handler(filename, line, function, err, fmt):
        """No-op error handler for ALSA"""
        pass
    
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    
    try:
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
        ALSA_HANDLER_AVAILABLE = True
    except Exception:
        ALSA_HANDLER_AVAILABLE = False
except Exception:
    ALSA_HANDLER_AVAILABLE = False

@contextmanager
def suppress_alsa_errors(debug_audio: bool = False):
    """Suppress ALSA error messages during PyAudio operations
    
    Args:
        debug_audio: If True, allow ALSA errors to be displayed (for troubleshooting)
    """
    if debug_audio or not ALSA_HANDLER_AVAILABLE:
        yield
        return
    
    try:
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
    except Exception:
        pass

@dataclass
class AudioDeviceInfo:
    """Information about an audio input device"""
    index: int
    name: str
    channels: int
    sample_rate: int
    is_respeaker: bool

class AudioDeviceManager:
    """Manages audio device detection and configuration for voice input"""
    
    def __init__(self, debug_audio: bool = False):
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.respeaker_device: Optional[AudioDeviceInfo] = None
        self.default_device: Optional[AudioDeviceInfo] = None
        self.debug_audio = debug_audio or os.environ.get('PASCAL_DEBUG_AUDIO', '0') == '1'
        
    def initialize(self) -> bool:
        """Initialize PyAudio and detect devices (with ALSA error suppression)"""
        try:
            with suppress_alsa_errors(self.debug_audio):
                self.pyaudio_instance = pyaudio.PyAudio()
                self._detect_devices()
            return True
        except Exception as e:
            print(f"[AUDIO] Failed to initialize PyAudio: {e}")
            return False
    
    def _detect_devices(self):
        """Detect all available audio input devices across ALL host APIs"""
        if not self.pyaudio_instance:
            return
        
        try:
            num_devices = self.pyaudio_instance.get_device_count()
            
            if self.debug_audio:
                print(f"[AUDIO_DEBUG] Scanning {num_devices} total devices across all host APIs...")
            
            for i in range(num_devices):
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_index(i)
                    
                    if device_info.get('maxInputChannels', 0) > 0:
                        device_name = device_info.get('name', '')
                        channels = device_info.get('maxInputChannels', 0)
                        sample_rate = int(device_info.get('defaultSampleRate', 16000))
                        
                        if self.debug_audio:
                            print(f"[AUDIO_DEBUG] Device {i}: {device_name} (channels: {channels}, rate: {sample_rate})")
                        
                        is_respeaker = self._is_respeaker_device(device_name)
                        
                        device = AudioDeviceInfo(
                            index=i,
                            name=device_name,
                            channels=channels,
                            sample_rate=sample_rate,
                            is_respeaker=is_respeaker
                        )
                        
                        if is_respeaker:
                            self.respeaker_device = device
                            print(f"[AUDIO] ✅ Found ReSpeaker: {device_name} (index: {i}, channels: {channels})")
                        
                        # Set first REAL (non-virtual) device as default
                        if self.default_device is None:
                            self.default_device = device
                            
                except Exception as e:
                    if self.debug_audio:
                        print(f"[AUDIO_DEBUG] Error reading device {i}: {e}")
                    continue
            
            if self.respeaker_device is None and self.default_device:
                print(f"[AUDIO] ⚠️  ReSpeaker not found, using default: {self.default_device.name}")
            elif self.respeaker_device is None and self.default_device is None:
                print(f"[AUDIO] ❌ No usable audio input devices found (only virtual devices detected)")
                
        except Exception as e:
            print(f"[AUDIO] Error detecting devices: {e}")
    
    def _is_respeaker_device(self, device_name: str) -> bool:
        """Check if device is a ReSpeaker USB Mic Array"""
        respeaker_identifiers = [
            'respeaker',
            'usb mic array',
            '4 mic array',
            'xmos',
        ]
        
        device_name_lower = device_name.lower()
        return any(identifier in device_name_lower for identifier in respeaker_identifiers)
    
    def get_input_device(self) -> Optional[AudioDeviceInfo]:
        """Get the best available input device (ReSpeaker preferred)"""
        if self.respeaker_device:
            return self.respeaker_device
        return self.default_device
    
    def list_devices(self) -> List[AudioDeviceInfo]:
        """List all available input devices across ALL host APIs"""
        devices = []
        
        if not self.pyaudio_instance:
            return devices
        
        try:
            num_devices = self.pyaudio_instance.get_device_count()
            
            for i in range(num_devices):
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_index(i)
                    
                    if device_info.get('maxInputChannels', 0) > 0:
                        device = AudioDeviceInfo(
                            index=i,
                            name=device_info.get('name', ''),
                            channels=device_info.get('maxInputChannels', 0),
                            sample_rate=int(device_info.get('defaultSampleRate', 16000)),
                            is_respeaker=self._is_respeaker_device(device_info.get('name', ''))
                        )
                        devices.append(device)
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"[AUDIO] Error listing devices: {e}")
        
        return devices
    
    def get_recommended_settings(self) -> Dict[str, int]:
        """Get recommended audio settings for the current device"""
        device = self.get_input_device()
        
        if not device:
            return {
                'rate': 16000,
                'channels': 1,
                'chunk': 4096,
                'format': pyaudio.paInt16
            }
        
        if device.is_respeaker:
            return {
                'rate': 16000,
                'channels': 1,
                'chunk': 4096,
                'format': pyaudio.paInt16
            }
        else:
            return {
                'rate': min(device.sample_rate, 16000),
                'channels': min(device.channels, 2),
                'chunk': 4096,
                'format': pyaudio.paInt16
            }
    
    def open_stream(self, callback=None) -> Optional[pyaudio.Stream]:
        """Open an audio input stream with recommended settings (with ALSA error suppression)"""
        device = self.get_input_device()
        
        if not device or not self.pyaudio_instance:
            print("[AUDIO] No input device available")
            return None
        
        settings = self.get_recommended_settings()
        stream = None
        
        try:
            with suppress_alsa_errors(self.debug_audio):
                stream = self.pyaudio_instance.open(
                    format=settings['format'],
                    channels=settings['channels'],
                    rate=settings['rate'],
                    input=True,
                    input_device_index=device.index,
                    frames_per_buffer=settings['chunk'],
                    stream_callback=callback
                )
            
            if stream is not None:
                print(f"[AUDIO] ✅ Stream opened: {device.name} @ {settings['rate']}Hz, {settings['channels']}ch")
                return stream
            else:
                raise Exception("PyAudio stream creation failed (returned None)")
            
        except Exception as e:
            error_msg = str(e)
            
            if "REPLIT" in os.environ or not os.path.exists("/dev/snd"):
                print(f"[AUDIO] ❌ No physical audio hardware detected")
                print(f"[AUDIO] 💡 Voice mode requires real microphone hardware (ReSpeaker on Raspberry Pi 5)")
                print(f"[AUDIO] 💡 This feature will work when you run on your Pi 5 with ReSpeaker")
            else:
                print(f"[AUDIO] ❌ Failed to open stream: {error_msg}")
                print(f"[AUDIO] 💡 Try: Device '{device.name}' may be in use by another app")
                print(f"[AUDIO] 💡 Try: Check 'lsof /dev/snd/*' to see what's using the device")
            
            return None
    
    def close(self):
        """Clean up PyAudio resources"""
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
                print("[AUDIO] ✅ PyAudio terminated")
            except Exception as e:
                print(f"[AUDIO] Error terminating PyAudio: {e}")
            finally:
                self.pyaudio_instance = None
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.close()
