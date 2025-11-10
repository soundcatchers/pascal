"""
Audio Device Manager - ReSpeaker USB Mic Array Detection and Configuration
Handles PyAudio setup, device detection, and audio stream management for Pascal voice input
"""

import pyaudio
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

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
    
    def __init__(self):
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.respeaker_device: Optional[AudioDeviceInfo] = None
        self.default_device: Optional[AudioDeviceInfo] = None
        
    def initialize(self) -> bool:
        """Initialize PyAudio and detect devices"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self._detect_devices()
            return True
        except Exception as e:
            print(f"[AUDIO] Failed to initialize PyAudio: {e}")
            return False
    
    def _detect_devices(self):
        """Detect all available audio input devices"""
        if not self.pyaudio_instance:
            return
        
        try:
            host_api_info = self.pyaudio_instance.get_host_api_info_by_index(0)
            num_devices = host_api_info.get('deviceCount', 0)
            
            for i in range(num_devices):
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_host_api_device_index(0, i)
                    
                    if device_info.get('maxInputChannels', 0) > 0:
                        device_name = device_info.get('name', '')
                        channels = device_info.get('maxInputChannels', 0)
                        sample_rate = int(device_info.get('defaultSampleRate', 16000))
                        
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
                        
                        if self.default_device is None:
                            self.default_device = device
                            
                except Exception as e:
                    continue
            
            if self.respeaker_device is None and self.default_device:
                print(f"[AUDIO] ⚠️  ReSpeaker not found, using default: {self.default_device.name}")
                
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
        """List all available input devices"""
        devices = []
        
        if not self.pyaudio_instance:
            return devices
        
        try:
            host_api_info = self.pyaudio_instance.get_host_api_info_by_index(0)
            num_devices = host_api_info.get('deviceCount', 0)
            
            for i in range(num_devices):
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_host_api_device_index(0, i)
                    
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
        """Open an audio input stream with recommended settings"""
        device = self.get_input_device()
        
        if not device or not self.pyaudio_instance:
            print("[AUDIO] No input device available")
            return None
        
        settings = self.get_recommended_settings()
        
        try:
            stream = self.pyaudio_instance.open(
                format=settings['format'],
                channels=settings['channels'],
                rate=settings['rate'],
                input=True,
                input_device_index=device.index,
                frames_per_buffer=settings['chunk'],
                stream_callback=callback
            )
            
            print(f"[AUDIO] ✅ Stream opened: {device.name} @ {settings['rate']}Hz, {settings['channels']}ch")
            return stream
            
        except Exception as e:
            print(f"[AUDIO] Failed to open stream: {e}")
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
