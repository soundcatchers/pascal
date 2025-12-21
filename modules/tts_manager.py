"""
TTS Manager for Pascal AI Assistant

Orchestrates text-to-speech output with personality-aware voice switching,
LED feedback, and interruptible streaming audio.
"""

import asyncio
import threading
from typing import Optional, Callable
from pathlib import Path

from config.settings import settings


class TTSManager:
    """
    Main TTS orchestrator for Pascal.
    
    Features:
    - Personality-aware voice switching
    - Interruptible streaming audio
    - LED feedback coordination
    - Async-friendly interface
    """
    
    def __init__(self, 
                 voices_dir: str = "config/tts_voices",
                 led_controller = None,
                 debug: bool = False):
        self.debug = debug or settings.debug_mode
        self.voices_dir = Path(voices_dir)
        self.led_controller = led_controller
        
        # State
        self.enabled = True
        self.current_personality = "default"
        self._is_speaking = False
        
        # Components (lazy loaded)
        self._piper: Optional['PiperTTS'] = None
        self._voice_manager: Optional['VoiceManager'] = None
        self._initialized = False
        
        # Callbacks
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None
        
    async def initialize(self) -> bool:
        """Initialize TTS system"""
        if self._initialized:
            return True
        
        try:
            # Initialize voice manager
            from modules.tts_voices import VoiceManager
            self._voice_manager = VoiceManager(str(self.voices_dir))
            
            # Initialize Piper TTS
            from modules.tts_piper import PiperTTS
            self._piper = PiperTTS(debug=self.debug)
            
            if not self._piper.available:
                if self.debug:
                    print("[TTS] ⚠️  Piper not available - TTS disabled")
                self.enabled = False
                return False
            
            # Set up callbacks for LED feedback
            self._piper.set_callbacks(
                on_start=self._handle_speech_start,
                on_stop=self._handle_speech_stop
            )
            
            # Load default voice
            if not await self.set_voice(self.current_personality):
                # Try fallback to any available voice
                available = self._voice_manager.get_available_voices()
                for personality, is_available in available.items():
                    if is_available:
                        if await self.set_voice(personality):
                            break
            
            self._initialized = True
            
            if self.debug:
                info = self._piper.get_info()
                print(f"[TTS] ✅ Initialized ({info['mode']} mode)")
                
            return True
            
        except Exception as e:
            if self.debug:
                print(f"[TTS] ❌ Initialization error: {e}")
            self.enabled = False
            return False
    
    async def set_voice(self, personality: str) -> bool:
        """Switch to a different personality's voice"""
        if not self._voice_manager or not self._piper:
            return False
        
        # Check if voice is available
        if not self._voice_manager.is_voice_available(personality):
            if self.debug:
                print(f"[TTS] ⚠️  Voice not available for: {personality}")
            
            # Try default
            if personality != "default" and self._voice_manager.is_voice_available("default"):
                return await self.set_voice("default")
            return False
        
        # Get model path
        model_path = self._voice_manager.get_model_path(personality)
        if not model_path:
            return False
        
        # Load model
        voice_profile = self._voice_manager.get_voice(personality)
        if self._piper.load_model(model_path):
            self.current_personality = personality
            if self.debug:
                print(f"[TTS] 🎤 Voice set to: {personality} ({voice_profile.model_name})")
            return True
        
        return False
    
    async def speak(self, text: str, blocking: bool = True) -> bool:
        """
        Speak text using current personality's voice.
        
        Args:
            text: Text to speak
            blocking: If True, wait until speech completes
            
        Returns:
            True if speech started/completed successfully
        """
        if not self.enabled or not self._piper:
            return False
        
        if not self._initialized:
            await self.initialize()
        
        if not text or not text.strip():
            return False
        
        # Clean text for speech
        clean_text = self._prepare_text_for_speech(text)
        
        if blocking:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self._piper.speak(clean_text, blocking=True)
            )
        else:
            return self._piper.speak_async(clean_text)
    
    async def speak_streaming(self, text: str) -> bool:
        """
        Speak text with streaming output (interruptible).
        Preferred method for long responses.
        """
        return await self.speak(text, blocking=False)
    
    def stop(self):
        """Stop current speech immediately"""
        if self._piper:
            self._piper.stop()
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        if self._piper:
            return self._piper.is_speaking()
        return False
    
    def _prepare_text_for_speech(self, text: str) -> str:
        """Clean and prepare text for TTS - removes emojis, normalizes ranges, etc."""
        import re
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]*`', '', text)
        
        # Remove markdown links, keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove citation references like [1], [2]
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove bold/italic markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove bullet points
        text = re.sub(r'^[\s]*[-*•]\s*', '', text, flags=re.MULTILINE)
        
        # Remove numbered lists at start of lines
        text = re.sub(r'^[\s]*\d+\.\s*', '', text, flags=re.MULTILINE)
        
        # ===== EMOJI REMOVAL =====
        # Remove all emoji characters (comprehensive pattern)
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251"  # Enclosed characters
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)
        
        # ===== TEXT NORMALIZATION FOR NATURAL SPEECH =====
        # Convert "1-2 hours" to "1 to 2 hours" (number ranges)
        text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 to \2', text)
        
        # Convert "55-60 minutes" style ranges
        text = re.sub(r'(\d+)\s*–\s*(\d+)', r'\1 to \2', text)  # en-dash
        text = re.sub(r'(\d+)\s*—\s*(\d+)', r'\1 to \2', text)  # em-dash
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove parenthetical kilometer conversions like "(90 kilometers)"
        # Keep the main info, remove redundant conversions
        text = re.sub(r'\s*\(\d+\s*(?:km|kilometers?|kilometres?)\)', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _handle_speech_start(self):
        """Called when speech starts"""
        self._is_speaking = True
        
        # LED feedback
        if self.led_controller:
            try:
                self.led_controller.set_mode('speaking')
            except:
                pass
        
        if self._on_speech_start:
            self._on_speech_start()
    
    def _handle_speech_stop(self):
        """Called when speech stops"""
        self._is_speaking = False
        
        # LED feedback
        if self.led_controller:
            try:
                self.led_controller.set_mode('idle')
            except:
                pass
        
        if self._on_speech_end:
            self._on_speech_end()
    
    def set_callbacks(self, on_start: Optional[Callable] = None,
                     on_end: Optional[Callable] = None):
        """Set callbacks for speech events"""
        self._on_speech_start = on_start
        self._on_speech_end = on_end
    
    def get_available_voices(self) -> dict:
        """Get availability status of all configured voices"""
        if self._voice_manager:
            return self._voice_manager.get_available_voices()
        return {}
    
    def get_status(self) -> dict:
        """Get TTS system status"""
        status = {
            'enabled': self.enabled,
            'initialized': self._initialized,
            'current_personality': self.current_personality,
            'is_speaking': self.is_speaking(),
            'available_voices': self.get_available_voices()
        }
        
        if self._piper:
            status['piper'] = self._piper.get_info()
        
        return status
    
    async def close(self):
        """Clean up resources"""
        if self._piper:
            self._piper.stop()
            self._piper.unload_model()
        
        self._initialized = False
        
        if self.debug:
            print("[TTS] 👋 Shutdown complete")


# Convenience function for quick TTS
async def speak_text(text: str, personality: str = "default") -> bool:
    """
    Quick helper to speak text without managing a TTSManager instance.
    
    Note: Creates a new manager each call - use TTSManager directly for efficiency.
    """
    manager = TTSManager()
    try:
        await manager.initialize()
        await manager.set_voice(personality)
        return await manager.speak(text)
    finally:
        await manager.close()
