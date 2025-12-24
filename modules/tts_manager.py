"""
TTS Manager for Pascal AI Assistant

Orchestrates text-to-speech output with personality-aware voice switching,
LED feedback, and interruptible streaming audio.
"""

import asyncio
import re
import time
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
        
        # Check if streaming TTS is enabled
        use_streaming = getattr(settings, 'tts_streaming', False)
        
        if blocking:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            if use_streaming:
                # Use streaming speak for lower latency
                return await loop.run_in_executor(
                    None, 
                    lambda: self._piper.speak_streaming(clean_text, blocking=True)
                )
            else:
                # Use traditional speak (full synthesis before playback)
                return await loop.run_in_executor(
                    None, 
                    lambda: self._piper.speak(clean_text, blocking=True)
                )
        else:
            if use_streaming:
                return self._piper.speak_streaming(clean_text, blocking=False)
            else:
                return self._piper.speak_async(clean_text)
    
    async def speak_streaming(self, text: str) -> bool:
        """
        Speak text with streaming output (interruptible).
        Preferred method for long responses.
        """
        return await self.speak(text, blocking=False)
    
    def create_sentence_streamer(self):
        """
        Create a SentenceStreamer for sentence-by-sentence TTS during LLM streaming.
        
        Usage:
            streamer = tts_manager.create_sentence_streamer()
            async for chunk in llm_response:
                await streamer.add_chunk(chunk)
            await streamer.finish()
        """
        return SentenceStreamer(self)
    
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
        
        # ===== REMOVE REFERENCE SECTIONS =====
        # Remove "References:" section and everything after
        text = re.sub(r'\n*\s*References?:?\s*\n[\s\S]*$', '', text, flags=re.IGNORECASE)
        
        # Remove "Sources:" section and everything after
        text = re.sub(r'\n*\s*Sources?:?\s*\n[\s\S]*$', '', text, flags=re.IGNORECASE)
        
        # Remove "[Source: ...]" inline citations at end of text
        text = re.sub(r'\s*\[Source:?\s*[^\]]+\]\s*$', '', text, flags=re.IGNORECASE)
        
        # Remove standalone reference lines like "[1] - Wikipedia" or "[1]: NASA.gov"
        text = re.sub(r'^\s*\[\d+\]\s*[-:]\s*.*$', '', text, flags=re.MULTILINE)
        
        # Remove horizontal rules (often used before references)
        text = re.sub(r'\n\s*---+\s*\n?', '\n', text)
        text = re.sub(r'\n\s*\*\*\*+\s*\n?', '\n', text)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]*`', '', text)
        
        # Remove markdown links, keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove citation references like [1], [2], [1, 2], [1-3]
        text = re.sub(r'\s*\[\d+(?:\s*[-,]\s*\d+)*\]', '', text)
        
        # Remove any remaining square bracket content with numbers
        text = re.sub(r'\s*\[[^\]]*\d+[^\]]*\]', '', text)
        
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
        
        # Remove parenthetical conversions like "(90 kilometers)" or "(44 mi/s)"
        text = re.sub(r'\s*\([^)]*(?:km|mi|mph|kph|ft|lb|kg|oz|°[CF])[^)]*\)', '', text, flags=re.IGNORECASE)
        
        # ===== ABBREVIATION EXPANSION FOR NATURAL SPEECH =====
        # Speed/velocity units
        text = re.sub(r'(\d+)\s*km/s\b', r'\1 kilometers per second', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*km/h\b', r'\1 kilometers per hour', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*kph\b', r'\1 kilometers per hour', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*mi/s\b', r'\1 miles per second', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*mi/h\b', r'\1 miles per hour', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*mph\b', r'\1 miles per hour', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*m/s\b', r'\1 meters per second', text, flags=re.IGNORECASE)
        
        # Distance units
        text = re.sub(r'(\d+)\s*km\b', r'\1 kilometers', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*mi\b', r'\1 miles', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*m\b(?!\w)', r'\1 meters', text)  # Avoid matching "5 minutes"
        text = re.sub(r'(\d+)\s*ft\b', r'\1 feet', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*in\b', r'\1 inches', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*cm\b', r'\1 centimeters', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*mm\b', r'\1 millimeters', text, flags=re.IGNORECASE)
        
        # Weight units
        text = re.sub(r'(\d+)\s*kg\b', r'\1 kilograms', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*lb\b', r'\1 pounds', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*lbs\b', r'\1 pounds', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*oz\b', r'\1 ounces', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*g\b(?!\w)', r'\1 grams', text)  # Avoid matching words starting with g
        
        # Temperature
        text = re.sub(r'(\d+)\s*°C\b', r'\1 degrees Celsius', text)
        text = re.sub(r'(\d+)\s*°F\b', r'\1 degrees Fahrenheit', text)
        text = re.sub(r'(\d+)\s*C\b(?=\s|$|,|\.)', r'\1 degrees Celsius', text)  # "25C" context
        text = re.sub(r'(\d+)\s*F\b(?=\s|$|,|\.)', r'\1 degrees Fahrenheit', text)
        
        # Time abbreviations
        text = re.sub(r'(\d+)\s*hrs?\b', r'\1 hours', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*mins?\b', r'\1 minutes', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*secs?\b', r'\1 seconds', text, flags=re.IGNORECASE)
        
        # Common abbreviations
        text = re.sub(r'\betc\.\b', 'etcetera', text, flags=re.IGNORECASE)
        text = re.sub(r'\be\.g\.\b', 'for example', text, flags=re.IGNORECASE)
        text = re.sub(r'\bi\.e\.\b', 'that is', text, flags=re.IGNORECASE)
        text = re.sub(r'\bvs\.?\b', 'versus', text, flags=re.IGNORECASE)
        text = re.sub(r'\bapprox\.?\b', 'approximately', text, flags=re.IGNORECASE)
        text = re.sub(r'\bw/\b', 'with', text, flags=re.IGNORECASE)
        text = re.sub(r'\bw/o\b', 'without', text, flags=re.IGNORECASE)
        
        # Scientific notation (basic)
        text = re.sub(r'(\d+)\s*x\s*10\^(\d+)', r'\1 times 10 to the power of \2', text)
        
        # Percentage
        text = re.sub(r'(\d+)%', r'\1 percent', text)
        
        # Currency (basic)
        text = re.sub(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD)?', r'\1 dollars', text)
        text = re.sub(r'£(\d+(?:,\d{3})*(?:\.\d{2})?)', r'\1 pounds', text)
        text = re.sub(r'€(\d+(?:,\d{3})*(?:\.\d{2})?)', r'\1 euros', text)
        
        # ===== YEAR PRONUNCIATION =====
        # Convert years like 1976 to "nineteen seventy six" instead of "nineteen hundred..."
        # Matches standalone 4-digit years from 1000-2099
        def year_to_words(match):
            year = match.group(0)
            year_int = int(year)
            
            # Handle years 1000-1999 and 2000-2099
            if 1000 <= year_int <= 1999:
                century = year_int // 100
                decade = year_int % 100
                
                # Convert century (10-19)
                century_words = {
                    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
                    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
                    18: "eighteen", 19: "nineteen"
                }
                
                if decade == 0:
                    # 1900 -> "nineteen hundred"
                    return f"{century_words.get(century, str(century))} hundred"
                elif decade < 10:
                    # 1906 -> "nineteen oh six"
                    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
                    return f"{century_words.get(century, str(century))} oh {ones[decade]}"
                else:
                    # 1976 -> "nineteen seventy six"
                    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
                    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                           "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                           "seventeen", "eighteen", "nineteen"]
                    
                    if decade < 20:
                        return f"{century_words.get(century, str(century))} {ones[decade]}"
                    else:
                        ten_part = tens[decade // 10]
                        one_part = ones[decade % 10]
                        if one_part:
                            return f"{century_words.get(century, str(century))} {ten_part} {one_part}"
                        else:
                            return f"{century_words.get(century, str(century))} {ten_part}"
                            
            elif 2000 <= year_int <= 2009:
                # 2000-2009 -> "two thousand", "two thousand one", etc.
                ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
                if year_int == 2000:
                    return "two thousand"
                else:
                    return f"two thousand {ones[year_int - 2000]}"
                    
            elif 2010 <= year_int <= 2099:
                # 2010-2099 -> "twenty ten", "twenty twenty five", etc.
                decade = year_int % 100
                tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
                ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                       "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                       "seventeen", "eighteen", "nineteen"]
                
                if decade < 20:
                    return f"twenty {ones[decade]}"
                else:
                    ten_part = tens[decade // 10]
                    one_part = ones[decade % 10]
                    if one_part:
                        return f"twenty {ten_part} {one_part}"
                    else:
                        return f"twenty {ten_part}"
            
            return year
        
        # Match standalone 4-digit years (not part of larger numbers)
        text = re.sub(r'\b(1\d{3}|20\d{2})\b', year_to_words, text)
        
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


class SentenceStreamer:
    """
    Streams TTS sentence-by-sentence as LLM generates text with PRE-SYNTHESIS.
    
    This enables speaking to start before the full LLM response is complete,
    and pre-synthesizes upcoming sentences while the current one is playing.
    
    Architecture:
        1. LLM chunks → sentence_queue (complete sentences)
        2. synthesizer_task: sentence_queue → audio_queue (pre-synthesized audio)
        3. speaker_task: audio_queue → playback
        
    The synthesizer works ahead of the speaker, so audio is ready when needed.
    
    Usage:
        streamer = tts_manager.create_sentence_streamer()
        async for chunk in llm_response:
            print(chunk, end="")
            await streamer.add_chunk(chunk)
        await streamer.finish()
    """
    
    def __init__(self, tts_manager: TTSManager):
        self.tts_manager = tts_manager
        self.buffer = ""
        self.sentence_queue = asyncio.Queue()  # Text sentences waiting for synthesis
        self.audio_queue = asyncio.Queue()      # Pre-synthesized audio waiting for playback
        self.is_speaking = False
        self.synthesizer_task = None
        self.speaker_task = None
        self.finished = False
        self.synthesis_done = False
        self.debug = tts_manager.debug
        self.sentences_spoken = 0
        self.first_sentence_time = None
        self.start_time = None
        
        # Sentence ending patterns
        self.sentence_endings = re.compile(r'[.!?]\s*$|[.!?]["\']\s*$')
        self.min_sentence_length = 20
        
    async def add_chunk(self, chunk: str):
        """Add a chunk of text from the LLM stream"""
        if not chunk:
            return
            
        if self.start_time is None:
            self.start_time = time.time()
        
        self.buffer += chunk
        
        # Check for sentence completion
        await self._check_and_queue_sentences()
        
        # Start pipeline tasks if not running
        if self.synthesizer_task is None:
            self.synthesizer_task = asyncio.create_task(self._synthesizer_loop())
        if self.speaker_task is None:
            self.speaker_task = asyncio.create_task(self._speaker_loop())
    
    async def _check_and_queue_sentences(self):
        """Check buffer for complete sentences and queue them for synthesis"""
        while len(self.buffer) >= self.min_sentence_length:
            match = self.sentence_endings.search(self.buffer)
            if match:
                end_pos = match.end()
                sentence = self.buffer[:end_pos].strip()
                self.buffer = self.buffer[end_pos:].lstrip()
                
                if sentence:
                    await self.sentence_queue.put(sentence)
                    if self.debug:
                        print(f"\n[TTS-STREAM] Queued for synthesis: {sentence[:40]}...")
            else:
                break
    
    async def _synthesizer_loop(self):
        """
        Background task that pre-synthesizes sentences into audio buffers.
        Runs ahead of playback so audio is ready when needed.
        """
        loop = asyncio.get_event_loop()
        
        try:
            while True:
                if self.finished and self.sentence_queue.empty():
                    break
                    
                try:
                    sentence = await asyncio.wait_for(
                        self.sentence_queue.get(), 
                        timeout=0.2
                    )
                    
                    if self.first_sentence_time is None:
                        self.first_sentence_time = time.time()
                        if self.debug and self.start_time:
                            latency = (self.first_sentence_time - self.start_time) * 1000
                            print(f"\n[TTS-STREAM] ⚡ First sentence latency: {latency:.0f}ms")
                    
                    # Prepare text (strip references, convert years, clean up)
                    clean_text = self.tts_manager._prepare_text_for_speech(sentence)
                    if not clean_text:
                        continue
                    
                    # Pre-synthesize to buffer (runs in thread pool)
                    if self.debug:
                        print(f"\n[TTS-SYNTH] Synthesizing: {clean_text[:35]}...")
                    
                    piper = self.tts_manager._piper
                    if piper:
                        audio_buffer = await loop.run_in_executor(
                            None,
                            piper.synthesize_to_buffer,
                            clean_text
                        )
                        
                        if audio_buffer is not None:
                            # Queue pre-synthesized audio for playback
                            await self.audio_queue.put((sentence, audio_buffer))
                            if self.debug:
                                print(f"\n[TTS-SYNTH] ✅ Ready: {sentence[:35]}...")
                        else:
                            if self.debug:
                                print(f"\n[TTS-SYNTH] ⚠️  Synthesis failed")
                    
                except asyncio.TimeoutError:
                    if self.finished:
                        break
                    continue
                except asyncio.CancelledError:
                    raise
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.debug:
                print(f"\n[TTS-SYNTH] ❌ Error: {e}")
        finally:
            self.synthesis_done = True
    
    async def _speaker_loop(self):
        """
        Background task that plays pre-synthesized audio buffers.
        Audio should already be ready, so playback starts immediately.
        """
        loop = asyncio.get_event_loop()
        
        try:
            while True:
                # Exit when both synthesis is done AND audio queue is empty
                if self.synthesis_done and self.audio_queue.empty():
                    break
                    
                try:
                    sentence, audio_buffer = await asyncio.wait_for(
                        self.audio_queue.get(), 
                        timeout=0.2
                    )
                    
                    if self.debug:
                        print(f"\n[TTS-PLAY] ▶️  Playing: {sentence[:35]}...")
                    
                    self.is_speaking = True
                    
                    # Play pre-synthesized audio (runs in thread pool)
                    piper = self.tts_manager._piper
                    if piper:
                        # Handle LED feedback
                        if self.tts_manager.led_controller:
                            try:
                                self.tts_manager.led_controller.set_mode('speaking')
                            except:
                                pass
                        
                        await loop.run_in_executor(
                            None,
                            lambda: piper.play_buffer(audio_buffer, blocking=True)
                        )
                        
                        if self.tts_manager.led_controller:
                            try:
                                self.tts_manager.led_controller.set_mode('idle')
                            except:
                                pass
                    
                    self.is_speaking = False
                    self.sentences_spoken += 1
                    
                except asyncio.TimeoutError:
                    # No audio ready yet, check if we should exit
                    if self.synthesis_done and self.audio_queue.empty():
                        break
                    continue
                except asyncio.CancelledError:
                    raise
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.debug:
                print(f"\n[TTS-PLAY] ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    async def finish(self):
        """Signal that all text has been received and speak any remaining buffer"""
        self.finished = True
        
        # Queue any remaining text
        remaining = self.buffer.strip()
        if remaining:
            await self.sentence_queue.put(remaining)
            self.buffer = ""
        
        # Wait for synthesizer to finish
        if self.synthesizer_task:
            try:
                await asyncio.wait_for(self.synthesizer_task, timeout=60.0)
            except asyncio.TimeoutError:
                self.synthesizer_task.cancel()
        
        # Wait for speaker to finish
        if self.speaker_task:
            try:
                await asyncio.wait_for(self.speaker_task, timeout=60.0)
            except asyncio.TimeoutError:
                self.speaker_task.cancel()
        
        if self.debug:
            print(f"\n[TTS-STREAM] ✅ Complete: {self.sentences_spoken} sentences spoken")
    
    def stop(self):
        """Stop streaming immediately"""
        self.finished = True
        self.synthesis_done = True
        self.tts_manager.stop()
        if self.synthesizer_task:
            self.synthesizer_task.cancel()
        if self.speaker_task:
            self.speaker_task.cancel()


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
