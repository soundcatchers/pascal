"""
TTS Voice Configuration for Pascal AI Assistant

Maps personalities to voice models and provides voice settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path


@dataclass
class VoiceProfile:
    """Configuration for a single voice"""
    model_name: str
    model_path: Optional[str] = None
    description: str = ""
    speed: float = 1.0
    speaker_id: Optional[int] = None
    
    # Download info
    huggingface_url: Optional[str] = None
    model_size_mb: int = 0


# Pre-configured voice profiles for each personality
VOICE_PROFILES: Dict[str, VoiceProfile] = {
    "pascal": VoiceProfile(
        model_name="en_US-amy-medium",
        description="Friendly, warm female voice - matches Pascal's helpful personality",
        speed=1.0,
        huggingface_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium",
        model_size_mb=75
    ),
    "jarvis": VoiceProfile(
        model_name="en_GB-alan-medium",
        description="Formal British male voice - matches JARVIS's butler-like demeanor",
        speed=0.95,
        huggingface_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium",
        model_size_mb=75
    ),
    "rick": VoiceProfile(
        model_name="en_US-lessac-medium",
        description="Expressive male voice - closest match for Rick's sardonic tone",
        speed=1.1,
        huggingface_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium",
        model_size_mb=75
    ),
    "default": VoiceProfile(
        model_name="en_US-lessac-medium",
        description="Default fallback voice - neutral and clear",
        speed=1.0,
        huggingface_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium",
        model_size_mb=75
    )
}

# Alternative voices that can be swapped in
ALTERNATIVE_VOICES: Dict[str, VoiceProfile] = {
    "en_US-lessac-low": VoiceProfile(
        model_name="en_US-lessac-low",
        description="Lower quality but faster - good for testing",
        speed=1.0,
        huggingface_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low",
        model_size_mb=30
    ),
    "en_US-lessac-high": VoiceProfile(
        model_name="en_US-lessac-high",
        description="Higher quality but slower - best for final output",
        speed=1.0,
        huggingface_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high",
        model_size_mb=100
    ),
    "en_US-libritts-high": VoiceProfile(
        model_name="en_US-libritts-high",
        description="Multi-speaker voice model with variety",
        speed=1.0,
        huggingface_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high",
        model_size_mb=120
    ),
    "en_GB-cori-medium": VoiceProfile(
        model_name="en_GB-cori-medium",
        description="British female voice - alternative for JARVIS",
        speed=1.0,
        huggingface_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/cori/medium",
        model_size_mb=75
    ),
}


class VoiceManager:
    """Manages voice profiles and model paths"""
    
    def __init__(self, voices_dir: str = "config/tts_voices"):
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.profiles = VOICE_PROFILES.copy()
        self._resolve_model_paths()
    
    def _resolve_model_paths(self):
        """Set model paths for each profile"""
        for name, profile in self.profiles.items():
            model_dir = self.voices_dir / name
            model_file = model_dir / f"{profile.model_name}.onnx"
            profile.model_path = str(model_file)
    
    def get_voice(self, personality: str) -> VoiceProfile:
        """Get voice profile for a personality"""
        personality_lower = personality.lower()
        if personality_lower in self.profiles:
            return self.profiles[personality_lower]
        return self.profiles.get("default", list(self.profiles.values())[0])
    
    def get_model_path(self, personality: str) -> Optional[str]:
        """Get the model path for a personality's voice"""
        voice = self.get_voice(personality)
        if voice.model_path and Path(voice.model_path).exists():
            return voice.model_path
        return None
    
    def is_voice_available(self, personality: str) -> bool:
        """Check if voice model is downloaded"""
        model_path = self.get_model_path(personality)
        if model_path:
            json_path = model_path + ".json"
            return Path(model_path).exists() and Path(json_path).exists()
        return False
    
    def get_available_voices(self) -> Dict[str, bool]:
        """Get availability status of all voices"""
        return {name: self.is_voice_available(name) for name in self.profiles}
    
    def get_download_info(self, personality: str) -> Dict[str, str]:
        """Get download URLs for a voice model"""
        voice = self.get_voice(personality)
        if voice.huggingface_url:
            return {
                "onnx": f"{voice.huggingface_url}/{voice.model_name}.onnx",
                "json": f"{voice.huggingface_url}/{voice.model_name}.onnx.json",
                "dest_dir": str(self.voices_dir / personality)
            }
        return {}
    
    def set_custom_voice(self, personality: str, model_path: str, speed: float = 1.0):
        """Set a custom voice model for a personality"""
        self.profiles[personality] = VoiceProfile(
            model_name=Path(model_path).stem,
            model_path=model_path,
            description=f"Custom voice for {personality}",
            speed=speed
        )
    
    def list_personalities(self) -> list:
        """List all configured personalities"""
        return list(self.profiles.keys())
