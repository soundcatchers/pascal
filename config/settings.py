"""
Pascal AI Assistant - SIMPLIFIED Settings Configuration
Streamlined for Pi 5 with Groq-only online and Ollama offline
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimplifiedSettings:
    """Simplified settings focused on performance and reliability"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.config_dir = self.base_dir / "config"
        self.data_dir = self.base_dir / "data"
        
        # Ensure directories exist
        self._create_directories()
        
        # Pascal identity
        self.name = "Pascal"
        self.version = "3.0.0"  # Simplified version
        
        # Debug settings
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # SIMPLIFIED: Groq ONLY for online
        self.groq_api_key = self._load_groq_api_key()
        self.online_available = bool(self.groq_api_key and self._validate_groq_key(self.groq_api_key))
        
        # Performance settings (optimized for Pi 5)
        self.target_response_time = float(os.getenv("TARGET_RESPONSE_TIME", "2.0"))
        self.streaming_enabled = os.getenv("STREAMING_ENABLED", "true").lower() == "true"
        self.keep_alive_enabled = os.getenv("KEEP_ALIVE_ENABLED", "true").lower() == "true"
        
        # Ollama settings
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "30"))
        self.ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        
        # Memory settings
        self.memory_limit = int(os.getenv("MEMORY_LIMIT", "5"))
        self.context_window = int(os.getenv("CONTEXT_WINDOW", "1024"))
        
        # Pi 5 hardware optimization
        self.max_tokens = int(os.getenv("MAX_TOKENS", "200"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # SIMPLIFIED: Single model preferences
        self.preferred_offline_model = "nemotron-mini:4b-instruct-q4_K_M"
        self.preferred_online_model = "llama-3.1-8b-instant"
        
        if self.debug_mode:
            print(f"[SETTINGS] Pascal v{self.version} - Simplified Configuration")
            print(f"[SETTINGS] Online available: {self.online_available}")
            print(f"[SETTINGS] Debug mode: {self.debug_mode}")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config_dir,
            self.data_dir,
            self.data_dir / "memory",
            self.data_dir / "cache",
            self.config_dir / "personalities"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_groq_api_key(self) -> Optional[str]:
        """Load Groq API key with fallback support"""
        # Try GROQ_API_KEY first (preferred)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and self._validate_groq_key(groq_key):
            return groq_key
        
        # Fallback to legacy GROK_API_KEY
        legacy_key = os.getenv("GROK_API_KEY")
        if legacy_key and self._validate_groq_key(legacy_key):
            if self.debug_mode:
                print("[SETTINGS] Using legacy GROK_API_KEY - consider renaming to GROQ_API_KEY")
            return legacy_key
        
        return None
    
    def _validate_groq_key(self, key: str) -> bool:
        """Validate Groq API key format"""
        if not key or not isinstance(key, str):
            return False
        
        key = key.strip()
        
        # Check for placeholder values
        invalid_values = [
            '', 'your_groq_api_key_here', 'your_grok_api_key_here',
            'gsk_your_groq_api_key_here', 'gsk-your_groq_api_key_here'
        ]
        
        if key.lower() in [v.lower() for v in invalid_values]:
            return False
        
        # Accept both gsk_ (new) and gsk- (deprecated)
        if key.startswith('gsk_') or key.startswith('gsk-'):
            return len(key) > 20
        
        return False
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration for Pi 5"""
        return {
            "target_response_time": self.target_response_time,
            "streaming_enabled": self.streaming_enabled,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "context_window": self.context_window,
            "keep_alive": self.ollama_keep_alive
        }
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama configuration"""
        return {
            "host": self.ollama_host,
            "timeout": self.ollama_timeout,
            "keep_alive": self.ollama_keep_alive,
            "preferred_model": self.preferred_offline_model
        }
    
    def get_groq_config(self) -> Dict[str, Any]:
        """Get Groq configuration"""
        return {
            "api_key": self.groq_api_key,
            "model": self.preferred_online_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "available": self.online_available
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "version": self.version,
            "debug_mode": self.debug_mode,
            "online_available": self.online_available,
            "groq_configured": bool(self.groq_api_key),
            "streaming_enabled": self.streaming_enabled,
            "target_response_time": self.target_response_time,
            "preferred_models": {
                "offline": self.preferred_offline_model,
                "online": self.preferred_online_model
            }
        }

# Global instance
settings = SimplifiedSettings()

# Export paths for compatibility
BASE_DIR = settings.base_dir
CONFIG_DIR = settings.config_dir
DATA_DIR = settings.data_dir
