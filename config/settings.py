"""
Pascal AI Assistant - Global Configuration
Manages all system settings and environment variables
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Global settings for Pascal"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.config_dir = self.base_dir / "config"
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.data_dir / "models"
        self.memory_dir = self.data_dir / "memory"
        self.cache_dir = self.data_dir / "cache"
        
        # Ensure directories exist
        self._create_directories()
        
        # Pascal identity
        self.name = "Pascal"
        self.version = "1.0.0"
        
        # LLM Configuration
        self.default_personality = "default"
        self.max_context_length = 4096
        self.max_response_tokens = 100
        
        # Online LLM APIs
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Local LLM Settings
        self.local_model_path = self.models_dir / "local_model.gguf"
        self.local_model_threads = 4  # Use all Pi 5 cores
        self.local_model_context = 1024  # Reduce from 2048 to 1024
        
        # Router Settings
        self.prefer_offline = True  # Default to offline when possible
        self.online_timeout = 10.0  # Seconds
        self.fallback_to_offline = True
        
        # Memory Settings
        self.short_term_memory_limit = 2  # Number of messages
        self.long_term_memory_enabled = True
        self.memory_save_interval = 300  # Seconds
        
        # Performance Settings
        self.enable_caching = True
        self.cache_expiry = 3600  # Seconds
        self.max_concurrent_requests = 3
        
        # Debug Settings
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.verbose_logging = self.debug_mode
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.config_dir,
            self.data_dir,
            self.models_dir,
            self.memory_dir,
            self.cache_dir,
            self.config_dir / "personalities"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_personality_path(self, personality_name: str) -> Path:
        """Get path to personality configuration file"""
        return self.config_dir / "personalities" / f"{personality_name}.json"
    
    def get_memory_path(self, session_id: str = "default") -> Path:
        """Get path to memory file for a session"""
        return self.memory_dir / f"{session_id}_memory.json"
    
    def is_online_available(self) -> bool:
        """Check if any online API keys are configured"""
        return any([
            self.openai_api_key,
            self.anthropic_api_key,
            self.google_api_key
        ])
    
    def is_local_model_available(self) -> bool:
        """Check if local model file exists"""
        return self.local_model_path.exists()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for status display"""
        return {
            "pascal_version": self.version,
            "base_directory": str(self.base_dir),
            "personality": self.default_personality,
            "online_apis_configured": self.is_online_available(),
            "local_model_available": self.is_local_model_available(),
            "prefer_offline": self.prefer_offline,
            "debug_mode": self.debug_mode,
            "memory_enabled": self.long_term_memory_enabled
        }

# Global settings instance
settings = Settings()

# Export commonly used paths
BASE_DIR = settings.base_dir
CONFIG_DIR = settings.config_dir
DATA_DIR = settings.data_dir
