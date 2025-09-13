"""
Pascal AI Assistant - FIXED Settings Configuration
Corrected for Groq-only online and Ollama offline
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Fixed settings focused on Groq + Ollama only"""
    
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
        self.version = "3.0.0"
        
        # Debug settings
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.verbose_logging = self.debug_mode
        
        # FIXED: Groq ONLY for online (corrected API key loading)
        self.groq_api_key = self._load_groq_api_key()
        
        # Performance settings (optimized for Pi 5)
        self.performance_mode = os.getenv("PERFORMANCE_MODE", "balanced")
        self.target_response_time = float(os.getenv("TARGET_RESPONSE_TIME", "2.0"))
        self.streaming_enabled = os.getenv("STREAMING_ENABLED", "true").lower() == "true"
        self.keep_alive_enabled = os.getenv("KEEP_ALIVE_ENABLED", "true").lower() == "true"
        self.max_response_tokens = int(os.getenv("MAX_RESPONSE_TOKENS", "200"))
        
        # LLM Configuration
        self.default_personality = "default"
        self.max_context_length = int(os.getenv("CONTEXT_WINDOW", "1024"))
        
        # Ollama settings
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "30"))
        self.ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        
        # Pi 5 hardware optimization
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Memory settings
        self.short_term_memory_limit = int(os.getenv("MEMORY_LIMIT", "5"))
        self.long_term_memory_enabled = True
        self.memory_save_interval = 300
        
        # FIXED: Preferred offline model
        self.preferred_offline_model = "nemotron-mini:4b-instruct-q4_K_M"
        
        # Enhanced current info settings
        self.auto_route_current_info = True
        self.force_online_current_info = True
        self.enhance_current_info_prompts = True
        
        # Pi 5 Hardware Detection
        self.is_raspberry_pi = self._detect_raspberry_pi()
        self.pi_model = self._get_pi_model()
        self.available_ram_gb = self._get_available_ram()
        
        if self.debug_mode:
            print(f"[SETTINGS] Pascal v{self.version} - Fixed Configuration")
            print(f"[SETTINGS] Groq configured: {bool(self.groq_api_key)}")
            print(f"[SETTINGS] Debug mode: {self.debug_mode}")
    
    def _create_directories(self):
        """Create necessary directories"""
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
    
    def _load_groq_api_key(self) -> Optional[str]:
        """FIXED: Load Groq API key with proper validation"""
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
        """FIXED: Validate Groq API key format"""
        if not key or not isinstance(key, str):
            return False
        
        key = key.strip()
        
        # Check for placeholder values
        invalid_values = [
            '', 'your_groq_api_key_here', 'your_grok_api_key_here',
            'gsk_your_groq_api_key_here', 'gsk-your_groq_api_key_here',
            'gsk_your-groq-api-key-here'
        ]
        
        if key.lower() in [v.lower() for v in invalid_values]:
            return False
        
        # Accept both gsk_ (new) and gsk- (deprecated) but prefer gsk_
        if key.startswith('gsk_') or key.startswith('gsk-'):
            return len(key) > 20
        
        return False
    
    def _detect_raspberry_pi(self) -> bool:
        """Detect if running on Raspberry Pi"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                return 'Raspberry Pi' in model
        except FileNotFoundError:
            return False
    
    def _get_pi_model(self) -> str:
        """Get specific Raspberry Pi model"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'Raspberry Pi 5' in model:
                    return 'Pi 5'
                elif 'Raspberry Pi 4' in model:
                    return 'Pi 4'
                else:
                    return 'Unknown Pi'
        except FileNotFoundError:
            return 'Not Pi'
    
    def _get_available_ram(self) -> float:
        """Get available RAM in GB"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        return round(kb / 1024 / 1024, 1)
        except FileNotFoundError:
            return 8.0
        return 8.0
    
    def validate_groq_api_key(self, api_key: str) -> bool:
        """Validate Groq API key format (public method)"""
        return self._validate_groq_key(api_key)
    
    def is_online_available(self) -> bool:
        """Check if Groq API key is configured"""
        return bool(self.groq_api_key and self.validate_groq_api_key(self.groq_api_key))
    
    def get_personality_path(self, personality_name: str) -> Path:
        """Get path to personality configuration file"""
        return self.config_dir / "personalities" / f"{personality_name}.json"
    
    def get_memory_path(self, session_id: str = "default") -> Path:
        """Get path to memory file for a session"""
        return self.memory_dir / f"{session_id}_memory.json"
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.debug_mode
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information for optimization"""
        return {
            'is_raspberry_pi': self.is_raspberry_pi,
            'pi_model': self.pi_model,
            'available_ram_gb': self.available_ram_gb,
            'cpu_cores': os.cpu_count() or 4,
            'performance_mode': self.performance_mode,
            'streaming_enabled': self.streaming_enabled,
            'keep_alive_enabled': self.keep_alive_enabled
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for status display"""
        return {
            "pascal_version": self.version,
            "personality": self.default_personality,
            "groq_configured": bool(self.groq_api_key),
            "online_available": self.is_online_available(),
            "auto_route_current_info": self.auto_route_current_info,
            "force_online_current_info": self.force_online_current_info,
            "debug_mode": self.debug_mode,
            "memory_enabled": self.long_term_memory_enabled,
            "performance_mode": self.performance_mode,
            "hardware_info": self.get_hardware_info(),
            "streaming_enabled": self.streaming_enabled,
            "target_response_time": self.target_response_time,
            "preferred_offline_model": self.preferred_offline_model,
            "supported_providers": ["Groq"]  # Only Groq now
        }
    
    def set_performance_mode(self, mode: str):
        """Set performance mode"""
        if mode in ['speed', 'balanced', 'quality']:
            self.performance_mode = mode
            if self.debug_mode:
                print(f"Performance mode set to: {mode}")

# Global settings instance
settings = Settings()

# Export commonly used paths
BASE_DIR = settings.base_dir
CONFIG_DIR = settings.config_dir
DATA_DIR = settings.data_dir
