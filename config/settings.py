"""
Pascal AI Assistant - Global Configuration
FIXED: Enhanced API key validation and current info handling
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Global settings for Pascal - Enhanced for reliable current info handling"""
    
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
        self.version = "2.1.1"  # Updated version with enhanced fixes
        
        # Debug Settings - MOVED BEFORE FIRST USE
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.verbose_logging = self.debug_mode
        self.performance_logging = os.getenv("PERF_LOG", "false").lower() == "true"
        
        # FIXED: Enhanced API Key Loading with proper gsk_ support
        self.groq_api_key = self._load_groq_api_key()
        self.openai_api_key = self._load_openai_api_key()
        self.gemini_api_key = self._load_gemini_api_key()
        
        # Debug API key loading with enhanced validation
        if self.debug_mode:
            print(f"[DEBUG] Enhanced API Keys Status:")
            print(f"  Groq: {'✅ Loaded' if self.groq_api_key else '❌ Missing'}")
            print(f"  OpenAI: {'✅ Loaded' if self.openai_api_key else '❌ Missing'}")
            print(f"  Gemini: {'✅ Loaded' if self.gemini_api_key else '❌ Missing'}")
            
            if self.groq_api_key:
                if self.groq_api_key.startswith('gsk_'):
                    print(f"  Groq format: ✅ Correct (gsk_)")
                elif self.groq_api_key.startswith('gsk-'):
                    print(f"  Groq format: ⚠️ Deprecated (gsk-) - still works but update recommended")
                else:
                    print(f"  Groq format: ❌ Invalid - should start with gsk_")
        
        # LLM Configuration
        self.default_personality = "default"
        self.max_context_length = 2048
        self.max_response_tokens = 300  # Increased for better responses
        
        # Local LLM Settings
        self.local_model_path = self.models_dir / "local_model.gguf"
        self.local_model_threads = 4
        self.local_model_context = 2048
        self.streaming_enabled = True
        self.keep_alive_enabled = True
        
        # Performance Settings
        self.performance_mode = os.getenv("PERFORMANCE_MODE", "balanced")
        self.target_response_time = float(os.getenv("TARGET_RESPONSE_TIME", "3.0"))
        self.enable_streaming = os.getenv("STREAMING_ENABLED", "true").lower() == "true"
        self.first_token_target = 1.0
        
        # Pi 5 Hardware Detection
        self.is_raspberry_pi = self._detect_raspberry_pi()
        self.pi_model = self._get_pi_model()
        self.available_ram_gb = self._get_available_ram()
        
        # ENHANCED Router Settings for current info
        self.prefer_offline = False
        self.online_timeout = 45.0  # Increased timeout for better reliability
        self.fallback_to_offline = True
        self.smart_routing = True
        self.auto_route_current_info = True  # CRITICAL: Always route current info online
        self.force_online_current_info = True  # NEW: Force online for current info
        
        # Enhanced Current Information Settings
        self.enhance_current_info_prompts = True
        self.current_info_system_instructions = True
        self.current_info_priority = True  # NEW: Prioritize current info routing
        
        # Memory Settings
        self.short_term_memory_limit = 5
        self.long_term_memory_enabled = True
        self.memory_save_interval = 300
        
        # Model Preferences
        self.preferred_models = [
            "nemotron-mini:4b-instruct-q4_K_M",
            "qwen2.5:3b",
            "phi3:mini",
            "llama3.2:3b",
            "gemma2:2b"
        ]
        self.max_model_ram_usage = 6.0
        self.auto_model_selection = True
        
        # Voice Settings (future)
        self.voice_enabled = False
        self.voice_model = "whisper-tiny"
        self.tts_model = "coqui-tts"
    
    def _load_groq_api_key(self) -> Optional[str]:
        """ENHANCED: Load Groq API key with comprehensive validation"""
        # First try GROQ_API_KEY (preferred new naming)
        groq_key = os.getenv("GROQ_API_KEY")
        
        if groq_key and self._is_valid_api_key(groq_key, "groq"):
            if self.debug_mode:
                print(f"[DEBUG] Using GROQ_API_KEY: {groq_key[:15]}...")
            return groq_key
        
        # Fallback to legacy GROK_API_KEY for backward compatibility
        legacy_key = os.getenv("GROK_API_KEY")
        if legacy_key and self._is_valid_api_key(legacy_key, "groq"):
            if self.debug_mode:
                print(f"[DEBUG] Using legacy GROK_API_KEY (rename to GROQ_API_KEY): {legacy_key[:15]}...")
            return legacy_key
        
        return None
    
    def _load_openai_api_key(self) -> Optional[str]:
        """Load and validate OpenAI API key"""
        key = os.getenv("OPENAI_API_KEY")
        if key and self._is_valid_api_key(key, "openai"):
            return key
        return None
    
    def _load_gemini_api_key(self) -> Optional[str]:
        """Load and validate Gemini API key"""
        # Try both possible environment variable names
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key and self._is_valid_api_key(key, "gemini"):
            return key
        return None
    
    def _is_valid_api_key(self, key: str, provider: str) -> bool:
        """Enhanced API key validation"""
        if not key or not isinstance(key, str):
            return False
        
        # Remove any whitespace
        key = key.strip()
        
        # Check for obvious placeholder values
        placeholder_patterns = [
            '', 'your_api_key_here', 'your_groq_api_key_here', 'your_openai_api_key_here', 
            'your_gemini_api_key_here', 'your_google_api_key_here', 'gsk_your_groq_api_key_here',
            'sk_your_openai_api_key_here', 'gsk-your_groq_api_key_here'
        ]
        
        if key.lower() in [p.lower() for p in placeholder_patterns]:
            return False
        
        # Provider-specific validation
        if provider == "groq":
            # Accept both gsk_ (new) and gsk- (deprecated but working)
            if key.startswith('gsk_') or key.startswith('gsk-'):
                return len(key) > 20  # Reasonable minimum length
            return False
        
        elif provider == "openai":
            if key.startswith('sk-'):
                return len(key) > 20
            return False
        
        elif provider == "gemini":
            # Gemini keys don't have a specific prefix but should be reasonably long
            return len(key) > 20
        
        return False
    
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
        """ENHANCED: Validate Groq API key format with new gsk_ standard"""
        return self._is_valid_api_key(api_key, "groq")
    
    def validate_openai_api_key(self, api_key: str) -> bool:
        """Validate OpenAI API key format"""
        return self._is_valid_api_key(api_key, "openai")
    
    def validate_gemini_api_key(self, api_key: str) -> bool:
        """Validate Gemini API key format"""
        return self._is_valid_api_key(api_key, "gemini")
    
    def is_online_available(self) -> bool:
        """ENHANCED: Check if any online API keys are configured"""
        has_groq = self.validate_groq_api_key(self.groq_api_key)
        has_openai = self.validate_openai_api_key(self.openai_api_key)
        has_gemini = self.validate_gemini_api_key(self.gemini_api_key)
        
        if self.debug_mode:
            print(f"[DEBUG] Enhanced online availability check:")
            print(f"  Groq configured: {has_groq}")
            print(f"  OpenAI configured: {has_openai}")
            print(f"  Gemini configured: {has_gemini}")
            print(f"  Online available: {has_groq or has_openai or has_gemini}")
        
        return has_groq or has_openai or has_gemini
    
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
            'cpu_cores': self.local_model_threads,
            'performance_mode': self.performance_mode,
            'streaming_enabled': self.streaming_enabled,
            'keep_alive_enabled': self.keep_alive_enabled
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """ENHANCED: Get configuration summary for status display"""
        groq_configured = self.validate_groq_api_key(self.groq_api_key)
        gemini_configured = self.validate_gemini_api_key(self.gemini_api_key)
        openai_configured = self.validate_openai_api_key(self.openai_api_key)
        
        return {
            "pascal_version": self.version,
            "personality": self.default_personality,
            "online_apis_configured": self.is_online_available(),
            "groq_configured": groq_configured,
            "gemini_configured": gemini_configured,
            "openai_configured": openai_configured,
            "auto_route_current_info": self.auto_route_current_info,
            "force_online_current_info": self.force_online_current_info,
            "enhance_current_info_prompts": self.enhance_current_info_prompts,
            "current_info_priority": self.current_info_priority,
            "debug_mode": self.debug_mode,
            "memory_enabled": self.long_term_memory_enabled,
            "performance_mode": self.performance_mode,
            "hardware_info": self.get_hardware_info(),
            "streaming_enabled": self.streaming_enabled,
            "target_response_time": self.target_response_time,
            "preferred_models": self.preferred_models,
            "online_timeout": self.online_timeout
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
