"""
Pascal AI Assistant - ULTRA-SPEED Settings Configuration
Extreme optimizations for sub-2 second responses on Pi 5
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """ULTRA-SPEED optimized settings for sub-2-second responses"""
    
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
        self.version = "4.3.0"  # Ultra-speed-optimized version
        
        # Debug settings
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.verbose_logging = self.debug_mode
        
        # API Configuration - Groq only for online
        self.groq_api_key = self._load_groq_api_key()
        
        # ULTRA-SPEED PERFORMANCE SETTINGS
        self.performance_mode = os.getenv("PERFORMANCE_MODE", "speed")  # Default to speed
        self.target_response_time = float(os.getenv("TARGET_RESPONSE_TIME", "1.5"))  # Ultra-aggressive 1.5s target
        self.max_response_time = float(os.getenv("MAX_RESPONSE_TIME", "4.0"))  # Hard limit
        self.streaming_enabled = os.getenv("STREAMING_ENABLED", "true").lower() == "true"
        self.max_response_tokens = int(os.getenv("MAX_RESPONSE_TOKENS", "60"))  # Very short for speed
        
        # LLM Configuration - ULTRA-SPEED OPTIMIZED
        self.default_personality = "default"
        self.max_context_length = 128  # Ultra-aggressive reduction for speed
        self.temperature = 0.1  # Very low for faster, more focused responses
        
        # OLLAMA SETTINGS - ULTRA-SPEED CONFIGURATION
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "6"))  # Ultra-aggressive timeout
        self.ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        
        # Ultra-speed Ollama settings
        self.ollama_num_parallel = 1  # Single request at a time
        self.ollama_max_loaded_models = 1  # One model only
        self.ollama_num_thread = 4  # Use all 4 Pi 5 cores
        self.ollama_flash_attention = False  # Disabled for compatibility
        
        # Memory settings - INFINITE LONG-TERM MEMORY (Friend with no amnesia!)
        self.short_term_memory_limit = 100  # Last 100 interactions in working memory
        self.long_term_memory_enabled = True  # Enable infinite long-term memory
        self.memory_save_interval = 300  # Auto-save every 5 minutes
        self.long_term_memory_retention_days = 3650  # 10 years (effectively infinite)
        
        # ULTRA-SPEED Model preferences
        self.preferred_offline_models = [
            "nemotron-fast",                    # Speed-optimized custom version
            "nemotron-mini:4b-instruct-q4_K_M", # Original optimized model
            "qwen2.5:3b",                       # Fast alternative
            "phi3:mini",                        # Compact option
            "gemma2:2b",                        # Ultra-compact
        ]
        self.preferred_offline_model = self.preferred_offline_models[0]
        
        # Current info settings - simplified but aggressive routing
        self.auto_route_current_info = True
        self.force_online_current_info = True
        self.current_info_confidence_threshold = 0.7  # Lower threshold for more aggressive routing
        
        # Performance monitoring - minimal overhead
        self.enable_performance_monitoring = True
        self.performance_log_interval = 900  # Log every 15 minutes
        self.response_time_threshold = 2.0   # Warn if responses > 2s
        
        # Connection optimization - ULTRA-AGGRESSIVE
        self.connection_pool_size = 1  # Single connection for Pi 5
        self.connection_timeout = 1.5  # Ultra-fast connection timeout
        self.read_timeout = 4          # Fast read timeout
        
        # Hardware detection and optimization
        self.is_raspberry_pi = self._detect_raspberry_pi()
        self.pi_model = self._get_pi_model()
        self.available_ram_gb = self._get_available_ram()
        self.cpu_cores = self._get_cpu_cores()
        
        # Auto-optimize based on hardware
        if self.is_raspberry_pi:
            self._apply_ultra_speed_optimizations()
        
        # Speed-specific settings
        self.enable_turbo_mode = os.getenv("ENABLE_TURBO_MODE", "false").lower() == "true"
        self.minimal_prompts = True
        self.aggressive_timeouts = True
        self.skip_context_for_short_queries = True
        self.short_query_threshold = 8  # Words
        
        # Voice Input Post-Processing Settings
        self.voice_enable_spell_check = os.getenv("VOICE_ENABLE_SPELL_CHECK", "true").lower() == "true"
        self.voice_enable_confidence_filter = os.getenv("VOICE_ENABLE_CONFIDENCE_FILTER", "true").lower() == "true"
        self.voice_enable_punctuation = os.getenv("VOICE_ENABLE_PUNCTUATION", "true").lower() == "true"
        self.voice_confidence_threshold = float(os.getenv("VOICE_CONFIDENCE_THRESHOLD", "0.80"))
        self.voice_spell_check_max_distance = int(os.getenv("VOICE_SPELL_CHECK_MAX_DISTANCE", "3"))
        
        # Voice AI Correction Settings (context-aware fixing of valid-but-wrong words)
        self.voice_enable_ai_correction = os.getenv("VOICE_ENABLE_AI_CORRECTION", "true").lower() == "true"
        self.voice_ai_correction_model = os.getenv("VOICE_AI_CORRECTION_MODEL", "gemma2:2b")
        self.voice_ai_correction_timeout = float(os.getenv("VOICE_AI_CORRECTION_TIMEOUT", "2.0"))
        
        if self.debug_mode:
            print(f"[SETTINGS] Pascal v{self.version} - ULTRA-SPEED OPTIMIZED")
            print(f"[SETTINGS] Hardware: {self.pi_model} ({self.available_ram_gb}GB RAM, {self.cpu_cores} cores)")
            print(f"[SETTINGS] Target response time: {self.target_response_time}s (ULTRA-AGGRESSIVE)")
            print(f"[SETTINGS] Groq configured: {bool(self.groq_api_key)}")
            print(f"[SETTINGS] Performance mode: {self.performance_mode}")
            print(f"[SETTINGS] Speed optimizations: ULTRA-ENABLED")
            if self.enable_turbo_mode:
                print(f"[SETTINGS] 🚀 TURBO MODE ENABLED")
    
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
        """Load Groq API key with validation"""
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
            'gsk_your_groq_api_key_here', 'gsk-your_groq_api_key_here',
            'gsk_your-groq-api-key-here'
        ]
        
        if key.lower() in [v.lower() for v in invalid_values]:
            return False
        
        # Accept both gsk_ (new) and gsk- (deprecated)
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
    
    def _get_cpu_cores(self) -> int:
        """Get number of CPU cores"""
        return os.cpu_count() or 4
    
    def _apply_ultra_speed_optimizations(self):
        """Apply ULTRA-AGGRESSIVE speed optimizations for Pi"""
        if self.pi_model == 'Pi 5':
            # Pi 5 ULTRA-SPEED optimizations
            self.target_response_time = 1.5  # Ultra-aggressive target
            self.max_context_length = 128    # Ultra-small context
            self.max_response_tokens = 50    # Very short responses
            self.ollama_timeout = 5          # Ultra-fast timeout
            
            if self.available_ram_gb >= 16:
                # 16GB Pi 5 - can afford slightly more
                self.max_context_length = 256
                self.max_response_tokens = 60
                self.target_response_time = 1.3  # Even faster for 16GB
            elif self.available_ram_gb >= 8:
                # 8GB Pi 5 - balanced ultra-speed settings
                self.max_context_length = 128
                self.max_response_tokens = 50
                self.target_response_time = 1.5
            else:
                # 4GB Pi 5 - ultra-conservative for speed
                self.max_context_length = 64
                self.max_response_tokens = 40
                self.target_response_time = 2.0
                
        elif self.pi_model == 'Pi 4':
            # Pi 4 ultra-conservative settings
            self.target_response_time = 3.0
            self.max_context_length = 64
            self.max_response_tokens = 40
            self.ollama_timeout = 8
        
        if self.debug_mode:
            print(f"[SETTINGS] Applied ULTRA-SPEED optimizations for {self.pi_model}")
    
    def get_ollama_speed_config(self) -> Dict[str, Any]:
        """Get Ollama ULTRA-SPEED configuration"""
        return {
            "num_parallel": self.ollama_num_parallel,
            "max_loaded_models": self.ollama_max_loaded_models,
            "num_thread": self.ollama_num_thread,
            "flash_attention": self.ollama_flash_attention,
            "keep_alive": self.ollama_keep_alive,
            "host": self.ollama_host,
            "timeout": self.ollama_timeout,
            "ultra_speed_optimizations": {
                "aggressive_timeouts": True,
                "minimal_context": True,
                "ultra_short_responses": True,
                "single_model_focus": True,
                "turbo_mode": self.enable_turbo_mode,
                "skip_context_short_queries": self.skip_context_for_short_queries,
                "minimal_prompts": self.minimal_prompts
            }
        }
    
    def get_speed_config(self) -> Dict[str, Any]:
        """Get ULTRA-SPEED configuration"""
        return {
            "target_response_time": self.target_response_time,
            "max_response_time": self.max_response_time,
            "max_context_length": self.max_context_length,
            "max_response_tokens": self.max_response_tokens,
            "performance_mode": self.performance_mode,
            "streaming_enabled": self.streaming_enabled,
            "connection_timeout": self.connection_timeout,
            "read_timeout": self.read_timeout,
            "turbo_mode": self.enable_turbo_mode,
            "memory_optimizations": {
                "ultra_minimal_memory": True,
                "no_long_term": not self.long_term_memory_enabled,
                "ultra_reduced_context": True,
                "skip_context_short_queries": self.skip_context_for_short_queries
            },
            "ultra_speed_features": [
                f"Target: <{self.target_response_time}s responses",
                "Ultra-aggressive timeouts",
                "Minimal context windows (64-256 tokens)",
                "Ultra-short response limits (40-60 tokens)",
                "Single connection pooling",
                "Compiled regex patterns",
                "Direct model access",
                "Turbo mode support",
                "Context skipping for short queries"
            ]
        }
    
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
    
    def should_skip_context(self, query: str) -> bool:
        """Check if we should skip context for this query (speed optimization)"""
        if not self.skip_context_for_short_queries:
            return False
        
        word_count = len(query.split())
        return word_count <= self.short_query_threshold
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        return {
            'is_raspberry_pi': self.is_raspberry_pi,
            'pi_model': self.pi_model,
            'available_ram_gb': self.available_ram_gb,
            'cpu_cores': self.cpu_cores,
            'performance_mode': self.performance_mode,
            'streaming_enabled': self.streaming_enabled,
            'target_response_time': self.target_response_time,
            'optimization_level': 'ultra_speed_pi5' if self.pi_model == 'Pi 5' else 'ultra_speed_pi4'
        }
    
    def get_voice_postprocessing_config(self) -> Dict[str, Any]:
        """Get voice post-processing configuration"""
        return {
            'spell_check': self.voice_enable_spell_check,
            'confidence_filter': self.voice_enable_confidence_filter,
            'punctuation': self.voice_enable_punctuation,
            'confidence_threshold': self.voice_confidence_threshold,
            'spell_check_max_distance': self.voice_spell_check_max_distance,
            'ai_correction': self.voice_enable_ai_correction,
            'ai_correction_model': self.voice_ai_correction_model,
            'ai_correction_timeout': self.voice_ai_correction_timeout
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'pascal_version': self.version,
            'performance_mode': self.performance_mode,
            'streaming_enabled': self.streaming_enabled,
            'target_response_time': self.target_response_time,
            'max_response_tokens': self.max_response_tokens,
            'debug_mode': self.debug_mode
        }

# Create global settings instance
settings = Settings()
