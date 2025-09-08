"""
Pascal AI Assistant - Global Configuration
Manages all system settings and environment variables - Lightning-fast for Pi 5
FIXED: Consistent API key naming and better fallback support
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Global settings for Pascal - Optimized for sub-3-second responses"""
    
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
        self.version = "2.0.1"  # Fixed version with Groq integration
        
        # LLM Configuration - Optimized for speed
        self.default_personality = "default"
        self.max_context_length = 2048
        self.max_response_tokens = 200  # Limited for faster responses
        
        # Debug Settings - MOVED BEFORE FIRST USE
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.verbose_logging = self.debug_mode
        self.performance_logging = os.getenv("PERF_LOG", "false").lower() == "true"
        
        # Online LLM APIs - FIXED: Support both GROQ and GROK for compatibility
        # Check for both GROQ_API_KEY and GROK_API_KEY (user might have either)
        self.groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        # Warn about deprecated GROK naming
        if os.getenv("GROK_API_KEY") and not os.getenv("GROQ_API_KEY"):
            if self.is_debug_mode():
                print("[WARNING] Found GROK_API_KEY - please rename to GROQ_API_KEY in .env file")
        
        # Debug API key loading
        if self.is_debug_mode():
            if self.groq_api_key:
                print(f"[DEBUG] Groq API key loaded: {self.groq_api_key[:20]}...")
            else:
                print("[DEBUG] No Groq API key found in environment")
                print("[DEBUG] Set GROQ_API_KEY=gsk-your-key in .env file for fastest online responses")
        
        # Local LLM Settings - Lightning optimized
        self.local_model_path = self.models_dir / "local_model.gguf"
        self.local_model_threads = 4  # Use all Pi 5 cores
        self.local_model_context = 2048
        self.streaming_enabled = True  # Enable streaming for perceived speed
        self.keep_alive_enabled = True  # Keep models loaded
        
        # Lightning Performance Settings
        self.target_response_time = 3.0  # Target max 3 seconds
        self.enable_streaming = True
        self.first_token_target = 1.0  # Target 1 second to first token
        
        # Performance Settings - Pi 5 Specific
        self.performance_mode = os.getenv("PERFORMANCE_MODE", "balanced")  # Default to balanced
        self.use_arm_optimizations = True
        self.memory_optimization = True
        self.context_caching = True
        
        # Pi 5 Hardware Detection
        self.is_raspberry_pi = self._detect_raspberry_pi()
        self.pi_model = self._get_pi_model()
        self.available_ram_gb = self._get_available_ram()
        
        # Router Settings - IMPORTANT: Intelligent routing with online preference for current info
        self.prefer_offline = False  # Changed to False to use intelligent routing
        self.online_timeout = 30.0  # Increased timeout for online services
        self.fallback_to_offline = True
        self.smart_routing = True
        self.auto_route_current_info = True  # Automatically route current info queries to online
        
        # Memory Settings
        self.short_term_memory_limit = 5  # Keep recent context
        self.long_term_memory_enabled = True
        self.memory_save_interval = 300  # Seconds
        
        # Performance Settings
        self.enable_caching = True
        self.cache_expiry = 3600  # Seconds
        self.max_concurrent_requests = 2  # Limited for Pi 5
        
        # ARM-specific optimizations
        self.arm_cpu_optimization = {
            'enable_neon': True,
            'thread_affinity': True,
            'cache_line_size': 64,
            'prefetch_distance': 16,
        }
        
        # Model Selection Preferences - Updated for current models
        self.preferred_models = [
            "nemotron-mini:4b-instruct-q4_K_M",  # Primary - fastest
            "qwen2.5:3b",                         # Updated Qwen model
            "phi3:mini",                          # Reliable fallback
            "llama3.2:3b",                        # Current Llama model
            "gemma2:2b"                           # Lightweight option
        ]
        self.max_model_ram_usage = 6.0  # Max GB for model
        self.auto_model_selection = True
        
        # Voice Settings (for future phases)
        self.voice_enabled = False
        self.voice_model = "whisper-tiny"
        self.tts_model = "coqui-tts"
        self.voice_activation_threshold = 0.7
        
        # Display Settings (for future visual phase)
        self.visual_display_enabled = False
        self.display_fps = 30
        self.display_resolution = (1920, 1080)
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.debug_mode
    
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
                        return round(kb / 1024 / 1024, 1)  # Convert KB to GB
        except FileNotFoundError:
            return 8.0  # Default assumption
        return 8.0
    
    def get_personality_path(self, personality_name: str) -> Path:
        """Get path to personality configuration file"""
        return self.config_dir / "personalities" / f"{personality_name}.json"
    
    def get_memory_path(self, session_id: str = "default") -> Path:
        """Get path to memory file for a session"""
        return self.memory_dir / f"{session_id}_memory.json"
    
    def is_online_available(self) -> bool:
        """Check if any online API keys are configured"""
        # Check if any API key is properly configured
        has_groq = self.groq_api_key and self.groq_api_key not in ['', 'your_groq_api_key_here', 'gsk-your_groq_api_key_here', 'your_grok_api_key_here']
        has_openai = self.openai_api_key and self.openai_api_key not in ['', 'your_openai_api_key_here', 'sk-your_openai_api_key_here']
        has_gemini = self.gemini_api_key and self.gemini_api_key not in ['', 'your_gemini_api_key_here', 'your_google_api_key_here']
        
        if self.debug_mode:
            print(f"[DEBUG] Online availability check:")
            print(f"  Groq configured: {has_groq}")
            print(f"  OpenAI configured: {has_openai}")
            print(f"  Gemini configured: {has_gemini}")
        
        return has_groq or has_openai or has_gemini
    
    def is_local_model_available(self) -> bool:
        """Check if any local model files exist"""
        if not self.models_dir.exists():
            return False
        
        # Check for any GGUF files
        gguf_files = list(self.models_dir.glob("*.gguf"))
        return len(gguf_files) > 0
    
    def get_optimal_context_size(self, model_ram_usage: float) -> int:
        """Calculate optimal context size based on available RAM"""
        available_for_context = self.available_ram_gb - model_ram_usage - 2  # Reserve 2GB for system
        
        if available_for_context <= 2:
            return 1024
        elif available_for_context <= 4:
            return 2048
        elif available_for_context <= 6:
            return 3072
        else:
            return 4096
    
    def get_performance_profile(self) -> Dict[str, Any]:
        """Get current performance profile settings optimized for speed"""
        profiles = {
            'speed': {
                'max_tokens': 100,
                'context_size': 1024,
                'temperature': 0.5,
                'priority': 'response_time',
                'streaming': True
            },
            'balanced': {
                'max_tokens': 150,
                'context_size': 2048,
                'temperature': 0.7,
                'priority': 'balanced',
                'streaming': True
            },
            'quality': {
                'max_tokens': 200,
                'context_size': 2048,
                'temperature': 0.8,
                'priority': 'output_quality',
                'streaming': False
            }
        }
        
        return profiles.get(self.performance_mode, profiles['balanced'])
    
    def set_performance_mode(self, mode: str):
        """Set performance mode (speed/balanced/quality)"""
        if mode in ['speed', 'balanced', 'quality']:
            self.performance_mode = mode
            if self.debug_mode:
                print(f"Performance mode set to: {mode}")
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information for optimization"""
        return {
            'is_raspberry_pi': self.is_raspberry_pi,
            'pi_model': self.pi_model,
            'available_ram_gb': self.available_ram_gb,
            'cpu_cores': self.local_model_threads,
            'arm_optimizations_enabled': self.use_arm_optimizations,
            'performance_mode': self.performance_mode,
            'streaming_enabled': self.streaming_enabled,
            'keep_alive_enabled': self.keep_alive_enabled
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for status display"""
        # Check API keys with improved validation
        groq_configured = bool(self.groq_api_key and 
                              self.groq_api_key not in ['', 'your_groq_api_key_here', 'gsk-your_groq_api_key_here', 'your_grok_api_key_here'] and
                              len(self.groq_api_key) > 10)
        
        gemini_configured = bool(self.gemini_api_key and 
                               self.gemini_api_key not in ['', 'your_gemini_api_key_here', 'your_google_api_key_here'] and
                               len(self.gemini_api_key) > 10)
        
        openai_configured = bool(self.openai_api_key and 
                               self.openai_api_key not in ['', 'your_openai_api_key_here', 'sk-your_openai_api_key_here'] and
                               len(self.openai_api_key) > 10)
        
        return {
            "pascal_version": self.version,
            "base_directory": str(self.base_dir),
            "personality": self.default_personality,
            "online_apis_configured": self.is_online_available(),
            "groq_configured": groq_configured,
            "gemini_configured": gemini_configured,
            "openai_configured": openai_configured,
            "local_model_available": self.is_local_model_available(),
            "prefer_offline": self.prefer_offline,
            "auto_route_current_info": self.auto_route_current_info,
            "debug_mode": self.debug_mode,
            "memory_enabled": self.long_term_memory_enabled,
            "performance_mode": self.performance_mode,
            "hardware_info": self.get_hardware_info(),
            "arm_optimizations": self.use_arm_optimizations,
            "streaming_enabled": self.streaming_enabled,
            "target_response_time": self.target_response_time,
            "preferred_models": self.preferred_models,
            "available_models": len(list(self.models_dir.glob("*.gguf"))) if self.models_dir.exists() else 0
        }
    
    def update_from_env(self):
        """Update settings from environment variables"""
        # Performance mode
        env_perf_mode = os.getenv("PERFORMANCE_MODE")
        if env_perf_mode in ['speed', 'balanced', 'quality']:
            self.performance_mode = env_perf_mode
        
        # Thread count
        env_threads = os.getenv("LLM_THREADS")
        if env_threads and env_threads.isdigit():
            self.local_model_threads = min(int(env_threads), 4)  # Max 4 for Pi 5
        
        # Context size
        env_context = os.getenv("LLM_CONTEXT")
        if env_context and env_context.isdigit():
            self.local_model_context = int(env_context)
        
        # Max response tokens
        env_max_tokens = os.getenv("MAX_RESPONSE_TOKENS")
        if env_max_tokens and env_max_tokens.isdigit():
            self.max_response_tokens = int(env_max_tokens)
        
        # Streaming
        env_streaming = os.getenv("STREAMING_ENABLED")
        if env_streaming:
            self.streaming_enabled = env_streaming.lower() == "true"
        
        # Keep alive
        env_keep_alive = os.getenv("KEEP_ALIVE_ENABLED")
        if env_keep_alive:
            self.keep_alive_enabled = env_keep_alive.lower() == "true"
        
        # Target response time
        env_target_time = os.getenv("TARGET_RESPONSE_TIME")
        if env_target_time:
            try:
                self.target_response_time = float(env_target_time)
            except ValueError:
                pass
        
        # Re-check API keys after env update
        self.groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.debug_mode and self.groq_api_key:
            print(f"[DEBUG] Groq API key updated: {self.groq_api_key[:20]}...")
    
    def save_performance_settings(self):
        """Save current performance settings to file"""
        settings_file = self.config_dir / "performance_settings.json"
        settings_data = {
            'performance_mode': self.performance_mode,
            'local_model_threads': self.local_model_threads,
            'local_model_context': self.local_model_context,
            'max_response_tokens': self.max_response_tokens,
            'use_arm_optimizations': self.use_arm_optimizations,
            'streaming_enabled': self.streaming_enabled,
            'keep_alive_enabled': self.keep_alive_enabled,
            'target_response_time': self.target_response_time,
            'preferred_models': self.preferred_models,
            'auto_route_current_info': self.auto_route_current_info,
            'updated_timestamp': 0  # Simplified for compatibility
        }
        
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings_data, f, indent=2)
        except Exception as e:
            if self.debug_mode:
                print(f"Failed to save performance settings: {e}")
    
    def load_performance_settings(self):
        """Load performance settings from file"""
        settings_file = self.config_dir / "performance_settings.json"
        
        if not settings_file.exists():
            return
        
        try:
            with open(settings_file, 'r') as f:
                settings_data = json.load(f)
            
            self.performance_mode = settings_data.get('performance_mode', self.performance_mode)
            self.local_model_threads = settings_data.get('local_model_threads', self.local_model_threads)
            self.local_model_context = settings_data.get('local_model_context', self.local_model_context)
            self.max_response_tokens = settings_data.get('max_response_tokens', self.max_response_tokens)
            self.use_arm_optimizations = settings_data.get('use_arm_optimizations', self.use_arm_optimizations)
            self.streaming_enabled = settings_data.get('streaming_enabled', self.streaming_enabled)
            self.keep_alive_enabled = settings_data.get('keep_alive_enabled', self.keep_alive_enabled)
            self.target_response_time = settings_data.get('target_response_time', self.target_response_time)
            self.preferred_models = settings_data.get('preferred_models', self.preferred_models)
            self.auto_route_current_info = settings_data.get('auto_route_current_info', self.auto_route_current_info)
            
        except Exception as e:
            if self.debug_mode:
                print(f"Failed to load performance settings: {e}")

# Global settings instance
settings = Settings()

# Load any saved performance settings
settings.load_performance_settings()

# Update from environment variables
settings.update_from_env()

# Export commonly used paths
BASE_DIR = settings.base_dir
CONFIG_DIR = settings.config_dir
DATA_DIR = settings.data_dir
