"""
Pascal AI Assistant - Global Configuration
Manages all system settings and environment variables - Lightning-fast for Pi 5
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
        self.version = "2.0.0"  # Lightning version with Grok
        
        # LLM Configuration - Optimized for speed
        self.default_personality = "default"
        self.max_context_length = 2048
        self.max_response_tokens = 150  # Limited for faster responses
        
        # Online LLM APIs - Grok as primary, Gemini as alternative
        self.grok_api_key = os.getenv("GROK_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")  # Support both names
        
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
        self.performance_mode = os.getenv("PERFORMANCE_MODE", "speed")  # Default to speed
        self.use_arm_optimizations = True
        self.memory_optimization = True
        self.context_caching = True
        
        # Pi 5 Hardware Detection
        self.is_raspberry_pi = self._detect_raspberry_pi()
        self.pi_model = self._get_pi_model()
        self.available_ram_gb = self._get_available_ram()
        
        # Router Settings
        self.prefer_offline = True  # Default to offline for speed
        self.online_timeout = 10.0  # Seconds
        self.fallback_to_offline = True
        self.smart_routing = True
        
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
        
        # Model Selection Preferences - Updated for new models
        self.preferred_models = [
            "nemotron-mini:4b-instruct-q4_K_M",
            "qwen3:4b-instruct",
            "gemma3:4b-it-q4_K_M"
        ]
        self.max_model_ram_usage = 6.0  # Max GB for model
        self.auto_model_selection = True
        
        # Debug Settings
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.verbose_logging = self.debug_mode
        self.performance_logging = os.getenv("PERF_LOG", "false").lower() == "true"
        
        # Voice Settings (for future phases)
        self.voice_enabled = False
        self.voice_model = "whisper-tiny"
        self.tts_model = "coqui-tts"
        self.voice_activation_threshold = 0.7
        
        # Display Settings (for future visual phase)
        self.visual_display_enabled = False
        self.display_fps = 30
        self.display_resolution = (1920, 1080)
    
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
        return any([
            self.grok_api_key,
            self.openai_api_key,
            self.gemini_api_key
        ])
    
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
        
        return profiles
