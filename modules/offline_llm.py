"""
Pascal AI Assistant - SPEED-OPTIMIZED Offline LLM
Ultra-fast Nemotron integration targeting <3 second responses
FOCUS: Minimal overhead, smart prompt optimization, aggressive caching
"""

import asyncio
import json
import time
from typing import Optional, AsyncGenerator, Dict, Any, List

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

class LightningOfflineLLM:
    """Ultra-fast offline LLM optimized for sub-3-second responses"""
    
    def __init__(self):
        from config.settings import settings
        self.settings = settings
        
        # Connection management - optimized for speed
        self.session = None
        self.connector = None
        self.available = False
        self.model_loaded = False
        self.current_model = None
        self.keep_alive_task = None
        
        # Ollama configuration - SPEED FOCUSED
        self.host = settings.ollama_host
        self.keep_alive_duration = "30m"
        
        # Performance tracking
        self.request_count = 0
        self.total_time = 0.0
        self.error_count = 0
        self.last_error = None
        self.response_times = []
        
        # SPEED-OPTIMIZED generation settings
        self.ultra_fast_config = {
            'temperature': 0.3,          # Lower for faster, focused responses
            'top_p': 0.8,               # Reduced for speed
            'top_k': 20,                # Significantly reduced for speed
            'repeat_penalty': 1.05,
            'num_predict': 50,          # Very short for speed profile
            'num_ctx': 256,             # Minimal context for speed
            'num_thread': 4,            # Use all Pi 5 cores
            'num_gpu': 0,               # CPU only
            'stop': ["</s>", "<|end|>", "Human:", "User:", "\n\nHuman:", "\n\nUser:"]
        }
        
        # Performance profiles - AGGRESSIVE SPEED FOCUS
        self.performance_profiles = {
            'speed': {
                'num_predict': 30,       # Ultra-short responses
                'temperature': 0.2,      # Very focused
                'num_ctx': 128,          # Tiny context for maximum speed
                'timeout': 5,
                'description': 'Ultra-fast (<2s)',
                'skip_personality': True,
                'skip_memory': True
            },
            'balanced': {
                'num_predict': 80,       # Reasonable length
                'temperature': 0.5,
                'num_ctx': 256,          # Small context
                'timeout': 8,
                'description': 'Fast (2-3s)',
                'skip_personality': False,
                'skip_memory': True
            },
            'quality': {
                'num_predict': 150,      # Normal length
                'temperature': 0.7,
                'num_ctx': 512,          # Full context
                'timeout': 15,
                'description': 'Detailed (3-6s)',
                'skip_personality': False,
                'skip_memory': False
            }
        }
        self.current_profile = 'speed'  # Default to ultra-fast
        
        # Model preferences - prioritize speed-optimized models
        self.preferred_models = [
            'nemotron-fast',                    # Our speed-optimized model
            'nemotron-mini:4b-instruct-q4_K_M', # Base model
            'qwen2.5:3b',                       # Fast alternative
            'phi3:mini',                        # Compact option
        ]
        
        # Simple query detection for minimal prompts
        self.simple_patterns = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon',
            'how are you', 'what\'s up', 'thanks', 'thank you',
            'yes', 'no', 'ok', 'okay', 'bye', 'goodbye'
        ]
    
    async def initialize(self) -> bool:
        """SPEED-OPTIMIZED initialization with minimal overhead"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not available"
            return False
        
        try:
            # SPEED-OPTIMIZED connection settings
            timeout = aiohttp.ClientTimeout(
                total=10,           # Reduced from 20
                connect=2,          # Reduced from 3
                sock_read=8         # Reduced from 15
            )
            
            # Fast connector with keep-alive
            self.connector = aiohttp.TCPConnector(
                limit=1,                    # Single connection for Pi
                limit_per_host=1,           # One connection per host
                enable_cleanup_closed=True,
                use_dns_cache=True,
                keepalive_timeout=300,      # 5 minute keep-alive
                force_close=False           # Reuse connections
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout, 
                connector=self.connector,
                headers={'Connection': 'keep-alive'}  # HTTP keep-alive
            )
            
            # Fast connection test
            if not await self._test_connection_fast():
                self.last_error = "Cannot connect to Ollama"
                return False
            
            # Load and warm up model
            if await self._load_and_warm_model():
                self.available = True
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ⚡ Speed-optimized model ready: {self.current_model}")
                    print(f"[OLLAMA] 🚀 Profile: {self.current_profile} ({self.performance_profiles[self.current_profile]['description']})")
                return True
            else:
                self.last_error = "Failed to load speed-optimized model"
                return False
                
        except Exception as e:
            self.last_error = f"Initialization failed: {str(e)}"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Init error: {e}")
            return False
    
    async def _test_connection_fast(self) -> bool:
        """Ultra-fast connection test"""
        try:
            async with self.session.get(
                f"{self.host}/api/version",
                timeout=aiohttp.ClientTimeout(total=3)
            ) as response:
                if response.status == 200:
                    if self.settings.debug_mode:
                        data = await response.json()
                        print(f"[OLLAMA] ✅ Connected - Ollama v{data.get('version', 'unknown')}")
                    return True
                return False
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Connection failed: {e}")
            return False
    
    async def _get_available_models_fast(self) -> List[str]:
        """Get models with aggressive timeout"""
        try:
            async with self.session.get(
                f"{self.host}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] Available models: {models}")
                    return models
                return []
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ List models error: {e}")
            return []
    
    async def _load_and_warm_model(self) -> bool:
        """Load model and warm it up for fast responses"""
        try:
            available_models = await self._get_available_models_fast()
            
            if not available_models:
                self.last_error = "No Ollama models found"
                return False
            
            # Find best speed-optimized model
            model_to_load = None
            for preferred in self.preferred_models:
                for available in available_models:
                    if preferred == available or preferred in available:
                        model_to_load = available
                        break
                if model_to_load:
                    break
            
            if not model_to_load:
                model_to_load = available_models[0]
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ⚠️ Using first available: {model_to_load}")
            
            # Load model with speed config
            if await self._load_model_ultra_fast(model_to_load):
                # Warm up with a tiny query
                await self._warm_up_model()
                return True
            return False
            
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Load error: {e}")
            self.last_error = f"Load error: {str(e)}"
            return False
    
    async def _load_model_ultra_fast(self, model_name: str) -> bool:
        """Load model with ultra-fast settings"""
        try:
            if self.settings.debug_mode:
                print(f"[OLLAMA] Loading ultra-fast model: {model_name}")
            
            # Get speed profile settings
            profile = self.performance_profiles[self.current_profile]
            config = {**self.ultra_fast_config}
            config.update({k: v for k, v in profile.items() 
                          if k not in ['timeout', 'description', 'skip_personality', 'skip_memory']})
            
            # Ultra-minimal test payload
            payload = {
                "model": model_name,
                "prompt": "Hi",  # Single word
                "options": config,
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            timeout_val = profile.get('timeout', 5)
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_val)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'response' in data:
                        self.current_model = model_name
                        self.model_loaded = True
                        
                        if self.settings.debug_mode:
                            load_time = data.get('total_duration', 0) / 1e9 if data.get('total_duration') else 0
                            print(f"[OLLAMA] ✅ Model loaded in {load_time:.2f}s: {model_name}")
                        
                        return True
                    else:
                        self.last_error = "Invalid model response"
                        return False
                else:
                    self.last_error = f"Model load failed: HTTP {response.status}"
                    return False
                    
        except asyncio.TimeoutError:
            self.last_error = f"Model load timeout: {model_name}"
            return False
        except Exception as e:
            self.last_error = f"Model load error: {str(e)}"
            return False
    
    async def _warm_up_model(self):
        """Warm up model with tiny query to reduce first-response latency"""
        try:
            config = {**self.ultra_fast_config}
            config['num_predict'] = 5  # Just a few tokens
            
            payload = {
                "model": self.current_model,
                "prompt": "Say hi",
                "options": config,
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if self.settings.debug_mode and response.status == 200:
                    print(f"[OLLAMA] 🔥 Model warmed up")
                        
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ⚠️ Warm-up failed: {e}")
            # Non-critical failure
    
    def _build_ultra_fast_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Build minimal prompt optimized for speed"""
        query_lower = query.lower().strip()
        
        # Check if this is a simple query that needs minimal context
        is_simple = any(pattern in query_lower for pattern in self.simple_patterns)
        profile = self.performance_profiles[self.current_profile]
        
        # For simple queries or speed profile, use minimal prompt
        if is_simple or profile.get('skip_personality', False):
            if self.current_model and 'nemotron' in self.current_model.lower():
                return f"User: {query}\nAssistant:"
            else:
                return f"User: {query}\nAssistant:"
        
        # For other queries, add minimal context if not skipped
        parts = []
        
        if not profile.get('skip_personality', False) and personality_context:
            # Use only the first 100 chars of personality
            parts.append(personality_context[:100])
        
        if not profile.get('skip_memory', False) and memory_context:
            # Use only the last 50 chars of memory
            parts.append(memory_context[-50:] if len(memory_context) > 50 else memory_context)
        
        if parts:
            context = " ".join(parts)
            if self.current_model and 'nemotron' in self.current_model.lower():
                return f"System: {context}\n\nUser: {query}\nAssistant:"
            else:
                return f"{context}\n\nUser: {query}\nAssistant:"
        else:
            # Fallback to minimal prompt
            return f"User: {query}\nAssistant:"
    
    def set_performance_profile(self, profile: str):
        """Set performance profile - optimized for immediate effect"""
        if profile in self.performance_profiles:
            self.current_profile = profile
            
            # Update ultra_fast_config immediately
            profile_config = self.performance_profiles[profile]
            self.ultra_fast_config.update({
                k: v for k, v in profile_config.items() 
                if k not in ['timeout', 'description', 'skip_personality', 'skip_memory']
            })
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ⚡ Profile: {profile} - {profile_config['description']}")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """ULTRA-FAST response generation"""
        if not self.available or not self.model_loaded:
            return "Offline model unavailable."
        
        start_time = time.time()
        
        try:
            # Build minimal prompt
            prompt = self._build_ultra_fast_prompt(query, personality_context, memory_context)
            
            # Use current profile
            profile = self.performance_profiles[self.current_profile]
            config = {**self.ultra_fast_config}
            config.update({k: v for k, v in profile.items() 
                          if k not in ['timeout', 'description', 'skip_personality', 'skip_memory']})
            
            # Ultra-minimal payload
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": config,
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            timeout_val = profile.get('timeout', 5)
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_val)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '').strip()
                    
                    # Update performance tracking
                    elapsed = time.time() - start_time
                    self._update_performance_stats(elapsed, True)
                    
                    if self.settings.debug_mode:
                        eval_count = data.get('eval_count', 0)
                        eval_duration = data.get('eval_duration', 1)
                        tokens_per_sec = eval_count / max(eval_duration / 1e9, 0.001) if eval_count > 0 else 0
                        print(f"[OLLAMA] ✅ Response in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
                    
                    return response_text or "No response generated."
                    
                else:
                    elapsed = time.time() - start_time
                    self._update_performance_stats(elapsed, False)
                    self.last_error = f"HTTP {response.status}"
                    return f"Model error: {response.status}"
                    
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Timeout after {elapsed:.1f}s")
            
            return f"Response timed out after {elapsed:.1f}s. Try 'speed' profile."
            
        except Exception as e:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            self.last_error = str(e)
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Error: {e}")
            
            return f"Model error: {str(e)[:50]}"
    
    async def generate_response_stream(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """ULTRA-FAST streaming with immediate first token"""
        if not self.available or not self.model_loaded:
            yield "Offline model unavailable."
            return
        
        start_time = time.time()
        
        try:
            prompt = self._build_ultra_fast_prompt(query, personality_context, memory_context)
            
            profile = self.performance_profiles[self.current_profile]
            config = {**self.ultra_fast_config}
            config.update({k: v for k, v in profile.items() 
                          if k not in ['timeout', 'description', 'skip_personality', 'skip_memory']})
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": config,
                "stream": True,
                "keep_alive": self.keep_alive_duration
            }
            
            timeout_val = profile.get('timeout', 5)
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_val + 5)
            ) as response:
                if response.status == 200:
                    first_chunk = True
                    response_received = False
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                
                                if 'response' in data and data['response']:
                                    chunk = data['response']
                                    yield chunk
                                    response_received = True
                                    
                                    if first_chunk and self.settings.debug_mode:
                                        first_chunk_time = time.time() - start_time
                                        print(f"[OLLAMA] ⚡ First chunk in {first_chunk_time:.2f}s")
                                        first_chunk = False
                                
                                if data.get('done', False):
                                    elapsed = time.time() - start_time
                                    self._update_performance_stats(elapsed, True)
                                    
                                    if self.settings.debug_mode:
                                        print(f"[OLLAMA] ✅ Streaming complete in {elapsed:.2f}s")
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    if not response_received:
                        yield "No response generated."
                        
                else:
                    yield f"Streaming error: {response.status}"
                    
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            yield f"\n\nTimed out after {elapsed:.1f}s."
            
        except Exception as e:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            yield f"Error: {str(e)[:50]}"
    
    def _update_performance_stats(self, response_time: float, success: bool):
        """Update performance statistics"""
        self.request_count += 1
        self.total_time += response_time
        
        if not success:
            self.error_count += 1
        
        # Keep rolling window of recent response times
        self.response_times.append(response_time)
        if len(self.response_times) > 20:
            self.response_times = self.response_times[-20:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status with performance metrics"""
        avg_time = self.total_time / max(self.request_count, 1)
        success_rate = ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100
        
        recent_avg = 0
        if self.response_times:
            recent_avg = sum(self.response_times[-5:]) / len(self.response_times[-5:])
        
        return {
            'available': self.available,
            'model_loaded': self.model_loaded,
            'current_model': self.current_model,
            'performance_profile': self.current_profile,
            'profile_description': self.performance_profiles[self.current_profile]['description'],
            'optimization_level': 'ultra_fast_pi5',
            'stats': {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'avg_response_time': f"{avg_time:.2f}s",
                'recent_avg_time': f"{recent_avg:.2f}s",
                'success_rate': f"{success_rate:.1f}%",
                'target_time': f"<{self.performance_profiles[self.current_profile].get('timeout', 5)}s",
                'speed_grade': self._calculate_speed_grade(recent_avg)
            },
            'current_settings': {
                'num_predict': self.ultra_fast_config['num_predict'],
                'num_ctx': self.ultra_fast_config['num_ctx'],
                'temperature': self.ultra_fast_config['temperature'],
                'profile_optimizations': self.performance_profiles[self.current_profile]
            },
            'last_error': self.last_error,
            'preferred_models': self.preferred_models,
            'speed_optimizations': [
                'Minimal prompt building',
                'Context skipping for simple queries',
                'Connection keep-alive',
                'Model warm-up',
                'Aggressive timeouts',
                'Ultra-fast config profiles'
            ]
        }
    
    def _calculate_speed_grade(self, avg_time: float) -> str:
        """Calculate speed grade"""
        if avg_time < 1:
            return "A+ (Lightning)"
        elif avg_time < 2:
            return "A (Excellent)"
        elif avg_time < 3:
            return "B (Good)"
        elif avg_time < 5:
            return "C (Fair)"
        else:
            return "D (Slow)"
    
    async def close(self):
        """Clean close with performance summary"""
        if self.session:
            await self.session.close()
            self.session = None
        
        if self.connector:
            await self.connector.close()
            self.connector = None
        
        self.available = False
        self.model_loaded = False
        self.current_model = None
        
        if self.settings.debug_mode:
            if self.request_count > 0:
                avg_time = self.total_time / self.request_count
                print(f"[OLLAMA] 📊 Session stats: {self.request_count} requests, {avg_time:.2f}s avg")
            print("[OLLAMA] 🔌 Ultra-fast connection closed")

# Maintain compatibility
OptimizedOfflineLLM = LightningOfflineLLM
