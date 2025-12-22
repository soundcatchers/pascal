"""
Pascal AI Assistant - FIXED Offline LLM Module
Resolves aiohttp import issues and improves initialization reliability
"""

import asyncio
import json
import time
import re
from typing import Optional, AsyncGenerator, Dict, Any, List

# FIXED: More robust aiohttp import with better error handling
AIOHTTP_AVAILABLE = False
aiohttp = None

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
    print("[DEBUG] aiohttp imported successfully")
except ImportError as e:
    print(f"[DEBUG] aiohttp import failed: {e}")
    AIOHTTP_AVAILABLE = False
    aiohttp = None

class LightningOfflineLLM:
    """FIXED: Speed-optimized offline LLM with improved aiohttp handling"""
    
    def __init__(self):
        from config.settings import settings
        self.settings = settings
        
        # Connection management
        self.session = None
        self.connector = None
        self.available = False
        self.model_loaded = False
        self.current_model = None
        self.keep_alive_task = None
        
        # Ollama configuration
        self.host = settings.ollama_host
        self.keep_alive_duration = "30m"
        self.keep_alive_interval = 30
        
        # Performance tracking
        self.request_count = 0
        self.total_time = 0.0
        self.error_count = 0
        self.last_error = None
        self.response_times = []
        self.consecutive_errors = 0
        self.last_successful_time = time.time()
        
        # FIXED: Model preferences with better fallbacks
        self.preferred_models = [
            'nemotron-mini:4b-instruct-q4_K_M',
            'nemotron-fast',
            'qwen2.5:3b-instruct',
            'qwen2.5:3b',
            'phi3:mini',
            'gemma2:2b',
        ]
        
        # Performance profiles
        self.profiles = {
            'speed': {
                'num_predict': 80,
                'temperature': 0.3,
                'num_ctx': 512,
                'timeout': 15,
                'description': 'Fast (2-4s)',
                'top_p': 0.8,
                'top_k': 25,
                'repeat_penalty': 1.05,
                'num_thread': 4,
                'num_gpu': 0,
                'target_time': 3.0
            },
            'balanced': {
                'num_predict': 150,
                'temperature': 0.5,
                'num_ctx': 1024,
                'timeout': 25,
                'description': 'Balanced (3-6s)',
                'top_p': 0.9,
                'top_k': 40,
                'repeat_penalty': 1.1,
                'num_thread': 4,
                'num_gpu': 0,
                'target_time': 5.0
            },
            'quality': {
                'num_predict': 300,
                'temperature': 0.7,
                'num_ctx': 2048,
                'timeout': 45,
                'description': 'Quality (5-10s)',
                'top_p': 0.9,
                'top_k': 50,
                'repeat_penalty': 1.1,
                'num_thread': 4,
                'num_gpu': 0,
                'target_time': 8.0
            }
        }
        self.current_profile = 'balanced'
    
    async def initialize(self) -> bool:
        """FIXED: More robust initialization with better error handling"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not available - run: pip install aiohttp==3.9.5"
            print(f"[OLLAMA] ❌ {self.last_error}")
            return False
        
        try:
            # Create HTTP session with better error handling
            if not await self._create_session():
                return False
            
            # Test connection with retries
            if not await self._test_connection():
                return False
            
            # Load best available model
            if not await self._load_model():
                return False
            
            # Start keep-alive
            await self._start_keep_alive()
            
            # Warmup with full context to pre-allocate memory
            await self._warmup_model()
            
            self.available = True
            self.consecutive_errors = 0
            self.last_successful_time = time.time()
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ✅ Initialized: {self.current_model} ({self.current_profile})")
            
            return True
            
        except Exception as e:
            self.last_error = f"Initialization error: {str(e)}"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Init failed: {e}")
                import traceback
                traceback.print_exc()
            return False
    
    async def _create_session(self) -> bool:
        """FIXED: Create HTTP session with better error handling"""
        try:
            if not AIOHTTP_AVAILABLE or aiohttp is None:
                self.last_error = "aiohttp module not available"
                return False
            
            # Create timeout
            timeout = aiohttp.ClientTimeout(
                total=60,
                connect=10,
                sock_read=45
            )
            
            # Create connector
            self.connector = aiohttp.TCPConnector(
                limit=2,
                limit_per_host=2,
                enable_cleanup_closed=True,
                use_dns_cache=True,
                keepalive_timeout=300,
                force_close=False,
                ttl_dns_cache=300
            )
            
            # Create session
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=self.connector,
                headers={
                    'Connection': 'keep-alive',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
            return True
            
        except Exception as e:
            self.last_error = f"Session creation failed: {str(e)}"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Session creation error: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test connection to Ollama service"""
        if not self.session:
            return False
        
        try:
            async with self.session.get(
                f"{self.host}/api/version",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ⚡ Connected - Ollama v{data.get('version', 'unknown')}")
                    return True
                else:
                    self.last_error = f"Ollama API returned HTTP {response.status}"
                    return False
        except Exception as e:
            self.last_error = f"Connection test failed: {str(e)}"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Connection error: {e}")
            return False
    
    async def _load_model(self) -> bool:
        """Load the best available model"""
        try:
            # Get available models
            async with self.session.get(f"{self.host}/api/tags") as response:
                if response.status != 200:
                    self.last_error = f"Failed to list models: HTTP {response.status}"
                    return False
                
                data = await response.json()
                models = data.get('models', [])
                
                if not models:
                    self.last_error = "No models found"
                    return False
                
                model_names = [model.get('name', '') for model in models]
                
                # Find best model
                for preferred in self.preferred_models:
                    for model_name in model_names:
                        if preferred == model_name or preferred in model_name:
                            if await self._test_model(model_name):
                                self.current_model = model_name
                                self.model_loaded = True
                                if self.settings.debug_mode:
                                    print(f"[OLLAMA] ⚡ Loaded: {model_name}")
                                return True
                
                # Try any available model
                for model in models:
                    model_name = model.get('name', '')
                    if model_name and await self._test_model(model_name):
                        self.current_model = model_name
                        self.model_loaded = True
                        return True
                
                self.last_error = "No working models found"
                return False
                
        except Exception as e:
            self.last_error = f"Model loading error: {str(e)}"
            return False
    
    async def _test_model(self, model_name: str) -> bool:
        """Test if a model works properly"""
        try:
            payload = {
                "model": model_name,
                "prompt": "Hello! Please respond with just 'Hi'.",
                "options": {
                    "num_predict": 10,
                    "temperature": 0.1,
                    "num_ctx": 256,
                    "num_thread": 4,
                    "top_p": 0.8,
                    "top_k": 20
                },
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '').strip()
                    
                    if response_text and len(response_text) > 0:
                        if not self._is_nonsense_response(response_text):
                            return True
                
                return False
                
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] Model test failed for {model_name}: {e}")
            return False
    
    def _is_nonsense_response(self, response: str) -> bool:
        """Check if response is nonsense"""
        response_lower = response.lower().strip()
        
        if len(response_lower) < 1:
            return True
        
        # Check for repeated characters
        if len(set(response_lower.replace(' ', ''))) <= 2 and len(response_lower) > 5:
            return True
        
        # Check for broken responses
        nonsense_patterns = [
            r'^[\s\n\r]*$',
            r'^[^\w\s]+$',
            r'(.)\1{10,}',
            r'^\s*error\s*$',
            r'^\s*null\s*$',
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, response_lower):
                return True
        
        return False
    
    async def _warmup_model(self):
        """Warmup model with full context size to pre-allocate memory.
        This eliminates the delay on first real query.
        """
        if not self.current_model or not self.session:
            return
        
        print(f"[OLLAMA] ⚡ Warming up {self.current_model}...")
        
        try:
            # Use the same settings as real queries to pre-allocate full memory
            profile = self.profiles.get(self.current_profile, self.profiles['balanced'])
            
            payload = {
                "model": self.current_model,
                "prompt": "Hello! Respond with a brief greeting.",
                "system": "You are a helpful AI assistant. Be concise.",
                "options": {
                    "num_predict": 20,  # Short response for speed
                    "temperature": profile['temperature'],
                    "num_ctx": profile['num_ctx'],  # Full context size
                    "num_thread": profile['num_thread'],
                    "top_p": profile['top_p'],
                    "top_k": profile['top_k']
                },
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            start_time = time.time()
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)  # Allow time for first load
            ) as response:
                if response.status == 200:
                    warmup_time = time.time() - start_time
                    print(f"[OLLAMA] ✅ Warmup complete ({warmup_time:.1f}s) - model ready in memory")
                else:
                    print(f"[OLLAMA] ⚠️  Warmup returned HTTP {response.status}")
                    
        except Exception as e:
            print(f"[OLLAMA] ⚠️  Warmup failed: {e}")
    
    async def _start_keep_alive(self):
        """Start keep-alive task"""
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
        
        self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())
    
    async def _keep_alive_loop(self):
        """Keep-alive loop"""
        while self.model_loaded and self.current_model:
            try:
                await asyncio.sleep(self.keep_alive_interval)
                
                payload = {
                    "model": self.current_model,
                    "prompt": "",
                    "options": {"num_predict": 1},
                    "stream": False,
                    "keep_alive": self.keep_alive_duration
                }
                
                async with self.session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    pass
                    
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    def set_performance_profile(self, profile: str):
        """Set performance profile"""
        if profile in self.profiles:
            self.current_profile = profile
            if self.settings.debug_mode:
                target_time = self.profiles[profile]['target_time']
                print(f"[OLLAMA] ⚡ Profile: {profile} - Target: <{target_time}s")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str, user_name: str = None) -> str:
        """Generate response with proper error handling"""
        if not self.available or not self.model_loaded:
            return "Offline model unavailable. Please check Ollama service."
        
        start_time = time.time()
        
        try:
            prompt = self._build_prompt(query, personality_context, memory_context, user_name)
            profile = self.profiles[self.current_profile]
            
            options = {
                "num_predict": profile['num_predict'],
                "temperature": profile['temperature'],
                "num_ctx": profile['num_ctx'],
                "top_p": profile['top_p'],
                "top_k": profile['top_k'],
                "repeat_penalty": profile['repeat_penalty'],
                "num_thread": profile['num_thread'],
                "num_gpu": profile['num_gpu']
            }
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": options,
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=profile['timeout'])
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '').strip()
                    
                    elapsed = time.time() - start_time
                    
                    if response_text and not self._is_nonsense_response(response_text):
                        self._update_stats(elapsed, True)
                        
                        if self.settings.debug_mode:
                            target = profile['target_time']
                            status = "⚡" if elapsed < target else "✅" if elapsed < target * 1.5 else "⚠️"
                            print(f"[OLLAMA] {status} Response in {elapsed:.2f}s")
                        
                        self.consecutive_errors = 0
                        return response_text
                    else:
                        self.last_error = "Empty or invalid response"
                        return "I'm having trouble generating a response right now."
                else:
                    self.last_error = f"HTTP {response.status}"
                    return "I'm experiencing technical difficulties."
                    
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self.last_error = f"Timeout after {elapsed:.1f}s"
            return "Response timed out. The model may be loading."
        except Exception as e:
            self.last_error = f"Generation error: {str(e)}"
            return "I'm having trouble responding right now."
        
        finally:
            elapsed = time.time() - start_time
            self._update_stats(elapsed, False)
    
    async def generate_response_stream(self, query: str, personality_context: str, memory_context: str, user_name: str = None) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        if not self.available or not self.model_loaded:
            yield "Offline model unavailable."
            return
        
        try:
            prompt = self._build_prompt(query, personality_context, memory_context, user_name)
            profile = self.profiles[self.current_profile]
            
            options = {
                "num_predict": profile['num_predict'],
                "temperature": profile['temperature'],
                "num_ctx": profile['num_ctx'],
                "top_p": profile['top_p'],
                "top_k": profile['top_k'],
                "repeat_penalty": profile['repeat_penalty'],
                "num_thread": profile['num_thread'],
                "num_gpu": profile['num_gpu']
            }
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": options,
                "stream": True,
                "keep_alive": self.keep_alive_duration
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=profile['timeout'] + 10)
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                
                                if 'response' in data and data['response']:
                                    yield data['response']
                                
                                if data.get('done', False):
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                else:
                    yield f"Model error: HTTP {response.status}"
                    
        except Exception as e:
            yield f"Error: {str(e)[:50]}"
    
    def _build_prompt(self, query: str, personality_context: str, memory_context: str, user_name: str = None) -> str:
        """Build optimized prompt"""
        prompt_parts = []
        
        # Add user name instruction if known
        if user_name:
            prompt_parts.append(f"[User's name is {user_name}. Address them by name, not 'user'.]")
        
        if len(query.split()) <= 8 and not user_name:
            return f"User: {query}\nAssistant:"
        
        if personality_context and len(personality_context) < 500:
            prompt_parts.append(personality_context[:300])
        
        if memory_context and len(memory_context) < 300:
            prompt_parts.append(f"Context: {memory_context}")
        
        prompt_parts.append(f"User: {query}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _update_stats(self, response_time: float, success: bool):
        """Update performance statistics"""
        self.request_count += 1
        self.total_time += response_time
        
        if not success:
            self.error_count += 1
            self.consecutive_errors += 1
        else:
            self.last_successful_time = time.time()
            self.consecutive_errors = 0
        
        self.response_times.append(response_time)
        if len(self.response_times) > 20:
            self.response_times = self.response_times[-20:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        avg_time = self.total_time / max(self.request_count, 1)
        success_rate = ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100
        
        recent_avg = 0
        if self.response_times:
            recent_avg = sum(self.response_times[-5:]) / len(self.response_times[-5:])
        
        target_time = self.profiles[self.current_profile]['target_time']
        speed_grade = "A+" if recent_avg < target_time else "A" if recent_avg < target_time * 1.5 else "B"
        
        return {
            'available': self.available,
            'model_loaded': self.model_loaded,
            'current_model': self.current_model,
            'performance_profile': self.current_profile,
            'profile_description': self.profiles[self.current_profile]['description'],
            'target_time': f"{target_time}s",
            'aiohttp_available': AIOHTTP_AVAILABLE,
            'last_error': self.last_error,
            'stats': {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'avg_response_time': f"{avg_time:.2f}s",
                'recent_avg_time': f"{recent_avg:.2f}s",
                'success_rate': f"{success_rate:.1f}%",
                'speed_grade': speed_grade
            },
            'improvements': [
                'FIXED: Better aiohttp import handling',
                'FIXED: Improved error handling and fallbacks',
                'FIXED: More robust session creation',
                'FIXED: Better model testing and selection'
            ]
        }
    
    async def close(self):
        """Clean shutdown"""
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
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
            print("[OLLAMA] 🔌 Connection closed")

# Maintain compatibility
OptimizedOfflineLLM = LightningOfflineLLM
