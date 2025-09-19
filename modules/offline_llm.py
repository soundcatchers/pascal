"""
Pascal AI Assistant - FIXED Offline LLM Module
Resolved the critical aiohttp scoping error and improved error handling
"""

import asyncio
import json
import time
import re
from typing import Optional, AsyncGenerator, Dict, Any, List

# Import aiohttp at module level to avoid scoping issues
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

class LightningOfflineLLM:
    """Fixed offline LLM with resolved aiohttp scoping error"""
    
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
        
        # Response quality tracking
        self.response_cache = {}
        self.cache_max_size = 50
        
        # Model preferences
        self.preferred_models = [
            'nemotron-mini:4b-instruct-q4_K_M',
            'nemotron-fast',
            'qwen2.5:3b',
            'phi3:mini',
            'llama3.2:3b',
            'gemma2:2b',
        ]
        
        # Performance profiles
        self.profiles = {
            'speed': {
                'num_predict': 50,
                'temperature': 0.1,
                'num_ctx': 256,
                'timeout': 8,
                'description': 'Ultra-fast (<2s)',
                'top_p': 0.7,
                'top_k': 15,
                'repeat_penalty': 1.02,
            },
            'balanced': {
                'num_predict': 100,
                'temperature': 0.3,
                'num_ctx': 512,
                'timeout': 12,
                'description': 'Balanced (2-4s)',
                'top_p': 0.8,
                'top_k': 25,
                'repeat_penalty': 1.05,
            },
            'quality': {
                'num_predict': 200,
                'temperature': 0.7,
                'num_ctx': 1024,
                'timeout': 20,
                'description': 'Quality (4-8s)',
                'top_p': 0.9,
                'top_k': 40,
                'repeat_penalty': 1.1,
            }
        }
        self.current_profile = 'speed'
    
    async def initialize(self) -> bool:
        """Fixed initialization with proper error handling"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not available - install with: pip install aiohttp==3.8.6"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ {self.last_error}")
            return False
        
        try:
            # Create session with fixed connector settings
            await self._create_session()
            
            # Test connection
            if not await self._test_connection():
                self.last_error = "Cannot connect to Ollama service"
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ❌ Connection failed")
                return False
            
            # Load best model
            if not await self._load_best_model():
                self.last_error = "No working models found"
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ❌ Model loading failed")
                return False
            
            # Start keep-alive
            await self._start_keep_alive()
            
            self.available = True
            self.consecutive_errors = 0
            self.last_successful_time = time.time()
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ✅ Fixed LLM ready: {self.current_model}")
                print(f"[OLLAMA] 🚀 Profile: {self.current_profile}")
            
            return True
            
        except Exception as e:
            self.last_error = f"Initialization error: {str(e)}"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Init failed: {e}")
            return False
    
    async def _create_session(self):
        """Create aiohttp session with proper configuration for Pi 5"""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available")
        
        # Create timeout with reasonable values for Pi 5
        timeout = aiohttp.ClientTimeout(
            total=20,
            connect=5,
            sock_read=15
        )
        
        # Create connector with Pi 5 optimized settings
        # Removed tcp_nodelay to avoid compatibility issues
        connector_kwargs = {
            'limit': 2,
            'limit_per_host': 2,
            'enable_cleanup_closed': True,
            'use_dns_cache': True,
            'keepalive_timeout': 300,
            'force_close': False
        }
        
        self.connector = aiohttp.TCPConnector(**connector_kwargs)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=self.connector,
            headers={
                'Connection': 'keep-alive',
                'Content-Type': 'application/json'
            }
        )
    
    async def _test_connection(self) -> bool:
        """Test connection to Ollama service"""
        try:
            async with self.session.get(
                f"{self.host}/api/version",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ✅ Connected - Ollama v{data.get('version', 'unknown')}")
                    return True
                return False
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Connection test failed: {e}")
            return False
    
    async def _get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models from Ollama"""
        try:
            async with self.session.get(
                f"{self.host}/api/tags",
                timeout=aiohttp.ClientTimeout(total=8)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    if self.settings.debug_mode:
                        model_names = [m.get('name', 'unknown') for m in models]
                        print(f"[OLLAMA] Available models: {model_names}")
                    return models
                return []
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Failed to list models: {e}")
            return []
    
    async def _load_best_model(self) -> bool:
        """Find and load the best working model"""
        available_models = await self._get_available_models()
        
        if not available_models:
            return False
        
        model_names = [model.get('name', '') for model in available_models]
        
        # Try preferred models in order
        for preferred in self.preferred_models:
            for model_name in model_names:
                if preferred == model_name or preferred in model_name:
                    if await self._test_model(model_name):
                        self.current_model = model_name
                        self.model_loaded = True
                        if self.settings.debug_mode:
                            print(f"[OLLAMA] ✅ Loaded working model: {model_name}")
                        return True
        
        # Try any available model
        for model in available_models:
            model_name = model.get('name', '')
            if model_name and await self._test_model(model_name):
                self.current_model = model_name
                self.model_loaded = True
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ✅ Loaded fallback model: {model_name}")
                return True
        
        return False
    
    async def _test_model(self, model_name: str) -> bool:
        """Quick test of a model"""
        try:
            payload = {
                "model": model_name,
                "prompt": "Hi",
                "options": {
                    "num_predict": 5,
                    "temperature": 0.1,
                    "num_ctx": 128,
                    "num_thread": 4
                },
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '').strip()
                    
                    if response_text and len(response_text) > 0:
                        if self.settings.debug_mode:
                            print(f"[OLLAMA] ✅ Model test passed: {model_name}")
                        return True
                
                return False
                
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model test failed for {model_name}: {e}")
            return False
    
    async def _start_keep_alive(self):
        """Start background keep-alive task"""
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
        
        self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())
    
    async def _keep_alive_loop(self):
        """Background task to keep model loaded"""
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
                    timeout=aiohttp.ClientTimeout(total=5)
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
                print(f"[OLLAMA] ⚡ Profile: {profile} - {self.profiles[profile]['description']}")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate response with fixed error handling"""
        if not self.available or not self.model_loaded:
            return "Offline model unavailable. Please check Ollama service."
        
        start_time = time.time()
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Build simple prompt for speed
                prompt = f"User: {query}\nAssistant: "
                
                # Get profile settings
                profile = self.profiles[self.current_profile]
                options = {
                    "num_predict": profile['num_predict'],
                    "temperature": profile['temperature'],
                    "num_ctx": profile['num_ctx'],
                    "top_p": profile['top_p'],
                    "top_k": profile['top_k'],
                    "repeat_penalty": profile['repeat_penalty'],
                    "num_thread": 4,
                    "num_gpu": 0
                }
                
                payload = {
                    "model": self.current_model,
                    "prompt": prompt,
                    "options": options,
                    "stream": False,
                    "keep_alive": self.keep_alive_duration
                }
                
                timeout_val = profile['timeout']
                
                async with self.session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_val)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data.get('response', '').strip()
                        
                        elapsed = time.time() - start_time
                        
                        if response_text and len(response_text) > 1:
                            self._update_performance_stats(elapsed, True)
                            
                            if self.settings.debug_mode:
                                print(f"[OLLAMA] ✅ Response in {elapsed:.2f}s")
                            
                            self.consecutive_errors = 0
                            return response_text
                        else:
                            if self.settings.debug_mode:
                                print(f"[OLLAMA] ⚠️ Empty response (attempt {attempt + 1})")
                    else:
                        if self.settings.debug_mode:
                            print(f"[OLLAMA] ❌ HTTP {response.status} (attempt {attempt + 1})")
                        
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ❌ Timeout after {elapsed:.1f}s (attempt {attempt + 1})")
                
            except Exception as e:
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ❌ Error (attempt {attempt + 1}): {e}")
        
        # All attempts failed
        elapsed = time.time() - start_time
        self._update_performance_stats(elapsed, False)
        self.consecutive_errors += 1
        
        return "I'm having trouble responding right now. Please try again."
    
    async def generate_response_stream(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        if not self.available or not self.model_loaded:
            yield "Offline model unavailable. Please check Ollama service."
            return
        
        start_time = time.time()
        
        try:
            prompt = f"User: {query}\nAssistant: "
            profile = self.profiles[self.current_profile]
            
            options = {
                "num_predict": profile['num_predict'],
                "temperature": profile['temperature'],
                "num_ctx": profile['num_ctx'],
                "top_p": profile['top_p'],
                "top_k": profile['top_k'],
                "repeat_penalty": profile['repeat_penalty'],
                "num_thread": 4,
                "num_gpu": 0
            }
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": options,
                "stream": True,
                "keep_alive": self.keep_alive_duration
            }
            
            timeout_val = profile['timeout'] + 5
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_val)
            ) as response:
                if response.status == 200:
                    first_chunk = True
                    response_received = False
                    full_response = ""
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                
                                if 'response' in data and data['response']:
                                    chunk = data['response']
                                    full_response += chunk
                                    yield chunk
                                    response_received = True
                                    
                                    if first_chunk and self.settings.debug_mode:
                                        first_chunk_time = time.time() - start_time
                                        print(f"[OLLAMA] ⚡ First chunk in {first_chunk_time:.2f}s")
                                        first_chunk = False
                                
                                if data.get('done', False):
                                    elapsed = time.time() - start_time
                                    self._update_performance_stats(elapsed, True)
                                    self.consecutive_errors = 0
                                    
                                    if self.settings.debug_mode:
                                        print(f"[OLLAMA] ✅ Streaming complete in {elapsed:.2f}s")
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    if not response_received:
                        yield "No response generated."
                        
                else:
                    yield f"Streaming error: HTTP {response.status}"
                    
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            yield f"Timed out after {elapsed:.1f}s."
            
        except Exception as e:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            yield f"Error: {str(e)[:50]}"
    
    def _validate_response(self, response: str, query: str) -> bool:
        """Validate response quality"""
        if not response or len(response.strip()) < 2:
            return False
        
        # Check for common error patterns
        error_patterns = [
            'model error',
            'connection error',
            'timeout',
            'failed to',
            'unable to'
        ]
        
        response_lower = response.lower()
        if any(pattern in response_lower for pattern in error_patterns):
            return False
        
        # Check for repetitive patterns (sign of poor generation)
        words = response.split()
        if len(words) > 5:
            # Check for excessive repetition
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # If any word appears more than 30% of the time, it's likely repetitive
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.3:
                return False
        
        return True
    
    def _update_performance_stats(self, response_time: float, success: bool):
        """Update performance statistics"""
        self.request_count += 1
        self.total_time += response_time
        
        if not success:
            self.error_count += 1
        else:
            self.last_successful_time = time.time()
        
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
        
        return {
            'available': self.available,
            'model_loaded': self.model_loaded,
            'current_model': self.current_model,
            'performance_profile': self.current_profile,
            'profile_description': self.profiles[self.current_profile]['description'],
            'optimization_level': 'fixed_compatibility',
            'stats': {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'avg_response_time': f"{avg_time:.2f}s",
                'recent_avg_time': f"{recent_avg:.2f}s",
                'success_rate': f"{success_rate:.1f}%",
                'target_time': f"<{self.profiles[self.current_profile]['timeout']}s"
            },
            'last_error': self.last_error,
            'preferred_models': self.preferred_models
        }
    
    async def close(self):
        """Clean close with proper resource cleanup"""
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
            if self.request_count > 0:
                avg_time = self.total_time / self.request_count
                success_rate = ((self.request_count - self.error_count) / self.request_count) * 100
                print(f"[OLLAMA] 📊 Session stats: {self.request_count} requests, {avg_time:.2f}s avg, {success_rate:.1f}% success")
            print("[OLLAMA] 🔌 Fixed connection closed")

# Maintain compatibility
OptimizedOfflineLLM = LightningOfflineLLM
