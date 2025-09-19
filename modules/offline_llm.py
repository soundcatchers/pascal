"""
Pascal AI Assistant - ULTRA-FAST Offline LLM (Complete Rewrite)
Optimized for sub-2 second responses with robust error handling
FOCUS: Speed, reliability, and coherent responses
"""

import asyncio
import json
import time
import re
from typing import Optional, AsyncGenerator, Dict, Any, List

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

class LightningOfflineLLM:
    """Ultra-fast offline LLM with robust error handling and optimized prompts"""
    
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
        self.keep_alive_interval = 30  # seconds
        
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
        self.nonsense_patterns = [
            r'^\d+$',  # Just numbers
            r'^\d+\s*[\+\-\*\/]\s*\d+$',  # Math expressions
            r'^\d+:\d+\s*(AM|PM)?$',  # Time formats
            r'^[A-Za-z]$',  # Single letters
        ]
        
        # Model preferences - prioritize working models
        self.preferred_models = [
            'nemotron-mini:4b-instruct-q4_K_M',
            'nemotron-fast',
            'qwen2.5:3b',
            'phi3:mini',
            'llama3.2:3b',
            'gemma2:2b',
        ]
        
        # Optimized generation settings by profile
        self.profiles = {
            'speed': {
                'num_predict': 40,
                'temperature': 0.1,
                'num_ctx': 256,
                'timeout': 6,
                'description': 'Ultra-fast (<2s)',
                'top_p': 0.7,
                'top_k': 15,
                'repeat_penalty': 1.02,
            },
            'balanced': {
                'num_predict': 80,
                'temperature': 0.3,
                'num_ctx': 512,
                'timeout': 10,
                'description': 'Balanced (2-4s)',
                'top_p': 0.8,
                'top_k': 25,
                'repeat_penalty': 1.05,
            },
            'quality': {
                'num_predict': 150,
                'temperature': 0.7,
                'num_ctx': 1024,
                'timeout': 20,
                'description': 'Quality (4-8s)',
                'top_p': 0.9,
                'top_k': 40,
                'repeat_penalty': 1.1,
            }
        }
        self.current_profile = 'speed'  # Default to fastest
        
        # Smart prompt templates for different query types
        self.prompt_templates = {
            'greeting': {
                'template': "User: {query}\nAssistant: ",
                'system': "You are Pascal, a helpful AI assistant. Give a brief, friendly greeting."
            },
            'simple_question': {
                'template': "User: {query}\nAssistant: ",
                'system': "You are Pascal, a helpful AI assistant. Give a direct, concise answer."
            },
            'calculation': {
                'template': "User: {query}\nAssistant: ",
                'system': "You are Pascal. Solve this calculation and explain briefly."
            },
            'explanation': {
                'template': "System: You are Pascal, a helpful AI assistant. Explain things clearly and concisely.\n\nUser: {query}\nAssistant: ",
                'system': None  # Already in template
            },
            'conversation': {
                'template': "System: You are Pascal, a helpful and friendly AI assistant.\n\nUser: {query}\nAssistant: ",
                'system': None
            }
        }
    
    async def initialize(self) -> bool:
        """Ultra-fast initialization with robust error handling"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not available - install with: pip install aiohttp"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ {self.last_error}")
            return False
        
        try:
            # Create optimized HTTP session
            await self._create_optimized_session()
            
            # Test connection with short timeout
            if not await self._test_connection_fast():
                self.last_error = "Cannot connect to Ollama service"
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ❌ Connection failed")
                return False
            
            # Load and test best available model
            if not await self._load_best_model():
                self.last_error = "No working models found"
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ❌ Model loading failed")
                return False
            
            # Start keep-alive task
            await self._start_keep_alive()
            
            self.available = True
            self.consecutive_errors = 0
            self.last_successful_time = time.time()
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ✅ Ultra-fast LLM ready: {self.current_model}")
                print(f"[OLLAMA] 🚀 Profile: {self.current_profile} ({self.profiles[self.current_profile]['description']})")
            
            return True
            
        except Exception as e:
            self.last_error = f"Initialization error: {str(e)}"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Init failed: {e}")
            return False
    
    async def _create_optimized_session(self):
        """Create highly optimized HTTP session"""
        # Ultra-fast timeouts
        timeout = aiohttp.ClientTimeout(
            total=15,
            connect=3,
            sock_read=12
        )
        
        # Optimized connector
        self.connector = aiohttp.TCPConnector(
            limit=1,
            limit_per_host=1,
            enable_cleanup_closed=True,
            use_dns_cache=True,
            keepalive_timeout=600,  # 10 minutes
            force_close=False,
            tcp_nodelay=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=self.connector,
            headers={
                'Connection': 'keep-alive',
                'Content-Type': 'application/json'
            }
        )
    
    async def _test_connection_fast(self) -> bool:
        """Ultra-fast connection test"""
        try:
            async with self.session.get(
                f"{self.host}/api/version",
                timeout=aiohttp.ClientTimeout(total=3)
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
        """Get available models with error handling"""
        try:
            async with self.session.get(
                f"{self.host}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5)
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
                    if await self._test_model_fast(model_name):
                        self.current_model = model_name
                        self.model_loaded = True
                        if self.settings.debug_mode:
                            print(f"[OLLAMA] ✅ Loaded working model: {model_name}")
                        return True
        
        # Try any available model
        for model in available_models:
            model_name = model.get('name', '')
            if model_name and await self._test_model_fast(model_name):
                self.current_model = model_name
                self.model_loaded = True
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ✅ Loaded fallback model: {model_name}")
                return True
        
        return False
    
    async def _test_model_fast(self, model_name: str) -> bool:
        """Quick model test with timeout"""
        try:
            profile = self.profiles[self.current_profile]
            
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
                timeout=aiohttp.ClientTimeout(total=8)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '').strip()
                    
                    # Check if response is reasonable
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
                
                # Send minimal keep-alive request
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
                    pass  # Just keeping connection alive
                    
            except asyncio.CancelledError:
                break
            except Exception:
                # Keep-alive failure is not critical
                pass
    
    def _classify_query(self, query: str) -> str:
        """Classify query to choose optimal prompt template"""
        query_lower = query.lower().strip()
        
        # Greeting patterns
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in query_lower for greeting in greetings):
            return 'greeting'
        
        # Simple questions
        simple_patterns = ['what is', 'who is', 'where is', 'when is', 'how much', 'how many']
        if any(pattern in query_lower for pattern in simple_patterns) and len(query.split()) <= 6:
            return 'simple_question'
        
        # Calculations
        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', query) or 'calculate' in query_lower:
            return 'calculation'
        
        # Explanation requests
        explain_patterns = ['explain', 'describe', 'tell me about', 'what does', 'how does']
        if any(pattern in query_lower for pattern in explain_patterns):
            return 'explanation'
        
        # Default to conversation
        return 'conversation'
    
    def _build_optimized_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Build optimized prompt based on query classification"""
        query_type = self._classify_query(query)
        template_info = self.prompt_templates[query_type]
        
        # For speed profile, use minimal context
        if self.current_profile == 'speed':
            return template_info['template'].format(query=query)
        
        # For other profiles, add context if available
        if template_info['system'] and (personality_context or memory_context):
            context_parts = []
            if personality_context:
                context_parts.append(personality_context[:200])  # Limit context length
            if memory_context:
                context_parts.append(memory_context[-100:])  # Use recent memory only
            
            if context_parts:
                full_system = template_info['system'] + " " + " ".join(context_parts)
                return f"System: {full_system}\n\nUser: {query}\nAssistant: "
        
        return template_info['template'].format(query=query)
    
    def _validate_response(self, response: str, query: str) -> bool:
        """Validate response quality"""
        if not response or len(response.strip()) < 2:
            return False
        
        response_lower = response.lower().strip()
        
        # Check for nonsense patterns
        for pattern in self.nonsense_patterns:
            if re.match(pattern, response_lower):
                return False
        
        # Check if response seems unrelated to query
        query_words = set(query.lower().split())
        response_words = set(response_lower.split())
        
        # For greetings, check for appropriate response
        if any(word in query_words for word in ['hello', 'hi', 'hey']):
            greeting_words = ['hello', 'hi', 'hey', 'good', 'morning', 'help', 'assist']
            if not any(word in response_words for word in greeting_words):
                return False
        
        return True
    
    def _get_cached_response(self, query: str) -> Optional[str]:
        """Get cached response for similar queries"""
        query_key = query.lower().strip()
        
        # Exact match
        if query_key in self.response_cache:
            return self.response_cache[query_key]
        
        # Similar greeting patterns
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning']
        query_lower = query_key.lower()
        
        for pattern in greeting_patterns:
            if pattern in query_lower:
                for cached_query, cached_response in self.response_cache.items():
                    if pattern in cached_query:
                        return cached_response
        
        return None
    
    def _cache_response(self, query: str, response: str):
        """Cache response for future use"""
        if len(self.response_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[query.lower().strip()] = response
    
    def set_performance_profile(self, profile: str):
        """Set performance profile with immediate effect"""
        if profile in self.profiles:
            self.current_profile = profile
            if self.settings.debug_mode:
                print(f"[OLLAMA] ⚡ Profile: {profile} - {self.profiles[profile]['description']}")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate response with quality validation and caching"""
        if not self.available or not self.model_loaded:
            return "Offline model unavailable. Please check Ollama service."
        
        # Check cache first for common queries
        cached_response = self._get_cached_response(query)
        if cached_response and self.current_profile == 'speed':
            if self.settings.debug_mode:
                print(f"[OLLAMA] ⚡ Cache hit: {query[:30]}...")
            return cached_response
        
        start_time = time.time()
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Build optimized prompt
                prompt = self._build_optimized_prompt(query, personality_context, memory_context)
                
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
                        
                        # Validate response quality
                        if self._validate_response(response_text, query):
                            # Update performance stats
                            self._update_performance_stats(elapsed, True)
                            
                            # Cache good responses
                            if len(response_text) < 500:  # Don't cache very long responses
                                self._cache_response(query, response_text)
                            
                            if self.settings.debug_mode:
                                eval_count = data.get('eval_count', 0)
                                eval_duration = data.get('eval_duration', 1)
                                tokens_per_sec = eval_count / max(eval_duration / 1e9, 0.001) if eval_count > 0 else 0
                                print(f"[OLLAMA] ✅ Response in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
                            
                            self.consecutive_errors = 0
                            return response_text
                        else:
                            if self.settings.debug_mode:
                                print(f"[OLLAMA] ⚠️ Invalid response (attempt {attempt + 1}): '{response_text[:50]}...'")
                            if attempt < max_retries - 1:
                                continue  # Retry with different approach
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
        
        # Return helpful error message
        if self.consecutive_errors > 3:
            return "I'm having persistent issues. Please check the Ollama service status."
        else:
            return f"I'm having trouble responding right now. Please try rephrasing your question."
    
    async def generate_response_stream(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response with quality validation"""
        if not self.available or not self.model_loaded:
            yield "Offline model unavailable. Please check Ollama service."
            return
        
        start_time = time.time()
        
        try:
            prompt = self._build_optimized_prompt(query, personality_context, memory_context)
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
            
            timeout_val = profile['timeout'] + 5  # Extra time for streaming
            
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
                                    
                                    # Validate full response
                                    if self._validate_response(full_response, query):
                                        self._update_performance_stats(elapsed, True)
                                        self._cache_response(query, full_response)
                                        self.consecutive_errors = 0
                                    else:
                                        self._update_performance_stats(elapsed, False)
                                    
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
            yield f"\n\nTimed out after {elapsed:.1f}s. Try 'speed' profile for faster responses."
            
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
        else:
            self.last_successful_time = time.time()
        
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
            'profile_description': self.profiles[self.current_profile]['description'],
            'optimization_level': 'ultra_fast_rewrite',
            'stats': {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'avg_response_time': f"{avg_time:.2f}s",
                'recent_avg_time': f"{recent_avg:.2f}s",
                'success_rate': f"{success_rate:.1f}%",
                'target_time': f"<{self.profiles[self.current_profile]['timeout']}s",
                'speed_grade': self._calculate_speed_grade(recent_avg),
                'cache_size': len(self.response_cache)
            },
            'current_settings': {
                'num_predict': self.profiles[self.current_profile]['num_predict'],
                'num_ctx': self.profiles[self.current_profile]['num_ctx'],
                'temperature': self.profiles[self.current_profile]['temperature'],
                'profile_optimizations': self.profiles[self.current_profile]
            },
            'last_error': self.last_error,
            'preferred_models': self.preferred_models,
            'quality_features': [
                'Response validation',
                'Intelligent caching',
                'Query classification',
                'Retry with validation',
                'Nonsense detection',
                'Context optimization'
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
        # Cancel keep-alive task
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
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
            print("[OLLAMA] 🔌 Ultra-fast connection closed")

# Maintain compatibility
OptimizedOfflineLLM = LightningOfflineLLM
