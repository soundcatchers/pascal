"""
Pascal AI Assistant - FIXED Offline LLM Module
Resolves performance issues, model loading problems, and aiohttp compatibility
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
    """FIXED: Speed-optimized offline LLM with improved reliability"""
    
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
        self.keep_alive_interval = 30  # More conservative interval
        
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
            'nemotron-mini:4b-instruct-q4_K_M',  # Primary choice
            'qwen2.5:3b-instruct',               # Good fallback
            'qwen2.5:3b',                        # Alternative
            'phi3:mini',                         # Compact option
            'gemma2:2b',                         # Last resort
        ]
        
        # FIXED: More realistic performance profiles
        self.profiles = {
            'speed': {
                'num_predict': 80,              # Reasonable length
                'temperature': 0.3,             # Balanced creativity
                'num_ctx': 512,                 # Adequate context
                'timeout': 15,                  # More realistic timeout
                'description': 'Fast (2-4s)',
                'top_p': 0.8,                   
                'top_k': 25,                    
                'repeat_penalty': 1.05,         
                'num_thread': 4,                
                'num_gpu': 0,                   
                'target_time': 3.0              # Realistic target
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
        self.current_profile = 'balanced'  # Start with balanced
    
    async def initialize(self) -> bool:
        """FIXED: More robust initialization with better error handling"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not available - install with: pip install aiohttp==3.9.5"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ {self.last_error}")
            return False
        
        try:
            # FIXED: Create more robust session
            await self._create_robust_session()
            
            # FIXED: Better connection test with retries
            if not await self._test_connection_with_retries():
                self.last_error = "Cannot connect to Ollama service after retries"
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ❌ Connection failed after retries")
                return False
            
            # FIXED: More robust model loading
            if not await self._load_best_available_model():
                self.last_error = "No working models found after testing"
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ❌ Model loading failed")
                return False
            
            # Start keep-alive
            await self._start_keep_alive()
            
            self.available = True
            self.consecutive_errors = 0
            self.last_successful_time = time.time()
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ✅ FIXED LLM ready: {self.current_model}")
                print(f"[OLLAMA] ⚡ Profile: {self.current_profile} ({self.profiles[self.current_profile]['description']})")
            
            return True
            
        except Exception as e:
            self.last_error = f"Initialization error: {str(e)}"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Init failed: {e}")
            return False
    
    async def _create_robust_session(self):
        """FIXED: Create more robust HTTP session"""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available")
        
        # FIXED: More conservative timeouts
        timeout = aiohttp.ClientTimeout(
            total=60,        # Longer total timeout
            connect=10,      # Conservative connection timeout
            sock_read=45     # Conservative read timeout
        )
        
        # FIXED: More robust connector settings
        connector_kwargs = {
            'limit': 2,                    # Allow 2 connections
            'limit_per_host': 2,           
            'enable_cleanup_closed': True,
            'use_dns_cache': True,
            'keepalive_timeout': 300,      
            'force_close': False,
            'ttl_dns_cache': 300           
        }
        
        self.connector = aiohttp.TCPConnector(**connector_kwargs)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=self.connector,
            headers={
                'Connection': 'keep-alive',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        )
    
    async def _test_connection_with_retries(self, max_retries: int = 3) -> bool:
        """FIXED: Test connection with retries"""
        for attempt in range(max_retries):
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
            except Exception as e:
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ⚠️ Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    async def _get_available_models_robust(self) -> List[Dict[str, Any]]:
        """FIXED: More robust model listing"""
        try:
            async with self.session.get(
                f"{self.host}/api/tags",
                timeout=aiohttp.ClientTimeout(total=15)
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
    
    async def _load_best_available_model(self) -> bool:
        """FIXED: Better model selection and testing"""
        available_models = await self._get_available_models_robust()
        
        if not available_models:
            if self.settings.debug_mode:
                print("[OLLAMA] ❌ No models available")
            return False
        
        model_names = [model.get('name', '') for model in available_models]
        
        # FIXED: Try preferred models with better testing
        for preferred in self.preferred_models:
            for model_name in model_names:
                if preferred == model_name or preferred in model_name:
                    if await self._test_model_thoroughly(model_name):
                        self.current_model = model_name
                        self.model_loaded = True
                        if self.settings.debug_mode:
                            print(f"[OLLAMA] ⚡ Loaded model: {model_name}")
                        return True
        
        # FIXED: If no preferred models work, try any available model
        for model in available_models:
            model_name = model.get('name', '')
            if model_name and await self._test_model_thoroughly(model_name):
                self.current_model = model_name
                self.model_loaded = True
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ⚡ Loaded fallback model: {model_name}")
                return True
        
        return False
    
    async def _test_model_thoroughly(self, model_name: str) -> bool:
        """FIXED: More thorough model testing"""
        try:
            # FIXED: Better test payload
            payload = {
                "model": model_name,
                "prompt": "Hello! Please respond with just 'Hi' and nothing else.",
                "options": {
                    "num_predict": 10,       # Short response
                    "temperature": 0.1,      # Focused response
                    "num_ctx": 256,          # Minimal context
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
                timeout=aiohttp.ClientTimeout(total=30)  # More time for model loading
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '').strip()
                    
                    # FIXED: Better response validation
                    if response_text and len(response_text) > 0:
                        # Check if response makes sense
                        if not self._is_nonsense_response(response_text):
                            if self.settings.debug_mode:
                                print(f"[OLLAMA] ✅ Model test passed: {model_name}")
                            return True
                
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ⚠️ Model test gave poor response: {model_name}")
                return False
                
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model test failed for {model_name}: {e}")
            return False
    
    def _is_nonsense_response(self, response: str) -> bool:
        """FIXED: Check if response is nonsense"""
        response_lower = response.lower().strip()
        
        # Check for empty or very short responses
        if len(response_lower) < 1:
            return True
        
        # Check for repeated characters (like "aaaaaaa")
        if len(set(response_lower.replace(' ', ''))) <= 2 and len(response_lower) > 5:
            return True
        
        # Check for obviously broken responses
        nonsense_patterns = [
            r'^[\s\n\r]*$',              # Only whitespace
            r'^[^\w\s]+$',               # Only special characters
            r'(.)\1{10,}',               # Repeated character 10+ times
            r'^\s*error\s*$',            # Just "error"
            r'^\s*null\s*$',             # Just "null"
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, response_lower):
                return True
        
        return False
    
    async def _start_keep_alive(self):
        """Start keep-alive task"""
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
        
        self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())
    
    async def _keep_alive_loop(self):
        """Keep-alive loop to maintain model in memory"""
        while self.model_loaded and self.current_model:
            try:
                await asyncio.sleep(self.keep_alive_interval)
                
                # Minimal keep-alive payload
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
                    # Keep-alive sent successfully
                    pass
                    
            except asyncio.CancelledError:
                break
            except Exception:
                # Ignore keep-alive errors
                pass
    
    def set_performance_profile(self, profile: str):
        """Set performance profile"""
        if profile in self.profiles:
            self.current_profile = profile
            if self.settings.debug_mode:
                target_time = self.profiles[profile]['target_time']
                print(f"[OLLAMA] ⚡ Profile: {profile} - Target: <{target_time}s")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """FIXED: More robust response generation"""
        if not self.available or not self.model_loaded:
            return "Offline model unavailable. Please check Ollama service."
        
        start_time = time.time()
        max_retries = 2  # Allow one retry
        
        for attempt in range(max_retries):
            try:
                # FIXED: Build better prompt
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
                        
                        # FIXED: Better response validation
                        if response_text and not self._is_nonsense_response(response_text):
                            self._update_performance_stats(elapsed, True)
                            
                            if self.settings.debug_mode:
                                target = profile['target_time']
                                status = "⚡" if elapsed < target else "✅" if elapsed < target * 1.5 else "⚠️"
                                print(f"[OLLAMA] {status} Response in {elapsed:.2f}s (target: <{target}s)")
                            
                            self.consecutive_errors = 0
                            return response_text
                        else:
                            if self.settings.debug_mode:
                                print(f"[OLLAMA] ⚠️ Poor response quality (attempt {attempt + 1})")
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
    
    def _build_optimized_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """FIXED: Build better prompts"""
        # For simple queries, use minimal prompt
        if len(query.split()) <= 8:
            return f"User: {query}\nAssistant:"
        
        # For longer queries, include context but keep it short
        prompt_parts = []
        
        if personality_context and len(personality_context) < 500:
            # Only use first part of personality context
            prompt_parts.append(personality_context[:300])
        
        if memory_context and len(memory_context) < 300:
            prompt_parts.append(f"Context: {memory_context}")
        
        prompt_parts.append(f"User: {query}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    async def generate_response_stream(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """FIXED: More robust streaming"""
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
            
            timeout_val = profile['timeout'] + 10  # Extra time for streaming
            
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
                                        target = profile['target_time']
                                        status = "⚡" if elapsed < target else "✅" if elapsed < target * 1.5 else "⚠️"
                                        print(f"[OLLAMA] {status} Streaming complete in {elapsed:.2f}s")
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    if not response_received:
                        yield "No response generated - model may be overloaded."
                        
                else:
                    yield f"Model error: HTTP {response.status}"
                    
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            yield f"Response timed out after {elapsed:.1f}s. The model may be slow to load."
            
        except Exception as e:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            yield f"Error generating response: {str(e)[:50]}"
    
    def _update_performance_stats(self, response_time: float, success: bool):
        """Update performance statistics"""
        self.request_count += 1
        self.total_time += response_time
        
        if not success:
            self.error_count += 1
        else:
            self.last_successful_time = time.time()
        
        self.response_times.append(response_time)
        if len(self.response_times) > 20:  # Keep only recent times
            self.response_times = self.response_times[-20:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        avg_time = self.total_time / max(self.request_count, 1)
        success_rate = ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100
        
        recent_avg = 0
        if self.response_times:
            recent_avg = sum(self.response_times[-5:]) / len(self.response_times[-5:])
        
        # Performance analysis
        target_time = self.profiles[self.current_profile]['target_time']
        fast_responses = sum(1 for t in self.response_times if t < target_time)
        speed_grade = "A+" if recent_avg < target_time else "A" if recent_avg < target_time * 1.5 else "B" if recent_avg < target_time * 2 else "C"
        
        return {
            'available': self.available,
            'model_loaded': self.model_loaded,
            'current_model': self.current_model,
            'performance_profile': self.current_profile,
            'profile_description': self.profiles[self.current_profile]['description'],
            'target_time': f"{target_time}s",
            'optimization_level': 'fixed_for_reliability',
            'improvements': [
                'Fixed model loading and testing',
                'Improved timeout handling',
                'Better error recovery',
                'Enhanced response validation',
                'More robust HTTP session',
                'Better keep-alive management'
            ],
            'stats': {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'avg_response_time': f"{avg_time:.2f}s",
                'recent_avg_time': f"{recent_avg:.2f}s",
                'target_response_time': f"{target_time:.1f}s",
                'success_rate': f"{success_rate:.1f}%",
                'fast_responses': f"{fast_responses}/{len(self.response_times)}" if self.response_times else "0/0",
                'speed_grade': speed_grade
            },
            'last_error': self.last_error,
            'preferred_models': self.preferred_models
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
            if self.request_count > 0:
                avg_time = self.total_time / self.request_count
                success_rate = ((self.request_count - self.error_count) / self.request_count) * 100
                target = self.profiles[self.current_profile]['target_time']
                
                print(f"[OLLAMA] 📊 FIXED session stats:")
                print(f"  Requests: {self.request_count}")
                print(f"  Average time: {avg_time:.2f}s (target: <{target}s)")
                print(f"  Success rate: {success_rate:.1f}%")
                
                if self.response_times:
                    fast_responses = sum(1 for t in self.response_times if t < target)
                    print(f"  Fast responses: {fast_responses}/{len(self.response_times)}")
            
            print("[OLLAMA] 🔌 FIXED connection closed")

# Maintain compatibility
OptimizedOfflineLLM = LightningOfflineLLM
