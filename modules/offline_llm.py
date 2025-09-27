"""
Pascal AI Assistant - SPEED-OPTIMIZED Offline LLM Module
Aggressive optimizations for sub-2 second responses on Pi 5
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
    """Speed-optimized offline LLM with aggressive performance tuning for Pi 5"""
    
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
        self.keep_alive_interval = 20  # Reduced from 30s for better model retention
        
        # Performance tracking
        self.request_count = 0
        self.total_time = 0.0
        self.error_count = 0
        self.last_error = None
        self.response_times = []
        self.consecutive_errors = 0
        self.last_successful_time = time.time()
        
        # Speed optimization flags
        self.speed_mode_active = True
        self.use_minimal_context = True
        self.aggressive_timeouts = True
        
        # Model preferences - prioritize speed
        self.preferred_models = [
            'nemotron-fast',                    # Custom optimized model
            'nemotron-mini:4b-instruct-q4_K_M', # Standard model
            'qwen2.5:3b',                       # Fast alternative
            'phi3:mini',                        # Compact option
            'gemma2:2b',                        # Ultra-compact
        ]
        
        # SPEED-OPTIMIZED performance profiles
        self.profiles = {
            'speed': {
                'num_predict': 40,              # Very short responses
                'temperature': 0.1,             # Low temperature for speed
                'num_ctx': 128,                 # Minimal context
                'timeout': 6,                   # Aggressive timeout
                'description': 'Ultra-fast (<1.5s)',
                'top_p': 0.6,                   # Reduced for speed
                'top_k': 10,                    # Minimal for speed
                'repeat_penalty': 1.02,         # Minimal penalty
                'num_thread': 4,                # All Pi 5 cores
                'num_gpu': 0,                   # CPU only
                'target_time': 1.5
            },
            'balanced': {
                'num_predict': 80,              # Short responses
                'temperature': 0.3,             # Moderate temperature
                'num_ctx': 256,                 # Limited context
                'timeout': 10,                  # Fast timeout
                'description': 'Fast (1.5-3s)',
                'top_p': 0.7,                   # Moderate
                'top_k': 20,                    # Reasonable
                'repeat_penalty': 1.05,         # Light penalty
                'num_thread': 4,
                'num_gpu': 0,
                'target_time': 2.5
            },
            'quality': {
                'num_predict': 150,             # Longer responses
                'temperature': 0.7,             # Higher temperature
                'num_ctx': 512,                 # More context
                'timeout': 15,                  # Longer timeout
                'description': 'Quality (3-6s)',
                'top_p': 0.8,                   # More variety
                'top_k': 30,                    # More options
                'repeat_penalty': 1.1,          # Normal penalty
                'num_thread': 4,
                'num_gpu': 0,
                'target_time': 4.0
            }
        }
        self.current_profile = 'speed'  # Default to fastest
    
    async def initialize(self) -> bool:
        """Speed-optimized initialization"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not available - install with: pip install aiohttp==3.9.5"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ {self.last_error}")
            return False
        
        try:
            # Create optimized session
            await self._create_speed_optimized_session()
            
            # Quick connection test with minimal timeout
            if not await self._test_connection_fast():
                self.last_error = "Cannot connect to Ollama service"
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ❌ Connection failed")
                return False
            
            # Load fastest available model
            if not await self._load_fastest_model():
                self.last_error = "No working models found"
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ❌ Model loading failed")
                return False
            
            # Start aggressive keep-alive
            await self._start_aggressive_keep_alive()
            
            self.available = True
            self.consecutive_errors = 0
            self.last_successful_time = time.time()
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ✅ Speed-optimized LLM ready: {self.current_model}")
                print(f"[OLLAMA] ⚡ Profile: {self.current_profile} ({self.profiles[self.current_profile]['description']})")
            
            return True
            
        except Exception as e:
            self.last_error = f"Initialization error: {str(e)}"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Init failed: {e}")
            return False
    
    async def _create_speed_optimized_session(self):
        """Create session optimized for Pi 5 speed"""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available")
        
        # Ultra-aggressive timeouts for speed
        timeout = aiohttp.ClientTimeout(
            total=12,        # Shorter total timeout
            connect=3,       # Fast connection
            sock_read=8      # Fast read timeout
        )
        
        # Speed-optimized connector for Pi 5
        connector_kwargs = {
            'limit': 1,                    # Single connection for Pi 5
            'limit_per_host': 1,           # One connection per host
            'enable_cleanup_closed': True,
            'use_dns_cache': True,
            'keepalive_timeout': 600,      # Long keepalive
            'force_close': False,
            'ttl_dns_cache': 300          # DNS caching
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
                        print(f"[OLLAMA] ⚡ Connected - Ollama v{data.get('version', 'unknown')}")
                    return True
                return False
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Connection test failed: {e}")
            return False
    
    async def _get_available_models_fast(self) -> List[Dict[str, Any]]:
        """Fast model listing"""
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
    
    async def _load_fastest_model(self) -> bool:
        """Load the fastest available model"""
        available_models = await self._get_available_models_fast()
        
        if not available_models:
            return False
        
        model_names = [model.get('name', '') for model in available_models]
        
        # Try preferred models in speed order
        for preferred in self.preferred_models:
            for model_name in model_names:
                if preferred == model_name or preferred in model_name:
                    if await self._test_model_speed(model_name):
                        self.current_model = model_name
                        self.model_loaded = True
                        if self.settings.debug_mode:
                            print(f"[OLLAMA] ⚡ Loaded speed-optimized model: {model_name}")
                        return True
        
        # Try any available model if preferred ones fail
        for model in available_models:
            model_name = model.get('name', '')
            if model_name and await self._test_model_speed(model_name):
                self.current_model = model_name
                self.model_loaded = True
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ⚡ Loaded fallback model: {model_name}")
                return True
        
        return False
    
    async def _test_model_speed(self, model_name: str) -> bool:
        """Ultra-fast model test with minimal payload"""
        try:
            # Minimal test payload for speed
            payload = {
                "model": model_name,
                "prompt": "Hi",
                "options": {
                    "num_predict": 3,        # Minimal response
                    "temperature": 0.1,      # Fast generation
                    "num_ctx": 64,           # Minimal context
                    "num_thread": 4,         # All Pi 5 cores
                    "top_p": 0.5,            # Fast sampling
                    "top_k": 5               # Minimal choices
                },
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=6)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '').strip()
                    
                    if response_text and len(response_text) > 0:
                        if self.settings.debug_mode:
                            print(f"[OLLAMA] ⚡ Model test passed: {model_name}")
                        return True
                
                return False
                
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model test failed for {model_name}: {e}")
            return False
    
    async def _start_aggressive_keep_alive(self):
        """Start aggressive keep-alive to maintain model in memory"""
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
        
        self.keep_alive_task = asyncio.create_task(self._aggressive_keep_alive_loop())
    
    async def _aggressive_keep_alive_loop(self):
        """Aggressive keep-alive loop for speed"""
        while self.model_loaded and self.current_model:
            try:
                await asyncio.sleep(self.keep_alive_interval)
                
                # Ultra-minimal keep-alive payload
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
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    # Don't care about response, just keeping alive
                    pass
                    
            except asyncio.CancelledError:
                break
            except Exception:
                # Ignore keep-alive errors
                pass
    
    def set_performance_profile(self, profile: str):
        """Set performance profile with speed focus"""
        if profile in self.profiles:
            self.current_profile = profile
            if self.settings.debug_mode:
                target_time = self.profiles[profile]['target_time']
                print(f"[OLLAMA] ⚡ Profile: {profile} - Target: <{target_time}s")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Speed-optimized response generation"""
        if not self.available or not self.model_loaded:
            return "Offline model unavailable. Please check Ollama service."
        
        start_time = time.time()
        max_retries = 1  # Reduced retries for speed
        
        for attempt in range(max_retries):
            try:
                # Build minimal prompt for maximum speed
                if self.use_minimal_context or len(query.split()) <= 10:
                    # Ultra-minimal prompt for speed
                    prompt = query
                else:
                    # Standard prompt
                    prompt = f"User: {query}\nAssistant: "
                
                # Get aggressive profile settings
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
                        
                        if response_text and len(response_text) > 1:
                            self._update_performance_stats(elapsed, True)
                            
                            if self.settings.debug_mode:
                                target = profile['target_time']
                                status = "⚡" if elapsed < target else "⚠️" if elapsed < target * 1.5 else "❌"
                                print(f"[OLLAMA] {status} Response in {elapsed:.2f}s (target: <{target}s)")
                            
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
        
        return "I'm having trouble responding quickly. Please try again."
    
    async def generate_response_stream(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Speed-optimized streaming response"""
        if not self.available or not self.model_loaded:
            yield "Offline model unavailable. Please check Ollama service."
            return
        
        start_time = time.time()
        
        try:
            # Minimal prompt for streaming speed
            if self.use_minimal_context or len(query.split()) <= 10:
                prompt = query
            else:
                prompt = f"User: {query}\nAssistant: "
            
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
            
            timeout_val = profile['timeout'] + 3  # Slight buffer for streaming
            
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
                                        target = profile['target_time'] / 3  # Expect first chunk in 1/3 of target time
                                        status = "⚡" if first_chunk_time < target else "⚠️"
                                        print(f"[OLLAMA] {status} First chunk in {first_chunk_time:.2f}s")
                                        first_chunk = False
                                
                                if data.get('done', False):
                                    elapsed = time.time() - start_time
                                    self._update_performance_stats(elapsed, True)
                                    self.consecutive_errors = 0
                                    
                                    if self.settings.debug_mode:
                                        target = profile['target_time']
                                        status = "⚡" if elapsed < target else "⚠️" if elapsed < target * 1.5 else "❌"
                                        print(f"[OLLAMA] {status} Streaming complete in {elapsed:.2f}s")
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
            yield f"Response timed out after {elapsed:.1f}s."
            
        except Exception as e:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            yield f"Error: {str(e)[:50]}"
    
    def _validate_response(self, response: str, query: str) -> bool:
        """Fast response validation"""
        if not response or len(response.strip()) < 1:
            return False
        
        # Quick error pattern check
        error_patterns = ['model error', 'connection error', 'timeout', 'failed to']
        response_lower = response.lower()
        if any(pattern in response_lower for pattern in error_patterns):
            return False
        
        # Quick repetition check (simplified for speed)
        if len(response) > 50:
            words = response.split()
            if len(words) > 10:
                # Check if any word appears more than 40% of the time
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                max_count = max(word_counts.values())
                if max_count > len(words) * 0.4:
                    return False
        
        return True
    
    def _update_performance_stats(self, response_time: float, success: bool):
        """Update performance statistics with speed focus"""
        self.request_count += 1
        self.total_time += response_time
        
        if not success:
            self.error_count += 1
        else:
            self.last_successful_time = time.time()
        
        self.response_times.append(response_time)
        if len(self.response_times) > 10:  # Keep only recent times
            self.response_times = self.response_times[-10:]
        
        # Log speed issues
        if success and response_time > self.profiles[self.current_profile]['target_time'] * 2:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ⚠️ Slow response: {response_time:.2f}s (target: {self.profiles[self.current_profile]['target_time']}s)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status with speed metrics"""
        avg_time = self.total_time / max(self.request_count, 1)
        success_rate = ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100
        
        recent_avg = 0
        if self.response_times:
            recent_avg = sum(self.response_times[-5:]) / len(self.response_times[-5:])
        
        # Speed performance analysis
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
            'optimization_level': 'speed_optimized_pi5',
            'speed_features': [
                'Aggressive timeouts',
                'Minimal context windows',
                'Ultra-fast model parameters',
                'Optimized connection pooling',
                'Pi 5 CPU optimization',
                'Aggressive keep-alive'
            ],
            'stats': {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'avg_response_time': f"{avg_time:.2f}s",
                'recent_avg_time': f"{recent_avg:.2f}s",
                'target_response_time': f"{target_time:.1f}s",
                'success_rate': f"{success_rate:.1f}%",
                'fast_responses': f"{fast_responses}/{len(self.response_times)}",
                'speed_grade': speed_grade
            },
            'speed_optimizations': {
                'minimal_context': self.use_minimal_context,
                'aggressive_timeouts': self.aggressive_timeouts,
                'speed_mode_active': self.speed_mode_active,
                'keep_alive_interval': f"{self.keep_alive_interval}s"
            },
            'last_error': self.last_error,
            'preferred_models': self.preferred_models
        }
    
    def get_speed_recommendations(self) -> List[str]:
        """Get speed optimization recommendations"""
        recommendations = []
        
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            target_time = self.profiles[self.current_profile]['target_time']
            
            if avg_time > target_time * 2:
                recommendations.append(f"Average response time ({avg_time:.2f}s) is much slower than target ({target_time}s)")
                recommendations.append("Consider switching to 'speed' profile or check system resources")
            
            slow_responses = sum(1 for t in self.response_times if t > target_time * 1.5)
            if slow_responses > len(self.response_times) * 0.5:
                recommendations.append("Many responses are slower than target - check Pi 5 temperature and cooling")
        
        if self.consecutive_errors > 3:
            recommendations.append("Multiple consecutive errors - check Ollama service health")
        
        if self.current_profile != 'speed':
            recommendations.append("Switch to 'speed' profile for fastest responses")
        
        if not self.current_model or 'nemotron-fast' not in self.current_model:
            recommendations.append("Use optimized 'nemotron-fast' model for best performance")
        
        # System recommendations
        recommendations.append("Ensure Pi 5 has adequate cooling for sustained performance")
        recommendations.append("Use fast NVMe storage for optimal model loading")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def enable_turbo_mode(self):
        """Enable ultra-aggressive speed optimizations"""
        if self.settings.debug_mode:
            print("[OLLAMA] 🚀 Enabling TURBO mode for maximum speed")
        
        # Ultra-aggressive settings
        self.profiles['speed'].update({
            'num_predict': 30,              # Very short responses
            'temperature': 0.05,            # Minimal randomness
            'num_ctx': 64,                  # Minimal context
            'timeout': 4,                   # Very aggressive timeout
            'top_p': 0.5,                   # Minimal diversity
            'top_k': 5,                     # Minimal options
            'target_time': 1.0              # 1 second target
        })
        
        self.use_minimal_context = True
        self.aggressive_timeouts = True
        self.keep_alive_interval = 15  # More frequent keep-alive
        
        if self.current_profile == 'speed':
            if self.settings.debug_mode:
                print("[OLLAMA] ⚡ TURBO mode active - targeting <1s responses")
    
    def disable_turbo_mode(self):
        """Restore normal speed optimizations"""
        if self.settings.debug_mode:
            print("[OLLAMA] 🔄 Restoring normal speed mode")
        
        # Restore normal speed settings
        self.profiles['speed'].update({
            'num_predict': 40,
            'temperature': 0.1,
            'num_ctx': 128,
            'timeout': 6,
            'top_p': 0.6,
            'top_k': 10,
            'target_time': 1.5
        })
        
        self.keep_alive_interval = 20
    
    async def benchmark_speed(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """Run speed benchmark"""
        if not test_queries:
            test_queries = [
                "Hello",
                "Hi there",
                "How are you?",
                "What's 2+2?",
                "Say hello"
            ]
        
        if not self.available:
            return {"error": "LLM not available"}
        
        results = []
        total_time = 0
        
        print(f"[OLLAMA] 🧪 Running speed benchmark with {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            start_time = time.time()
            
            try:
                response = await self.generate_response(query, "", "")
                elapsed = time.time() - start_time
                
                results.append({
                    'query': query,
                    'time': elapsed,
                    'success': True,
                    'response_length': len(response)
                })
                total_time += elapsed
                
                target = self.profiles[self.current_profile]['target_time']
                status = "⚡" if elapsed < target else "⚠️" if elapsed < target * 1.5 else "❌"
                print(f"  {i}/{len(test_queries)}: {status} {elapsed:.2f}s - {query}")
                
            except Exception as e:
                elapsed = time.time() - start_time
                results.append({
                    'query': query,
                    'time': elapsed,
                    'success': False,
                    'error': str(e)
                })
                total_time += elapsed
                print(f"  {i}/{len(test_queries)}: ❌ {elapsed:.2f}s - Error: {str(e)[:30]}")
        
        # Calculate benchmark results
        successful_results = [r for r in results if r['success']]
        if successful_results:
            avg_time = sum(r['time'] for r in successful_results) / len(successful_results)
            min_time = min(r['time'] for r in successful_results)
            max_time = max(r['time'] for r in successful_results)
            success_rate = len(successful_results) / len(results) * 100
            
            target = self.profiles[self.current_profile]['target_time']
            fast_responses = sum(1 for r in successful_results if r['time'] < target)
            
            # Performance grade
            if avg_time < target and success_rate >= 90:
                grade = "A+ (Excellent)"
            elif avg_time < target * 1.5 and success_rate >= 80:
                grade = "A (Very Good)"
            elif avg_time < target * 2 and success_rate >= 70:
                grade = "B (Good)"
            else:
                grade = "C (Needs Optimization)"
            
            benchmark_result = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'success_rate': success_rate,
                'fast_responses': fast_responses,
                'total_queries': len(test_queries),
                'target_time': target,
                'grade': grade,
                'profile': self.current_profile,
                'model': self.current_model,
                'results': results
            }
            
            print(f"[OLLAMA] 📊 Benchmark Results:")
            print(f"  Average time: {avg_time:.2f}s (target: <{target}s)")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Fast responses: {fast_responses}/{len(successful_results)}")
            print(f"  Performance grade: {grade}")
            
            return benchmark_result
        else:
            return {"error": "No successful responses"}
    
    async def close(self):
        """Clean close with performance summary"""
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
                
                print(f"[OLLAMA] 📊 Speed-optimized session stats:")
                print(f"  Requests: {self.request_count}")
                print(f"  Average time: {avg_time:.2f}s (target: <{target}s)")
                print(f"  Success rate: {success_rate:.1f}%")
                
                if self.response_times:
                    fast_responses = sum(1 for t in self.response_times if t < target)
                    print(f"  Fast responses: {fast_responses}/{len(self.response_times)}")
            
            print("[OLLAMA] 🔌 Speed-optimized connection closed")

# Maintain compatibility
OptimizedOfflineLLM = LightningOfflineLLM
