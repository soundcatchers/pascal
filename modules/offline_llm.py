"""
Pascal AI Assistant - High-Performance Offline LLM (Ollama Integration)
OPTIMIZED for Raspberry Pi 5 - Target: 2-4 second responses
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
    """High-performance offline LLM optimized for Pi 5 with sub-4s responses"""
    
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
        
        # Ollama configuration - OPTIMIZED for Pi 5
        self.host = settings.ollama_host
        self.keep_alive_duration = "30m"  # Keep model loaded for 30 minutes
        
        # Performance tracking
        self.request_count = 0
        self.total_time = 0.0
        self.error_count = 0
        self.last_error = None
        self.response_times = []
        
        # OPTIMIZED generation settings for Pi 5 speed
        self.fast_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.05,
            'num_predict': 100,      # Reduced for speed
            'num_ctx': 512,          # Reduced context for Pi 5
            'num_thread': 4,         # Use all Pi 5 cores
            'num_gpu': 0,            # CPU only on Pi
            'stop': ["</s>", "<|end|>", "<|eot_id|>", "Human:", "User:", "\n\nHuman:", "\n\nUser:"]
        }
        
        # Performance profiles - OPTIMIZED
        self.performance_profiles = {
            'speed': {
                'num_predict': 50,       # Very short responses
                'temperature': 0.3,      # More focused
                'num_ctx': 256,          # Minimal context
                'timeout': 5,
                'description': 'Ultra-fast (1-2s)'
            },
            'balanced': {
                'num_predict': 100,      # Balanced length
                'temperature': 0.7,
                'num_ctx': 512,          # Reasonable context
                'timeout': 8,
                'description': 'Balanced (2-4s)'
            },
            'quality': {
                'num_predict': 200,      # Longer responses
                'temperature': 0.8,
                'num_ctx': 1024,         # Full context
                'timeout': 15,
                'description': 'Best quality (4-8s)'
            }
        }
        self.current_profile = 'balanced'  # Start with balanced
        
        # Model preferences - OPTIMIZED order
        self.preferred_models = [
            'nemotron-fast',                    # Our optimized model
            'nemotron-mini:4b-instruct-q4_K_M', # Original model
            'qwen2.5:3b',                       # Fallback 1
            'phi3:mini',                        # Fallback 2
            'llama3.2:3b',                      # Fallback 3
        ]
    
    async def initialize(self) -> bool:
        """Initialize with optimized connection settings"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not available - install: pip install aiohttp"
            return False
        
        try:
            # OPTIMIZED connection settings for Pi 5
            timeout = aiohttp.ClientTimeout(
                total=20,           # Reasonable total timeout
                connect=3,          # Fast connection timeout
                sock_read=15        # Read timeout
            )
            
            # Optimized connector for Pi 5
            self.connector = aiohttp.TCPConnector(
                limit=2,                    # Minimal connection pool
                limit_per_host=2,           # Per-host limit
                force_close=True,           # Always close connections
                enable_cleanup_closed=True, # Cleanup
                keepalive_timeout=30,       # Short keepalive
                use_dns_cache=True          # Cache DNS
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout, 
                connector=self.connector,
                headers={'Connection': 'close'}  # Don't keep connections open
            )
            
            # Test connection with fast timeout
            if not await self._test_connection_fast():
                self.last_error = "Cannot connect to Ollama - check service"
                return False
            
            # Load optimized model
            if await self._load_optimized_model():
                # Start lightweight keep-alive
                self.keep_alive_task = asyncio.create_task(self._keep_alive_lightweight())
                self.available = True
                
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ✅ Optimized model loaded: {self.current_model}")
                    print(f"[OLLAMA] ✅ Profile: {self.current_profile} ({self.performance_profiles[self.current_profile]['description']})")
                    print(f"[OLLAMA] ✅ Context: {self.fast_config['num_ctx']} tokens")
                
                return True
            else:
                self.last_error = "Failed to load optimized model"
                return False
                
        except Exception as e:
            self.last_error = f"Initialization failed: {str(e)}"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Initialization error: {e}")
            return False
    
    async def _test_connection_fast(self) -> bool:
        """Fast connection test with minimal timeout"""
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
    
    async def _get_available_models_fast(self) -> List[str]:
        """Get available models with fast timeout"""
        try:
            async with self.session.get(
                f"{self.host}/api/tags",
                timeout=aiohttp.ClientTimeout(total=8)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    return models
                return []
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Error listing models: {e}")
            return []
    
    async def _load_optimized_model(self) -> bool:
        """Load the best optimized model available"""
        try:
            available_models = await self._get_available_models_fast()
            
            if not available_models:
                if self.settings.debug_mode:
                    print("[OLLAMA] ❌ No models available")
                return False
            
            # Find best model
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
            
            # Load model with optimized settings
            return await self._load_model_optimized(model_to_load)
            
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model loading error: {e}")
            return False
    
    async def _load_model_optimized(self, model_name: str) -> bool:
        """Load model with Pi 5 optimized settings"""
        try:
            if self.settings.debug_mode:
                print(f"[OLLAMA] Loading optimized model: {model_name}")
            
            # Get current profile settings
            profile = self.performance_profiles[self.current_profile]
            config = {**self.fast_config}
            config.update({k: v for k, v in profile.items() 
                          if k not in ['timeout', 'description']})
            
            # Optimized test payload
            payload = {
                "model": model_name,
                "prompt": "Hi",  # Minimal test prompt
                "options": config,
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            timeout_val = profile.get('timeout', 8)
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_val + 5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'response' in data:
                        self.current_model = model_name
                        self.model_loaded = True
                        
                        if self.settings.debug_mode:
                            load_time = data.get('total_duration', 0) / 1e9
                            print(f"[OLLAMA] ✅ Model loaded in {load_time:.2f}s: {model_name}")
                        
                        return True
                    else:
                        if self.settings.debug_mode:
                            print(f"[OLLAMA] ❌ Invalid response: {data}")
                        return False
                else:
                    error_text = await response.text()
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ❌ Load failed: {response.status} - {error_text[:100]}")
                    return False
                    
        except asyncio.TimeoutError:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model load timeout: {model_name}")
            return False
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model load error: {e}")
            return False
    
    async def _keep_alive_lightweight(self):
        """Lightweight keep-alive to prevent model unloading"""
        while self.model_loaded and self.current_model:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                if not self.current_model:
                    break
                
                # Minimal keep-alive request
                payload = {
                    "model": self.current_model,
                    "keep_alive": self.keep_alive_duration,
                    "prompt": "",
                    "stream": False
                }
                
                async with self.session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=8)
                ) as response:
                    if self.settings.debug_mode and response.status != 200:
                        print(f"[OLLAMA] ⚠️ Keep-alive warning: {response.status}")
                        
            except asyncio.CancelledError:
                if self.settings.debug_mode:
                    print("[OLLAMA] Keep-alive cancelled")
                break
            except Exception as e:
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ⚠️ Keep-alive error: {e}")
                # Continue trying
                await asyncio.sleep(60)
    
    def _build_fast_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Build optimized prompt for fast inference"""
        # Keep prompt minimal for speed
        parts = []
        
        # Essential personality context only (truncated)
        if personality_context:
            parts.append(personality_context[:200])
        
        # Recent memory only (truncated)
        if memory_context:
            parts.append(f"Context: {memory_context[:150]}")
        
        # Build based on current model
        if self.current_model and 'nemotron' in self.current_model.lower():
            # Nemotron format - simple and fast
            if parts:
                context = " ".join(parts)
                prompt = f"System: {context}\n\nUser: {query}\nAssistant:"
            else:
                prompt = f"User: {query}\nAssistant:"
        else:
            # Generic format
            if parts:
                context = " ".join(parts)
                prompt = f"{context}\n\nUser: {query}\nAssistant:"
            else:
                prompt = f"User: {query}\nAssistant:"
        
        return prompt
    
    def set_performance_profile(self, profile: str):
        """Set performance profile with immediate effect"""
        if profile in self.performance_profiles:
            self.current_profile = profile
            
            # Update fast_config
            profile_config = self.performance_profiles[profile]
            self.fast_config.update({
                k: v for k, v in profile_config.items() 
                if k not in ['timeout', 'description']
            })
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ✅ Profile: {profile} - {profile_config['description']}")
        else:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Invalid profile: {profile}")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate response with Pi 5 optimization"""
        if not self.available or not self.model_loaded:
            return "Offline model unavailable. Check Ollama service and models."
        
        start_time = time.time()
        
        try:
            prompt = self._build_fast_prompt(query, personality_context, memory_context)
            
            # Use current profile
            profile = self.performance_profiles[self.current_profile]
            config = {**self.fast_config}
            config.update({k: v for k, v in profile.items() 
                          if k not in ['timeout', 'description']})
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": config,
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            timeout_val = profile.get('timeout', 8)
            
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
                        tokens_per_sec = data.get('eval_count', 0) / max(data.get('eval_duration', 1) / 1e9, 0.001)
                        print(f"[OLLAMA] ✅ Response in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
                    
                    return response_text or "I wasn't able to generate a response."
                    
                else:
                    error_text = await response.text()
                    self._update_performance_stats(time.time() - start_time, False)
                    self.last_error = f"HTTP {response.status}"
                    
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ❌ API error: {response.status}")
                    
                    return f"Model error: {response.status}"
                    
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            self.last_error = f"Timeout after {elapsed:.1f}s"
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Timeout after {elapsed:.1f}s")
            
            # Suggest faster profile
            if self.current_profile != 'speed':
                return f"Response timed out. Try 'speed' profile for faster responses."
            else:
                return f"Response timed out even on speed profile. Check Ollama performance."
            
        except Exception as e:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            self.last_error = str(e)
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Error: {e}")
            
            return f"Model error: {str(e)[:50]}"
    
    async def generate_response_stream(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response with Pi 5 optimization"""
        if not self.available or not self.model_loaded:
            yield "Offline model unavailable. Check Ollama service and models."
            return
        
        start_time = time.time()
        
        try:
            prompt = self._build_fast_prompt(query, personality_context, memory_context)
            
            # Use current profile
            profile = self.performance_profiles[self.current_profile]
            config = {**self.fast_config}
            config.update({k: v for k, v in profile.items() 
                          if k not in ['timeout', 'description']})
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": config,
                "stream": True,
                "keep_alive": self.keep_alive_duration
            }
            
            timeout_val = profile.get('timeout', 8)
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_val + 10)  # Extra time for streaming
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
                                    
                                    # Log first chunk timing
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
                    error_text = await response.text()
                    self._update_performance_stats(time.time() - start_time, False)
                    
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ❌ Streaming error: {response.status}")
                    
                    yield f"Streaming error: {response.status}"
                    
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Streaming timeout after {elapsed:.1f}s")
            
            yield f"\n\nStreaming timed out after {elapsed:.1f}s."
            
        except Exception as e:
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed, False)
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Streaming error: {e}")
            
            yield f"Streaming error: {str(e)[:50]}"
    
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
            'host': self.host,
            'keep_alive_duration': self.keep_alive_duration,
            'optimization_level': 'high_performance_pi5',
            'stats': {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'avg_response_time': f"{avg_time:.2f}s",
                'recent_avg_time': f"{recent_avg:.2f}s",
                'success_rate': f"{success_rate:.1f}%",
                'target_time': f"{self.performance_profiles[self.current_profile].get('timeout', 8)}s"
            },
            'current_settings': {
                'num_predict': self.fast_config['num_predict'],
                'num_ctx': self.fast_config['num_ctx'],
                'temperature': self.fast_config['temperature'],
                'num_thread': self.fast_config['num_thread']
            },
            'last_error': self.last_error,
            'keep_alive_active': self.keep_alive_task and not self.keep_alive_task.done() if self.keep_alive_task else False,
            'available_profiles': list(self.performance_profiles.keys()),
            'preferred_models': self.preferred_models,
            'performance_tips': self._get_performance_tips()
        }
    
    def _get_performance_tips(self) -> List[str]:
        """Get performance optimization tips"""
        tips = []
        
        if self.response_times:
            recent_avg = sum(self.response_times[-5:]) / len(self.response_times[-5:])
            
            if recent_avg > 8:
                tips.append("Responses are slow - try 'speed' profile")
            elif recent_avg > 4 and self.current_profile != 'speed':
                tips.append("Consider 'speed' profile for faster responses")
            elif recent_avg < 2:
                tips.append("Great performance! Consider 'quality' profile for better responses")
        
        if self.error_count > 0:
            error_rate = (self.error_count / max(self.request_count, 1)) * 100
            if error_rate > 20:
                tips.append("High error rate - check Ollama service")
        
        if not self.model_loaded:
            tips.append("No model loaded - run optimization script")
        
        return tips
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        if not self.available:
            return False
        
        try:
            if await self._load_model_optimized(model_name):
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ✅ Switched to: {model_name}")
                return True
            return False
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model switch error: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models"""
        if not self.available:
            return []
        return await self._get_available_models_fast()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        if not self.response_times:
            return {"message": "No performance data available"}
        
        response_times = self.response_times[-10:]  # Last 10 requests
        
        return {
            "current_profile": self.current_profile,
            "target_time": f"{self.performance_profiles[self.current_profile].get('timeout', 8)}s",
            "recent_responses": len(response_times),
            "avg_time": f"{sum(response_times) / len(response_times):.2f}s",
            "min_time": f"{min(response_times):.2f}s",
            "max_time": f"{max(response_times):.2f}s",
            "under_4s": sum(1 for t in response_times if t < 4),
            "under_2s": sum(1 for t in response_times if t < 2),
            "performance_grade": self._calculate_performance_grade(response_times),
            "optimization_status": "Pi 5 optimized",
            "recommendations": self._get_performance_tips()
        }
    
    def _calculate_performance_grade(self, response_times: List[float]) -> str:
        """Calculate performance grade based on response times"""
        if not response_times:
            return "N/A"
        
        avg_time = sum(response_times) / len(response_times)
        under_4s_percent = (sum(1 for t in response_times if t < 4) / len(response_times)) * 100
        
        if avg_time < 2 and under_4s_percent >= 90:
            return "A+ (Excellent)"
        elif avg_time < 3 and under_4s_percent >= 80:
            return "A (Very Good)"
        elif avg_time < 4 and under_4s_percent >= 70:
            return "B (Good)"
        elif avg_time < 6:
            return "C (Fair)"
        else:
            return "D (Poor - needs optimization)"
    
    async def close(self):
        """Close with optimized cleanup"""
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
                print(f"[OLLAMA] 📊 Session stats: {self.request_count} requests, {avg_time:.2f}s avg")
            print("[OLLAMA] 🔌 Optimized connection closed")

# Maintain compatibility
OptimizedOfflineLLM = LightningOfflineLLM
