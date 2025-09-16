"""
Pascal AI Assistant - OPTIMIZED Offline LLM (Ollama Only)
Fast, efficient Ollama client optimized for Pi 5 performance
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
    """OPTIMIZED: Fast Ollama client with aggressive performance tuning"""
    
    def __init__(self):
        from config.settings import settings
        self.settings = settings
        self.session = None
        self.available = False
        self.model_loaded = False
        self.current_model = None
        self.keep_alive_task = None
        
        # Ollama configuration - OPTIMIZED for speed
        self.host = settings.ollama_host
        self.timeout = min(settings.ollama_timeout, 15)  # Cap at 15s max
        self.keep_alive_duration = settings.ollama_keep_alive
        
        # Performance tracking
        self.request_count = 0
        self.total_time = 0.0
        self.error_count = 0
        
        # OPTIMIZED generation settings for Pi 5 speed
        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.05,  # Reduced for speed
            'num_predict': 100,      # REDUCED for faster responses
            'num_ctx': 512,          # REDUCED context for speed
            'num_thread': 4,         # Use all Pi 5 cores
            'num_gpu': 0,            # CPU only on Pi
            'stop': ["</s>", "<|end|>", "<|eot_id|>", "Human:", "User:", "\n\nHuman:", "\n\nUser:", "Q:", "\nQ:"]
        }
        
        # OPTIMIZED performance profiles for Pi 5
        self.performance_profiles = {
            'speed': {
                'num_predict': 50,       # Very short responses
                'temperature': 0.1,      # Low creativity for consistency
                'top_p': 0.7,
                'num_ctx': 256,          # Minimal context
                'timeout': 8,            # Short timeout
                'preferred_models': ['phi3:mini', 'qwen2.5:3b', 'gemma2:2b']
            },
            'balanced': {
                'num_predict': 100,      # Balanced length
                'temperature': 0.7,
                'top_p': 0.9,
                'num_ctx': 512,          # Moderate context
                'timeout': 12,           # Moderate timeout
                'preferred_models': ['nemotron-mini:4b-instruct-q4_K_M', 'qwen2.5:3b', 'phi3:mini']
            },
            'quality': {
                'num_predict': 200,      # Longer responses
                'temperature': 0.8,
                'top_p': 0.95,
                'num_ctx': 1024,         # Full context
                'timeout': 20,           # Longer timeout allowed
                'preferred_models': ['qwen2.5:7b', 'llama3.2:3b', 'nemotron-mini:4b-instruct-q4_K_M']
            }
        }
        self.current_profile = 'balanced'
    
    async def initialize(self) -> bool:
        """Initialize Ollama client with optimized settings"""
        if not AIOHTTP_AVAILABLE:
            if self.settings.debug_mode:
                print("[OLLAMA] aiohttp not available - install with: pip install aiohttp")
            return False
        
        try:
            # Create OPTIMIZED session with aggressive timeouts
            timeout = aiohttp.ClientTimeout(
                total=20,           # Overall timeout
                connect=3,          # Quick connection timeout
                sock_read=15        # Read timeout
            )
            connector = aiohttp.TCPConnector(
                limit=3,            # Fewer connections
                force_close=True,   # Always close connections
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            
            # Test Ollama connection with short timeout
            if not await self._test_connection():
                if self.settings.debug_mode:
                    print("[OLLAMA] Connection test failed")
                return False
            
            # Load preferred model with optimization
            if await self._load_best_model():
                # Start optimized keep-alive task
                self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())
                self.available = True
                if self.settings.debug_mode:
                    print(f"[OLLAMA] Model loaded: {self.current_model}")
                return True
            else:
                if self.settings.debug_mode:
                    print("[OLLAMA] No suitable model found")
                return False
                
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] Initialization failed: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test Ollama connection with short timeout"""
        try:
            async with self.session.get(
                f"{self.host}/api/version",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except:
            return False
    
    async def _load_best_model(self) -> bool:
        """Load the best available model with optimization"""
        try:
            # Check available models
            available_models = await self._get_available_models()
            
            if not available_models:
                if self.settings.debug_mode:
                    print("[OLLAMA] No models available")
                return False
            
            # Try to find preferred model for current profile
            preferred_models = self.performance_profiles[self.current_profile]['preferred_models']
            model_to_load = None
            
            for preferred in preferred_models:
                for available in available_models:
                    if preferred in available:
                        model_to_load = available
                        break
                if model_to_load:
                    break
            
            if not model_to_load:
                model_to_load = available_models[0]
                if self.settings.debug_mode:
                    print(f"[OLLAMA] Using first available model: {model_to_load}")
            
            # Load the model with OPTIMIZED settings
            profile_config = self.performance_profiles[self.current_profile]
            config = {**self.generation_config}
            config.update({k: v for k, v in profile_config.items() if k not in ['preferred_models', 'timeout']})
            
            payload = {
                "model": model_to_load,
                "keep_alive": self.keep_alive_duration,
                "prompt": "Hi",  # Minimal test prompt
                "options": config,
                "stream": False
            }
            
            load_timeout = profile_config.get('timeout', 15)
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=load_timeout)
            ) as response:
                if response.status == 200:
                    self.current_model = model_to_load
                    self.model_loaded = True
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] Model loaded: {model_to_load}")
                    return True
                else:
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] Failed to load model: {response.status}")
                    return False
                    
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] Model loading failed: {e}")
            return False
    
    async def _get_available_models(self) -> List[str]:
        """Get list of available models with short timeout"""
        try:
            async with self.session.get(
                f"{self.host}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []
    
    async def _keep_alive_loop(self):
        """OPTIMIZED background task to keep model loaded"""
        while self.model_loaded and self.current_model:
            try:
                await asyncio.sleep(45)  # Check every 45 seconds (less frequent)
                
                if not self.current_model:
                    break
                
                # Send minimal keep-alive request
                payload = {
                    "model": self.current_model,
                    "keep_alive": self.keep_alive_duration,
                    "prompt": "",
                    "stream": False
                }
                
                async with self.session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=8)  # Short timeout for keep-alive
                ) as response:
                    if response.status != 200 and self.settings.debug_mode:
                        print(f"[OLLAMA] Keep-alive error: {response.status}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.settings.debug_mode:
                    print(f"[OLLAMA] Keep-alive error: {e}")
                # Don't break on errors - keep trying
    
    def _build_optimized_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Build optimized prompt for fast responses"""
        # OPTIMIZED: Keep prompts short for speed
        context_parts = []
        
        # Limit personality context for speed
        if personality_context:
            # Take only essential personality info
            context_parts.append(personality_context[:200])
        
        # Limit memory context for speed
        if memory_context:
            context_parts.append(f"Context: {memory_context[:100]}")
        
        # Build minimal prompt based on model
        if self.current_model and 'llama' in self.current_model.lower():
            # Llama format
            base_context = " ".join(context_parts)
            if base_context:
                prompt = f"{base_context}\n\nUser: {query}\nAssistant:"
            else:
                prompt = f"User: {query}\nAssistant:"
        elif self.current_model and 'gemma' in self.current_model.lower():
            # Gemma format
            prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        elif self.current_model and 'phi' in self.current_model.lower():
            # Phi format
            prompt = f"<|user|>\n{query}<|end|>\n<|assistant|>\n"
        else:
            # Generic format
            base_context = " ".join(context_parts)
            if base_context:
                prompt = f"{base_context}\n\nUser: {query}\nAssistant:"
            else:
                prompt = f"User: {query}\nAssistant:"
        
        return prompt
    
    def set_performance_profile(self, profile: str):
        """Set performance profile with immediate effect"""
        if profile in self.performance_profiles:
            self.current_profile = profile
            
            # Update generation config immediately
            profile_config = self.performance_profiles[profile]
            self.generation_config.update({
                k: v for k, v in profile_config.items() 
                if k not in ['preferred_models', 'timeout']
            })
            
            # Update timeout
            self.timeout = profile_config.get('timeout', 15)
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] Set performance profile to: {profile}")
                print(f"[OLLAMA] New settings: predict={self.generation_config['num_predict']}, ctx={self.generation_config['num_ctx']}, timeout={self.timeout}s")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate OPTIMIZED response from Ollama"""
        if not self.available or not self.model_loaded:
            return "Offline model is not available. Please check that Ollama is running and models are downloaded."
        
        try:
            start_time = time.time()
            
            prompt = self._build_optimized_prompt(query, personality_context, memory_context)
            
            # Use current profile settings
            profile_config = self.performance_profiles[self.current_profile].copy()
            config = {**self.generation_config}
            config.update({k: v for k, v in profile_config.items() if k not in ['preferred_models', 'timeout']})
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": config,
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            # Use profile-specific timeout
            request_timeout = profile_config.get('timeout', self.timeout)
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=request_timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '').strip()
                    
                    # Update stats
                    self.request_count += 1
                    elapsed_time = time.time() - start_time
                    self.total_time += elapsed_time
                    
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ✅ Response generated in {elapsed_time:.2f}s ({self.current_profile} profile)")
                    
                    return response_text or "I wasn't able to generate a response."
                else:
                    error_text = await response.text()
                    self.error_count += 1
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ❌ API error {response.status}: {error_text[:100]}")
                    return f"Ollama error: HTTP {response.status}"
                    
        except asyncio.TimeoutError:
            self.error_count += 1
            elapsed_time = time.time() - start_time
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Timeout after {elapsed_time:.1f}s (limit: {request_timeout}s)")
            return f"Response timed out after {elapsed_time:.1f}s. Try the 'speed' profile for faster responses."
        except Exception as e:
            self.error_count += 1
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Error: {e}")
            return f"Offline processing error: {str(e)[:100]}"
    
    async def generate_response_stream(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Generate OPTIMIZED streaming response from Ollama"""
        if not self.available or not self.model_loaded:
            yield "Offline model is not available. Please check that Ollama is running and models are downloaded."
            return
        
        try:
            start_time = time.time()
            
            prompt = self._build_optimized_prompt(query, personality_context, memory_context)
            
            # Use current profile settings
            profile_config = self.performance_profiles[self.current_profile].copy()
            config = {**self.generation_config}
            config.update({k: v for k, v in profile_config.items() if k not in ['preferred_models', 'timeout']})
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": config,
                "stream": True,
                "keep_alive": self.keep_alive_duration
            }
            
            # Use profile-specific timeout
            request_timeout = profile_config.get('timeout', self.timeout)
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=request_timeout)
            ) as response:
                if response.status == 200:
                    response_received = False
                    first_chunk_time = None
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                
                                if 'response' in data and data['response']:
                                    chunk = data['response']
                                    yield chunk
                                    
                                    if not response_received:
                                        first_chunk_time = time.time()
                                        if self.settings.debug_mode:
                                            print(f"[OLLAMA] 🌊 First chunk in {first_chunk_time - start_time:.2f}s")
                                        response_received = True
                                
                                if data.get('done', False):
                                    self.request_count += 1
                                    elapsed_time = time.time() - start_time
                                    self.total_time += elapsed_time
                                    
                                    if self.settings.debug_mode:
                                        print(f"[OLLAMA] ✅ Streaming complete in {elapsed_time:.2f}s ({self.current_profile} profile)")
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    if not response_received:
                        yield "I wasn't able to generate a response."
                        
                else:
                    error_text = await response.text()
                    self.error_count += 1
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ❌ Streaming error {response.status}: {error_text[:100]}")
                    yield f"Ollama error: HTTP {response.status}"
                    
        except asyncio.TimeoutError:
            self.error_count += 1
            elapsed_time = time.time() - start_time
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Streaming timeout after {elapsed_time:.1f}s")
            yield f"\n\nResponse timed out after {elapsed_time:.1f}s. Try 'speed' profile for faster responses."
        except Exception as e:
            self.error_count += 1
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Streaming error: {e}")
            yield f"Offline processing error: {str(e)[:100]}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        avg_time = self.total_time / self.request_count if self.request_count > 0 else 0
        success_rate = ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100
        
        return {
            'available': self.available,
            'model_loaded': self.model_loaded,
            'current_model': self.current_model,
            'performance_profile': self.current_profile,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'avg_response_time': f"{avg_time:.2f}s",
            'success_rate': f"{success_rate:.1f}%",
            'current_settings': {
                'num_predict': self.generation_config['num_predict'],
                'num_ctx': self.generation_config['num_ctx'],
                'temperature': self.generation_config['temperature'],
                'timeout': f"{self.timeout}s"
            },
            'keep_alive_active': self.keep_alive_task and not self.keep_alive_task.done() if self.keep_alive_task else False,
            'ollama_host': self.host,
            'ollama_enabled': True,
            'optimization_level': 'High (Pi 5 optimized)'
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models with Pi 5 performance ratings"""
        # This would normally query Ollama, but return template for now
        return [
            {
                'name': 'nemotron-mini:4b-instruct-q4_K_M',
                'size': '~2.7GB',
                'speed_rating': 'Fast',
                'quality_rating': 'High',
                'recommended_profile': 'balanced',
                'pi5_optimized': True
            }
        ]
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model with optimization"""
        try:
            # Use current profile settings for model switch
            profile_config = self.performance_profiles[self.current_profile]
            config = {**self.generation_config}
            config.update({k: v for k, v in profile_config.items() if k not in ['preferred_models', 'timeout']})
            
            payload = {
                "model": model_name,
                "keep_alive": self.keep_alive_duration,
                "prompt": "Hi",  # Minimal test
                "options": config,
                "stream": False
            }
            
            switch_timeout = profile_config.get('timeout', 15)
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=switch_timeout)
            ) as response:
                if response.status == 200:
                    self.current_model = model_name
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ✅ Switched to model: {model_name}")
                    return True
                else:
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ❌ Model switch failed: {response.status}")
                    return False
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model switch failed: {e}")
            return False
    
    def get_optimization_tips(self) -> List[str]:
        """Get Pi 5 optimization tips"""
        tips = []
        
        if self.request_count > 0:
            avg_time = self.total_time / self.request_count
            
            if avg_time > 5:
                tips.append("Responses are slow - try 'speed' profile or switch to a smaller model")
            elif avg_time > 3:
                tips.append("Consider using 'speed' profile for faster responses")
            
            if self.error_count / max(self.request_count, 1) > 0.1:
                tips.append("High error rate - check Ollama service and model availability")
        
        tips.append("For fastest responses: Use 'speed' profile with phi3:mini model")
        tips.append("For balanced performance: Current 'balanced' profile is optimized for Pi 5")
        tips.append("Monitor CPU temperature: 'vcgencmd measure_temp' - keep below 70°C")
        
        return tips
    
    async def close(self):
        """Close the client with cleanup"""
        # Cancel keep-alive task
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        # Close session
        if self.session:
            await self.session.close()
            self.session = None
        
        self.available = False
        self.model_loaded = False
        self.current_model = None
        
        if self.settings.debug_mode:
            print("[OLLAMA] 🔌 Optimized client closed")
