"""
Pascal AI Assistant - FIXED Offline LLM (Ollama Integration)
Properly named class with robust Ollama connectivity
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
    """FIXED: Properly named Lightning Offline LLM with robust Ollama integration"""
    
    def __init__(self):
        from config.settings import settings
        self.settings = settings
        self.session = None
        self.available = False
        self.model_loaded = False
        self.current_model = None
        self.keep_alive_task = None
        
        # Ollama configuration - FIXED with better defaults
        self.host = settings.ollama_host
        self.timeout = min(settings.ollama_timeout, 20)  # Reasonable timeout
        self.keep_alive_duration = settings.ollama_keep_alive
        
        # Performance tracking
        self.request_count = 0
        self.total_time = 0.0
        self.error_count = 0
        self.last_error = None
        
        # FIXED generation settings optimized for Pi 5
        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.05,
            'num_predict': 150,      # Balanced response length
            'num_ctx': 1024,         # Reasonable context for Pi 5
            'num_thread': 4,         # Use all Pi 5 cores
            'num_gpu': 0,            # CPU only on Pi
            'stop': ["</s>", "<|end|>", "<|eot_id|>", "Human:", "User:", "\n\nHuman:", "\n\nUser:"]
        }
        
        # Performance profiles optimized for different use cases
        self.performance_profiles = {
            'speed': {
                'num_predict': 80,       # Quick responses
                'temperature': 0.3,      # More focused
                'num_ctx': 512,          # Less context for speed
                'timeout': 10,
                'description': 'Fast responses (1-3s)'
            },
            'balanced': {
                'num_predict': 150,      # Balanced length
                'temperature': 0.7,
                'num_ctx': 1024,         # Good context
                'timeout': 15,
                'description': 'Balanced performance (2-5s)'
            },
            'quality': {
                'num_predict': 250,      # Longer responses
                'temperature': 0.8,
                'num_ctx': 2048,         # Full context
                'timeout': 25,
                'description': 'Best quality (3-8s)'
            }
        }
        self.current_profile = 'balanced'
    
    async def initialize(self) -> bool:
        """FIXED: Initialize Ollama client with proper error handling"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not available - install with: pip install aiohttp"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ {self.last_error}")
            return False
        
        try:
            # Create session with reasonable timeouts
            timeout = aiohttp.ClientTimeout(
                total=30,           # Overall timeout
                connect=5,          # Connection timeout
                sock_read=20        # Read timeout
            )
            connector = aiohttp.TCPConnector(
                limit=5,            # Connection pool limit
                force_close=True,   # Always close connections
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            
            # Test Ollama connection
            if not await self._test_ollama_connection():
                self.last_error = "Cannot connect to Ollama service"
                return False
            
            # Load best available model
            if await self._load_best_model():
                # Start keep-alive task
                self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())
                self.available = True
                
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ✅ Model loaded: {self.current_model}")
                    print(f"[OLLAMA] ✅ Profile: {self.current_profile}")
                    print(f"[OLLAMA] ✅ Context: {self.generation_config['num_ctx']} tokens")
                
                return True
            else:
                self.last_error = "No suitable model found or failed to load"
                return False
                
        except Exception as e:
            self.last_error = f"Initialization failed: {str(e)}"
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Initialization failed: {e}")
            return False
    
    async def _test_ollama_connection(self) -> bool:
        """FIXED: Test Ollama connection with better error handling"""
        try:
            if self.settings.debug_mode:
                print(f"[OLLAMA] Testing connection to {self.host}")
            
            async with self.session.get(
                f"{self.host}/api/version",
                timeout=aiohttp.ClientTimeout(total=8)
            ) as response:
                if response.status == 200:
                    version_data = await response.json()
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ✅ Connected - Ollama version: {version_data.get('version', 'unknown')}")
                    return True
                else:
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ❌ Connection failed: HTTP {response.status}")
                    return False
                    
        except asyncio.TimeoutError:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Connection timeout to {self.host}")
            return False
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Connection error: {e}")
            return False
    
    async def _get_available_models(self) -> List[str]:
        """FIXED: Get available models with error handling"""
        try:
            async with self.session.get(
                f"{self.host}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] Available models: {models}")
                    return models
                else:
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ❌ Failed to get models: HTTP {response.status}")
                    return []
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Error getting models: {e}")
            return []
    
    async def _load_best_model(self) -> bool:
        """FIXED: Load the best available model with fallbacks"""
        try:
            available_models = await self._get_available_models()
            
            if not available_models:
                if self.settings.debug_mode:
                    print("[OLLAMA] ❌ No models available")
                    print("[OLLAMA] 💡 Try: ollama pull nemotron-mini:4b-instruct-q4_K_M")
                return False
            
            # Preferred models in order of preference
            preferred_models = [
                'nemotron-mini:4b-instruct-q4_K_M',  # Primary choice
                'nemotron-mini',                      # Alternative naming
                'qwen2.5:3b',                        # Fallback 1
                'phi3:mini',                         # Fallback 2
                'llama3.2:3b',                       # Fallback 3
                'gemma2:2b'                          # Fallback 4
            ]
            
            model_to_load = None
            
            # Find best available model
            for preferred in preferred_models:
                for available in available_models:
                    if preferred in available or available in preferred:
                        model_to_load = available
                        break
                if model_to_load:
                    break
            
            # If no preferred model found, use first available
            if not model_to_load:
                model_to_load = available_models[0]
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ⚠️ Using first available model: {model_to_load}")
            else:
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ✅ Selected preferred model: {model_to_load}")
            
            # Load the model
            return await self._load_specific_model(model_to_load)
            
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model loading failed: {e}")
            return False
    
    async def _load_specific_model(self, model_name: str) -> bool:
        """FIXED: Load a specific model with proper configuration"""
        try:
            if self.settings.debug_mode:
                print(f"[OLLAMA] Loading model: {model_name}")
            
            # Use current profile settings
            profile_config = self.performance_profiles[self.current_profile].copy()
            config = {**self.generation_config}
            config.update({k: v for k, v in profile_config.items() 
                          if k not in ['timeout', 'description']})
            
            payload = {
                "model": model_name,
                "keep_alive": self.keep_alive_duration,
                "prompt": "Hello",  # Simple test prompt
                "options": config,
                "stream": False
            }
            
            load_timeout = profile_config.get('timeout', 15)
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=load_timeout + 10)  # Extra time for loading
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'response' in data:
                        self.current_model = model_name
                        self.model_loaded = True
                        if self.settings.debug_mode:
                            print(f"[OLLAMA] ✅ Model loaded successfully: {model_name}")
                        return True
                    else:
                        if self.settings.debug_mode:
                            print(f"[OLLAMA] ❌ Invalid response from model: {data}")
                        return False
                else:
                    error_text = await response.text()
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ❌ Failed to load model: {response.status} - {error_text[:200]}")
                    return False
                    
        except asyncio.TimeoutError:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model loading timeout for {model_name}")
            return False
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model loading error: {e}")
            return False
    
    async def _keep_alive_loop(self):
        """FIXED: Keep-alive loop with better error handling"""
        while self.model_loaded and self.current_model:
            try:
                await asyncio.sleep(60)  # Check every minute
                
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
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200 and self.settings.debug_mode:
                        print(f"[OLLAMA] ⚠️ Keep-alive warning: {response.status}")
                        
            except asyncio.CancelledError:
                if self.settings.debug_mode:
                    print("[OLLAMA] Keep-alive task cancelled")
                break
            except Exception as e:
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ⚠️ Keep-alive error: {e}")
                # Continue trying - don't break on errors
                await asyncio.sleep(30)  # Wait before retrying
    
    def set_performance_profile(self, profile: str):
        """FIXED: Set performance profile with validation"""
        if profile in self.performance_profiles:
            self.current_profile = profile
            
            # Update generation config
            profile_config = self.performance_profiles[profile]
            self.generation_config.update({
                k: v for k, v in profile_config.items() 
                if k not in ['timeout', 'description']
            })
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ✅ Set profile: {profile} - {profile_config['description']}")
                print(f"[OLLAMA] Settings: predict={self.generation_config['num_predict']}, "
                      f"ctx={self.generation_config['num_ctx']}")
        else:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Invalid profile: {profile}")
    
    def _build_optimized_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """FIXED: Build optimized prompt for the current model"""
        # Keep contexts short for performance
        context_parts = []
        
        if personality_context:
            context_parts.append(personality_context[:300])  # Limit personality context
        
        if memory_context:
            context_parts.append(f"Recent context: {memory_context[:200]}")  # Limit memory context
        
        # Build prompt based on model type
        if self.current_model:
            model_lower = self.current_model.lower()
            
            if 'nemotron' in model_lower:
                # Nemotron format
                base_context = " ".join(context_parts)
                if base_context:
                    prompt = f"System: {base_context}\n\nUser: {query}\nAssistant:"
                else:
                    prompt = f"User: {query}\nAssistant:"
                    
            elif 'gemma' in model_lower:
                # Gemma format
                base_context = " ".join(context_parts)
                if base_context:
                    prompt = f"<start_of_turn>system\n{base_context}<end_of_turn>\n<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
                else:
                    prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
                    
            elif 'phi' in model_lower:
                # Phi format
                base_context = " ".join(context_parts)
                if base_context:
                    prompt = f"<|system|>\n{base_context}<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>\n"
                else:
                    prompt = f"<|user|>\n{query}<|end|>\n<|assistant|>\n"
                    
            elif 'llama' in model_lower:
                # Llama format
                base_context = " ".join(context_parts)
                if base_context:
                    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{base_context}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                else:
                    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            else:
                # Generic format
                base_context = " ".join(context_parts)
                if base_context:
                    prompt = f"{base_context}\n\nUser: {query}\nAssistant:"
                else:
                    prompt = f"User: {query}\nAssistant:"
        else:
            # Fallback format
            prompt = f"User: {query}\nAssistant:"
        
        return prompt
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """FIXED: Generate response with proper error handling"""
        if not self.available or not self.model_loaded:
            return "Offline model is not available. Please check that Ollama is running and a model is loaded."
        
        try:
            start_time = time.time()
            
            prompt = self._build_optimized_prompt(query, personality_context, memory_context)
            
            # Use current profile settings
            profile_config = self.performance_profiles[self.current_profile].copy()
            config = {**self.generation_config}
            config.update({k: v for k, v in profile_config.items() 
                          if k not in ['timeout', 'description']})
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": config,
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            request_timeout = profile_config.get('timeout', 15)
            
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
                        print(f"[OLLAMA] ✅ Response in {elapsed_time:.2f}s (profile: {self.current_profile})")
                    
                    return response_text or "I wasn't able to generate a response."
                    
                else:
                    error_text = await response.text()
                    self.error_count += 1
                    self.last_error = f"HTTP {response.status}: {error_text[:100]}"
                    
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ❌ API error: {self.last_error}")
                    
                    return f"I encountered an error: {self.last_error}"
                    
        except asyncio.TimeoutError:
            self.error_count += 1
            elapsed_time = time.time() - start_time
            self.last_error = f"Timeout after {elapsed_time:.1f}s"
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ {self.last_error}")
            
            return f"Response timed out after {elapsed_time:.1f}s. Try the 'speed' profile for faster responses."
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Error: {e}")
            
            return f"I encountered an error: {str(e)[:100]}"
    
    async def generate_response_stream(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """FIXED: Generate streaming response with proper error handling"""
        if not self.available or not self.model_loaded:
            yield "Offline model is not available. Please check that Ollama is running and a model is loaded."
            return
        
        try:
            start_time = time.time()
            
            prompt = self._build_optimized_prompt(query, personality_context, memory_context)
            
            # Use current profile settings
            profile_config = self.performance_profiles[self.current_profile].copy()
            config = {**self.generation_config}
            config.update({k: v for k, v in profile_config.items() 
                          if k not in ['timeout', 'description']})
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": config,
                "stream": True,
                "keep_alive": self.keep_alive_duration
            }
            
            request_timeout = profile_config.get('timeout', 15)
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=request_timeout + 5)  # Extra time for streaming
            ) as response:
                if response.status == 200:
                    response_received = False
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                
                                if 'response' in data and data['response']:
                                    chunk = data['response']
                                    yield chunk
                                    response_received = True
                                
                                if data.get('done', False):
                                    self.request_count += 1
                                    elapsed_time = time.time() - start_time
                                    self.total_time += elapsed_time
                                    
                                    if self.settings.debug_mode:
                                        print(f"\n[OLLAMA] ✅ Streaming complete in {elapsed_time:.2f}s")
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    if not response_received:
                        yield "I wasn't able to generate a response."
                        
                else:
                    error_text = await response.text()
                    self.error_count += 1
                    self.last_error = f"HTTP {response.status}: {error_text[:100]}"
                    
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] ❌ Streaming error: {self.last_error}")
                    
                    yield f"I encountered an error: {self.last_error}"
                    
        except asyncio.TimeoutError:
            self.error_count += 1
            elapsed_time = time.time() - start_time
            self.last_error = f"Streaming timeout after {elapsed_time:.1f}s"
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ {self.last_error}")
            
            yield f"\n\nResponse timed out after {elapsed_time:.1f}s. Try the 'speed' profile for faster responses."
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Streaming error: {e}")
            
            yield f"I encountered an error: {str(e)[:100]}"
    
    def get_status(self) -> Dict[str, Any]:
        """FIXED: Get comprehensive status information"""
        avg_time = self.total_time / self.request_count if self.request_count > 0 else 0
        success_rate = ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100
        
        return {
            'available': self.available,
            'model_loaded': self.model_loaded,
            'current_model': self.current_model,
            'performance_profile': self.current_profile,
            'host': self.host,
            'keep_alive_duration': self.keep_alive_duration,
            'stats': {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'avg_response_time': f"{avg_time:.2f}s",
                'success_rate': f"{success_rate:.1f}%"
            },
            'current_settings': {
                'num_predict': self.generation_config['num_predict'],
                'num_ctx': self.generation_config['num_ctx'],
                'temperature': self.generation_config['temperature'],
                'timeout': f"{self.performance_profiles[self.current_profile].get('timeout', 15)}s"
            },
            'last_error': self.last_error,
            'keep_alive_active': self.keep_alive_task and not self.keep_alive_task.done() if self.keep_alive_task else False,
            'available_profiles': list(self.performance_profiles.keys())
        }
    
    async def switch_model(self, model_name: str) -> bool:
        """FIXED: Switch to a different model"""
        if not self.available:
            return False
        
        try:
            if await self._load_specific_model(model_name):
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ✅ Switched to model: {model_name}")
                return True
            else:
                if self.settings.debug_mode:
                    print(f"[OLLAMA] ❌ Failed to switch to model: {model_name}")
                return False
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] ❌ Model switch error: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """FIXED: List available models"""
        if not self.available:
            return []
        return await self._get_available_models()
    
    async def close(self):
        """FIXED: Close with proper cleanup"""
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
            self.session = None
        
        self.available = False
        self.model_loaded = False
        self.current_model = None
        
        if self.settings.debug_mode:
            print("[OLLAMA] 🔌 Connection closed")

# FIXED: Ensure compatibility with existing imports
OptimizedOfflineLLM = LightningOfflineLLM
