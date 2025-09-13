"""
Pascal AI Assistant - FIXED Offline LLM (Ollama Only)
Simple, fast Ollama client with the correct class name
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
    """FIXED: Correct class name for Ollama client"""
    
    def __init__(self):
        from config.settings import settings
        self.settings = settings
        self.session = None
        self.available = False
        self.model_loaded = False
        self.current_model = None
        self.keep_alive_task = None
        
        # Ollama configuration
        self.host = settings.ollama_host
        self.timeout = settings.ollama_timeout
        self.keep_alive_duration = settings.ollama_keep_alive
        
        # Performance tracking
        self.request_count = 0
        self.total_time = 0.0
        self.error_count = 0
        
        # Generation settings optimized for Pi 5
        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.1,
            'num_predict': 150,  # Balanced for Pi 5
            'num_ctx': 1024,     # Optimized context window
            'stop': ["</s>", "<|end|>", "<|eot_id|>", "Human:", "User:", "\n\nHuman:", "\n\nUser:"]
        }
        
        # Performance profiles
        self.performance_profiles = {
            'speed': {
                'num_predict': 100,
                'temperature': 0.3,
                'top_p': 0.8,
                'preferred_models': ['phi3:mini', 'qwen2.5:3b', 'gemma2:2b']
            },
            'balanced': {
                'num_predict': 150,
                'temperature': 0.7,
                'top_p': 0.9,
                'preferred_models': ['nemotron-mini:4b-instruct-q4_K_M', 'qwen2.5:3b', 'phi3:mini']
            },
            'quality': {
                'num_predict': 250,
                'temperature': 0.8,
                'top_p': 0.95,
                'preferred_models': ['qwen2.5:7b', 'llama3.2:3b', 'nemotron-mini:4b-instruct-q4_K_M']
            }
        }
        self.current_profile = 'balanced'
    
    async def initialize(self) -> bool:
        """Initialize Ollama client"""
        if not AIOHTTP_AVAILABLE:
            if self.settings.debug_mode:
                print("[OLLAMA] aiohttp not available - install with: pip install aiohttp")
            return False
        
        try:
            # Create session
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            connector = aiohttp.TCPConnector(limit=5, force_close=True)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            
            # Test Ollama connection
            if not await self._test_connection():
                if self.settings.debug_mode:
                    print("[OLLAMA] Connection test failed")
                return False
            
            # Load preferred model
            if await self._load_best_model():
                # Start keep-alive task
                self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())
                self.available = True
                if self.settings.debug_mode:
                    print(f"[OLLAMA] Initialized successfully with model: {self.current_model}")
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
        """Test Ollama connection"""
        try:
            async with self.session.get(f"{self.host}/api/version") as response:
                return response.status == 200
        except:
            return False
    
    async def _load_best_model(self) -> bool:
        """Load the best available model"""
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
            
            # Load the model
            payload = {
                "model": model_to_load,
                "keep_alive": self.keep_alive_duration,
                "prompt": "",
                "stream": False
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
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
        """Get list of available models"""
        try:
            async with self.session.get(f"{self.host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []
    
    async def _keep_alive_loop(self):
        """Background task to keep model loaded"""
        while self.model_loaded and self.current_model:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self.current_model:
                    break
                
                # Send keep-alive request
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
                        print(f"[OLLAMA] Keep-alive failed")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.settings.debug_mode:
                    print(f"[OLLAMA] Keep-alive error: {e}")
    
    def _build_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Build optimized prompt for Ollama"""
        context_parts = []
        
        if personality_context:
            context_parts.append(personality_context[:400])
        
        if memory_context:
            context_parts.append(f"Context: {memory_context[:200]}")
        
        # Build final prompt
        if self.current_model and 'llama' in self.current_model.lower():
            prompt = f"{chr(10).join(context_parts)}\n\nUser: {query}\n\nAssistant:"
        elif self.current_model and 'gemma' in self.current_model.lower():
            prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = f"{chr(10).join(context_parts)}\n\nUser: {query}\n\nAssistant:"
        
        return prompt
    
    def set_performance_profile(self, profile: str):
        """Set performance profile"""
        if profile in self.performance_profiles:
            self.current_profile = profile
            self.generation_config.update(self.performance_profiles[profile])
            if self.settings.debug_mode:
                print(f"[OLLAMA] Set performance profile to: {profile}")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate response from Ollama"""
        if not self.available or not self.model_loaded:
            return "Offline model is not available. Please check that Ollama is running and models are downloaded."
        
        try:
            start_time = time.time()
            
            prompt = self._build_prompt(query, personality_context, memory_context)
            
            # Use current profile settings
            profile_config = self.performance_profiles[self.current_profile].copy()
            config = {**self.generation_config}
            config.update({k: v for k, v in profile_config.items() if k not in ['preferred_models']})
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": config,
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '').strip()
                    
                    # Update stats
                    self.request_count += 1
                    self.total_time += time.time() - start_time
                    
                    return response_text or "I wasn't able to generate a response."
                else:
                    error_text = await response.text()
                    self.error_count += 1
                    return f"Ollama error: {error_text[:100]}"
                    
        except asyncio.TimeoutError:
            self.error_count += 1
            return "Response timed out. The query may be too complex for offline processing."
        except Exception as e:
            self.error_count += 1
            if self.settings.debug_mode:
                print(f"[OLLAMA] Error: {e}")
            return f"Offline processing error: {str(e)[:100]}"
    
    async def generate_response_stream(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response from Ollama"""
        if not self.available or not self.model_loaded:
            yield "Offline model is not available. Please check that Ollama is running and models are downloaded."
            return
        
        try:
            start_time = time.time()
            
            prompt = self._build_prompt(query, personality_context, memory_context)
            
            # Use current profile settings
            profile_config = self.performance_profiles[self.current_profile].copy()
            config = {**self.generation_config}
            config.update({k: v for k, v in profile_config.items() if k not in ['preferred_models']})
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": config,
                "stream": True,
                "keep_alive": self.keep_alive_duration
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
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
                                    self.total_time += time.time() - start_time
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    if not response_received:
                        yield "I wasn't able to generate a response."
                        
                else:
                    error_text = await response.text()
                    self.error_count += 1
                    yield f"Ollama error: {error_text[:100]}"
                    
        except asyncio.TimeoutError:
            self.error_count += 1
            yield "\n\nResponse timed out. The query may be too complex."
        except Exception as e:
            self.error_count += 1
            if self.settings.debug_mode:
                print(f"[OLLAMA] Streaming error: {e}")
            yield f"Offline processing error: {str(e)[:100]}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = self.total_time / self.request_count if self.request_count > 0 else 0
        
        return {
            'available': self.available,
            'model_loaded': self.model_loaded,
            'current_model': self.current_model,
            'performance_profile': self.current_profile,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'avg_response_time': f"{avg_time:.2f}s",
            'success_rate': f"{((self.request_count - self.error_count) / max(self.request_count, 1)) * 100:.1f}%",
            'keep_alive_active': self.keep_alive_task and not self.keep_alive_task.done() if self.keep_alive_task else False,
            'ollama_host': self.host,
            'ollama_enabled': True
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models with metadata"""
        # This would normally query Ollama, but for now return empty list
        # The actual implementation would call _get_available_models() and add metadata
        return []
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        try:
            payload = {
                "model": model_name,
                "keep_alive": self.keep_alive_duration,
                "prompt": "",
                "stream": False
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    self.current_model = model_name
                    if self.settings.debug_mode:
                        print(f"[OLLAMA] Switched to model: {model_name}")
                    return True
                else:
                    return False
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] Model switch failed: {e}")
            return False
    
    async def close(self):
        """Close the client"""
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
