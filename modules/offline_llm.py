"""
Pascal AI Assistant - Fast Ollama Client
Optimized for Pi 5 with single model focus and aggressive keep-alive
"""

import asyncio
import json
import time
from typing import Optional, AsyncGenerator, Dict, Any

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

class FastOllamaClient:
    """Simplified, fast Ollama client optimized for Pi 5"""
    
    def __init__(self, settings):
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
        self.preferred_model = settings.preferred_offline_model
        
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
    
    async def initialize(self) -> bool:
        """Initialize Ollama client"""
        if not AIOHTTP_AVAILABLE:
            return False
        
        try:
            # Create session
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            connector = aiohttp.TCPConnector(limit=5, force_close=True)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            
            # Test Ollama connection
            if not await self._test_connection():
                return False
            
            # Load preferred model
            if await self._load_model():
                # Start keep-alive task
                self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())
                self.available = True
                return True
            else:
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
    
    async def _load_model(self) -> bool:
        """Load the preferred model with keep-alive"""
        try:
            # Check available models
            available_models = await self._get_available_models()
            
            if not available_models:
                if self.settings.debug_mode:
                    print(f"[OLLAMA] No models available")
                return False
            
            # Try to find preferred model or use first available
            model_to_load = None
            for model in available_models:
                if self.preferred_model in model:
                    model_to_load = model
                    break
            
            if not model_to_load:
                model_to_load = available_models[0]
                if self.settings.debug_mode:
                    print(f"[OLLAMA] Preferred model not found, using: {model_to_load}")
            
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
                    return False
                    
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[OLLAMA] Model loading failed: {e}")
            return False
    
    async def _get_available_models(self) -> list:
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
    
    def _build_optimized_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Build optimized prompt for Ollama"""
        # Keep prompts minimal for speed
        context_parts = []
        
        if personality_context:
            context_parts.append(personality_context[:400])  # Limit context size
        
        if memory_context:
            context_parts.append(f"Recent context: {memory_context[:200]}")
        
        # Build final prompt based on model type
        if self.current_model and 'llama' in self.current_model.lower():
            prompt = f"{chr(10).join(context_parts)}\n\nUser: {query}\n\nAssistant:"
        elif self.current_model and 'gemma' in self.current_model.lower():
            prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        else:
            # Generic format
            prompt = f"{chr(10).join(context_parts)}\n\nUser: {query}\n\nAssistant:"
        
        return prompt
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate response from Ollama"""
        if not self.available or not self.model_loaded:
            return "Offline model is not available. Please check that Ollama is running and models are downloaded."
        
        try:
            start_time = time.time()
            
            prompt = self._build_optimized_prompt(query, personality_context, memory_context)
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": self.generation_config,
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
            
            prompt = self._build_optimized_prompt(query, personality_context, memory_context)
            
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": self.generation_config,
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
                    token_count = 0
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                
                                if 'response' in data and data['response']:
                                    chunk = data['response']
                                    yield chunk
                                    response_received = True
                                    token_count += 1
                                
                                # Check if done
                                if data.get('done', False):
                                    # Update stats
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
    
    def get_stats(self) -> dict:
        """Get client statistics"""
        avg_time = self.total_time / self.request_count if self.request_count > 0 else 0
        
        return {
            'available': self.available,
            'model_loaded': self.model_loaded,
            'current_model': self.current_model,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'avg_response_time': f"{avg_time:.2f}s",
            'success_rate': f"{((self.request_count - self.error_count) / max(self.request_count, 1)) * 100:.1f}%",
            'keep_alive_active': self.keep_alive_task and not self.keep_alive_task.done() if self.keep_alive_task else False
        }
    
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
