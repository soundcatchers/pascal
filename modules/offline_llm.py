"""
Pascal AI Assistant - Lightning Fast Offline LLM with Streaming
Optimized for 1-3 second responses on Raspberry Pi 5
"""

import asyncio
import time
import json
import aiohttp
from typing import Optional, Dict, Any, List, AsyncGenerator
from pathlib import Path

from config.settings import settings

class ModelInfo:
    """Information about available Ollama models"""
    def __init__(self, name: str, size: str = "Unknown", loaded: bool = False):
        self.name = name
        self.size = size
        self.loaded = loaded
        
        # Model priority (1 = highest)
        if "nemotron-mini" in name.lower():
            self.priority = 1
            self.display_name = "Nemotron Mini 4B"
        elif "qwen3:4b" in name.lower():
            self.priority = 2
            self.display_name = "Qwen3 4B"
        elif "gemma3:4b" in name.lower():
            self.priority = 3
            self.display_name = "Gemma3 4B"
        else:
            self.priority = 99
            self.display_name = name

class OptimizedOfflineLLM:
    """Lightning-fast Ollama integration with streaming and keep-alive"""
    
    def __init__(self):
        self.ollama_host = "http://localhost:11434"
        self.session = None
        self.current_model = None
        self.available_models = []
        self.model_loaded = False
        self.keep_alive_task = None
        
        # Primary models in priority order
        self.priority_models = [
            "nemotron-mini:4b-instruct-q4_K_M",  # Primary - fastest
            "qwen3:4b-instruct",                  # Secondary fallback
            "gemma3:4b-it-q4_K_M"                 # Tertiary fallback
        ]
        
        # Streaming configuration for speed
        self.stream_config = {
            'enabled': True,
            'chunk_timeout': 0.1,  # 100ms per chunk for responsiveness
            'first_token_target': 0.5  # Target 500ms to first token
        }
        
        # Keep-alive configuration (only for primary model)
        self.keep_alive_config = {
            'enabled': True,
            'interval': 30,  # Ping every 30 seconds
            'timeout': 300   # Keep model loaded for 5 minutes
        }
        
        # Performance tracking
        self.response_metrics = {
            'first_token_times': [],
            'total_times': [],
            'tokens_per_second': []
        }
    
    async def initialize(self) -> bool:
        """Initialize Ollama with focus on speed"""
        try:
            # Create aiohttp session with optimized settings
            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=2,
                sock_read=5
            )
            connector = aiohttp.TCPConnector(
                limit=10,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            
            # Test Ollama connection
            if not await self._test_ollama_connection():
                print("❌ Ollama not running. Start with: sudo systemctl start ollama")
                return False
            
            # Check for priority models
            await self._check_priority_models()
            
            if not self.available_models:
                print("❌ No priority models found. Downloading required models...")
                await self._download_priority_models()
            
            # Load the primary model and keep it warm
            if await self._load_primary_model():
                # Start keep-alive task
                if self.keep_alive_config['enabled']:
                    self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())
                
                print(f"⚡ Lightning mode activated with {self.current_model.display_name}")
                return True
            
            return False
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def _test_ollama_connection(self) -> bool:
        """Quick connection test"""
        try:
            async with self.session.get(f"{self.ollama_host}/api/version") as response:
                return response.status == 200
        except:
            return False
    
    async def _check_priority_models(self):
        """Check which priority models are available"""
        try:
            async with self.session.get(f"{self.ollama_host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    available_model_names = [m['name'] for m in data.get('models', [])]
                    
                    self.available_models = []
                    for priority_model in self.priority_models:
                        if priority_model in available_model_names:
                            model_info = ModelInfo(priority_model)
                            self.available_models.append(model_info)
                            print(f"✅ Found model: {model_info.display_name}")
                    
                    if not self.available_models and available_model_names:
                        # Fallback to any available model
                        print("⚠️ No priority models found, using available models")
                        for model_name in available_model_names[:3]:  # Take first 3
                            self.available_models.append(ModelInfo(model_name))
                
        except Exception as e:
            print(f"Error checking models: {e}")
    
    async def _download_priority_models(self):
        """Download the primary model if not available"""
        primary_model = self.priority_models[0]
        print(f"📥 Downloading {primary_model} for optimal performance...")
        
        try:
            payload = {"name": primary_model, "stream": False}
            async with self.session.post(f"{self.ollama_host}/api/pull", json=payload) as response:
                if response.status == 200:
                    print(f"✅ Downloaded {primary_model}")
                    # Refresh available models
                    await self._check_priority_models()
                    return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
        
        return False
    
    async def _load_primary_model(self) -> bool:
        """Load and warm up the primary model"""
        if not self.available_models:
            return False
        
        # Select the highest priority model
        self.current_model = self.available_models[0]
        
        try:
            # Warm up the model with a quick test
            print(f"🔥 Warming up {self.current_model.display_name}...")
            
            warmup_payload = {
                "model": self.current_model.name,
                "prompt": "Hi",
                "stream": False,
                "options": {
                    "num_predict": 5,
                    "temperature": 0.1
                },
                "keep_alive": "5m"  # Keep model loaded for 5 minutes
            }
            
            start_time = time.time()
            async with self.session.post(
                f"{self.ollama_host}/api/generate",
                json=warmup_payload
            ) as response:
                if response.status == 200:
                    warmup_time = time.time() - start_time
                    self.model_loaded = True
                    self.current_model.loaded = True
                    print(f"⚡ Model ready! Warmup time: {warmup_time:.2f}s")
                    return True
                
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
        
        return False
    
    async def _keep_alive_loop(self):
        """Keep the primary model loaded in memory"""
        while self.model_loaded and self.current_model:
            try:
                await asyncio.sleep(self.keep_alive_config['interval'])
                
                # Send keep-alive request
                keep_alive_payload = {
                    "model": self.current_model.name,
                    "keep_alive": f"{self.keep_alive_config['timeout']}s"
                }
                
                async with self.session.post(
                    f"{self.ollama_host}/api/generate",
                    json=keep_alive_payload
                ) as response:
                    if response.status != 200:
                        print(f"⚠️ Keep-alive failed, reloading model...")
                        await self._load_primary_model()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                if settings.debug_mode:
                    print(f"Keep-alive error: {e}")
    
    async def generate_response_stream(self, query: str, personality_context: str, 
                                     memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response for perceived speed (1-3 second target)"""
        if not self.model_loaded or not self.current_model:
            yield "I'm starting up, just a moment..."
            if not await self._load_primary_model():
                yield "\nHaving trouble loading the model. Let me try a fallback..."
                await self._try_fallback_models()
                if not self.model_loaded:
                    yield "\nI'm having technical difficulties. Please try again."
                    return
        
        try:
            # Build optimized prompt
            prompt = self._build_fast_prompt(query, personality_context, memory_context)
            
            # Streaming request for fast first token
            payload = {
                "model": self.current_model.name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": 150,  # Reasonable response length
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "seed": -1,
                    "stop": ["User:", "Human:", "Question:", "\n\n\n"]
                },
                "keep_alive": "5m"
            }
            
            first_token_time = None
            start_time = time.time()
            full_response = []
            
            async with self.session.post(
                f"{self.ollama_host}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line)
                                if 'response' in chunk:
                                    token = chunk['response']
                                    
                                    # Track first token time
                                    if first_token_time is None:
                                        first_token_time = time.time() - start_time
                                        if settings.debug_mode:
                                            print(f"⚡ First token: {first_token_time:.3f}s")
                                    
                                    full_response.append(token)
                                    yield token
                                    
                                    # Check if response is complete
                                    if chunk.get('done', False):
                                        break
                                        
                            except json.JSONDecodeError:
                                continue
                    
                    # Track metrics
                    total_time = time.time() - start_time
                    self._update_metrics(first_token_time, total_time, len(full_response))
                    
                else:
                    yield "I encountered an error. Let me try again..."
                    # Try fallback model
                    await self._try_fallback_models()
                    if self.model_loaded:
                        async for token in self.generate_response_stream(query, personality_context, memory_context):
                            yield token
                            
        except Exception as e:
            if settings.debug_mode:
                print(f"Stream error: {e}")
            yield f"I apologize, but I encountered an error: {str(e)}"
    
    async def generate_response(self, query: str, personality_context: str, 
                              memory_context: str, profile: str = None) -> str:
        """Non-streaming response (fallback method)"""
        full_response = []
        async for token in self.generate_response_stream(query, personality_context, memory_context):
            full_response.append(token)
        return ''.join(full_response)
    
    async def _try_fallback_models(self):
        """Try fallback models if primary fails"""
        for i, model_name in enumerate(self.priority_models[1:], start=2):
            print(f"🔄 Trying fallback model {i}: {model_name}")
            
            # Check if model exists
            model_exists = False
            for available_model in self.available_models:
                if available_model.name == model_name:
                    model_exists = True
                    self.current_model = available_model
                    break
            
            if not model_exists:
                # Try to download it
                print(f"📥 Downloading fallback model: {model_name}")
                try:
                    payload = {"name": model_name, "stream": False}
                    async with self.session.post(f"{self.ollama_host}/api/pull", json=payload) as response:
                        if response.status == 200:
                            self.current_model = ModelInfo(model_name)
                            self.available_models.append(self.current_model)
                        else:
                            continue
                except:
                    continue
            
            # Try to load the fallback model
            if await self._load_primary_model():
                print(f"✅ Fallback model loaded: {self.current_model.display_name}")
                return
        
        print("❌ All fallback models failed")
    
    def _build_fast_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Build minimal, fast prompt optimized for speed"""
        # Keep prompts short for faster processing
        if not personality_context:
            personality_context = "You are Pascal, a helpful and fast AI assistant."
        
        # Truncate context for speed
        if len(personality_context) > 150:
            personality_context = personality_context[:150]
        
        if memory_context and len(memory_context) > 100:
            memory_context = memory_context[:100]
        
        # Model-specific formatting (minimal for speed)
        if self.current_model and "nemotron" in self.current_model.name.lower():
            # Nemotron format
            prompt = f"System: {personality_context}\nUser: {query}\nAssistant:"
        elif self.current_model and "qwen" in self.current_model.name.lower():
            # Qwen format
            prompt = f"<|im_start|>system\n{personality_context}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        elif self.current_model and "gemma" in self.current_model.name.lower():
            # Gemma format
            prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        else:
            # Generic format
            prompt = f"{personality_context}\n\nUser: {query}\nAssistant:"
        
        return prompt
    
    def _update_metrics(self, first_token_time: float, total_time: float, token_count: int):
        """Track performance metrics"""
        if first_token_time:
            self.response_metrics['first_token_times'].append(first_token_time)
            if len(self.response_metrics['first_token_times']) > 20:
                self.response_metrics['first_token_times'] = self.response_metrics['first_token_times'][-20:]
        
        self.response_metrics['total_times'].append(total_time)
        if len(self.response_metrics['total_times']) > 20:
            self.response_metrics['total_times'] = self.response_metrics['total_times'][-20:]
        
        if total_time > 0:
            tps = token_count / total_time
            self.response_metrics['tokens_per_second'].append(tps)
            if len(self.response_metrics['tokens_per_second']) > 20:
                self.response_metrics['tokens_per_second'] = self.response_metrics['tokens_per_second'][-20:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "ollama_enabled": True,
            "model_loaded": self.model_loaded,
            "current_model": self.current_model.name if self.current_model else None,
            "model_display_name": self.current_model.display_name if self.current_model else None,
            "streaming_enabled": self.stream_config['enabled'],
            "keep_alive_enabled": self.keep_alive_config['enabled'],
            "available_models": len(self.available_models),
            "ollama_host": self.ollama_host
        }
        
        # Add performance metrics
        if self.response_metrics['first_token_times']:
            stats['avg_first_token_time'] = f"{sum(self.response_metrics['first_token_times']) / len(self.response_metrics['first_token_times']):.3f}s"
        
        if self.response_metrics['total_times']:
            stats['avg_total_time'] = f"{sum(self.response_metrics['total_times']) / len(self.response_metrics['total_times']):.2f}s"
        
        if self.response_metrics['tokens_per_second']:
            stats['avg_tokens_per_second'] = f"{sum(self.response_metrics['tokens_per_second']) / len(self.response_metrics['tokens_per_second']):.1f}"
        
        return stats
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        return [
            {
                "name": model.name,
                "display_name": model.display_name,
                "priority": model.priority,
                "loaded": model.loaded,
                "size": model.size
            }
            for model in sorted(self.available_models, key=lambda x: x.priority)
        ]
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        for model in self.available_models:
            if model_name in model.name:
                old_model = self.current_model
                self.current_model = model
                if await self._load_primary_model():
                    print(f"✅ Switched from {old_model.display_name if old_model else 'None'} to {model.display_name}")
                    return True
                else:
                    self.current_model = old_model
                    return False
        return False
    
    async def pull_model(self, model_name: str) -> bool:
        """Download a new model"""
        try:
            print(f"📥 Downloading {model_name}...")
            payload = {"name": model_name, "stream": False}
            
            async with self.session.post(f"{self.ollama_host}/api/pull", json=payload) as response:
                if response.status == 200:
                    await self._check_priority_models()
                    print(f"✅ Downloaded {model_name}")
                    return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
        return False
    
    async def remove_model(self, model_name: str) -> bool:
        """Remove a model"""
        # Don't remove current model
        if self.current_model and model_name in self.current_model.name:
            print("❌ Cannot remove currently loaded model")
            return False
        
        try:
            payload = {"name": model_name}
            async with self.session.delete(f"{self.ollama_host}/api/delete", json=payload) as response:
                if response.status == 200:
                    # Update available models
                    self.available_models = [m for m in self.available_models if model_name not in m.name]
                    print(f"✅ Removed {model_name}")
                    return True
        except Exception as e:
            print(f"❌ Remove failed: {e}")
        return False
    
    def is_available(self) -> bool:
        """Check if ready for lightning-fast responses"""
        return self.model_loaded and self.current_model is not None
    
    def set_performance_profile(self, profile: str):
        """Compatibility method - all profiles optimized for speed"""
        # In lightning mode, we always optimize for speed
        if settings.debug_mode:
            print(f"⚡ Lightning mode active - all profiles optimized for 1-3s responses")
    
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
        
        self.model_loaded = False
        self.current_model = None
