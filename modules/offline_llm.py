"""
Pascal AI Assistant - Lightning-Fast Offline LLM with Streaming
Optimized for sub-3-second responses on Raspberry Pi 5 with keep-alive
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
    def __init__(self, name: str, size: str, parameters: str, modified: str = ""):
        self.name = name
        self.size = size
        self.parameters = parameters
        self.modified = modified
        self.priority = self._get_priority()
    
    def _get_priority(self) -> int:
        """Get model priority (1 = highest)"""
        # Priority order for our specific models
        if "nemotron-mini:4b-instruct-q4_K_M" in self.name:
            return 1
        elif "qwen3:4b-instruct" in self.name:
            return 2
        elif "gemma3:4b-it-q4_K_M" in self.name:
            return 3
        else:
            return 99  # Other models have lowest priority

class LightningOfflineLLM:
    """Lightning-fast Ollama-based offline LLM with streaming and keep-alive"""
    
    def __init__(self):
        self.ollama_host = "http://localhost:11434"
        self.session = None
        self.current_model = None
        self.available_models = []
        self.model_loaded = False
        self.keep_alive_task = None
        
        # Primary and fallback models
        self.model_hierarchy = [
            "nemotron-mini:4b-instruct-q4_K_M",  # Primary - fastest
            "qwen3:4b-instruct",                  # Fallback 1
            "gemma3:4b-it-q4_K_M"                  # Fallback 2
        ]
        
        # Performance tracking
        self.inference_times = []
        self.first_token_times = []
        
        # Optimized settings for sub-3-second responses
        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.1,
            'num_predict': 150,  # Limit tokens for faster responses
            'num_ctx': 2048,     # Context window
            'num_batch': 512,    # Batch size for processing
            'num_thread': 4,     # Use all Pi 5 cores
            'seed': -1,
            'stop': ["</s>", "<|end|>", "<|eot_id|>", "Human:", "User:", "\n\n"]
        }
        
        # Keep-alive configuration
        self.keep_alive_interval = 30  # seconds
        self.keep_alive_duration = "5m"  # Keep model loaded for 5 minutes
    
    async def initialize(self) -> bool:
        """Initialize with lightning-fast configuration"""
        try:
            # Create optimized aiohttp session
            timeout = aiohttp.ClientTimeout(total=60, sock_read=30)
            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            
            # Test Ollama connection
            if not await self._test_ollama_connection():
                print("❌ Could not connect to Ollama. Is it running?")
                print("   Start with: sudo systemctl start ollama")
                return False
            
            # Scan for available models
            await self._scan_available_models()
            
            # Load primary model with keep-alive
            if await self._load_primary_model():
                # Start keep-alive task
                self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())
                print(f"⚡ Lightning mode activated with {self.current_model.name}")
                return True
            else:
                print("❌ Failed to load any models")
                return False
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def _test_ollama_connection(self) -> bool:
        """Test connection to Ollama service"""
        try:
            async with self.session.get(f"{self.ollama_host}/api/version") as response:
                if response.status == 200:
                    version_data = await response.json()
                    if settings.debug_mode:
                        print(f"Connected to Ollama version: {version_data.get('version', 'unknown')}")
                    return True
                return False
        except Exception as e:
            if settings.debug_mode:
                print(f"Ollama connection test failed: {e}")
            return False
    
    async def _scan_available_models(self):
        """Scan for available models with priority sorting"""
        try:
            async with self.session.get(f"{self.ollama_host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    
                    self.available_models = []
                    for model_data in models:
                        model_info = ModelInfo(
                            name=model_data.get('name', ''),
                            size=self._format_size(model_data.get('size', 0)),
                            parameters=model_data.get('details', {}).get('parameter_size', 'Unknown'),
                            modified=model_data.get('modified_at', '')
                        )
                        self.available_models.append(model_info)
                    
                    # Sort by priority for our specific models
                    self.available_models.sort(key=lambda x: x.priority)
                    
                    if settings.debug_mode:
                        print(f"Found {len(self.available_models)} models")
                        for model in self.available_models[:3]:
                            print(f"  • {model.name} (Priority: {model.priority})")
                
        except Exception as e:
            if settings.debug_mode:
                print(f"Failed to scan models: {e}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"
    
    async def _load_primary_model(self) -> bool:
        """Load the primary model or first available fallback"""
        for model_name in self.model_hierarchy:
            # Check if model is available
            model = next((m for m in self.available_models if model_name in m.name), None)
            if model:
                # Load the model
                if await self._load_model(model):
                    self.current_model = model
                    self.model_loaded = True
                    print(f"✅ Loaded model: {model.name}")
                    return True
                else:
                    print(f"⚠️ Failed to load {model_name}, trying fallback...")
        
        # If none of our preferred models are available, try any available model
        if self.available_models:
            for model in self.available_models:
                if await self._load_model(model):
                    self.current_model = model
                    self.model_loaded = True
                    print(f"✅ Loaded fallback model: {model.name}")
                    return True
        
        return False
    
    async def _load_model(self, model: ModelInfo) -> bool:
        """Load a specific model with keep-alive"""
        try:
            # First, ensure the model is loaded with keep-alive
            payload = {
                "model": model.name,
                "keep_alive": self.keep_alive_duration
            }
            
            async with self.session.post(
                f"{self.ollama_host}/api/generate",
                json={**payload, "prompt": "", "stream": False}
            ) as response:
                if response.status == 200:
                    return True
                    
        except Exception as e:
            if settings.debug_mode:
                print(f"Failed to load model {model.name}: {e}")
        
        return False
    
    async def _keep_alive_loop(self):
        """Background task to keep model loaded in memory"""
        while self.model_loaded and self.current_model:
            try:
                await asyncio.sleep(self.keep_alive_interval)
                
                # Send keep-alive request
                payload = {
                    "model": self.current_model.name,
                    "keep_alive": self.keep_alive_duration
                }
                
                async with self.session.post(
                    f"{self.ollama_host}/api/generate",
                    json={**payload, "prompt": "", "stream": False}
                ) as response:
                    if response.status != 200:
                        print(f"⚠️ Keep-alive failed for {self.current_model.name}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                if settings.debug_mode:
                    print(f"Keep-alive error: {e}")
    
    async def generate_response_stream(self, query: str, personality_context: str, 
                                     memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response for instant feedback"""
        if not self.model_loaded or not self.current_model:
            yield "I'm having trouble connecting to my language model. Please wait..."
            return
        
        try:
            start_time = time.time()
            first_token_received = False
            
            # Build optimized prompt
            prompt = self._build_lightning_prompt(query, personality_context, memory_context)
            
            # Prepare streaming request
            payload = {
                "model": self.current_model.name,
                "prompt": prompt,
                "options": self.generation_config,
                "stream": True,  # Enable streaming
                "keep_alive": self.keep_alive_duration
            }
            
            # Stream response
            async with self.session.post(
                f"{self.ollama_host}/api/generate", 
                json=payload
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        try:
                            if line:
                                data = json.loads(line)
                                
                                # Track first token time
                                if not first_token_received:
                                    first_token_time = time.time() - start_time
                                    self.first_token_times.append(first_token_time)
                                    first_token_received = True
                                    if settings.debug_mode:
                                        print(f"⚡ First token: {first_token_time:.2f}s")
                                
                                # Yield the response chunk
                                if 'response' in data:
                                    yield data['response']
                                
                                # Check if done
                                if data.get('done', False):
                                    # Track total time
                                    total_time = time.time() - start_time
                                    self.inference_times.append(total_time)
                                    if settings.debug_mode:
                                        print(f"⏱️ Total time: {total_time:.2f}s")
                                    break
                                    
                        except json.JSONDecodeError:
                            continue
                else:
                    yield f"Error: Received status {response.status}"
                    
        except asyncio.TimeoutError:
            yield "Response timeout - trying faster model..."
            # Try to switch to a faster fallback
            await self._fallback_to_faster_model()
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Stream generation error: {e}")
            yield f"I encountered an error. Let me try again..."
    
    async def generate_response(self, query: str, personality_context: str, 
                              memory_context: str, profile: str = None) -> str:
        """Generate complete response (non-streaming fallback)"""
        if not self.model_loaded or not self.current_model:
            return await self._fallback_response(query)
        
        try:
            start_time = time.time()
            
            # Build optimized prompt
            prompt = self._build_lightning_prompt(query, personality_context, memory_context)
            
            # Generate response
            payload = {
                "model": self.current_model.name,
                "prompt": prompt,
                "options": self.generation_config,
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            async with self.session.post(
                f"{self.ollama_host}/api/generate", 
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    inference_time = time.time() - start_time
                    self.inference_times.append(inference_time)
                    
                    # Keep only last 20 measurements
                    if len(self.inference_times) > 20:
                        self.inference_times = self.inference_times[-20:]
                    
                    if settings.debug_mode:
                        print(f"⏱️ Response time: {inference_time:.2f}s")
                    
                    return data.get('response', '')
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                    
        except Exception as e:
            if settings.debug_mode:
                print(f"Generation error: {e}")
            
            # Try fallback model
            if await self._fallback_to_faster_model():
                return await self.generate_response(query, personality_context, memory_context)
            else:
                return await self._fallback_response(query)
    
    async def _fallback_to_faster_model(self) -> bool:
        """Switch to next available model in hierarchy"""
        if not self.current_model:
            return False
        
        current_priority = self.current_model.priority
        
        # Find next model in hierarchy
        for model in self.available_models:
            if model.priority > current_priority and model.priority <= 3:
                if await self._load_model(model):
                    self.current_model = model
                    print(f"⚡ Switched to fallback: {model.name}")
                    return True
        
        return False
    
    def _build_lightning_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Build optimized prompt for lightning-fast responses"""
        # Determine model type for optimal prompting
        if not self.current_model:
            return query
        
        model_name = self.current_model.name.lower()
        
        # Keep prompts minimal for speed
        if 'nemotron' in model_name:
            return self._build_nemotron_prompt(query, personality_context)
        elif 'qwen' in model_name:
            return self._build_qwen_prompt(query, personality_context)
        elif 'gemma' in model_name:
            return self._build_gemma_prompt(query, personality_context)
        else:
            return self._build_generic_prompt(query, personality_context)
    
    def _build_nemotron_prompt(self, query: str, personality_context: str) -> str:
        """Optimized prompt for Nemotron models"""
        # Nemotron typically uses a simple format
        if personality_context:
            return f"System: {personality_context[:100]}\n\nUser: {query}\n\nAssistant:"
        else:
            return f"User: {query}\n\nAssistant:"
    
    def _build_qwen_prompt(self, query: str, personality_context: str) -> str:
        """Optimized prompt for Qwen models"""
        # Qwen uses special tokens
        system_content = personality_context[:100] if personality_context else "You are Pascal, a helpful AI assistant."
        return f"<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    
    def _build_gemma_prompt(self, query: str, personality_context: str) -> str:
        """Optimized prompt for Gemma models"""
        # Gemma format
        if personality_context:
            return f"<start_of_turn>user\n{personality_context[:100]}\n\n{query}<end_of_turn>\n<start_of_turn>model\n"
        else:
            return f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
    
    def _build_generic_prompt(self, query: str, personality_context: str) -> str:
        """Generic prompt format"""
        if personality_context:
            return f"{personality_context[:100]}\n\nUser: {query}\n\nAssistant:"
        else:
            return f"User: {query}\n\nAssistant:"
    
    async def _fallback_response(self, query: str) -> str:
        """Provide fallback response when model unavailable"""
        query_lower = query.lower()
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm Pascal, but I'm currently having trouble with my language model. Please give me a moment..."
        elif '?' in query:
            return "I'd like to help with that, but my language model isn't responding right now. Please try again in a moment."
        else:
            return "I'm experiencing a technical issue. Please try again shortly."
    
    def set_performance_profile(self, profile: str):
        """Adjust settings for different performance profiles"""
        if profile == 'speed':
            self.generation_config['num_predict'] = 100
            self.generation_config['temperature'] = 0.5
        elif profile == 'balanced':
            self.generation_config['num_predict'] = 150
            self.generation_config['temperature'] = 0.7
        elif profile == 'quality':
            self.generation_config['num_predict'] = 200
            self.generation_config['temperature'] = 0.8
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        target_model = next((m for m in self.available_models if model_name in m.name), None)
        
        if not target_model:
            return False
        
        if await self._load_model(target_model):
            self.current_model = target_model
            print(f"✅ Switched to model: {target_model.name}")
            return True
        
        return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        stats = {
            "ollama_enabled": True,
            "model_loaded": self.model_loaded,
            "current_model": self.current_model.name if self.current_model else None,
            "model_priority": self.current_model.priority if self.current_model else None,
            "keep_alive_active": self.keep_alive_task and not self.keep_alive_task.done() if self.keep_alive_task else False,
            "ollama_host": self.ollama_host
        }
        
        if self.inference_times:
            stats.update({
                "avg_inference_time": f"{sum(self.inference_times) / len(self.inference_times):.2f}s",
                "min_inference_time": f"{min(self.inference_times):.2f}s",
                "max_inference_time": f"{max(self.inference_times):.2f}s",
                "total_inferences": len(self.inference_times)
            })
        
        if self.first_token_times:
            stats["avg_first_token_time"] = f"{sum(self.first_token_times) / len(self.first_token_times):.2f}s"
        
        return stats
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their stats"""
        return [
            {
                "name": model.name,
                "size": model.size,
                "priority": model.priority,
                "loaded": model == self.current_model,
                "is_primary": model.priority == 1,
                "is_fallback": model.priority in [2, 3]
            }
            for model in self.available_models
        ]
    
    async def pull_model(self, model_name: str) -> bool:
        """Download a new model using Ollama"""
        try:
            payload = {"name": model_name, "stream": False}
            
            async with self.session.post(
                f"{self.ollama_host}/api/pull", 
                json=payload
            ) as response:
                if response.status == 200:
                    await self._scan_available_models()
                    print(f"✅ Downloaded model: {model_name}")
                    return True
                else:
                    print(f"❌ Failed to download model: {model_name}")
                    return False
                    
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False
    
    async def remove_model(self, model_name: str) -> bool:
        """Remove a model using Ollama"""
        try:
            # Don't remove current model
            if self.current_model and model_name in self.current_model.name:
                print("❌ Cannot remove currently loaded model")
                return False
            
            payload = {"name": model_name}
            
            async with self.session.delete(
                f"{self.ollama_host}/api/delete", 
                json=payload
            ) as response:
                if response.status == 200:
                    await self._scan_available_models()
                    print(f"✅ Removed model: {model_name}")
                    return True
                else:
                    print(f"❌ Failed to remove model: {model_name}")
                    return False
                    
        except Exception as e:
            print(f"❌ Remove error: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if offline LLM is ready"""
        return self.model_loaded and self.current_model is not None
    
    async def close(self):
        """Clean shutdown"""
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
        
        self.model_loaded = False
        self.current_model = None

# For backwards compatibility with existing code
OptimizedOfflineLLM = LightningOfflineLLM
