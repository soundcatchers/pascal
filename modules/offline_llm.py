"""
Pascal AI Assistant - Lightning-Fast Offline LLM with Streaming
Optimized for sub-3-second responses on Raspberry Pi 5 with keep-alive
"""

import asyncio
import time
import json
import aiohttp
import os
from typing import Optional, Dict, Any, List, AsyncGenerator
from pathlib import Path

from config.settings import settings

class ModelInfo:
    """Information about available Ollama models"""
    def __init__(self, name: str, size: str = "Unknown", parameters: str = "Unknown", modified: str = ""):
        self.name = name
        self.size = size
        self.parameters = parameters
        self.modified = modified
        self.priority = self._get_priority()
        self.loaded = False
    
    def _get_priority(self) -> int:
        """Get model priority (1 = highest)"""
        # Priority order for specific models
        if "nemotron-mini:4b-instruct-q4_K_M" in self.name:
            return 1
        elif "qwen3:4b-instruct" in self.name:
            return 2
        elif "gemma3:4b-it-q4_K_M" in self.name:
            return 3
        elif "phi" in self.name.lower():
            return 4
        elif "llama" in self.name.lower():
            return 5
        elif "gemma" in self.name.lower():
            return 6
        else:
            return 99

class LightningOfflineLLM:
    """Lightning-fast Ollama-based offline LLM with streaming and keep-alive"""
    
    def __init__(self):
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.session = None
        self.current_model = None
        self.available_models = []
        self.model_loaded = False
        self.keep_alive_task = None
        self.ollama_available = False
        
        # Primary and fallback models
        self.model_hierarchy = [
            "nemotron-mini:4b-instruct-q4_K_M",  # Primary - fastest
            "qwen3:4b-instruct",                  # Fallback 1
            "gemma3:4b-it-q4_K_M",                # Fallback 2
            "phi3:mini",                          # Fallback 3
            "llama3.2:3b",                        # Fallback 4
            "gemma2:2b"                           # Fallback 5
        ]
        
        # Performance tracking
        self.inference_times = []
        self.first_token_times = []
        
        # Dynamic timeout management
        self.timeout_config = {
            'simple': 15.0,    # Simple queries get 15 seconds
            'medium': 25.0,    # Medium complexity gets 25 seconds
            'complex': 40.0,   # Complex queries get 40 seconds
            'default': 20.0    # Default timeout
        }
        
        # Smart response configuration based on query type
        self.response_configs = {
            'speed': {
                'temperature': 0.5,
                'top_p': 0.9,
                'top_k': 30,
                'repeat_penalty': 1.2,
                'num_predict': 100,  # Very concise
                'num_ctx': 1024,
                'seed': -1,
                'stop': ["</s>", "<|end|>", "<|eot_id|>", "Human:", "User:", "\n\n", "\n\nIs there", "\n\nWould you"]
            },
            'balanced': {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'repeat_penalty': 1.1,
                'num_predict': 200,  # Moderate length
                'num_ctx': 2048,
                'seed': -1,
                'stop': ["</s>", "<|end|>", "<|eot_id|>", "Human:", "User:", "\n\n"]
            },
            'quality': {
                'temperature': 0.8,
                'top_p': 0.95,
                'top_k': 50,
                'repeat_penalty': 1.0,
                'num_predict': 400,  # More detailed - standard is 300 for complex queries make larger for more detail
                'num_ctx': 2048,
                'seed': -1,
                'stop': ["</s>", "<|end|>", "<|eot_id|>", "Human:", "User:"]
            }
        }
        
        # Default to balanced
        self.generation_config = self.response_configs['balanced'].copy()
        
        # Keep-alive configuration
        self.keep_alive_interval = 30  # seconds
        self.keep_alive_duration = "30m"  # Keep model loaded for 5 minutes (5m)
    
    def analyze_query_complexity(self, query: str) -> tuple[str, float]:
        """Analyze query to determine complexity and appropriate timeout"""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Keywords indicating complexity
        complex_keywords = ['explain', 'analyze', 'compare', 'describe', 'how does', 'what is the', 
                          'tell me about', 'can you explain', 'detailed', 'comprehensive', 'elaborate',
                          'walk me through', 'step by step', 'photosynthesis', 'physics', 'algorithm']
        
        simple_keywords = ['hi', 'hello', 'thanks', 'yes', 'no', 'ok', 'what time', 'what day',
                         'what color', 'how far', 'capital of', 'who is', 'when was']
        
        technical_keywords = ['code', 'programming', 'function', 'algorithm', 'implement', 'debug',
                            'syntax', 'compile', 'database', 'api', 'framework']
        
        # Check for question types
        is_complex = any(keyword in query_lower for keyword in complex_keywords)
        is_simple = any(keyword in query_lower for keyword in simple_keywords)
        is_technical = any(keyword in query_lower for keyword in technical_keywords)
        
        # Determine complexity and timeout
        if is_simple and word_count < 10:
            return 'simple', self.timeout_config['simple']
        elif is_complex or is_technical or word_count > 30:
            return 'complex', self.timeout_config['complex']
        elif word_count > 15:
            return 'medium', self.timeout_config['medium']
        else:
            return 'simple', self.timeout_config['simple']
    
    async def initialize(self) -> bool:
        """Initialize with lightning-fast configuration"""
        try:
            # Create optimized aiohttp session with longer timeout for complex queries
            timeout = aiohttp.ClientTimeout(total=60, sock_connect=5, sock_read=45)
            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            
            # Test Ollama connection with retries
            if not await self._test_ollama_connection():
                print("❌ Could not connect to Ollama. Is it running?")
                print("   Start with: sudo systemctl start ollama")
                self.ollama_available = False
                return False
            
            self.ollama_available = True
            
            # Scan for available models
            await self._scan_available_models()
            
            if not self.available_models:
                print("❌ No models found in Ollama")
                print("   Download models with: ./download_models.sh")
                return False
            
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
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
            return False
    
    async def _test_ollama_connection(self) -> bool:
        """Test connection to Ollama service with retries"""
        for attempt in range(3):
            try:
                async with self.session.get(f"{self.ollama_host}/api/version") as response:
                    if response.status == 200:
                        version_data = await response.json()
                        if settings.debug_mode:
                            print(f"✅ Connected to Ollama version: {version_data.get('version', 'unknown')}")
                        return True
            except aiohttp.ClientError as e:
                if settings.debug_mode:
                    print(f"Ollama connection attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(2)
            except Exception as e:
                if settings.debug_mode:
                    print(f"Unexpected error connecting to Ollama: {e}")
                if attempt < 2:
                    await asyncio.sleep(2)
        
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
                        print(f"Found {len(self.available_models)} models:")
                        for model in self.available_models[:5]:
                            print(f"  • {model.name} (Priority: {model.priority}, Size: {model.size})")
                
        except Exception as e:
            print(f"Failed to scan models: {e}")
            self.available_models = []
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format"""
        if size_bytes == 0:
            return "Unknown"
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"
    
    async def _load_primary_model(self) -> bool:
        """Load the primary model or first available fallback"""
        # First try our preferred models
        for model_name in self.model_hierarchy:
            # Check if model is available
            model = next((m for m in self.available_models if model_name in m.name), None)
            if model:
                if await self._load_model(model):
                    self.current_model = model
                    self.model_loaded = True
                    model.loaded = True
                    print(f"✅ Loaded model: {model.name}")
                    return True
                else:
                    print(f"⚠️ Failed to load {model_name}, trying next...")
        
        # If none of our preferred models are available, try any available model
        if self.available_models:
            print("⚠️ No preferred models found, trying any available model...")
            for model in self.available_models:
                if await self._load_model(model):
                    self.current_model = model
                    self.model_loaded = True
                    model.loaded = True
                    print(f"✅ Loaded fallback model: {model.name}")
                    return True
        
        return False
    
    async def _load_model(self, model: ModelInfo) -> bool:
        """Load a specific model with keep-alive"""
        try:
            # First, ensure the model is loaded with keep-alive
            payload = {
                "model": model.name,
                "keep_alive": self.keep_alive_duration,
                "prompt": "",
                "stream": False
            }
            
            async with self.session.post(
                f"{self.ollama_host}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    # Model loaded successfully
                    return True
                else:
                    error_text = await response.text()
                    if settings.debug_mode:
                        print(f"Failed to load model {model.name}: {error_text}")
                    return False
                    
        except asyncio.TimeoutError:
            print(f"⚠️ Timeout loading model {model.name}")
            return False
        except Exception as e:
            if settings.debug_mode:
                print(f"Failed to load model {model.name}: {e}")
        
        return False
    
    async def _keep_alive_loop(self):
        """Background task to keep model loaded in memory"""
        while self.model_loaded and self.current_model:
            try:
                await asyncio.sleep(self.keep_alive_interval)
                
                if not self.current_model:
                    break
                
                # Send keep-alive request
                payload = {
                    "model": self.current_model.name,
                    "keep_alive": self.keep_alive_duration,
                    "prompt": "",
                    "stream": False
                }
                
                async with self.session.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload
                ) as response:
                    if response.status != 200:
                        if settings.debug_mode:
                            print(f"⚠️ Keep-alive failed for {self.current_model.name}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                if settings.debug_mode:
                    print(f"Keep-alive error: {e}")
    
    async def generate_response_stream(self, query: str, personality_context: str, 
                                     memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response for instant feedback with smart timeout"""
        if not self.model_loaded or not self.current_model:
            yield "Offline model is not available. Please check Ollama is running and models are downloaded."
            return
        
        try:
            start_time = time.time()
            first_token_received = False
            
            # Analyze query complexity and get appropriate timeout
            complexity, timeout_seconds = self.analyze_query_complexity(query)
            
            # Adjust generation config based on complexity
            if complexity == 'simple':
                self.generation_config = self.response_configs['speed'].copy()
                if settings.debug_mode:
                    print(f"📝 Simple query - concise mode (timeout: {timeout_seconds}s)")
            elif complexity == 'complex':
                self.generation_config = self.response_configs['quality'].copy()
                if settings.debug_mode:
                    print(f"📚 Complex query - detailed mode (timeout: {timeout_seconds}s)")
            else:
                self.generation_config = self.response_configs['balanced'].copy()
                if settings.debug_mode:
                    print(f"⚖️ Medium query - balanced mode (timeout: {timeout_seconds}s)")
            
            # Build optimized prompt
            prompt = self._build_lightning_prompt(query, personality_context, memory_context, complexity)
            
            # Prepare streaming request
            payload = {
                "model": self.current_model.name,
                "prompt": prompt,
                "options": self.generation_config,
                "stream": True,
                "keep_alive": self.keep_alive_duration
            }
            
            # Create timeout for this specific query
            query_timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            
            # Stream response with dynamic timeout
            async with self.session.post(
                f"{self.ollama_host}/api/generate", 
                json=payload,
                timeout=query_timeout
            ) as response:
                if response.status == 200:
                    response_buffer = []
                    token_count = 0
                    last_meaningful_time = start_time
                    
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
                                    chunk = data['response']
                                    response_buffer.append(chunk)
                                    token_count += 1
                                    
                                    # Track if we're getting meaningful content
                                    if chunk.strip():
                                        last_meaningful_time = time.time()
                                    
                                    yield chunk
                                    
                                    # Check if we've been generating for too long without meaningful content
                                    if time.time() - last_meaningful_time > 5.0:
                                        if settings.debug_mode:
                                            print("⚠️ No meaningful content for 5s, stopping")
                                        break
                                
                                # Check if done
                                if data.get('done', False):
                                    # Track total time
                                    total_time = time.time() - start_time
                                    self.inference_times.append(total_time)
                                    if settings.debug_mode:
                                        print(f"⏱️ Total time: {total_time:.2f}s, Tokens: {token_count}")
                                    break
                                    
                        except json.JSONDecodeError:
                            continue
                else:
                    error_text = await response.text()
                    yield f"Ollama error (status {response.status}): {error_text[:100]}"
                    
        except asyncio.TimeoutError:
            total_time = time.time() - start_time
            if settings.debug_mode:
                print(f"⏱️ Timeout after {total_time:.2f}s (limit was {timeout_seconds}s)")
            
            # Don't show timeout message for complex queries that got a good amount of response
            if complexity == 'complex' and first_token_received:
                yield "\n\n[Response completed]"
            else:
                yield "\n\n[Response time limit reached. Try rephrasing for a more concise answer.]"
            
        except aiohttp.ClientError as e:
            yield f"Connection error with Ollama: {str(e)}"
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Stream generation error: {e}")
                import traceback
                traceback.print_exc()
            yield f"An error occurred while generating response: {str(e)}"
    
    async def generate_response(self, query: str, personality_context: str, 
                              memory_context: str, profile: str = None) -> str:
        """Generate complete response (non-streaming fallback) with smart timeout"""
        if not self.model_loaded or not self.current_model:
            return "Offline model is not available. Please check Ollama is running and models are downloaded."
        
        try:
            start_time = time.time()
            
            # Analyze query complexity
            complexity, timeout_seconds = self.analyze_query_complexity(query)
            
            # Set appropriate config
            if profile:
                self.set_performance_profile(profile)
            elif complexity == 'simple':
                self.generation_config = self.response_configs['speed'].copy()
            elif complexity == 'complex':
                self.generation_config = self.response_configs['quality'].copy()
            else:
                self.generation_config = self.response_configs['balanced'].copy()
            
            # Build optimized prompt
            prompt = self._build_lightning_prompt(query, personality_context, memory_context, complexity)
            
            # Generate response
            payload = {
                "model": self.current_model.name,
                "prompt": prompt,
                "options": self.generation_config,
                "stream": False,
                "keep_alive": self.keep_alive_duration
            }
            
            # Use dynamic timeout
            query_timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            
            async with self.session.post(
                f"{self.ollama_host}/api/generate", 
                json=payload,
                timeout=query_timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    inference_time = time.time() - start_time
                    self.inference_times.append(inference_time)
                    
                    # Keep only last 20 measurements
                    if len(self.inference_times) > 20:
                        self.inference_times = self.inference_times[-20:]
                    
                    if settings.debug_mode:
                        print(f"⏱️ Response time: {inference_time:.2f}s (complexity: {complexity})")
                    
                    return data.get('response', 'No response generated')
                else:
                    error_text = await response.text()
                    return f"Ollama API error {response.status}: {error_text[:200]}"
                    
        except asyncio.TimeoutError:
            return f"Response exceeded time limit ({timeout_seconds}s). Try asking for a more concise answer or breaking the question into parts."
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Generation error: {e}")
                import traceback
                traceback.print_exc()
            
            return f"An error occurred: {str(e)}"
    
    def _build_lightning_prompt(self, query: str, personality_context: str, memory_context: str, complexity: str = 'medium') -> str:
        """Build optimized prompt for lightning-fast responses"""
        # Keep prompts minimal for speed
        if not self.current_model:
            return query
        
        model_name = self.current_model.name.lower()
        
        # Add instructions based on complexity
        instructions = ""
        if complexity == 'simple':
            instructions = "Provide a brief, direct answer. "
        elif complexity == 'complex':
            instructions = "Provide a comprehensive but well-structured answer. "
        else:
            instructions = "Provide a clear, balanced response. "
        
        # Very minimal context for speed
        context = personality_context[:200] if personality_context else ""
        
        # Model-specific formatting with instructions
        if 'llama' in model_name:
            return f"{context}\n\n{instructions}User: {query}\n\nAssistant:"
        elif 'gemma' in model_name:
            return f"<start_of_turn>user\n{instructions}{query}<end_of_turn>\n<start_of_turn>model\n"
        elif 'phi' in model_name:
            return f"{instructions}User: {query}\n\nAssistant:"
        else:
            # Generic format
            return f"{context}\n\n{instructions}User: {query}\n\nAssistant:"
    
    def set_performance_profile(self, profile: str):
        """Adjust settings for different performance profiles"""
        if profile == 'speed':
            self.generation_config = self.response_configs['speed'].copy()
        elif profile == 'balanced':
            self.generation_config = self.response_configs['balanced'].copy()
        elif profile == 'quality':
            self.generation_config = self.response_configs['quality'].copy()
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        # Find the model
        target_model = None
        for model in self.available_models:
            if model_name.lower() in model.name.lower():
                target_model = model
                break
        
        if not target_model:
            print(f"Model {model_name} not found")
            return False
        
        # Unload current model
        if self.current_model:
            self.current_model.loaded = False
        
        # Load new model
        if await self._load_model(target_model):
            self.current_model = target_model
            target_model.loaded = True
            print(f"✅ Switched to model: {target_model.name}")
            return True
        
        return False
    
    async def pull_model(self, model_name: str) -> bool:
        """Download a new model using Ollama"""
        try:
            print(f"📥 Pulling model {model_name}...")
            payload = {"name": model_name, "stream": False}
            
            async with self.session.post(
                f"{self.ollama_host}/api/pull", 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600)  # 10 minutes for download
            ) as response:
                if response.status == 200:
                    # Rescan available models
                    await self._scan_available_models()
                    print(f"✅ Downloaded model: {model_name}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to download model: {error_text}")
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
                    print(f"❌ Failed to remove model")
                    return False
                    
        except Exception as e:
            print(f"❌ Remove error: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        stats = {
            "ollama_enabled": self.ollama_available,
            "ollama_host": self.ollama_host,
            "model_loaded": self.model_loaded,
            "current_model": self.current_model.name if self.current_model else None,
            "model_priority": self.current_model.priority if self.current_model else None,
            "keep_alive_active": self.keep_alive_task and not self.keep_alive_task.done() if self.keep_alive_task else False
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
                "loaded": model.loaded,
                "is_primary": model.priority == 1,
                "is_fallback": model.priority in [2, 3, 4, 5, 6]
            }
            for model in self.available_models
        ]
    
    def is_available(self) -> bool:
        """Check if offline LLM is ready"""
        return self.ollama_available and self.model_loaded and self.current_model is not None
    
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
        self.ollama_available = False

# For backwards compatibility
OptimizedOfflineLLM = LightningOfflineLLM
