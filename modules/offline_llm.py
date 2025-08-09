"""
Pascal AI Assistant - Optimized Offline LLM for Raspberry Pi 5 (Ollama Version)
High-performance local model inference using Ollama with ARM-specific optimizations
"""

import asyncio
import time
import json
import aiohttp
from typing import Optional, Dict, Any, List
from pathlib import Path

from config.settings import settings

class ModelInfo:
    """Information about available Ollama models"""
    def __init__(self, name: str, size: str, parameters: str, modified: str = ""):
        self.name = name
        self.size = size
        self.parameters = parameters
        self.modified = modified
        
        # Estimate performance characteristics
        self.speed_rating = self._estimate_speed_rating()
        self.quality_rating = self._estimate_quality_rating()
        self.ram_usage = self._estimate_ram_usage()
    
    def _estimate_speed_rating(self) -> int:
        """Estimate speed rating 1-10 based on model size"""
        if "2b" in self.name.lower():
            return 9
        elif "3b" in self.name.lower():
            return 8
        elif "7b" in self.name.lower():
            return 6
        elif "mini" in self.name.lower():
            return 9
        else:
            return 7
    
    def _estimate_quality_rating(self) -> int:
        """Estimate quality rating 1-10 based on model type"""
        if "qwen" in self.name.lower():
            return 9
        elif "llama" in self.name.lower():
            return 8
        elif "phi" in self.name.lower():
            return 7
        elif "gemma" in self.name.lower():
            return 7
        else:
            return 6
    
    def _estimate_ram_usage(self) -> float:
        """Estimate RAM usage in GB"""
        if "2b" in self.name.lower():
            return 2.0
        elif "3b" in self.name.lower():
            return 2.5
        elif "7b" in self.name.lower():
            return 4.5
        elif "mini" in self.name.lower():
            return 2.3
        else:
            return 3.0

class OptimizedOfflineLLM:
    """Ollama-based offline LLM with intelligent model management for Pi 5"""
    
    def __init__(self):
        self.ollama_host = "http://localhost:11434"
        self.session = None
        self.current_model = None
        self.available_models = []
        self.model_loaded = False
        
        # Performance tracking
        self.inference_times = []
        self.tokens_per_second = []
        
        # Performance profiles optimized for Pi 5
        self.performance_profiles = {
            'speed': {
                'temperature': 0.3,
                'top_p': 0.8,
                'max_tokens': 100,
                'preferred_models': ['phi3:mini', 'gemma2:2b'],
                'stream': False
            },
            'balanced': {
                'temperature': 0.7,
                'top_p': 0.9,
                'max_tokens': 200,
                'preferred_models': ['llama3.2:3b', 'phi3:mini'],
                'stream': False
            },
            'quality': {
                'temperature': 0.8,
                'top_p': 0.95,
                'max_tokens': 300,
                'preferred_models': ['qwen2.5:7b', 'llama3.2:3b'],
                'stream': False
            }
        }
        
        self.current_profile = 'balanced'
        
        # Load Ollama configuration if available
        self._load_ollama_config()
    
    def _load_ollama_config(self):
        """Load Ollama configuration from file"""
        config_file = settings.models_dir / "ollama_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                if config.get('ollama_host'):
                    self.ollama_host = config['ollama_host']
                
                # Update performance profiles from config
                if 'performance_profiles' in config:
                    for profile_name, profile_config in config['performance_profiles'].items():
                        if profile_name in self.performance_profiles:
                            self.performance_profiles[profile_name].update(profile_config)
                
                if settings.debug_mode:
                    print(f"Loaded Ollama config from {config_file}")
                    
            except Exception as e:
                if settings.debug_mode:
                    print(f"Failed to load Ollama config: {e}")
    
    async def initialize(self) -> bool:
        """Initialize Ollama connection and scan for models"""
        try:
            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Test Ollama connection
            if not await self._test_ollama_connection():
                print("❌ Could not connect to Ollama. Is it running?")
                print("   Start with: sudo systemctl start ollama")
                return False
            
            # Scan for available models
            await self._scan_available_models()
            
            if not self.available_models:
                print("❌ No models found. Download models first:")
                print("   ./download_models.sh")
                return False
            
            # Select best initial model
            best_model = await self._select_best_model()
            if best_model:
                self.current_model = best_model
                self.model_loaded = True
                print(f"✅ Ollama initialized with model: {best_model.name}")
                return True
            else:
                print("❌ No suitable model found")
                return False
                
        except Exception as e:
            print(f"❌ Ollama initialization failed: {e}")
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
        """Scan for available Ollama models"""
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
                    
                    # Sort by speed rating for Pi 5
                    self.available_models.sort(key=lambda x: x.speed_rating, reverse=True)
                    
                    if settings.debug_mode:
                        print(f"Found {len(self.available_models)} models")
                        for model in self.available_models:
                            print(f"  • {model.name} ({model.size}) - Speed: {model.speed_rating}/10")
                
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
    
    async def _select_best_model(self) -> Optional[ModelInfo]:
        """Select the best model for current profile and Pi 5"""
        if not self.available_models:
            return None
        
        profile = self.performance_profiles[self.current_profile]
        preferred_models = profile.get('preferred_models', [])
        
        # Try to find preferred models first
        for preferred in preferred_models:
            for model in self.available_models:
                if preferred in model.name:
                    return model
        
        # Fall back to fastest available model
        return self.available_models[0]  # Already sorted by speed rating
    
    async def generate_response(self, query: str, personality_context: str, 
                              memory_context: str, profile: str = None) -> str:
        """Generate response using Ollama"""
        if not self.model_loaded or not self.current_model:
            return await self._fallback_response(query)
        
        # Use specified profile or current default
        profile_name = profile or self.current_profile
        profile_settings = self.performance_profiles.get(profile_name, 
                                                       self.performance_profiles['balanced'])
        
        try:
            start_time = time.time()
            
            # Build optimized prompt
            prompt = self._build_optimized_prompt(query, personality_context, memory_context)
            
            # Generate response using Ollama API
            response = await self._call_ollama_generate(prompt, profile_settings)
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only last 20 measurements
            if len(self.inference_times) > 20:
                self.inference_times = self.inference_times[-20:]
            
            # Clean response
            cleaned_response = self._clean_response(response)
            
            return cleaned_response
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Generation error: {e}")
            return await self._fallback_response(query)
    
    async def _call_ollama_generate(self, prompt: str, profile_settings: Dict[str, Any]) -> str:
        """Call Ollama generate API"""
        payload = {
            "model": self.current_model.name,
            "prompt": prompt,
            "options": {
                "temperature": profile_settings.get('temperature', 0.7),
                "top_p": profile_settings.get('top_p', 0.9),
                "num_predict": profile_settings.get('max_tokens', 200),
                "stop": ["</s>", "<|end|>", "<|eot_id|>", "Human:", "User:"]
            },
            "stream": profile_settings.get('stream', False)
        }
        
        async with self.session.post(
            f"{self.ollama_host}/api/generate", 
            json=payload
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('response', '')
            else:
                error_text = await response.text()
                raise Exception(f"Ollama API error {response.status}: {error_text}")
    
    def _build_optimized_prompt(self, query: str, personality_context: str, 
                               memory_context: str) -> str:
        """Build optimized prompt based on model type"""
        if not self.current_model:
            return query
        
        model_name = self.current_model.name.lower()
        
        # Different prompt formats for different models
        if 'phi' in model_name:
            return self._build_phi_prompt(query, personality_context, memory_context)
        elif 'llama' in model_name:
            return self._build_llama_prompt(query, personality_context, memory_context)
        elif 'gemma' in model_name:
            return self._build_gemma_prompt(query, personality_context, memory_context)
        elif 'qwen' in model_name:
            return self._build_qwen_prompt(query, personality_context, memory_context)
        else:
            return self._build_generic_prompt(query, personality_context, memory_context)
    
    def _build_phi_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Optimized prompt for Phi models"""
        if personality_context:
            return f"<|system|>\n{personality_context}<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>\n"
        else:
            return f"<|user|>\n{query}<|end|>\n<|assistant|>\n"
    
    def _build_llama_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Optimized prompt for Llama models"""
        system_content = personality_context if personality_context else "You are Pascal, a helpful AI assistant."
        
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    
    def _build_gemma_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Optimized prompt for Gemma models"""
        prompt_parts = ["<bos>"]
        
        if personality_context:
            prompt_parts.extend([
                "<start_of_turn>user",
                f"You are Pascal. {personality_context[:200]}",
                "<end_of_turn>",
                "<start_of_turn>model",
                "I understand. I'm Pascal, ready to help!",
                "<end_of_turn>"
            ])
        
        prompt_parts.extend([
            "<start_of_turn>user",
            query,
            "<end_of_turn>",
            "<start_of_turn>model"
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_qwen_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Optimized prompt for Qwen models"""
        system_content = personality_context if personality_context else "You are Pascal, a helpful AI assistant."
        
        return f"<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    
    def _build_generic_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generic prompt format"""
        prompt_parts = []
        
        if personality_context:
            prompt_parts.append(f"System: {personality_context}")
        
        prompt_parts.append(f"User: {query}")
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format response"""
        # Remove common artifacts
        artifacts = ['</s>', '<|end|>', '<|eot_id|>', '<|endoftext|>', '<|im_end|>', '<|end_of_turn|>']
        for artifact in artifacts:
            response = response.replace(artifact, '')
        
        # Clean up whitespace
        response = response.strip()
        
        # Remove incomplete sentences at the end
        if response and not response[-1] in '.!?':
            sentences = response.split('.')
            if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
                response = '.'.join(sentences[:-1]) + '.'
        
        return response
    
    async def _fallback_response(self, query: str) -> str:
        """Provide fallback response when model unavailable"""
        fallback_map = {
            'greeting': "Hello! I'm Pascal, but I'm currently running in limited mode.",
            'question': "I'd like to help with that question, but my language model isn't fully loaded right now.",
            'default': "I'm running in limited mode right now. Please try again in a moment."
        }
        
        query_lower = query.lower()
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return fallback_map['greeting']
        elif '?' in query:
            return fallback_map['question']
        else:
            return fallback_map['default']
    
    def set_performance_profile(self, profile: str):
        """Set performance profile and switch to optimal model if needed"""
        if profile in self.performance_profiles:
            old_profile = self.current_profile
            self.current_profile = profile
            
            # Check if we should switch models for this profile
            asyncio.create_task(self._switch_to_optimal_model_async(profile))
            
            if settings.debug_mode:
                print(f"Set performance profile: {old_profile} → {profile}")
    
    async def _switch_to_optimal_model_async(self, profile: str):
        """Switch to optimal model for profile (async)"""
        profile_settings = self.performance_profiles[profile]
        preferred_models = profile_settings.get('preferred_models', [])
        
        # Find best available model for this profile
        for preferred in preferred_models:
            for model in self.available_models:
                if preferred in model.name and model != self.current_model:
                    if await self.switch_model(model.name):
                        if settings.debug_mode:
                            print(f"Switched to {model.name} for {profile} profile")
                        return
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different available model"""
        target_model = next((m for m in self.available_models if model_name in m.name), None)
        
        if not target_model:
            return False
        
        try:
            # Test the model by making a simple request
            test_payload = {
                "model": target_model.name,
                "prompt": "Hello",
                "options": {"num_predict": 5}
            }
            
            async with self.session.post(
                f"{self.ollama_host}/api/generate", 
                json=test_payload
            ) as response:
                if response.status == 200:
                    self.current_model = target_model
                    if settings.debug_mode:
                        print(f"Switched to model: {target_model.name}")
                    return True
                else:
                    if settings.debug_mode:
                        print(f"Model switch failed: {response.status}")
                    return False
                    
        except Exception as e:
            if settings.debug_mode:
                print(f"Model switch error: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        if not self.inference_times:
            return {"status": "No inference data available"}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        min_time = min(self.inference_times)
        max_time = max(self.inference_times)
        
        return {
            "ollama_enabled": True,
            "model_loaded": self.model_loaded,
            "current_model": self.current_model.name if self.current_model else None,
            "model_size": self.current_model.size if self.current_model else None,
            "model_ram_usage": f"{self.current_model.ram_usage:.1f}GB" if self.current_model else None,
            "performance_profile": self.current_profile,
            "avg_inference_time": f"{avg_time:.2f}s",
            "min_inference_time": f"{min_time:.2f}s",
            "max_inference_time": f"{max_time:.2f}s",
            "total_inferences": len(self.inference_times),
            "available_models": len(self.available_models),
            "ollama_host": self.ollama_host
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their stats"""
        return [
            {
                "name": model.name,
                "size": model.size,
                "speed_rating": f"{model.speed_rating}/10",
                "quality_rating": f"{model.quality_rating}/10",
                "ram_usage": f"{model.ram_usage:.1f}GB",
                "loaded": model == self.current_model
            }
            for model in self.available_models
        ]
    
    async def pull_model(self, model_name: str) -> bool:
        """Download a new model using Ollama"""
        try:
            payload = {"name": model_name}
            
            async with self.session.post(
                f"{self.ollama_host}/api/pull", 
                json=payload
            ) as response:
                if response.status == 200:
                    # Refresh available models
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
            payload = {"name": model_name}
            
            async with self.session.delete(
                f"{self.ollama_host}/api/delete", 
                json=payload
            ) as response:
                if response.status == 200:
                    # Refresh available models
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
        if self.session:
            await self.session.close()
        self.model_loaded = False
        self.current_model = None
