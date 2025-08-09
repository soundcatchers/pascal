"""
Pascal AI Assistant - Optimized Offline LLM for Raspberry Pi 5
High-performance local model inference with ARM-specific optimizations
"""

import asyncio
import time
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings

class ModelInfo:
    """Information about available models"""
    def __init__(self, name: str, file_path: str, context_size: int, 
                 ram_usage: float, speed_rating: int, quality_rating: int):
        self.name = name
        self.file_path = file_path
        self.context_size = context_size
        self.ram_usage = ram_usage  # GB
        self.speed_rating = speed_rating  # 1-10
        self.quality_rating = quality_rating  # 1-10

class OptimizedOfflineLLM:
    """ARM-optimized offline LLM with intelligent model management"""
    
    def __init__(self):
        self.model = None
        self.model_info = None
        self.model_loaded = False
        self.generation_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
        
        # Performance tracking
        self.inference_times = []
        self.tokens_per_second = []
        
        # ARM-optimized settings
        self.arm_optimizations = {
            'n_threads': 4,  # Pi 5 has 4 cores
            'n_threads_batch': 4,
            'use_mmap': True,
            'use_mlock': False,  # Don't lock pages on Pi
            'numa': False,  # No NUMA on Pi
            'low_vram': True,
            'f16_kv': True,  # FP16 key-value cache
            'logits_all': False,
            'vocab_only': False,
            'mul_mat_q': True,  # Quantized matrix multiplication
        }
        
        # Available models (auto-detected)
        self.available_models = self._scan_available_models()
        
        # Performance profiles for different use cases
        self.performance_profiles = {
            'speed': {
                'max_tokens': 50,
                'temperature': 0.3,
                'top_p': 0.8,
                'repeat_penalty': 1.1,
                'context_window': 512
            },
            'balanced': {
                'max_tokens': 100,
                'temperature': 0.7,
                'top_p': 0.9,
                'repeat_penalty': 1.05,
                'context_window': 1024
            },
            'quality': {
                'max_tokens': 200,
                'temperature': 0.8,
                'top_p': 0.95,
                'repeat_penalty': 1.02,
                'context_window': 2048
            }
        }
        
        self.current_profile = 'balanced'
    
    def _scan_available_models(self) -> List[ModelInfo]:
        """Scan for available GGUF models and return sorted by performance"""
        models = []
        models_dir = settings.models_dir
        
        if not models_dir.exists():
            return models
        
        # Define known good models for Pi 5
        known_models = {
            'gemma-2-9b': {
                'patterns': ['gemma-2-9b', 'gemma2-9b'],
                'context_size': 8192,
                'ram_usage': 5.5,
                'speed_rating': 7,
                'quality_rating': 9
            },
            'qwen2.5-7b': {
                'patterns': ['qwen2.5-7b', 'qwen2-7b'],
                'context_size': 32768,
                'ram_usage': 4.5,
                'speed_rating': 8,
                'quality_rating': 8
            },
            'phi-3-mini': {
                'patterns': ['phi-3-mini', 'phi3-mini'],
                'context_size': 4096,
                'ram_usage': 3.0,
                'speed_rating': 9,
                'quality_rating': 7
            },
            'llama-3.2-3b': {
                'patterns': ['llama-3.2-3b', 'llama3.2-3b'],
                'context_size': 2048,
                'ram_usage': 2.5,
                'speed_rating': 9,
                'quality_rating': 6
            }
        }
        
        # Scan for GGUF files
        for gguf_file in models_dir.glob("*.gguf"):
            filename_lower = gguf_file.name.lower()
            
            # Match against known models
            for model_key, model_data in known_models.items():
                for pattern in model_data['patterns']:
                    if pattern in filename_lower:
                        # Prefer Q4_K_M quantization for Pi 5
                        if 'q4_k_m' in filename_lower or 'q5_k_m' in filename_lower:
                            models.append(ModelInfo(
                                name=model_key,
                                file_path=str(gguf_file),
                                context_size=model_data['context_size'],
                                ram_usage=model_data['ram_usage'],
                                speed_rating=model_data['speed_rating'],
                                quality_rating=model_data['quality_rating']
                            ))
                            break
        
        # Sort by combined speed + quality score
        models.sort(key=lambda m: m.speed_rating + m.quality_rating, reverse=True)
        return models
    
    async def initialize(self) -> bool:
        """Initialize with best available model"""
        try:
            if not self.available_models:
                print("❌ No compatible models found")
                return False
            
            # Try models in order of performance
            for model_info in self.available_models:
                print(f"🔄 Trying to load {model_info.name}...")
                
                if await self._load_model(model_info):
                    print(f"✅ Successfully loaded {model_info.name}")
                    return True
                else:
                    print(f"❌ Failed to load {model_info.name}")
            
            return False
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def _load_model(self, model_info: ModelInfo) -> bool:
        """Load specific model with optimizations"""
        try:
            # Import llama-cpp-python
            from llama_cpp import Llama
            
            # Calculate optimal context size based on available RAM
            available_ram_gb = 16  # Pi 5 RAM
            model_ram_gb = model_info.ram_usage
            
            # Reserve 4GB for system, use rest for context
            context_ram_gb = available_ram_gb - model_ram_gb - 4
            
            # Estimate context size (roughly 1GB per 1000 tokens for Q4)
            optimal_context = min(
                model_info.context_size,
                int(context_ram_gb * 800),  # Conservative estimate
                2048  # Max for responsive performance
            )
            
            print(f"Loading {model_info.name} with context size: {optimal_context}")
            
            # Load model in executor to avoid blocking
            self.model = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._create_model,
                model_info.file_path,
                optimal_context
            )
            
            if self.model:
                self.model_info = model_info
                self.model_loaded = True
                
                # Test model with simple prompt
                test_successful = await self._test_model()
                if test_successful:
                    print(f"✅ Model test passed for {model_info.name}")
                    return True
                else:
                    print(f"❌ Model test failed for {model_info.name}")
                    self._unload_model()
                    return False
            
            return False
            
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return False
    
    def _create_model(self, model_path: str, context_size: int):
        """Create Llama model with ARM optimizations"""
        try:
            from llama_cpp import Llama
            
            # ARM-specific optimizations
            model = Llama(
                model_path=model_path,
                n_ctx=context_size,
                verbose=settings.debug_mode,
                
                # Thread configuration for Pi 5 ARM Cortex-A76
                n_threads=self.arm_optimizations['n_threads'],
                n_threads_batch=self.arm_optimizations['n_threads_batch'],
                
                # Memory optimizations for Pi 5
                use_mmap=self.arm_optimizations['use_mmap'],
                use_mlock=self.arm_optimizations['use_mlock'],
                low_vram=self.arm_optimizations['low_vram'],
                
                # Performance optimizations
                f16_kv=self.arm_optimizations['f16_kv'],
                logits_all=self.arm_optimizations['logits_all'],
                vocab_only=self.arm_optimizations['vocab_only'],
                mul_mat_q=self.arm_optimizations['mul_mat_q'],
                
                # ARM-specific settings
                numa=self.arm_optimizations['numa'],
                
                # Batch size optimization for ARM
                n_batch=256,  # Smaller batches work better on ARM
                
                # Disable GPU layers (CPU only on Pi)
                n_gpu_layers=0,
                
                # Additional ARM optimizations
                rope_scaling_type=0,  # Default ROPE scaling
                rope_freq_base=10000.0,
                rope_freq_scale=1.0,
            )
            
            return model
            
        except Exception as e:
            print(f"Model creation failed: {e}")
            return None
    
    async def _test_model(self) -> bool:
        """Test model with simple prompt"""
        try:
            test_prompt = "Hello! Please respond with exactly: 'Test successful'"
            
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_sync,
                test_prompt,
                30,  # max tokens
                0.1  # low temperature
            )
            
            return "test" in response.lower() and "successful" in response.lower()
            
        except Exception as e:
            print(f"Model test failed: {e}")
            return False
    
    async def generate_response(self, query: str, personality_context: str, 
                              memory_context: str, profile: str = None) -> str:
        """Generate response with performance profiling"""
        if not self.model_loaded or not self.model:
            return await self._fallback_response(query)
        
        # Use specified profile or current default
        profile_name = profile or self.current_profile
        profile_settings = self.performance_profiles.get(profile_name, 
                                                       self.performance_profiles['balanced'])
        
        try:
            start_time = time.time()
            
            # Build optimized prompt
            prompt = self._build_optimized_prompt(query, personality_context, memory_context)
            
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_sync,
                prompt,
                profile_settings['max_tokens'],
                profile_settings['temperature'],
                profile_settings['top_p'],
                profile_settings['repeat_penalty']
            )
            
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
            print(f"Generation error: {e}")
            return await self._fallback_response(query)
    
    def _generate_sync(self, prompt: str, max_tokens: int, temperature: float = 0.7,
                      top_p: float = 0.9, repeat_penalty: float = 1.05) -> str:
        """Synchronous generation for executor"""
        with self.generation_lock:
            try:
                # Stop sequences for clean responses
                stop_sequences = [
                    '</s>', '<|end|>', '<|eot_id|>', 
                    '\n\nUser:', '\n\nHuman:', '\n\nAssistant:',
                    '<|endoftext|>'
                ]
                
                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop_sequences,
                    echo=False,
                    stream=False
                )
                
                if 'choices' in output and len(output['choices']) > 0:
                    return output['choices'][0]['text']
                else:
                    return "I apologize, but I couldn't generate a proper response."
                    
            except Exception as e:
                print(f"Sync generation error: {e}")
                return "I encountered an error while processing your request."
    
    def _build_optimized_prompt(self, query: str, personality_context: str, 
                               memory_context: str) -> str:
        """Build optimized prompt based on model type"""
        if not self.model_info:
            return query
        
        # Different prompt formats for different models
        if 'gemma' in self.model_info.name:
            return self._build_gemma_prompt(query, personality_context, memory_context)
        elif 'qwen' in self.model_info.name:
            return self._build_qwen_prompt(query, personality_context, memory_context)
        elif 'phi' in self.model_info.name:
            return self._build_phi_prompt(query, personality_context, memory_context)
        else:
            return self._build_generic_prompt(query, personality_context, memory_context)
    
    def _build_gemma_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Optimized prompt for Gemma models"""
        prompt_parts = ["<bos>"]
        
        if personality_context:
            prompt_parts.extend([
                "<start_of_turn>user",
                f"You are Pascal. {personality_context[:200]}",  # Limit context
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
    
    def _build_phi_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Optimized prompt for Phi models"""
        if personality_context:
            return f"<|system|>\n{personality_context}<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>\n"
        else:
            return f"<|user|>\n{query}<|end|>\n<|assistant|>\n"
    
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
        artifacts = ['</s>', '<|end|>', '<|eot_id|>', '<|endoftext|>', '<|im_end|>']
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
        """Set performance profile (speed/balanced/quality)"""
        if profile in self.performance_profiles:
            self.current_profile = profile
            print(f"Set performance profile to: {profile}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        if not self.inference_times:
            return {"status": "No inference data available"}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        min_time = min(self.inference_times)
        max_time = max(self.inference_times)
        
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.model_info.name if self.model_info else None,
            "model_ram_usage": f"{self.model_info.ram_usage:.1f}GB" if self.model_info else None,
            "performance_profile": self.current_profile,
            "avg_inference_time": f"{avg_time:.2f}s",
            "min_inference_time": f"{min_time:.2f}s",
            "max_inference_time": f"{max_time:.2f}s",
            "total_inferences": len(self.inference_times),
            "available_models": len(self.available_models)
        }
    
    def _unload_model(self):
        """Unload current model"""
        if self.model:
            self.model = None
        self.model_loaded = False
        self.model_info = None
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different available model"""
        target_model = next((m for m in self.available_models if m.name == model_name), None)
        
        if not target_model:
            return False
        
        # Unload current model
        self._unload_model()
        
        # Load new model
        return await self._load_model(target_model)
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their stats"""
        return [
            {
                "name": model.name,
                "ram_usage": f"{model.ram_usage:.1f}GB",
                "speed_rating": f"{model.speed_rating}/10",
                "quality_rating": f"{model.quality_rating}/10",
                "context_size": model.context_size,
                "loaded": model == self.model_info
            }
            for model in self.available_models
        ]
    
    def is_available(self) -> bool:
        """Check if offline LLM is ready"""
        return self.model_loaded and self.model is not None
    
    async def close(self):
        """Clean shutdown"""
        self._unload_model()
        self.executor.shutdown(wait=True)
