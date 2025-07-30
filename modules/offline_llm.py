"""
Pascal AI Assistant - Offline LLM Integration
Handles local language model inference
"""

import asyncio
import time
from typing import Optional, Dict, Any
from pathlib import Path

from config.settings import settings

class OfflineLLM:
    """Manages local LLM inference"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_path = settings.local_model_path
        self.generation_params = {
            'max_tokens': settings.max_response_tokens,
            'temperature': 0.7,
            'top_p': 0.9,
            'stop': ['\n\nUser:', '\n\nHuman:', '\n\nAssistant:']
        }
    
    async def initialize(self) -> bool:
        """Initialize the offline LLM"""
        try:
            print(f"DEBUG: Checking model path: {self.model_path}")
            print(f"DEBUG: Model exists: {self.model_path.exists()}")
        
            if not self.model_path.exists():
                print(f"DEBUG: Local model not found at: {self.model_path}")
                return False
        
            print("DEBUG: Model file found, attempting to import llama_cpp...")
        
            # Try to import and initialize llama-cpp-python
             try:
                 from llama_cpp import Llama
                 print("DEBUG: llama_cpp imported successfully")
            
                 print("DEBUG: About to load model in executor...")
                 print(f"DEBUG: Model path: {str(self.model_path)}")
                 print(f"DEBUG: Context length: {settings.local_model_context}")
                 print(f"DEBUG: Threads: {settings.local_model_threads}")
            
                 # Initialize model in a separate thread to avoid blocking
                 self.model = await asyncio.get_event_loop().run_in_executor(
                     None, 
                     self._load_model,
                     str(self.model_path)
                 )
            
                 print("DEBUG: Model loading completed")
                 print(f"DEBUG: Model object: {type(self.model)}")
            
                 self.model_loaded = True
            
                 if settings.debug_mode:
                     print("✅ Offline LLM initialized successfully")
            
                 return True
            
             except ImportError as e:
                 print(f"DEBUG: ImportError - {e}")
                 if settings.debug_mode:
                     print("❌ llama-cpp-python not installed")
                 return False
        
         except Exception as e:
             print(f"DEBUG: Exception in initialize: {e}")
             import traceback
             traceback.print_exc()
             if settings.debug_mode:
                 print(f"❌ Failed to initialize offline LLM: {e}")
             return False
    
    def _load_model(self, model_path: str):
        """Load the model (runs in executor to avoid blocking)"""
        from llama_cpp import Llama
        
        return Llama(
            model_path=model_path,
            n_ctx=settings.local_model_context,
            n_threads=settings.local_model_threads,
            verbose=settings.debug_mode
        )
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate response using local LLM"""
        if not self.model_loaded or not self.model:
            return await self._fallback_response(query)
        
        try:
            # Construct prompt
            prompt = self._build_prompt(query, personality_context, memory_context)
            
            if settings.debug_mode:
                print(f"Offline LLM prompt length: {len(prompt)} characters")
            
            # Generate response in executor to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self._generate_text,
                prompt
            )
            
            # Clean up response
            cleaned_response = self._clean_response(response)
            
            return cleaned_response
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Offline LLM generation error: {e}")
            return await self._fallback_response(query)
    
    def _build_prompt(self, query: str, personality_context: str, memory_context: str) -> str:
        """Build the complete prompt for the LLM"""
        prompt_parts = []
        
        # Add personality context
        if personality_context:
            prompt_parts.append(f"System: {personality_context}")
        
        # Add memory context
        if memory_context:
            prompt_parts.append(f"Context: {memory_context}")
        
        # Add current query
        prompt_parts.append(f"User: {query}")
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the model (runs in executor)"""
        try:
            # Generate response
            output = self.model(
                prompt,
                max_tokens=self.generation_params['max_tokens'],
                temperature=self.generation_params['temperature'],
                top_p=self.generation_params['top_p'],
                stop=self.generation_params['stop'],
                echo=False
            )
            
            # Extract generated text
            if 'choices' in output and len(output['choices']) > 0:
                return output['choices'][0]['text']
            else:
                return "I apologize, but I couldn't generate a proper response."
                
        except Exception as e:
            if settings.debug_mode:
                print(f"Text generation error: {e}")
            return "I encountered an error while processing your request."
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the model response"""
        # Remove leading/trailing whitespace
        response = response.strip()
        
        # Remove any remaining stop sequences
        for stop_seq in self.generation_params['stop']:
            if stop_seq in response:
                response = response.split(stop_seq)[0]
        
        # Remove common artifacts
        response = response.replace("</s>", "")
        response = response.replace("<|endoftext|>", "")
        
        # Ensure response ends with proper punctuation
        if response and not response[-1] in '.!?':
            response += "."
        
        return response
    
    async def _fallback_response(self, query: str) -> str:
        """Provide fallback response when model is unavailable"""
        fallback_responses = {
            'greeting': "Hello! I'm Pascal, but I'm currently running in limited mode. How can I help you?",
            'question': "I'd like to help you with that question, but I'm currently running in limited mode. Could you try rephrasing or asking something simpler?",
            'default': "I'm currently running in limited offline mode. My full capabilities aren't available right now, but I'm here to help as best I can."
        }
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return fallback_responses['greeting']
        elif '?' in query:
            return fallback_responses['question']
        else:
            return fallback_responses['default']
    
    def is_available(self) -> bool:
        """Check if offline LLM is available"""
        return self.model_loaded and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_loaded': self.model_loaded,
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists(),
            'context_length': settings.local_model_context,
            'threads': settings.local_model_threads,
            'generation_params': self.generation_params
        }
    
    async def test_model(self) -> bool:
        """Test if the model is working correctly"""
        if not self.is_available():
            return False
        
        try:
            test_response = await self.generate_response(
                "Hello, please respond with 'Test successful'",
                "You are a helpful assistant.",
                ""
            )
            
            return "test" in test_response.lower() or "hello" in test_response.lower()
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Model test failed: {e}")
            return False
    
    def update_generation_params(self, **kwargs):
        """Update generation parameters"""
        for key, value in kwargs.items():
            if key in self.generation_params:
                self.generation_params[key] = value
    
    async def reload_model(self) -> bool:
        """Reload the model (useful for switching models)"""
        if self.model:
            # Clean up current model
            self.model = None
            self.model_loaded = False
        
        return await self.initialize()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        # Basic stats - could be expanded with actual timing data
        return {
            'model_loaded': self.model_loaded,
            'available': self.is_available(),
            'context_length': settings.local_model_context,
            'max_tokens': self.generation_params['max_tokens']
        }
