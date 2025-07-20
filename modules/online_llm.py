"""
Pascal AI Assistant - Online LLM Integration
Handles API calls to external LLM services
"""

import asyncio
import aiohttp
import time
from typing import Optional, Dict, Any, List
from enum import Enum

from config.settings import settings

class APIProvider(Enum):
    """Available API providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

class OnlineLLM:
    """Manages online LLM API calls"""
    
    def __init__(self):
        self.session = None
        self.available_providers = []
        self.preferred_provider = None
        self.api_configs = {
            APIProvider.OPENAI: {
                'base_url': 'https://api.openai.com/v1/chat/completions',
                'model': 'gpt-3.5-turbo',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {settings.openai_api_key}'
                }
            },
            APIProvider.ANTHROPIC: {
                'base_url': 'https://api.anthropic.com/v1/messages',
                'model': 'claude-3-haiku-20240307',
                'headers': {
                    'Content-Type': 'application/json',
                    'x-api-key': settings.anthropic_api_key,
                    'anthropic-version': '2023-06-01'
                }
            },
            APIProvider.GOOGLE: {
                'base_url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
                'model': 'gemini-pro',
                'headers': {
                    'Content-Type': 'application/json'
                }
            }
        }
        
        # Performance tracking
        self.response_times = {provider: [] for provider in APIProvider}
        self.failure_counts = {provider: 0 for provider in APIProvider}
        self.success_counts = {provider: 0 for provider in APIProvider}
    
    async def initialize(self) -> bool:
        """Initialize online LLM connections"""
        try:
            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=settings.online_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Check which providers are available
            await self._check_available_providers()
            
            if not self.available_providers:
                if settings.debug_mode:
                    print("❌ No online LLM providers available")
                return False
            
            # Set preferred provider (first available)
            self.preferred_provider = self.available_providers[0]
            
            if settings.debug_mode:
                print(f"✅ Online LLM initialized with providers: {[p.value for p in self.available_providers]}")
                print(f"Preferred provider: {self.preferred_provider.value}")
            
            return True
            
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Failed to initialize online LLM: {e}")
            return False
    
    async def _check_available_providers(self):
        """Check which API providers are configured and available"""
        self.available_providers = []
        
        # Check OpenAI
        if settings.openai_api_key:
            self.available_providers.append(APIProvider.OPENAI)
        
        # Check Anthropic
        if settings.anthropic_api_key:
            self.available_providers.append(APIProvider.ANTHROPIC)
        
        # Check Google
        if settings.google_api_key:
            self.available_providers.append(APIProvider.GOOGLE)
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate response using online APIs"""
        if not self.available_providers:
            return "I'm sorry, but online services are not available right now."
        
        # Try providers in order of preference
        for provider in self.available_providers:
            try:
                response = await self._call_api(provider, query, personality_context, memory_context)
                if response:
                    self.success_counts[provider] += 1
                    return response
            except Exception as e:
                self.failure_counts[provider] += 1
                if settings.debug_mode:
                    print(f"Provider {provider.value} failed: {e}")
                continue
        
        return "I'm sorry, but I'm having trouble connecting to online services right now."
    
    async def _call_api(self, provider: APIProvider, query: str, personality_context: str, memory_context: str) -> Optional[str]:
        """Call specific API provider"""
        start_time = time.time()
        
        try:
            if provider == APIProvider.OPENAI:
                response = await self._call_openai(query, personality_context, memory_context)
            elif provider == APIProvider.ANTHROPIC:
                response = await self._call_anthropic(query, personality_context, memory_context)
            elif provider == APIProvider.GOOGLE:
                response = await self._call_google(query, personality_context, memory_context)
            else:
                return None
            
            # Record response time
            response_time = time.time() - start_time
            self.response_times[provider].append(response_time)
            
            # Keep only last 10 response times
            if len(self.response_times[provider]) > 10:
                self.response_times[provider] = self.response_times[provider][-10:]
            
            return response
            
        except Exception as e:
            if settings.debug_mode:
                print(f"API call to {provider.value} failed: {e}")
            raise
    
    async def _call_openai(self, query: str, personality_context: str, memory_context: str) -> str:
        """Call OpenAI API"""
        config = self.api_configs[APIProvider.OPENAI]
        
        messages = []
        
        # Add system message
        if personality_context:
            messages.append({"role": "system", "content": personality_context})
        
        # Add memory context as system message
        if memory_context:
            messages.append({"role": "system", "content": f"Previous context: {memory_context}"})
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": config['model'],
            "messages": messages,
            "max_tokens": settings.max_response_tokens,
            "temperature": 0.7
        }
        
        async with self.session.post(config['base_url'], headers=config['headers'], json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['choices'][0]['message']['content']
            else:
                raise Exception(f"OpenAI API error: {response.status}")
    
    async def _call_anthropic(self, query: str, personality_context: str, memory_context: str) -> str:
        """Call Anthropic API"""
        config = self.api_configs[APIProvider.ANTHROPIC]
        
        # Build system prompt
        system_parts = []
        if personality_context:
            system_parts.append(personality_context)
        if memory_context:
            system_parts.append(f"Previous context: {memory_context}")
        
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        
        payload = {
            "model": config['model'],
            "max_tokens": settings.max_response_tokens,
            "messages": [{"role": "user", "content": query}]
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        async with self.session.post(config['base_url'], headers=config['headers'], json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['content'][0]['text']
            else:
                raise Exception(f"Anthropic API error: {response.status}")
    
    async def _call_google(self, query: str, personality_context: str, memory_context: str) -> str:
        """Call Google Gemini API"""
        config = self.api_configs[APIProvider.GOOGLE]
        
        # Build prompt
        prompt_parts = []
        if personality_context:
            prompt_parts.append(personality_context)
        if memory_context:
            prompt_parts.append(f"Context: {memory_context}")
        prompt_parts.append(f"User: {query}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": settings.max_response_tokens,
                "temperature": 0.7
            }
        }
        
        url = f"{config['base_url']}?key={settings.google_api_key}"
        
        async with self.session.post(url, headers=config['headers'], json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                raise Exception(f"Google API error: {response.status}")
    
    async def test_provider(self, provider: APIProvider) -> bool:
        """Test if a specific provider is working"""
        try:
            response = await self._call_api(
                provider,
                "Hello, please respond with 'Test successful'",
                "You are a helpful assistant.",
                ""
            )
            return response and "test" in response.lower()
        except:
            return False
    
    async def test_all_providers(self) -> Dict[APIProvider, bool]:
        """Test all available providers"""
        results = {}
        for provider in self.available_providers:
            results[provider] = await self.test_provider(provider)
        return results
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers"""
        stats = {}
        
        for provider in APIProvider:
            provider_stats = {
                'available': provider in self.available_providers,
                'success_count': self.success_counts[provider],
                'failure_count': self.failure_counts[provider],
                'avg_response_time': 0
            }
            
            if self.response_times[provider]:
                provider_stats['avg_response_time'] = sum(self.response_times[provider]) / len(self.response_times[provider])
            
            stats[provider.value] = provider_stats
        
        return stats
    
    def set_preferred_provider(self, provider: APIProvider):
        """Set the preferred provider"""
        if provider in self.available_providers:
            self.preferred_provider = provider
            # Move to front of list
            self.available_providers.remove(provider)
            self.available_providers.insert(0, provider)
    
    def is_available(self) -> bool:
        """Check if any online provider is available"""
        return len(self.available_providers) > 0
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
