"""
Pascal AI Assistant - Online LLM Integration with Grok Support
Handles API calls to Grok, OpenAI, Anthropic, and Google
"""

import asyncio
import aiohttp
import time
from typing import Optional, Dict, Any, List, AsyncGenerator
from enum import Enum

from config.settings import settings

class APIProvider(Enum):
    """Available API providers"""
    GROK = "grok"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

class OnlineLLM:
    """Manages online LLM API calls with Grok as primary"""
    
    def __init__(self):
        self.session = None
        self.available_providers = []
        self.preferred_provider = None
        
        # API configurations with Grok as primary
        self.api_configs = {
            APIProvider.GROK: {
                'base_url': 'https://api.x.ai/v1/chat/completions',
                'model': 'grok-beta',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {settings.grok_api_key}'
                },
                'streaming': True
            },
            APIProvider.OPENAI: {
                'base_url': 'https://api.openai.com/v1/chat/completions',
                'model': 'gpt-4-turbo-preview',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {settings.openai_api_key}'
                },
                'streaming': True
            },
            APIProvider.ANTHROPIC: {
                'base_url': 'https://api.anthropic.com/v1/messages',
                'model': 'claude-3-sonnet-20240229',
                'headers': {
                    'Content-Type': 'application/json',
                    'x-api-key': settings.anthropic_api_key,
                    'anthropic-version': '2023-06-01'
                },
                'streaming': True
            },
            APIProvider.GOOGLE: {
                'base_url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent',
                'model': 'gemini-pro',
                'headers': {
                    'Content-Type': 'application/json'
                },
                'streaming': True
            }
        }
        
        # Performance tracking
        self.response_times = {provider: [] for provider in APIProvider}
        self.failure_counts = {provider: 0 for provider in APIProvider}
        self.success_counts = {provider: 0 for provider in APIProvider}
    
    async def initialize(self) -> bool:
        """Initialize online LLM connections"""
        try:
            # Create aiohttp session with optimized settings for speed
            timeout = aiohttp.ClientTimeout(
                total=10,  # 10 second total timeout for fast fallback
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
            
            # Check which providers are available
            await self._check_available_providers()
            
            if not self.available_providers:
                if settings.debug_mode:
                    print("❌ No online LLM providers configured")
                return False
            
            # Set Grok as preferred if available, otherwise first available
            if APIProvider.GROK in self.available_providers:
                self.preferred_provider = APIProvider.GROK
                print("⚡ Grok API configured as primary online provider")
            else:
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
        """Check which API providers are configured"""
        self.available_providers = []
        
        # Check Grok (primary)
        if hasattr(settings, 'grok_api_key') and settings.grok_api_key:
            self.available_providers.append(APIProvider.GROK)
        
        # Check OpenAI
        if settings.openai_api_key:
            self.available_providers.append(APIProvider.OPENAI)
        
        # Check Anthropic
        if settings.anthropic_api_key:
            self.available_providers.append(APIProvider.ANTHROPIC)
        
        # Check Google
        if settings.google_api_key:
            self.available_providers.append(APIProvider.GOOGLE)
    
    async def generate_response_stream(self, query: str, personality_context: str, 
                                     memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from online APIs for perceived speed"""
        if not self.available_providers:
            yield "Online services are not configured."
            return
        
        # Try preferred provider first, then others
        providers_to_try = [self.preferred_provider] + [p for p in self.available_providers if p != self.preferred_provider]
        
        for provider in providers_to_try:
            try:
                async for token in self._call_api_stream(provider, query, personality_context, memory_context):
                    yield token
                
                self.success_counts[provider] += 1
                return
                
            except Exception as e:
                self.failure_counts[provider] += 1
                if settings.debug_mode:
                    print(f"Provider {provider.value} failed: {e}")
                
                if provider != providers_to_try[-1]:
                    yield f"\n[Switching to backup provider...]\n"
                continue
        
        yield "I'm having trouble connecting to online services right now."
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Non-streaming response (compatibility method)"""
        full_response = []
        async for token in self.generate_response_stream(query, personality_context, memory_context):
            full_response.append(token)
        return ''.join(full_response)
    
    async def _call_api_stream(self, provider: APIProvider, query: str, 
                              personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from specific provider"""
        start_time = time.time()
        
        try:
            if provider == APIProvider.GROK:
                async for token in self._call_grok_stream(query, personality_context, memory_context):
                    yield token
            elif provider == APIProvider.OPENAI:
                async for token in self._call_openai_stream(query, personality_context, memory_context):
                    yield token
            elif provider == APIProvider.ANTHROPIC:
                async for token in self._call_anthropic_stream(query, personality_context, memory_context):
                    yield token
            elif provider == APIProvider.GOOGLE:
                async for token in self._call_google_stream(query, personality_context, memory_context):
                    yield token
            
            # Record response time
            response_time = time.time() - start_time
            self.response_times[provider].append(response_time)
            if len(self.response_times[provider]) > 10:
                self.response_times[provider] = self.response_times[provider][-10:]
                
        except Exception as e:
            if settings.debug_mode:
                print(f"API call to {provider.value} failed: {e}")
            raise
    
    async def _call_grok_stream(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from Grok API"""
        config = self.api_configs[APIProvider.GROK]
        
        messages = []
        
        # Add system message
        if personality_context:
            messages.append({"role": "system", "content": personality_context})
        
        # Add memory context
        if memory_context:
            messages.append({"role": "system", "content": f"Context: {memory_context}"})
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": config['model'],
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 150
        }
        
        async with self.session.post(config['base_url'], headers=config['headers'], json=payload) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        line_text = line.decode('utf-8').strip()
                        if line_text.startswith('data: '):
                            data_str = line_text[6:]
                            if data_str == '[DONE]':
                                break
                            
                            try:
