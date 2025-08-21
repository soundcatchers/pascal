"""
Pascal AI Assistant - Online LLM Integration with Grok
Handles API calls to Grok, OpenAI, and Anthropic with streaming support
"""

import asyncio
import aiohttp
import time
import json
from typing import Optional, Dict, Any, List, AsyncGenerator
from enum import Enum

from config.settings import settings

class APIProvider(Enum):
    """Available API providers"""
    GROK = "grok"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class OnlineLLM:
    """Manages online LLM API calls with Grok as primary"""
    
    def __init__(self):
        self.session = None
        self.available_providers = []
        self.preferred_provider = APIProvider.GROK
        
        # API configurations
        self.api_configs = {
            APIProvider.GROK: {
                'base_url': 'https://api.x.ai/v1/chat/completions',
                'model': 'grok-beta',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {settings.grok_api_key}'
                }
            },
            APIProvider.OPENAI: {
                'base_url': 'https://api.openai.com/v1/chat/completions',
                'model': 'gpt-4o-mini',
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
            }
        }
        
        # Performance tracking
        self.response_times = {provider: [] for provider in APIProvider}
        self.failure_counts = {provider: 0 for provider in APIProvider}
        self.success_counts = {provider: 0 for provider in APIProvider}
    
    async def initialize(self) -> bool:
        """Initialize online LLM connections"""
        try:
            # Create optimized aiohttp session
            timeout = aiohttp.ClientTimeout(total=settings.online_timeout)
            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            
            # Check which providers are available
            await self._check_available_providers()
            
            if not self.available_providers:
                if settings.debug_mode:
                    print("❌ No online LLM providers available")
                return False
            
            # Set Grok as preferred if available
            if APIProvider.GROK in self.available_providers:
                self.preferred_provider = APIProvider.GROK
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
        """Check which API providers are configured and available"""
        self.available_providers = []
        
        # Check Grok first (priority)
        if settings.grok_api_key:
            self.available_providers.append(APIProvider.GROK)
        
        # Check OpenAI
        if settings.openai_api_key:
            self.available_providers.append(APIProvider.OPENAI)
        
        # Check Anthropic
        if settings.anthropic_api_key:
            self.available_providers.append(APIProvider.ANTHROPIC)
    
    async def generate_response_stream(self, query: str, personality_context: str, 
                                     memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response from online API"""
        if not self.available_providers:
            yield "Online services are not available right now."
            return
        
        # Try preferred provider first
        providers_to_try = [self.preferred_provider] + [p for p in self.available_providers if p != self.preferred_provider]
        
        for provider in providers_to_try:
            try:
                async for chunk in self._stream_from_provider(provider, query, personality_context, memory_context):
                    yield chunk
                self.success_counts[provider] += 1
                return
            except Exception as e:
                self.failure_counts[provider] += 1
                if settings.debug_mode:
                    print(f"Provider {provider.value} streaming failed: {e}")
                continue
        
        yield "I'm having trouble connecting to online services right now."
    
    async def _stream_from_provider(self, provider: APIProvider, query: str, 
                                   personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from specific provider"""
        start_time = time.time()
        
        if provider == APIProvider.GROK:
            async for chunk in self._stream_grok(query, personality_context, memory_context):
                yield chunk
        elif provider == APIProvider.OPENAI:
            async for chunk in self._stream_openai(query, personality_context, memory_context):
                yield chunk
        elif provider == APIProvider.ANTHROPIC:
            async for chunk in self._stream_anthropic(query, personality_context, memory_context):
                yield chunk
        
        # Record response time
        response_time = time.time() - start_time
        self.response_times[provider].append(response_time)
        if len(self.response_times[provider]) > 10:
            self.response_times[provider] = self.response_times[provider][-10:]
    
    async def _stream_grok(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from Grok API"""
        config = self.api_configs[APIProvider.GROK]
        
        messages = []
        
        # Add system message
        if personality_context:
            messages.append({"role": "system", "content": personality_context[:500]})
        
        # Add memory context if available
        if memory_context:
            messages.append({"role": "system", "content": f"Previous context: {memory_context[:500]}"})
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": config['model'],
            "messages": messages,
            "max_tokens": settings.max_response_tokens,
            "temperature": 0.7,
            "stream": True
        }
        
        async with self.session.post(config['base_url'], headers=config['headers'], json=payload) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            line_str = line_str[6:]
                            if line_str == '[DONE]':
                                break
                            try:
                                data = json.loads(line_str)
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
            else:
                raise Exception(f"Grok API error: {response.status}")
    
    async def _stream_openai(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI API"""
        config = self.api_configs[APIProvider.OPENAI]
        
        messages = []
        
        # Add system message
        if personality_context:
            messages.append({"role": "system", "content": personality_context[:500]})
        
        # Add memory context
        if memory_context:
            messages.append({"role": "system", "content": f"Previous context: {memory_context[:500]}"})
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": config['model'],
            "messages": messages,
            "max_tokens": settings.max_response_tokens,
            "temperature": 0.7,
            "stream": True
        }
        
        async with self.session.post(config['base_url'], headers=config['headers'], json=payload) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            line_str = line_str[6:]
                            if line_str == '[DONE]':
                                break
                            try:
                                data = json.loads(line_str)
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
            else:
                raise Exception(f"OpenAI API error: {response.status}")
    
    async def _stream_anthropic(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from Anthropic API"""
        config = self.api_configs[APIProvider.ANTHROPIC]
        
        # Build system prompt
        system_parts = []
        if personality_context:
            system_parts.append(personality_context[:500])
        if memory_context:
            system_parts.append(f"Previous context: {memory_context[:500]}")
        
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        
        payload = {
            "model": config['model'],
            "max_tokens": settings.max_response_tokens,
            "messages": [{"role": "user", "content": query}],
            "stream": True
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        async with self.session.post(config['base_url'], headers=config['headers'], json=payload) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            line_str = line_str[6:]
                            try:
                                data = json.loads(line_str)
                                if data.get('type') == 'content_block_delta':
                                    delta = data.get('delta', {})
                                    if 'text' in delta:
                                        yield delta['text']
                            except json.JSONDecodeError:
                                continue
            else:
                raise Exception(f"Anthropic API error: {response.status}")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate response using online APIs (non-streaming fallback)"""
        if not self.available_providers:
            return "I'm sorry, but online services are not available right now."
        
        # Try providers in order of preference
        providers_to_try = [self.preferred_provider] + [p for p in self.available_providers if p != self.preferred_provider]
        
        for provider in providers_to_try:
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
        """Call specific API provider (non-streaming)"""
        start_time = time.time()
        
        try:
            if provider == APIProvider.GROK:
                response = await self._call_grok(query, personality_context, memory_context)
            elif provider == APIProvider.OPENAI:
                response = await self._call_openai(query, personality_context, memory_context)
            elif provider == APIProvider.ANTHROPIC:
                response = await self._call_anthropic(query, personality_context, memory_context)
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
    
    async def _call_grok(self, query: str, personality_context: str, memory_context: str) -> str:
        """Call Grok API (non-streaming)"""
        config = self.api_configs[APIProvider.GROK]
        
        messages = []
        
        # Add system message
        if personality_context:
            messages.append({"role": "system", "content": personality_context[:500]})
        
        # Add memory context
        if memory_context:
            messages.append({"role": "system", "content": f"Previous context: {memory_context[:500]}"})
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": config['model'],
            "messages": messages,
            "max_tokens": settings.max_response_tokens,
            "temperature": 0.7,
            "stream": False
        }
        
        async with self.session.post(config['base_url'], headers=config['headers'], json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['choices'][0]['message']['content']
            else:
                raise Exception(f"Grok API error: {response.status}")
    
    async def _call_openai(self, query: str, personality_context: str, memory_context: str) -> str:
        """Call OpenAI API (non-streaming)"""
        config = self.api_configs[APIProvider.OPENAI]
        
        messages = []
        
        # Add system message
        if personality_context:
            messages.append({"role": "system", "content": personality_context[:500]})
        
        # Add memory context as system message
        if memory_context:
            messages.append({"role": "system", "content": f"Previous context: {memory_context[:500]}"})
        
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
        """Call Anthropic API (non-streaming)"""
        config = self.api_configs[APIProvider.ANTHROPIC]
        
        # Build system prompt
        system_parts = []
        if personality_context:
            system_parts.append(personality_context[:500])
        if memory_context:
            system_parts.append(f"Previous context: {memory_context[:500]}")
        
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
