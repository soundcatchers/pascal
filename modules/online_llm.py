"""
Pascal AI Assistant - Fixed Online LLM Integration
Handles API calls to Grok, OpenAI, and Anthropic with streaming support
"""

import asyncio
import time
import json
from typing import Optional, Dict, Any, List, AsyncGenerator
from enum import Enum

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("⚠️ aiohttp not available - online APIs disabled")

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
        self.preferred_provider = None
        self.initialization_successful = False
        self.last_error = None
        
        # Check if aiohttp is available
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp module not installed"
            return
        
        # API configurations
        self.api_configs = {
            APIProvider.GROK: {
                'base_url': 'https://api.x.ai/v1/chat/completions',
                'model': 'grok-beta',
                'api_key': settings.grok_api_key
            },
            APIProvider.OPENAI: {
                'base_url': 'https://api.openai.com/v1/chat/completions',
                'model': 'gpt-4o-mini',
                'api_key': settings.openai_api_key
            },
            APIProvider.ANTHROPIC: {
                'base_url': 'https://api.anthropic.com/v1/messages',
                'model': 'claude-3-haiku-20240307',
                'api_key': settings.anthropic_api_key
            }
        }
        
        # Performance tracking
        self.response_times = {provider: [] for provider in APIProvider}
        self.failure_counts = {provider: 0 for provider in APIProvider}
        self.success_counts = {provider: 0 for provider in APIProvider}
    
    async def initialize(self) -> bool:
        """Initialize online LLM connections"""
        if not AIOHTTP_AVAILABLE:
            if settings.debug_mode:
                print("❌ Cannot initialize online LLM: aiohttp not installed")
                print("   Install with: pip install aiohttp")
            return False
        
        try:
            # Create optimized aiohttp session
            timeout = aiohttp.ClientTimeout(total=settings.online_timeout, connect=5.0)
            connector = aiohttp.TCPConnector(
                limit=10, 
                force_close=True,
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            
            # Check which providers are available
            await self._check_available_providers()
            
            if not self.available_providers:
                self.last_error = "No API keys configured or all providers failed connection test"
                if settings.debug_mode:
                    print("❌ No online LLM providers available")
                    print("   Configure API keys in .env file:")
                    print("   GROK_API_KEY=your_grok_key")
                    print("   OPENAI_API_KEY=your_openai_key") 
                    print("   ANTHROPIC_API_KEY=your_anthropic_key")
                return False
            
            # Set preferred provider (Grok first, then OpenAI, then Anthropic)
            for provider in [APIProvider.GROK, APIProvider.OPENAI, APIProvider.ANTHROPIC]:
                if provider in self.available_providers:
                    self.preferred_provider = provider
                    break
            
            self.initialization_successful = True
            
            if settings.debug_mode:
                provider_names = [p.value for p in self.available_providers]
                print(f"✅ Online LLM initialized with providers: {provider_names}")
                print(f"Preferred provider: {self.preferred_provider.value}")
            
            return True
            
        except Exception as e:
            self.last_error = str(e)
            if settings.debug_mode:
                print(f"❌ Failed to initialize online LLM: {e}")
            return False
    
    async def _check_available_providers(self):
        """Check which API providers are configured and working"""
        self.available_providers = []
        
        for provider, config in self.api_configs.items():
            api_key = config.get('api_key')
            
            # Skip if no API key configured
            if not api_key or api_key in ['', 'your_api_key_here', f'your_{provider.value}_api_key_here', None]:
                if settings.debug_mode:
                    print(f"⏭️ Skipping {provider.value} - no API key configured")
                continue
            
            # Test provider connection
            try:
                if await self._test_provider_quick(provider):
                    self.available_providers.append(provider)
                    if settings.debug_mode:
                        print(f"✅ {provider.value} - connection test passed")
                else:
                    if settings.debug_mode:
                        print(f"❌ {provider.value} - connection test failed")
            except Exception as e:
                if settings.debug_mode:
                    print(f"❌ {provider.value} - test error: {str(e)[:100]}")
    
    async def _test_provider_quick(self, provider: APIProvider) -> bool:
        """Quick test of provider connectivity"""
        try:
            config = self.api_configs[provider]
            
            # Prepare minimal test request
            if provider == APIProvider.ANTHROPIC:
                headers = {
                    'Content-Type': 'application/json',
                    'x-api-key': config['api_key'],
                    'anthropic-version': '2023-06-01'
                }
                payload = {
                    "model": config['model'],
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": "Hi"}]
                }
            else:
                # OpenAI/Grok compatible
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {config["api_key"]}'
                }
                payload = {
                    "model": config['model'],
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                }
            
            # Quick connection test with short timeout
            test_timeout = aiohttp.ClientTimeout(total=10, connect=3)
            async with self.session.post(
                config['base_url'], 
                headers=headers, 
                json=payload,
                timeout=test_timeout
            ) as response:
                # Accept any response that isn't a connection error
                return response.status in [200, 400, 401, 429]  # 400/401 = auth issue but server reachable
                
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False
    
    async def generate_response_stream(self, query: str, personality_context: str, 
                                     memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response from online API"""
        if not self.initialization_successful or not self.available_providers:
            yield "Online services are not available right now."
            return
        
        # Try providers in order of preference
        providers_to_try = [self.preferred_provider] if self.preferred_provider else []
        providers_to_try.extend([p for p in self.available_providers if p != self.preferred_provider])
        
        last_error = None
        
        for provider in providers_to_try:
            try:
                async for chunk in self._stream_from_provider(provider, query, personality_context, memory_context):
                    yield chunk
                self.success_counts[provider] += 1
                return
            except Exception as e:
                self.failure_counts[provider] += 1
                last_error = str(e)
                if settings.debug_mode:
                    print(f"Provider {provider.value} streaming failed: {str(e)[:100]}")
                continue
        
        # All providers failed
        self.last_error = last_error
        yield f"I'm having trouble connecting to online services right now."
    
    async def _stream_from_provider(self, provider: APIProvider, query: str, 
                                   personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from specific provider"""
        start_time = time.time()
        
        if provider == APIProvider.ANTHROPIC:
            async for chunk in self._stream_anthropic(query, personality_context, memory_context):
                yield chunk
        else:
            # OpenAI/Grok compatible streaming
            async for chunk in self._stream_openai_compatible(provider, query, personality_context, memory_context):
                yield chunk
        
        # Record response time
        response_time = time.time() - start_time
        self.response_times[provider].append(response_time)
        if len(self.response_times[provider]) > 10:
            self.response_times[provider] = self.response_times[provider][-10:]
    
    async def _stream_openai_compatible(self, provider: APIProvider, query: str, 
                                       personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI-compatible APIs (OpenAI, Grok)"""
        config = self.api_configs[provider]
        
        messages = []
        
        # Build system message
        system_content_parts = []
        if personality_context:
            system_content_parts.append(personality_context[:600])
        if memory_context:
            system_content_parts.append(f"Recent context: {memory_context[:400]}")
        
        if system_content_parts:
            system_content = "\n\n".join(system_content_parts)
            messages.append({"role": "system", "content": system_content})
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": config['model'],
            "messages": messages,
            "max_tokens": min(settings.max_response_tokens, 400),
            "temperature": 0.7,
            "stream": True
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {config["api_key"]}'
        }
        
        async with self.session.post(config['base_url'], headers=headers, json=payload) as response:
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
                                    if 'content' in delta and delta['content']:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
            else:
                error_text = await response.text()
                raise Exception(f"{provider.value} API error {response.status}: {error_text[:200]}")
    
    async def _stream_anthropic(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from Anthropic API"""
        config = self.api_configs[APIProvider.ANTHROPIC]
        
        # Build system prompt
        system_parts = []
        if personality_context:
            system_parts.append(personality_context[:600])
        if memory_context:
            system_parts.append(f"Recent context: {memory_context[:400]}")
        
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        
        payload = {
            "model": config['model'],
            "max_tokens": min(settings.max_response_tokens, 400),
            "messages": [{"role": "user", "content": query}],
            "stream": True
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': config['api_key'],
            'anthropic-version': '2023-06-01'
        }
        
        async with self.session.post(config['base_url'], headers=headers, json=payload) as response:
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
                error_text = await response.text()
                raise Exception(f"Anthropic API error {response.status}: {error_text[:200]}")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate non-streaming response (fallback)"""
        if not self.initialization_successful or not self.available_providers:
            return "I'm sorry, but online services are not available right now."
        
        # Collect streaming response
        response_parts = []
        try:
            async for chunk in self.generate_response_stream(query, personality_context, memory_context):
                response_parts.append(chunk)
            
            response = ''.join(response_parts)
            return response if response else "I couldn't generate a response from online services."
            
        except Exception as e:
            self.last_error = str(e)
            return f"I encountered an error with online services: {str(e)[:100]}"
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers"""
        stats = {
            'aiohttp_available': AIOHTTP_AVAILABLE,
            'initialization_successful': self.initialization_successful,
            'last_error': self.last_error,
            'available_providers': [p.value for p in self.available_providers],
            'preferred_provider': self.preferred_provider.value if self.preferred_provider else None,
            'providers': {}
        }
        
        for provider in APIProvider:
            api_key = self.api_configs[provider].get('api_key')
            api_key_configured = bool(api_key and api_key not in ['', 'your_api_key_here', f'your_{provider.value}_api_key_here', None])
            
            provider_stats = {
                'available': provider in self.available_providers,
                'success_count': self.success_counts[provider],
                'failure_count': self.failure_counts[provider],
                'avg_response_time': 0,
                'api_key_configured': api_key_configured
            }
            
            if self.response_times[provider]:
                provider_stats['avg_response_time'] = sum(self.response_times[provider]) / len(self.response_times[provider])
            
            stats['providers'][provider.value] = provider_stats
        
        return stats
    
    def is_available(self) -> bool:
        """Check if any online provider is available"""
        return self.initialization_successful and len(self.available_providers) > 0
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
