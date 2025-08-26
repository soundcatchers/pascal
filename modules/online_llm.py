"""
Pascal AI Assistant - Online LLM Integration with Grok Priority
Handles API calls to Grok, OpenAI, and Anthropic with streaming support - Fixed Priority
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
        
        # Check if aiohttp is available first
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp module not installed"
            if settings.debug_mode:
                print("❌ aiohttp not available - install with: pip install aiohttp")
            return
        
        # API configurations - CORRECTED Grok endpoint
        self.api_configs = {
            APIProvider.GROK: {
                'base_url': 'https://api.x.ai/v1/chat/completions',
                'model': 'grok-beta',
                'api_key': getattr(settings, 'grok_api_key', None)
            },
            APIProvider.OPENAI: {
                'base_url': 'https://api.openai.com/v1/chat/completions',
                'model': 'gpt-4o-mini',
                'api_key': getattr(settings, 'openai_api_key', None)
            },
            APIProvider.ANTHROPIC: {
                'base_url': 'https://api.anthropic.com/v1/messages',
                'model': 'claude-3-haiku-20240307',
                'api_key': getattr(settings, 'anthropic_api_key', None)
            }
        }
        
        # Performance tracking
        self.response_times = {provider: [] for provider in APIProvider}
        self.failure_counts = {provider: 0 for provider in APIProvider}
        self.success_counts = {provider: 0 for provider in APIProvider}
    
    async def initialize(self) -> bool:
        """Initialize online LLM connections"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp module not available"
            if settings.debug_mode:
                print("❌ Cannot initialize online LLM: aiohttp not installed")
            return False
        
        try:
            # Create aiohttp session with longer timeout
            timeout = aiohttp.ClientTimeout(total=45, connect=15)
            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            
            # Check which providers are available - PRIORITIZE GROK
            await self._check_available_providers()
            
            if not self.available_providers:
                self.last_error = "No API keys configured properly or all providers failed connection test"
                if settings.debug_mode:
                    print("❌ No online LLM providers available")
                    print("   Check API keys in .env file")
                return False
            
            # Set preferred provider - GROK FIRST!
            for provider in [APIProvider.GROK, APIProvider.OPENAI, APIProvider.ANTHROPIC]:
                if provider in self.available_providers:
                    self.preferred_provider = provider
                    break
            
            self.initialization_successful = True
            
            if settings.debug_mode:
                provider_names = [p.value for p in self.available_providers]
                print(f"✅ Online LLM initialized with: {provider_names}")
                print(f"🎯 Primary provider: {self.preferred_provider.value}")
            
            return True
            
        except Exception as e:
            self.last_error = f"Initialization error: {str(e)}"
            if settings.debug_mode:
                print(f"❌ Online LLM initialization failed: {e}")
                import traceback
                traceback.print_exc()
            return False
    
    async def _check_available_providers(self):
        """Check which API providers are configured and working"""
        self.available_providers = []
        
        # Check providers in priority order: Grok -> OpenAI -> Anthropic
        priority_order = [APIProvider.GROK, APIProvider.OPENAI, APIProvider.ANTHROPIC]
        
        for provider in priority_order:
            config = self.api_configs[provider]
            api_key = config.get('api_key')
            
            # Skip if no API key or placeholder
            invalid_keys = [None, '', 'your_api_key_here', f'your_{provider.value}_api_key_here']
            if api_key in invalid_keys:
                if settings.debug_mode:
                    print(f"⏭️ Skipping {provider.value} - no valid API key")
                continue
            
            # Test connectivity with proper error handling
            try:
                if await self._test_provider_connectivity(provider):
                    self.available_providers.append(provider)
                    if settings.debug_mode:
                        print(f"✅ {provider.value} - connection test passed")
                else:
                    if settings.debug_mode:
                        print(f"❌ {provider.value} - connection test failed")
            except Exception as e:
                if settings.debug_mode:
                    print(f"❌ {provider.value} - test error: {str(e)[:100]}")
    
    async def _test_provider_connectivity(self, provider: APIProvider) -> bool:
        """Test if provider is reachable and working"""
        try:
            config = self.api_configs[provider]
            
            if provider == APIProvider.ANTHROPIC:
                headers = {
                    'Content-Type': 'application/json',
                    'x-api-key': config['api_key'],
                    'anthropic-version': '2023-06-01'
                }
                payload = {
                    "model": config['model'],
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": "hi"}]
                }
            else:
                # OpenAI/Grok compatible
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {config["api_key"]}'
                }
                payload = {
                    "model": config['model'],
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                    "temperature": 0.5
                }
            
            # Quick test with short timeout
            test_timeout = aiohttp.ClientTimeout(total=20)
            async with self.session.post(
                config['base_url'],
                headers=headers,
                json=payload,
                timeout=test_timeout
            ) as response:
                # Accept various response codes that indicate the server is reachable
                if response.status in [200, 400, 401, 429]:
                    if response.status == 429:
                        # Rate limited but server is working
                        if settings.debug_mode:
                            print(f"⚠️ {provider.value} rate limited but reachable")
                        return True
                    elif response.status == 401:
                        # Invalid API key but server is working
                        if settings.debug_mode:
                            print(f"⚠️ {provider.value} invalid API key but server reachable")
                        return False  # Don't use if API key is invalid
                    return True
                else:
                    return False
                    
        except asyncio.TimeoutError:
            if settings.debug_mode:
                print(f"⚠️ {provider.value} timeout during test")
            return False
        except Exception as e:
            if settings.debug_mode:
                print(f"⚠️ {provider.value} connectivity test error: {str(e)[:100]}")
            return False
    
    async def generate_response_stream(self, query: str, personality_context: str, 
                                     memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response from online API - GROK PRIORITY"""
        if not self.initialization_successful or not self.available_providers:
            yield "Online services are not available right now."
            return
        
        # Try providers in strict priority order: Grok -> OpenAI -> Anthropic
        providers_to_try = []
        
        # Always try Grok first if available
        if APIProvider.GROK in self.available_providers:
            providers_to_try.append(APIProvider.GROK)
        
        # Then OpenAI
        if APIProvider.OPENAI in self.available_providers:
            providers_to_try.append(APIProvider.OPENAI)
        
        # Finally Anthropic
        if APIProvider.ANTHROPIC in self.available_providers:
            providers_to_try.append(APIProvider.ANTHROPIC)
        
        last_error = None
        
        for provider in providers_to_try:
            try:
                if settings.debug_mode:
                    print(f"🔄 Trying provider: {provider.value}")
                
                response_generated = False
                async for chunk in self._stream_from_provider(provider, query, personality_context, memory_context):
                    yield chunk
                    response_generated = True
                
                if response_generated:
                    self.success_counts[provider] += 1
                    if settings.debug_mode:
                        print(f"✅ Success with {provider.value}")
                    return
                
            except Exception as e:
                self.failure_counts[provider] += 1
                last_error = str(e)
                if settings.debug_mode:
                    print(f"❌ {provider.value} failed: {str(e)[:150]}")
                continue
        
        # All providers failed
        self.last_error = last_error
        yield "I'm having trouble connecting to online services right now."
    
    async def _stream_from_provider(self, provider: APIProvider, query: str, 
                                   personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from specific provider with better error handling"""
        start_time = time.time()
        
        try:
            if provider == APIProvider.ANTHROPIC:
                async for chunk in self._stream_anthropic(query, personality_context, memory_context):
                    yield chunk
            else:
                # OpenAI/Grok compatible streaming
                async for chunk in self._stream_openai_compatible(provider, query, personality_context, memory_context):
                    yield chunk
                    
        except asyncio.TimeoutError:
            raise Exception(f"{provider.value} request timeout")
        except aiohttp.ClientError as e:
            raise Exception(f"{provider.value} connection error: {str(e)}")
        except Exception as e:
            raise Exception(f"{provider.value} error: {str(e)}")
        
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
        
        # Build system message - keep it concise for better performance
        system_parts = []
        if personality_context:
            system_parts.append(personality_context[:600])  # Limit context for speed
        if memory_context:
            system_parts.append(f"Recent context: {memory_context[:300]}")
        
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        
        messages.append({"role": "user", "content": query})
        
        # Different parameters for different providers
        if provider == APIProvider.GROK:
            payload = {
                "model": config['model'],
                "messages": messages,
                "max_tokens": min(getattr(settings, 'max_response_tokens', 200), 300),
                "temperature": 0.7,
                "stream": True
            }
        else:  # OpenAI
            payload = {
                "model": config['model'], 
                "messages": messages,
                "max_tokens": min(getattr(settings, 'max_response_tokens', 200), 200),
                "temperature": 0.7,
                "stream": True
            }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {config["api_key"]}'
        }
        
        response_received = False
        error_content = ""
        
        try:
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
                                            response_received = True
                                except json.JSONDecodeError:
                                    continue
                    
                    if not response_received:
                        yield f"No response received from {provider.value}."
                
                else:
                    # Handle specific error codes
                    error_content = await response.text()
                    if response.status == 429:
                        if "quota" in error_content.lower():
                            raise Exception(f"API quota exceeded for {provider.value}")
                        else:
                            raise Exception(f"Rate limited by {provider.value}")
                    elif response.status == 401:
                        raise Exception(f"Invalid API key for {provider.value}")
                    elif response.status == 403:
                        raise Exception(f"Access denied for {provider.value}")
                    else:
                        raise Exception(f"{provider.value} API error {response.status}: {error_content[:200]}")
                        
        except aiohttp.ClientError as e:
            raise Exception(f"{provider.value} connection failed: {str(e)}")
    
    async def _stream_anthropic(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from Anthropic API"""
        config = self.api_configs[APIProvider.ANTHROPIC]
        
        # Build system prompt
        system_parts = []
        if personality_context:
            system_parts.append(personality_context[:600])
        if memory_context:
            system_parts.append(f"Recent context: {memory_context[:300]}")
        
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        
        payload = {
            "model": config['model'],
            "max_tokens": min(getattr(settings, 'max_response_tokens', 200), 250),
            "messages": [{"role": "user", "content": query}],
            "stream": True,
            "temperature": 0.7
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': config['api_key'],
            'anthropic-version': '2023-06-01'
        }
        
        response_received = False
        try:
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
                                            response_received = True
                                except json.JSONDecodeError:
                                    continue
                    
                    if not response_received:
                        yield "No response received from Anthropic."
                else:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error {response.status}: {error_text[:200]}")
        except aiohttp.ClientError as e:
            raise Exception(f"Anthropic connection failed: {str(e)}")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate non-streaming response (fallback)"""
        if not self.initialization_successful or not self.available_providers:
            return "I'm having trouble connecting to online services right now."
        
        # Collect streaming response
        response_parts = []
        try:
            async for chunk in self.generate_response_stream(query, personality_context, memory_context):
                response_parts.append(chunk)
            
            response = ''.join(response_parts)
            return response if response else "No response received from online services."
            
        except Exception as e:
            self.last_error = str(e)
            if settings.debug_mode:
                print(f"❌ Online response error: {str(e)[:150]}")
            return f"I'm having trouble connecting to online services right now."
    
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
            invalid_keys = [None, '', 'your_api_key_here', f'your_{provider.value}_api_key_here']
            api_key_configured = api_key and api_key not in invalid_keys
            
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
