"""
Pascal AI Assistant - Online LLM Integration with Groq Priority
Handles API calls to Groq (primary), Gemini (secondary), and OpenAI (fallback) with streaming support
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
    """Available API providers in priority order"""
    GROQ = "groq"      # Primary - fastest and most efficient
    GEMINI = "gemini"  # Secondary - good quality and free
    OPENAI = "openai"  # Fallback - reliable but paid

class OnlineLLM:
    """Manages online LLM API calls with Groq as primary, Gemini as secondary, OpenAI as fallback"""
    
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
            # Initialize api_configs even if aiohttp is not available
            self.api_configs = {}
            return
        
        # API configurations - Groq as primary, Gemini as secondary, OpenAI as fallback
        self.api_configs = {
            APIProvider.GROQ: {
                'base_url': 'https://api.groq.com/openai/v1/chat/completions',
                'model': 'llama-3.1-70b-versatile',  # Fast and capable model
                'api_key': getattr(settings, 'grok_api_key', None)  # Using grok_api_key for Groq
            },
            APIProvider.GEMINI: {
                'base_url': 'https://generativelanguage.googleapis.com/v1beta/models',
                'model': 'gemini-2.0-flash-exp',
                'api_key': getattr(settings, 'gemini_api_key', None) or getattr(settings, 'google_api_key', None)
            },
            APIProvider.OPENAI: {
                'base_url': 'https://api.openai.com/v1/chat/completions',
                'model': 'gpt-4o-mini',
                'api_key': getattr(settings, 'openai_api_key', None)
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
            
            # Check which providers are available - PRIORITIZE GROQ FIRST
            await self._check_available_providers()
            
            if not self.available_providers:
                self.last_error = "No API keys configured properly or all providers failed connection test"
                if settings.debug_mode:
                    print("❌ No online LLM providers available")
                    print("   Check API keys in .env file")
                return False
            
            # Set preferred provider - GROQ FIRST, then GEMINI, then OPENAI
            for provider in [APIProvider.GROQ, APIProvider.GEMINI, APIProvider.OPENAI]:
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
        
        # Check providers in priority order: Groq -> Gemini -> OpenAI
        priority_order = [APIProvider.GROQ, APIProvider.GEMINI, APIProvider.OPENAI]
        
        for provider in priority_order:
            config = self.api_configs[provider]
            api_key = config.get('api_key')
            
            # Skip if no API key or placeholder
            invalid_keys = [None, '', 'your_api_key_here', f'your_{provider.value}_api_key_here', 
                          'your_gemini_api_key_here', 'your_google_api_key_here',
                          'your_grok_api_key_here', 'your_groq_api_key_here']
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
            
            if provider == APIProvider.GEMINI:
                # Test Gemini API - just check if we can list models
                test_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={config['api_key']}"
                
                test_timeout = aiohttp.ClientTimeout(total=10)
                async with self.session.get(
                    test_url,
                    timeout=test_timeout
                ) as response:
                    if response.status in [200, 400, 401, 403, 429]:
                        if response.status == 401 or response.status == 403:
                            if settings.debug_mode:
                                print(f"⚠️ {provider.value} invalid API key")
                            return False
                        return True
                    return False
                    
            else:
                # OpenAI/Groq compatible test
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
                    # Accept various response codes but handle quota/permission errors
                    if response.status == 200:
                        return True
                    elif response.status == 429:
                        # Rate limit or quota exceeded
                        if settings.debug_mode:
                            print(f"⚠️ {provider.value} quota exceeded or rate limited")
                        return False  # Don't use if quota exceeded
                    elif response.status == 401:
                        # Invalid API key
                        if settings.debug_mode:
                            print(f"⚠️ {provider.value} invalid API key")
                        return False
                    elif response.status == 403:
                        # No permission/credits
                        if settings.debug_mode:
                            print(f"⚠️ {provider.value} no credits or permission denied")
                        return False
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
        """Generate streaming response from online API"""
        if not self.initialization_successful or not self.available_providers:
            yield "Online services are not available right now."
            return
        
        # Try providers in priority order: Groq -> Gemini -> OpenAI
        providers_to_try = []
        
        # Always try Groq first if available
        if APIProvider.GROQ in self.available_providers:
            providers_to_try.append(APIProvider.GROQ)
        
        # Then Gemini
        if APIProvider.GEMINI in self.available_providers:
            providers_to_try.append(APIProvider.GEMINI)
        
        # Finally OpenAI
        if APIProvider.OPENAI in self.available_providers:
            providers_to_try.append(APIProvider.OPENAI)
        
        last_error = None
        
        for provider in providers_to_try:
            try:
                if settings.debug_mode:
                    print(f"🔄 Trying provider: {provider.value}")
                
                response_generated = False
                response_buffer = []
                
                async for chunk in self._stream_from_provider(provider, query, personality_context, memory_context):
                    if chunk:  # Only yield non-empty chunks
                        yield chunk
                        response_buffer.append(chunk)
                        response_generated = True
                
                if response_generated and len(''.join(response_buffer)) > 0:
                    self.success_counts[provider] += 1
                    if settings.debug_mode:
                        print(f"✅ Success with {provider.value}")
                    return  # Successfully got response, exit
                else:
                    # No response from this provider, try next
                    if settings.debug_mode:
                        print(f"⚠️ No response from {provider.value}, trying next provider")
                    continue
                
            except Exception as e:
                self.failure_counts[provider] += 1
                last_error = str(e)
                if settings.debug_mode:
                    print(f"❌ {provider.value} failed: {str(e)[:150]}")
                continue
        
        # All providers failed
        self.last_error = last_error
        yield f"I'm having trouble connecting to online services. Error: {last_error}"
    
    async def _stream_from_provider(self, provider: APIProvider, query: str, 
                                   personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from specific provider"""
        start_time = time.time()
        
        try:
            if provider == APIProvider.GEMINI:
                async for chunk in self._stream_gemini(query, personality_context, memory_context):
                    yield chunk
            else:
                # OpenAI/Groq compatible streaming
                async for chunk in self._stream_openai_compatible(provider, query, personality_context, memory_context):
                    yield chunk
                    
        except asyncio.TimeoutError:
            raise Exception(f"{provider.value} request timeout")
        except aiohttp.ClientError as e:
            raise Exception(f"{provider.value} connection error: {str(e)}")
        except Exception as e:
            raise Exception(f"{provider.value} error: {str(e)}")
        finally:
            # Record response time
            response_time = time.time() - start_time
            self.response_times[provider].append(response_time)
            if len(self.response_times[provider]) > 10:
                self.response_times[provider] = self.response_times[provider][-10:]
    
    async def _stream_gemini(self, query: str, personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from Google Gemini API"""
        config = self.api_configs[APIProvider.GEMINI]
        
        # Build prompt with context
        prompt_parts = []
        if personality_context:
            prompt_parts.append(f"Context: {personality_context[:600]}")
        if memory_context:
            prompt_parts.append(f"Recent conversation: {memory_context[:300]}")
        prompt_parts.append(f"User: {query}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        # Gemini API format
        payload = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": min(getattr(settings, 'max_response_tokens', 200), 300),
                "topP": 0.9,
                "topK": 40
            }
        }
        
        # Use streaming endpoint with API key in URL
        stream_url = f"{config['base_url']}/{config['model']}:streamGenerateContent?key={config['api_key']}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response_received = False
        response_text = ""
        
        try:
            async with self.session.post(stream_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                # Parse the JSON response
                                line_str = line.decode('utf-8').strip()
                                if line_str:
                                    # Gemini sometimes sends multiple JSON objects in one line
                                    # Split by newline and parse each
                                    for json_str in line_str.split('\n'):
                                        if json_str.strip():
                                            try:
                                                data = json.loads(json_str)
                                                # Extract text from Gemini response format
                                                if 'candidates' in data:
                                                    for candidate in data['candidates']:
                                                        if 'content' in candidate and 'parts' in candidate['content']:
                                                            for part in candidate['content']['parts']:
                                                                if 'text' in part:
                                                                    text_chunk = part['text']
                                                                    response_text += text_chunk
                                                                    yield text_chunk
                                                                    response_received = True
                                            except json.JSONDecodeError:
                                                continue
                            except Exception as parse_error:
                                if settings.debug_mode:
                                    print(f"Parse error in Gemini response: {parse_error}")
                                continue
                    
                    if not response_received:
                        if settings.debug_mode:
                            print(f"No valid response from Gemini")
                        # Don't yield error message here, let the main function handle it
                        
                else:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error {response.status}: {error_text[:200]}")
                    
        except aiohttp.ClientError as e:
            raise Exception(f"Gemini connection failed: {str(e)}")
    
    async def _stream_openai_compatible(self, provider: APIProvider, query: str, 
                                       personality_context: str, memory_context: str) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI-compatible APIs (OpenAI, Groq)"""
        config = self.api_configs[provider]
        
        messages = []
        
        # Build system message
        system_parts = []
        if personality_context:
            system_parts.append(personality_context[:600])
        if memory_context:
            system_parts.append(f"Recent context: {memory_context[:300]}")
        
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": config['model'],
            "messages": messages,
            "max_tokens": min(getattr(settings, 'max_response_tokens', 200), 300),
            "temperature": 0.7,
            "stream": True
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {config["api_key"]}'
        }
        
        response_received = False
        response_text = ""
        
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
                                            text_chunk = delta['content']
                                            response_text += text_chunk
                                            yield text_chunk
                                            response_received = True
                                except json.JSONDecodeError:
                                    continue
                    
                    if not response_received:
                        if settings.debug_mode:
                            print(f"No valid response from {provider.value}")
                        # Don't yield error message here, let the main function handle it
                
                else:
                    error_content = await response.text()
                    if response.status == 429:
                        raise Exception(f"Rate limited by {provider.value}")
                    elif response.status == 401:
                        raise Exception(f"Invalid API key for {provider.value}")
                    else:
                        raise Exception(f"{provider.value} API error {response.status}: {error_content[:200]}")
                        
        except aiohttp.ClientError as e:
            raise Exception(f"{provider.value} connection failed: {str(e)}")
    
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
            
            # Check if we got a valid response
            if not response or response.startswith("I'm having trouble"):
                return "I'm having trouble connecting to online services right now. Please try again."
            
            return response
            
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
            api_key = self.api_configs[provider].get('api_key') if hasattr(self, 'api_configs') else None
            invalid_keys = [None, '', 'your_api_key_here', f'your_{provider.value}_api_key_here',
                          'your_gemini_api_key_here', 'your_google_api_key_here',
                          'your_grok_api_key_here', 'your_groq_api_key_here']
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
