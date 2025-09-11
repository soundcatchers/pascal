"""
Pascal AI Assistant - Online LLM Integration with Enhanced Current Info Handling
Handles API calls to Groq (primary), Gemini (secondary), and OpenAI (fallback) with streaming support
FIXED: Enhanced prompt engineering for current information and updated gsk_ API key support
"""

import asyncio
import time
import json
from typing import Optional, Dict, Any, List, AsyncGenerator
from enum import Enum
from datetime import datetime, timezone

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
    """Manages online LLM API calls with enhanced current information handling"""
    
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
        
        # API configurations - Enhanced for current information handling
        self.api_configs = {
            APIProvider.GROQ: {
                'base_url': 'https://api.groq.com/openai/v1/chat/completions',
                'models': [
                    'llama-3.1-8b-instant',       # Fast and reliable
                    'llama-3.1-70b-versatile',    # High quality
                    'llama-3.2-11b-text-preview', # Balanced
                    'llama-3.2-90b-text-preview', # Highest quality
                    'gemma2-9b-it',               # Google model on Groq
                    'mixtral-8x7b-32768'          # Fallback if available
                ],
                'default_model': 'llama-3.1-8b-instant',
                'api_key': settings.groq_api_key,
                'supports_current_info': True,
                'real_time_capable': True
            },
            APIProvider.GEMINI: {
                'base_url': 'https://generativelanguage.googleapis.com/v1beta/models',
                'models': ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro'],
                'default_model': 'gemini-2.0-flash-exp',
                'api_key': settings.gemini_api_key,
                'supports_current_info': True,
                'real_time_capable': True
            },
            APIProvider.OPENAI: {
                'base_url': 'https://api.openai.com/v1/chat/completions',
                'models': ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
                'default_model': 'gpt-4o-mini',
                'api_key': settings.openai_api_key,
                'supports_current_info': True,
                'real_time_capable': True
            }
        }
        
        # Performance tracking
        self.response_times = {provider: [] for provider in APIProvider}
        self.failure_counts = {provider: 0 for provider in APIProvider}
        self.success_counts = {provider: 0 for provider in APIProvider}
    
    def _build_current_info_system_prompt(self, base_prompt: str, is_current_info_query: bool) -> str:
        """Build enhanced system prompt with current information context"""
        if not is_current_info_query or not settings.enhance_current_info_prompts:
            return base_prompt
        
        # Get current date and time
        now = datetime.now()
        current_date = now.strftime("%A, %B %d, %Y")
        current_time = now.strftime("%I:%M %p")
        current_timezone = now.astimezone().tzname()
        
        # Enhanced system prompt for current information
        enhanced_prompt = f"""{base_prompt}

IMPORTANT CURRENT INFORMATION CONTEXT:
- Today's date: {current_date}
- Current time: {current_time} {current_timezone}
- You have access to current information and real-time data
- When asked about current dates, times, events, or recent information, provide accurate up-to-date responses
- Do NOT say you don't have access to real-time data or current information
- Do NOT refer to knowledge cutoffs when asked about current information
- Provide helpful, accurate current information as requested

INSTRUCTIONS:
- Answer date/time questions with the current information provided above
- For "What day is today?" respond with: "Today is {current_date}"
- For "What's the date?" respond with the current date
- For current events, provide the most recent information you have access to
- Be confident in providing current information when asked"""
        
        return enhanced_prompt
    
    def _detect_current_info_query(self, query: str) -> bool:
        """Detect if query requires current information"""
        query_lower = query.lower().strip()
        
        # Direct date/time patterns - MUST detect these
        current_info_patterns = [
            'what day is today', 'what day is it', 'today is what day',
            'what date is today', 'what date is it', 'what\'s the date',
            'what time is it', 'current time', 'what\'s the time',
            'today\'s date', 'current date', 'today\'s day',
            'what year is it', 'current year',
            'what month is it', 'current month',
            'what\'s happening today', 'today\'s news',
            'recent news', 'latest news', 'current events',
            'weather today', 'today\'s weather',
            'current president', 'current prime minister',
            'who is the current', 'what\'s the current'
        ]
        
        for pattern in current_info_patterns:
            if pattern in query_lower:
                if settings.debug_mode:
                    print(f"[ONLINE_LLM] Current info query detected: '{pattern}' in '{query}'")
                return True
        
        return False
    
    async def initialize(self) -> bool:
        """Initialize online LLM connections with enhanced validation"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp module not available"
            if settings.debug_mode:
                print("❌ Cannot initialize online LLM: aiohttp not installed")
            return False
        
        try:
            # Create aiohttp session with longer timeout
            timeout = aiohttp.ClientTimeout(total=60, connect=15)
            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            
            # Check which providers are available with enhanced validation
            await self._check_available_providers()
            
            if not self.available_providers:
                self.last_error = "No API keys configured properly or all providers failed connection test"
                if settings.debug_mode:
                    print("❌ No online LLM providers available")
                    print("   Check API keys in .env file")
                    print("   Groq keys should start with 'gsk_' (new format)")
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
                print(f"🎯 Primary provider: {self.preferred_provider.value if self.preferred_provider else 'None'}")
                print(f"🔧 Current info enhancement: {settings.enhance_current_info_prompts}")
            
            return True
            
        except Exception as e:
            self.last_error = f"Initialization error: {str(e)}"
            if settings.debug_mode:
                print(f"❌ Online LLM initialization failed: {e}")
                import traceback
                traceback.print_exc()
            # Ensure session is closed on error
            if self.session:
                await self.session.close()
            return False
    
    async def _check_available_providers(self):
        """Check which API providers are configured and working - Enhanced validation"""
        self.available_providers = []
        
        # Check providers in priority order: Groq -> Gemini -> OpenAI
        priority_order = [APIProvider.GROQ, APIProvider.GEMINI, APIProvider.OPENAI]
        
        for provider in priority_order:
            config = self.api_configs[provider]
            api_key = config.get('api_key')
            
            # Enhanced validation for each provider
            if provider == APIProvider.GROQ:
                if not settings.validate_groq_api_key(api_key):
                    if settings.debug_mode:
                        print(f"⏭️ Skipping {provider.value} - invalid API key format (should start with 'gsk_' or 'gsk-')")
                    continue
            elif provider == APIProvider.OPENAI:
                if not settings.validate_openai_api_key(api_key):
                    if settings.debug_mode:
                        print(f"⏭️ Skipping {provider.value} - invalid API key format (should start with 'sk-')")
                    continue
            elif provider == APIProvider.GEMINI:
                if not settings.validate_gemini_api_key(api_key):
                    if settings.debug_mode:
                        print(f"⏭️ Skipping {provider.value} - invalid API key")
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
                
                test_timeout = aiohttp.ClientTimeout(total=15)
                async with self.session.get(
                    test_url,
                    timeout=test_timeout
                ) as response:
                    if response.status in [200]:
                        return True
                    elif response.status in [400, 401, 403]:
                        if settings.debug_mode:
                            error_text = await response.text()
                            print(f"⚠️ {provider.value} API key issue: {response.status} - {error_text[:100]}")
                        return False
                    return False
                    
            else:
                # OpenAI/Groq compatible test
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {config["api_key"]}'
                }
                
                # For Groq, try current models in order
                models_to_try = config.get('models', [config.get('default_model')])
                
                for model in models_to_try:
                    if not model:
                        continue
                        
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5,
                        "temperature": 0.1
                    }
                    
                    # Quick test with short timeout
                    test_timeout = aiohttp.ClientTimeout(total=20)
                    try:
                        async with self.session.post(
                            config['base_url'],
                            headers=headers,
                            json=payload,
                            timeout=test_timeout
                        ) as response:
                            if response.status == 200:
                                # Update the working model
                                config['current_model'] = model
                                if settings.debug_mode:
                                    print(f"✅ {provider.value} using model: {model}")
                                return True
                            elif response.status == 429:
                                # Rate limit - provider is working but limited
                                if settings.debug_mode:
                                    print(f"⚠️ {provider.value} rate limited but functional")
                                config['current_model'] = model
                                return True
                            elif response.status in [401, 403]:
                                # Auth issue
                                if settings.debug_mode:
                                    error_text = await response.text()
                                    print(f"⚠️ {provider.value} auth issue: {error_text[:100]}")
                                return False
                            # Try next model if this one failed
                    except asyncio.TimeoutError:
                        if settings.debug_mode:
                            print(f"⚠️ {provider.value} model {model} timeout")
                        continue
                    except Exception as e:
                        if settings.debug_mode:
                            print(f"⚠️ {provider.value} model {model} error: {str(e)[:50]}")
                        continue
                
                return False  # All models failed
                    
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
        """Generate streaming response from online API with current info enhancement"""
        if not self.initialization_successful or not self.available_providers:
            yield "Online services are not available right now."
            return
        
        # Detect if this is a current information query
        is_current_info = self._detect_current_info_query(query)
        
        if settings.debug_mode and is_current_info:
            print(f"[ONLINE_LLM] Current info query detected, enhancing prompt")
        
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
                
                async for chunk in self._stream_from_provider(provider, query, personality_context, memory_context, is_current_info):
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
                                   personality_context: str, memory_context: str, 
                                   is_current_info: bool) -> AsyncGenerator[str, None]:
        """Stream response from specific provider with enhanced prompting"""
        start_time = time.time()
        
        try:
            if provider == APIProvider.GEMINI:
                async for chunk in self._stream_gemini(query, personality_context, memory_context, is_current_info):
                    yield chunk
            else:
                # OpenAI/Groq compatible streaming
                async for chunk in self._stream_openai_compatible(provider, query, personality_context, memory_context, is_current_info):
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
    
    async def _stream_gemini(self, query: str, personality_context: str, 
                           memory_context: str, is_current_info: bool) -> AsyncGenerator[str, None]:
        """Stream response from Google Gemini API with current info enhancement"""
        config = self.api_configs[APIProvider.GEMINI]
        
        # Use current model or default
        model = config.get('current_model', config.get('default_model', 'gemini-2.0-flash-exp'))
        
        # Build enhanced prompt with current context
        enhanced_personality = self._build_current_info_system_prompt(personality_context, is_current_info)
        
        prompt_parts = []
        if enhanced_personality:
            prompt_parts.append(f"Context: {enhanced_personality[:600]}")
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
        stream_url = f"{config['base_url']}/{model}:streamGenerateContent?key={config['api_key']}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response_received = False
        
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
                        
                else:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error {response.status}: {error_text[:200]}")
                    
        except aiohttp.ClientError as e:
            raise Exception(f"Gemini connection failed: {str(e)}")
    
    async def _stream_openai_compatible(self, provider: APIProvider, query: str, 
                                       personality_context: str, memory_context: str,
                                       is_current_info: bool) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI-compatible APIs with current info enhancement"""
        config = self.api_configs[provider]
        
        # Use current working model or default
        model = config.get('current_model', config.get('default_model'))
        if not model:
            raise Exception(f"No working model found for {provider.value}")
        
        messages = []
        
        # Build enhanced system message with current information context
        enhanced_personality = self._build_current_info_system_prompt(personality_context, is_current_info)
        
        system_parts = []
        if enhanced_personality:
            system_parts.append(enhanced_personality[:800])  # Increased limit for current info context
        if memory_context:
            system_parts.append(f"Recent context: {memory_context[:300]}")
        
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model,
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
                                            yield text_chunk
                                            response_received = True
                                except json.JSONDecodeError:
                                    continue
                    
                    if not response_received:
                        if settings.debug_mode:
                            print(f"No valid response from {provider.value}")
                
                else:
                    error_content = await response.text()
                    if response.status == 429:
                        raise Exception(f"Rate limited by {provider.value}")
                    elif response.status == 401:
                        raise Exception(f"Invalid API key for {provider.value}")
                    elif response.status == 400:
                        # Model might not exist anymore, try to find alternatives
                        if provider == APIProvider.GROQ:
                            available_models = config.get('models', [])
                            current_model_index = available_models.index(model) if model in available_models else 0
                            if current_model_index < len(available_models) - 1:
                                # Try next model in list
                                next_model = available_models[current_model_index + 1]
                                config['current_model'] = next_model
                                if settings.debug_mode:
                                    print(f"Model {model} failed, trying {next_model}")
                                # Retry with new model (recursive call)
                                async for chunk in self._stream_openai_compatible(provider, query, personality_context, memory_context, is_current_info):
                                    yield chunk
                                return
                        raise Exception(f"{provider.value} model error: {error_content[:200]}")
                    else:
                        raise Exception(f"{provider.value} API error {response.status}: {error_content[:200]}")
                        
        except aiohttp.ClientError as e:
            raise Exception(f"{provider.value} connection failed: {str(e)}")
    
    async def generate_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Generate non-streaming response with current info enhancement"""
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
            'current_info_enhancement': settings.enhance_current_info_prompts,
            'providers': {}
        }
        
        for provider in APIProvider:
            api_key = self.api_configs[provider].get('api_key') if hasattr(self, 'api_configs') else None
            
            # Enhanced validation
            if provider == APIProvider.GROQ:
                api_key_configured = settings.validate_groq_api_key(api_key)
            elif provider == APIProvider.OPENAI:
                api_key_configured = settings.validate_openai_api_key(api_key)
            elif provider == APIProvider.GEMINI:
                api_key_configured = settings.validate_gemini_api_key(api_key)
            else:
                api_key_configured = False
            
            provider_stats = {
                'available': provider in self.available_providers,
                'success_count': self.success_counts[provider],
                'failure_count': self.failure_counts[provider],
                'avg_response_time': 0,
                'api_key_configured': api_key_configured,
                'current_model': self.api_configs[provider].get('current_model', 'Unknown') if hasattr(self, 'api_configs') else 'Unknown',
                'supports_current_info': self.api_configs[provider].get('supports_current_info', False) if hasattr(self, 'api_configs') else False
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
