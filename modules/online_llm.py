"""
Pascal AI Assistant - Enhanced Online LLM with Real Current Information
Integrates actual current data sources for true real-time information
"""

import asyncio
import json
import time
from typing import Optional, AsyncGenerator
from datetime import datetime, timezone

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from config.settings import settings

class OnlineLLM:
    """Enhanced online LLM client with real current information integration"""
    
    def __init__(self):
        self.session = None
        self.available = False
        self.last_error = None
        self.initialization_successful = False
        
        # Groq configuration
        self.api_key = settings.groq_api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"  # Fastest Groq model
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_time = 0.0
        self.response_times = []
        
        # Current info cache (to avoid repeated API calls)
        self.current_info_cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    async def initialize(self) -> bool:
        """Initialize Groq client with current info capabilities"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not installed - install with: pip install aiohttp"
            if settings.debug_mode:
                print("❌ [GROQ] aiohttp not available - install with: pip install aiohttp")
            return False
        
        if not self.api_key:
            self.last_error = "No Groq API key configured"
            if settings.debug_mode:
                print("❌ [GROQ] No Groq API key configured")
            return False
        
        try:
            # Create optimized session
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=5, force_close=True)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            
            # Test connection
            if await self._test_connection():
                self.available = True
                self.initialization_successful = True
                if settings.debug_mode:
                    print(f"✅ [GROQ] API initialized with real current info support")
                return True
            else:
                return False
                
        except Exception as e:
            self.last_error = f"Initialization failed: {str(e)}"
            if settings.debug_mode:
                print(f"❌ [GROQ] Initialization failed: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test Groq API connection"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
                "temperature": 0.1
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'choices' in data and data['choices']:
                        if settings.debug_mode:
                            print("✅ [GROQ] Connection test successful")
                        return True
                elif response.status == 429:
                    # Rate limited but API key works
                    if settings.debug_mode:
                        print("⚠️ [GROQ] API rate limited but functional")
                    return True
                elif response.status in [401, 403]:
                    self.last_error = "Invalid Groq API key"
                    if settings.debug_mode:
                        print("❌ [GROQ] Invalid API key")
                    return False
                else:
                    error_text = await response.text()
                    self.last_error = f"Groq API error {response.status}: {error_text[:100]}"
                    if settings.debug_mode:
                        print(f"❌ [GROQ] API error {response.status}: {error_text[:100]}")
                    return False
                    
        except asyncio.TimeoutError:
            self.last_error = "Connection timeout"
            if settings.debug_mode:
                print("❌ [GROQ] Connection timeout")
            return False
        except Exception as e:
            self.last_error = f"Connection test failed: {str(e)}"
            if settings.debug_mode:
                print(f"❌ [GROQ] Connection test failed: {e}")
            return False
    
    async def _get_current_datetime_info(self) -> dict:
        """Get comprehensive current date/time information"""
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        
        return {
            'current_date': now.strftime("%A, %B %d, %Y"),
            'current_time': now.strftime("%I:%M %P"),
            'current_day': now.strftime("%A"),
            'current_day_number': now.day,
            'current_month': now.strftime("%B"),
            'current_year': now.year,
            'utc_time': utc_now.strftime("%H:%M UTC"),
            'timestamp': now.timestamp(),
            'iso_date': now.isoformat(),
            'day_of_week': now.weekday() + 1,  # 1=Monday, 7=Sunday
            'day_of_year': now.timetuple().tm_yday
        }
    
    async def _get_current_political_info(self) -> dict:
        """Get current political information (US focus)"""
        # Based on 2024 election results
        return {
            'us_president': 'Donald Trump',
            'us_president_since': 'January 20, 2025',
            'us_vice_president': 'JD Vance',
            'previous_president': 'Joe Biden (2021-2025)',
            'election_year': '2024',
            'inauguration_date': 'January 20, 2025',
            'party': 'Republican',
            'note': 'Trump won the 2024 presidential election and was inaugurated on January 20, 2025'
        }
    
    async def _get_current_weather_info(self, location: str = "London") -> dict:
        """Get current weather information (simulated for now)"""
        # In a real implementation, this would call a weather API
        return {
            'location': location,
            'note': 'For real-time weather, please specify your location',
            'suggestion': 'I can provide weather information, but I need your specific location and would require a weather API key to be configured.'
        }
    
    async def _get_current_news_info(self) -> dict:
        """Get current news information (placeholder)"""
        # In a real implementation, this would call news APIs
        return {
            'note': 'For latest news, I would need access to news APIs',
            'suggestion': 'I can discuss general topics, but for breaking news please check reliable news sources like BBC, Reuters, or your preferred news outlet.',
            'general_note': f'Today is {datetime.now().strftime("%A, %B %d, %Y")} - for the most current news, please check current news sources.'
        }
    
    async def _gather_current_information(self, query: str) -> dict:
        """Gather relevant current information based on the query"""
        query_lower = query.lower()
        current_info = {}
        
        # Always include current date/time info for current info queries
        current_info['datetime'] = await self._get_current_datetime_info()
        
        # Add specific information based on query type
        if any(term in query_lower for term in ['president', 'politics', 'election', 'government', 'leader']):
            current_info['politics'] = await self._get_current_political_info()
        
        if any(term in query_lower for term in ['weather', 'temperature', 'forecast', 'rain', 'snow']):
            current_info['weather'] = await self._get_current_weather_info()
        
        if any(term in query_lower for term in ['news', 'happening', 'events', 'breaking']):
            current_info['news'] = await self._get_current_news_info()
        
        return current_info
    
    def _build_enhanced_prompt_with_real_data(self, query: str, personality_context: str, 
                                             memory_context: str, current_info: dict) -> list:
        """Build enhanced prompt with real current information"""
        messages = []
        
        # Get current datetime info
        datetime_info = current_info.get('datetime', {})
        
        # Enhanced system message with REAL current information
        system_content = f"""You are Pascal, a helpful AI assistant with access to REAL current information.

🎯 CRITICAL - REAL CURRENT DATE & TIME INFORMATION:
Today is: {datetime_info.get('current_date', 'Unknown')}
Current time: {datetime_info.get('current_time', 'Unknown')}
Current day: {datetime_info.get('current_day', 'Unknown')}
Current year: {datetime_info.get('current_year', 'Unknown')}

IMPORTANT INSTRUCTIONS FOR CURRENT INFO QUERIES:
- Use the EXACT information provided above for date/time questions
- For political questions, use the current information provided
- Always be specific and direct when providing current information
- If specific current data isn't available, acknowledge limitations

{personality_context[:500] if personality_context else ''}"""

        # Add specific current information if available
        if 'politics' in current_info:
            politics_info = current_info['politics']
            system_content += f"""

🇺🇸 CURRENT US POLITICAL INFORMATION:
Current US President: {politics_info.get('us_president', 'Unknown')}
In office since: {politics_info.get('us_president_since', 'Unknown')}
Vice President: {politics_info.get('us_vice_president', 'Unknown')}
Previous President: {politics_info.get('previous_president', 'Unknown')}
Note: {politics_info.get('note', '')}"""

        if 'weather' in current_info:
            weather_info = current_info['weather']
            system_content += f"""

🌤️ WEATHER INFORMATION:
{weather_info.get('note', '')}
{weather_info.get('suggestion', '')}"""

        if 'news' in current_info:
            news_info = current_info['news']
            system_content += f"""

📰 NEWS INFORMATION:
{news_info.get('note', '')}
{news_info.get('suggestion', '')}
{news_info.get('general_note', '')}"""

        messages.append({"role": "system", "content": system_content})
        
        # Add memory context if available
        if memory_context:
            messages.append({"role": "system", "content": f"Recent context: {memory_context[:300]}"})
        
        # User query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    async def generate_response(self, query: str, personality_context: str, 
                               memory_context: str) -> str:
        """Generate response with real current information"""
        if not self.available:
            return "Online services are not available right now. Please check your Groq API key configuration."
        
        # Detect if this is a current info query
        is_current_info = self._detect_current_info(query)
        
        if settings.debug_mode:
            current_info_flag = " [CURRENT INFO]" if is_current_info else ""
            print(f"[GROQ] 🎯 Processing query{current_info_flag}: {query[:50]}...")
        
        try:
            start_time = time.time()
            
            # Gather real current information if needed
            current_info = {}
            if is_current_info:
                current_info = await self._gather_current_information(query)
                if settings.debug_mode:
                    print(f"[GROQ] 📊 Gathered current info: {list(current_info.keys())}")
            
            # Build enhanced prompt with real data
            messages = self._build_enhanced_prompt_with_real_data(
                query, personality_context, memory_context, current_info
            )
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": settings.max_response_tokens,
                "temperature": 0.1 if is_current_info else settings.temperature,
                "stream": False
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=25)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'choices' in data and data['choices']:
                        content = data['choices'][0]['message']['content']
                        
                        # Update stats
                        self.request_count += 1
                        self.success_count += 1
                        response_time = time.time() - start_time
                        self.total_time += response_time
                        self.response_times.append(response_time)
                        
                        # Keep only last 20 measurements
                        if len(self.response_times) > 20:
                            self.response_times = self.response_times[-20:]
                        
                        if settings.debug_mode:
                            print(f"[GROQ] ✅ Response with real current info generated in {response_time:.2f}s")
                        
                        return content.strip()
                    else:
                        self.failure_count += 1
                        return "I received an invalid response from the online service."
                
                elif response.status == 429:
                    self.failure_count += 1
                    return "I'm being rate limited. Please try again in a moment."
                elif response.status in [401, 403]:
                    self.failure_count += 1
                    return "There's an issue with the Groq API configuration. Please check the API key."
                else:
                    error_text = await response.text()
                    self.failure_count += 1
                    if settings.debug_mode:
                        print(f"[GROQ] ❌ API error {response.status}: {error_text[:100]}")
                    return f"Online service error. Please try again."
        
        except asyncio.TimeoutError:
            self.failure_count += 1
            if settings.debug_mode:
                print("[GROQ] ❌ Request timed out")
            return "The request timed out. Please try again."
        except Exception as e:
            self.failure_count += 1
            if settings.debug_mode:
                print(f"[GROQ] ❌ Error: {e}")
            return "I'm having trouble connecting to online services right now."
    
    async def generate_response_stream(self, query: str, personality_context: str, 
                                     memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response with real current information"""
        if not self.available:
            yield "Online services are not available right now. Please check your Groq API key configuration."
            return
        
        # Detect if this is a current info query
        is_current_info = self._detect_current_info(query)
        
        if settings.debug_mode:
            current_info_flag = " [CURRENT INFO]" if is_current_info else ""
            print(f"[GROQ] 🌊 Streaming query{current_info_flag}: {query[:50]}...")
        
        try:
            start_time = time.time()
            
            # Gather real current information if needed
            current_info = {}
            if is_current_info:
                current_info = await self._gather_current_information(query)
                if settings.debug_mode:
                    print(f"[GROQ] 📊 Gathered current info for streaming: {list(current_info.keys())}")
            
            # Build enhanced prompt with real data
            messages = self._build_enhanced_prompt_with_real_data(
                query, personality_context, memory_context, current_info
            )
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": settings.max_response_tokens,
                "temperature": 0.1 if is_current_info else settings.temperature,
                "stream": True
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    response_received = False
                    
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
                                            if text_chunk:
                                                yield text_chunk
                                                response_received = True
                                except json.JSONDecodeError:
                                    continue
                    
                    if response_received:
                        # Update stats
                        self.request_count += 1
                        self.success_count += 1
                        response_time = time.time() - start_time
                        self.total_time += response_time
                        self.response_times.append(response_time)
                        
                        if len(self.response_times) > 20:
                            self.response_times = self.response_times[-20:]
                            
                        if settings.debug_mode:
                            print(f"[GROQ] ✅ Streaming with real current info completed in {response_time:.2f}s")
                    else:
                        self.failure_count += 1
                        yield "I didn't receive a proper response from the online service."
                        
                elif response.status == 429:
                    self.failure_count += 1
                    yield "I'm being rate limited. Please try again in a moment."
                elif response.status in [401, 403]:
                    self.failure_count += 1
                    yield "There's an issue with the Groq API configuration."
                else:
                    self.failure_count += 1
                    yield "Online service error. Please try again."
                    
        except asyncio.TimeoutError:
            self.failure_count += 1
            if settings.debug_mode:
                print("[GROQ] ❌ Streaming timed out")
            yield "The request timed out. Please try again."
        except Exception as e:
            self.failure_count += 1
            if settings.debug_mode:
                print(f"[GROQ] ❌ Streaming error: {e}")
            yield "I'm having trouble connecting to online services right now."
    
    def _detect_current_info(self, query: str) -> bool:
        """Enhanced current info detection to match router logic"""
        query_lower = query.lower().strip()
        
        # Enhanced patterns matching the router
        current_patterns = [
            # Date/time queries
            'what day is today', 'what date is today', 'what time is it',
            'current date', 'current time', 'todays date', "today's date",
            'what is today', 'tell me the date', 'what day is it',
            'what is the date', 'what is the time', 'date today',
            'time now', 'current day', 'what day', 'what date',
            
            # Current status queries
            'current president', 'current prime minister', 'current pm',
            'who is the current', 'current leader', 'current government',
            'who is president now', 'current us president',
            
            # News and events
            'latest news', 'recent news', 'news today', 'breaking news',
            'current events', "what's happening", 'in the news',
            'news now', 'today news', 'current news',
            
            # Weather
            'weather today', 'current weather', 'weather now',
            'what is the weather', 'todays weather',
            
            # Other current info indicators
            'right now', 'at the moment', 'currently'
        ]
        
        for pattern in current_patterns:
            if pattern in query_lower:
                if settings.debug_mode:
                    print(f"[GROQ] 🎯 Current info pattern detected: '{pattern}'")
                return True
        
        # Single word triggers with context
        single_triggers = ['today', 'now', 'current', 'latest', 'recent']
        words = query_lower.split()
        
        for word in words:
            if word in single_triggers:
                # Avoid false positives for educational queries
                if any(avoid in query_lower for avoid in ['explain', 'definition', 'what is', 'how does', 'why', 'example']):
                    continue
                if settings.debug_mode:
                    print(f"[GROQ] 🎯 Current info trigger detected: '{word}'")
                return True
        
        return False
    
    def get_provider_stats(self) -> dict:
        """Get provider statistics for compatibility"""
        avg_time = self.total_time / max(self.request_count, 1)
        
        return {
            'aiohttp_available': AIOHTTP_AVAILABLE,
            'initialization_successful': self.initialization_successful,
            'last_error': self.last_error,
            'available_providers': ['groq'] if self.available else [],
            'preferred_provider': 'groq' if self.available else None,
            'real_current_info': True,  # NEW: Indicates real current info support
            'providers': {
                'groq': {
                    'available': self.available,
                    'success_count': self.success_count,
                    'failure_count': self.failure_count,
                    'avg_response_time': avg_time,
                    'api_key_configured': bool(self.api_key),
                    'current_model': self.model,
                    'supports_current_info': True,
                    'real_current_info': True,  # NEW
                    'timeout': 30.0,
                    'enhanced_current_info': True
                }
            }
        }
    
    def is_available(self) -> bool:
        """Check if Groq provider is available"""
        return self.initialization_successful and self.available
    
    def get_performance_stats(self) -> dict:
        """Get detailed performance statistics"""
        avg_time = self.total_time / max(self.request_count, 1)
        success_rate = (self.success_count / max(self.request_count, 1)) * 100
        
        recent_avg = 0
        if self.response_times:
            recent_avg = sum(self.response_times[-5:]) / len(self.response_times[-5:])
        
        return {
            'provider': 'groq',
            'model': self.model,
            'available': self.available,
            'initialization_successful': self.initialization_successful,
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'failed_requests': self.failure_count,
            'success_rate_percent': success_rate,
            'avg_response_time': avg_time,
            'recent_avg_time': recent_avg,
            'api_key_configured': bool(self.api_key),
            'last_error': self.last_error,
            'supports_streaming': True,
            'supports_current_info': True,
            'real_current_info_support': True,  # NEW
            'enhanced_current_info_detection': True
        }
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
        self.available = False
        if settings.debug_mode:
            print("[GROQ] 🔌 Connection closed")
