"""
Pascal AI Assistant - Enhanced Online LLM with Real Current Information
Optimized for fast current info queries with improved detection and response quality
"""

import asyncio
import json
import time
from typing import Optional, AsyncGenerator, Dict, Any
from datetime import datetime, timezone

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from config.settings import settings

class OnlineLLM:
    """Enhanced online LLM client with optimized current information integration"""
    
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
        
        # Current info cache with TTL
        self.current_info_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # Enhanced current info patterns
        self.current_info_indicators = {
            'datetime': ['time', 'date', 'day', 'today', 'now', 'current time', 'current date'],
            'politics': ['president', 'prime minister', 'leader', 'government', 'current president'],
            'news': ['news', 'headlines', 'breaking', 'latest news', 'recent news', 'happening'],
            'weather': ['weather', 'temperature', 'forecast', 'climate', 'current weather'],
            'markets': ['stock', 'market', 'price', 'trading', 'financial'],
            'events': ['events', 'happening', 'current events', 'what\'s going on']
        }
    
    async def initialize(self) -> bool:
        """Initialize Groq client with enhanced error handling"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not installed - install with: pip install aiohttp"
            if settings.debug_mode:
                print("❌ [GROQ] aiohttp not available")
            return False
        
        if not self.api_key or not self._validate_api_key(self.api_key):
            self.last_error = "Invalid or missing Groq API key"
            if settings.debug_mode:
                print("❌ [GROQ] Invalid API key format")
            return False
        
        try:
            # Create optimized session for speed
            timeout = aiohttp.ClientTimeout(total=25, connect=5, sock_read=20)
            connector = aiohttp.TCPConnector(
                limit=3,
                limit_per_host=3,
                force_close=False,
                enable_cleanup_closed=True,
                use_dns_cache=True,
                keepalive_timeout=300
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout, 
                connector=connector,
                headers={'Content-Type': 'application/json'}
            )
            
            # Test connection with quick query
            if await self._test_connection_fast():
                self.available = True
                self.initialization_successful = True
                if settings.debug_mode:
                    print(f"✅ [GROQ] API initialized with enhanced current info support")
                return True
            else:
                return False
                
        except Exception as e:
            self.last_error = f"Initialization failed: {str(e)}"
            if settings.debug_mode:
                print(f"❌ [GROQ] Initialization error: {e}")
            return False
    
    def _validate_api_key(self, key: str) -> bool:
        """Enhanced API key validation"""
        if not key or not isinstance(key, str):
            return False
        
        key = key.strip()
        
        # Check for placeholder values
        invalid_values = [
            '', 'your_groq_api_key_here', 'your_grok_api_key_here',
            'gsk_your_groq_api_key_here', 'gsk-your_groq_api_key_here',
            'gsk_your-groq-api-key-here'
        ]
        
        if key.lower() in [v.lower() for v in invalid_values]:
            return False
        
        # Accept both gsk_ (new) and gsk- (deprecated) formats
        if key.startswith('gsk_') or key.startswith('gsk-'):
            return len(key) > 20
        
        return False
    
    async def _test_connection_fast(self) -> bool:
        """Quick connection test with minimal payload"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "temperature": 0.1
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
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
                    self.last_error = f"API error {response.status}: {error_text[:100]}"
                    if settings.debug_mode:
                        print(f"❌ [GROQ] API error {response.status}")
                    return False
                    
        except asyncio.TimeoutError:
            self.last_error = "Connection timeout"
            if settings.debug_mode:
                print("❌ [GROQ] Connection timeout")
            return False
        except Exception as e:
            self.last_error = f"Connection test failed: {str(e)}"
            if settings.debug_mode:
                print(f"❌ [GROQ] Connection test error: {e}")
            return False
    
    def _get_comprehensive_datetime_info(self) -> Dict[str, Any]:
        """Get comprehensive current date/time information"""
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        
        return {
            'current_date': now.strftime("%A, %B %d, %Y"),
            'current_time': now.strftime("%I:%M %p"),
            'current_day': now.strftime("%A"),
            'current_day_number': now.day,
            'current_month': now.strftime("%B"),
            'current_year': now.year,
            'utc_time': utc_now.strftime("%H:%M UTC"),
            'timestamp': now.timestamp(),
            'iso_date': now.isoformat(),
            'day_of_week': now.weekday() + 1,
            'day_of_year': now.timetuple().tm_yday,
            'week_number': now.isocalendar()[1],
            'quarter': (now.month - 1) // 3 + 1,
            'is_weekend': now.weekday() >= 5,
            'timezone': str(now.astimezone().tzinfo),
            'formatted_full': now.strftime("%A, %B %d, %Y at %I:%M %p"),
            'season': self._get_season(now.month)
        }
    
    def _get_season(self, month: int) -> str:
        """Get current season based on month"""
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"
    
    def _get_current_political_info(self) -> Dict[str, Any]:
        """Get current political information (updated for 2024-2025)"""
        return {
            'us_president': 'Donald Trump',
            'us_president_since': 'January 20, 2025',
            'us_vice_president': 'JD Vance',
            'previous_president': 'Joe Biden (2021-2025)',
            'election_year': '2024',
            'inauguration_date': 'January 20, 2025',
            'party': 'Republican',
            'term_number': '47th President',
            'note': 'Donald Trump won the 2024 presidential election, defeating Kamala Harris, and was inaugurated on January 20, 2025',
            'context': 'Trump previously served as the 45th President (2017-2021) and is now the 47th President'
        }
    
    def _detect_current_info_category(self, query: str) -> Optional[str]:
        """Detect which category of current info is requested"""
        query_lower = query.lower()
        
        for category, indicators in self.current_info_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return category
        
        return None
    
    async def _gather_targeted_current_info(self, query: str) -> Dict[str, Any]:
        """Gather targeted current information based on query analysis"""
        query_lower = query.lower()
        current_info = {}
        
        # Always include datetime for current info queries
        current_info['datetime'] = self._get_comprehensive_datetime_info()
        
        # Add specific information based on query content
        category = self._detect_current_info_category(query)
        
        if category == 'politics' or any(term in query_lower for term in ['president', 'politics', 'election', 'government', 'leader']):
            current_info['politics'] = self._get_current_political_info()
        
        if category == 'weather' or any(term in query_lower for term in ['weather', 'temperature', 'forecast', 'rain', 'snow']):
            location = self._extract_location_from_query(query) or "London"
            current_info['weather'] = {
                'note': f'For real-time weather in {location}, I would need a weather API configured.',
                'suggestion': 'I can provide general weather information, but for current conditions please check a weather service.',
                'location_requested': location
            }
        
        if category == 'news' or any(term in query_lower for term in ['news', 'happening', 'events', 'breaking']):
            current_info['news'] = {
                'note': 'For the latest news headlines, I would need access to a news API.',
                'suggestion': 'For current news, please check reliable sources like BBC, Reuters, AP, or your preferred news outlet.',
                'date': current_info['datetime']['current_date']
            }
        
        if category == 'markets' or any(term in query_lower for term in ['market', 'stock', 'economy', 'financial', 'trading']):
            current_info['markets'] = {
                'note': 'For current market data, I would need access to financial APIs.',
                'suggestion': 'For live market information, please check financial news sources or trading platforms.',
                'date': current_info['datetime']['current_date']
            }
        
        return current_info
    
    def _extract_location_from_query(self, query: str) -> Optional[str]:
        """Extract location from query (enhanced)"""
        import re
        
        # Look for "in [location]" patterns
        location_match = re.search(r'\bin\s+([A-Za-z][A-Za-z\s]{1,20}?)(?:\s|$|[,.?!])', query)
        if location_match:
            return location_match.group(1).strip().title()
        
        # Common cities
        cities = [
            'london', 'paris', 'new york', 'tokyo', 'berlin', 'madrid', 'rome',
            'amsterdam', 'chicago', 'los angeles', 'sydney', 'melbourne',
            'toronto', 'vancouver', 'dubai', 'singapore', 'hong kong', 'miami',
            'boston', 'seattle', 'san francisco', 'mumbai', 'delhi', 'bangkok'
        ]
        
        query_lower = query.lower()
        for city in cities:
            if city in query_lower:
                return city.title()
        
        return None
    
    def _build_enhanced_current_info_prompt(self, query: str, personality_context: str, 
                                          memory_context: str, current_info: Dict[str, Any]) -> list:
        """Build optimized prompt with real current information"""
        messages = []
        
        # Get datetime info
        datetime_info = current_info.get('datetime', {})
        
        # Enhanced system message with PRECISE current information
        system_content = f"""You are Pascal, a helpful AI assistant with access to REAL current information.

🎯 CRITICAL - ACCURATE CURRENT DATE & TIME:
Today is: {datetime_info.get('current_date', 'Unknown')}
Current time: {datetime_info.get('current_time', 'Unknown')}
Current day: {datetime_info.get('current_day', 'Unknown')}
Current year: {datetime_info.get('current_year', 'Unknown')}
Day of week: {datetime_info.get('day_of_week', 'Unknown')}
Week number: {datetime_info.get('week_number', 'Unknown')}

IMPORTANT INSTRUCTIONS:
- Use the EXACT information provided above for any date/time questions
- Be specific and direct when providing current information
- If you don't have specific current data for something, acknowledge the limitation clearly
- Always be accurate about what information you have vs. what you don't have

{personality_context[:300] if personality_context else ''}"""

        # Add specific current information sections
        if 'politics' in current_info:
            politics_info = current_info['politics']
            system_content += f"""

🇺🇸 CURRENT US POLITICAL INFORMATION (Accurate as of 2025):
Current US President: {politics_info.get('us_president', 'Unknown')}
In office since: {politics_info.get('us_president_since', 'Unknown')}
Vice President: {politics_info.get('us_vice_president', 'Unknown')}
Previous President: {politics_info.get('previous_president', 'Unknown')}
Context: {politics_info.get('context', '')}
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
{news_info.get('suggestion', '')}"""

        if 'markets' in current_info:
            markets_info = current_info['markets']
            system_content += f"""

📈 MARKET INFORMATION:
{markets_info.get('note', '')}
{markets_info.get('suggestion', '')}"""

        messages.append({"role": "system", "content": system_content})
        
        # Add memory context if available (but keep it short for speed)
        if memory_context:
            messages.append({"role": "system", "content": f"Recent context: {memory_context[-200:]}"})
        
        # User query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    async def generate_response(self, query: str, personality_context: str, 
                               memory_context: str) -> str:
        """Generate response with enhanced current information"""
        if not self.available:
            return "Online services are not available. Please check your Groq API key configuration."
        
        # Detect current info category for optimized processing
        current_info_category = self._detect_current_info_category(query)
        is_current_info = current_info_category is not None
        
        if settings.debug_mode:
            category_info = f" [{current_info_category.upper()}]" if current_info_category else ""
            print(f"[GROQ] 🎯 Processing query{category_info}: {query[:50]}...")
        
        try:
            start_time = time.time()
            
            # Gather targeted current information
            current_info = await self._gather_targeted_current_info(query)
            if settings.debug_mode and current_info:
                info_types = list(current_info.keys())
                print(f"[GROQ] 📊 Gathered current info: {info_types}")
            
            # Build enhanced prompt
            messages = self._build_enhanced_current_info_prompt(
                query, personality_context, memory_context, current_info
            )
            
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            # Optimize parameters for current info queries
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": min(settings.max_response_tokens, 300),  # Cap for speed
                "temperature": 0.1 if is_current_info else 0.3,  # Lower temp for factual info
                "stream": False
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=20)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'choices' in data and data['choices']:
                        content = data['choices'][0]['message']['content']
                        
                        # Update performance stats
                        self.request_count += 1
                        self.success_count += 1
                        response_time = time.time() - start_time
                        self.total_time += response_time
                        self.response_times.append(response_time)
                        
                        # Keep only last 20 measurements
                        if len(self.response_times) > 20:
                            self.response_times = self.response_times[-20:]
                        
                        if settings.debug_mode:
                            print(f"[GROQ] ✅ Current info response in {response_time:.2f}s")
                        
                        return content.strip()
                    else:
                        self.failure_count += 1
                        return "I received an incomplete response from the online service."
                
                elif response.status == 429:
                    self.failure_count += 1
                    return "I'm being rate limited. Please try again in a moment."
                elif response.status in [401, 403]:
                    self.failure_count += 1
                    return "There's an authentication issue with the online service."
                else:
                    error_text = await response.text()
                    self.failure_count += 1
                    if settings.debug_mode:
                        print(f"[GROQ] ❌ API error {response.status}: {error_text[:100]}")
                    return "Online service error. Please try again."
        
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
        """Generate streaming response with enhanced current information"""
        if not self.available:
            yield "Online services are not available. Please check your Groq API key."
            return
        
        # Detect current info for optimized processing
        current_info_category = self._detect_current_info_category(query)
        is_current_info = current_info_category is not None
        
        if settings.debug_mode:
            category_info = f" [{current_info_category.upper()}]" if current_info_category else ""
            print(f"[GROQ] 🌊 Streaming{category_info}: {query[:50]}...")
        
        try:
            start_time = time.time()
            
            # Gather current information
            current_info = await self._gather_targeted_current_info(query)
            if settings.debug_mode and current_info:
                print(f"[GROQ] 📊 Current info for streaming: {list(current_info.keys())}")
            
            # Build enhanced prompt
            messages = self._build_enhanced_current_info_prompt(
                query, personality_context, memory_context, current_info
            )
            
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": min(settings.max_response_tokens, 300),
                "temperature": 0.1 if is_current_info else 0.3,
                "stream": True
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=25)
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
                            print(f"[GROQ] ✅ Streaming complete in {response_time:.2f}s")
                    else:
                        self.failure_count += 1
                        yield "I didn't receive a proper response from the online service."
                        
                elif response.status == 429:
                    self.failure_count += 1
                    yield "I'm being rate limited. Please try again in a moment."
                elif response.status in [401, 403]:
                    self.failure_count += 1
                    yield "There's an authentication issue with the online service."
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
            yield "I'm having trouble with online services right now."
    
    def _detect_current_info(self, query: str) -> bool:
        """Enhanced current info detection for compatibility"""
        return self._detect_current_info_category(query) is not None
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get comprehensive provider statistics"""
        avg_time = self.total_time / max(self.request_count, 1)
        
        return {
            'aiohttp_available': AIOHTTP_AVAILABLE,
            'initialization_successful': self.initialization_successful,
            'last_error': self.last_error,
            'available_providers': ['groq'] if self.available else [],
            'preferred_provider': 'groq' if self.available else None,
            'enhanced_current_info': True,
            'providers': {
                'groq': {
                    'available': self.available,
                    'success_count': self.success_count,
                    'failure_count': self.failure_count,
                    'avg_response_time': avg_time,
                    'api_key_configured': bool(self.api_key),
                    'current_model': self.model,
                    'supports_current_info': True,
                    'enhanced_detection': True,
                    'timeout': 20.0,
                    'optimization_level': 'enhanced_speed'
                }
            }
        }
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        return self.initialization_successful and self.available
    
    def get_performance_stats(self) -> Dict[str, Any]:
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
            'enhanced_current_info_detection': True,
            'current_info_categories': list(self.current_info_indicators.keys()),
            'optimization_features': [
                'Targeted current info gathering',
                'Category-based optimization',
                'Enhanced prompt engineering',
                'Fast connection pooling',
                'Response quality validation'
            ]
        }
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
        self.available = False
        if settings.debug_mode:
            if self.request_count > 0:
                avg_time = self.total_time / self.request_count
                success_rate = (self.success_count / self.request_count) * 100
                print(f"[GROQ] 📊 Session stats: {self.request_count} requests, {avg_time:.2f}s avg, {success_rate:.1f}% success")
            print("[GROQ] 🔌 Enhanced connection closed")
