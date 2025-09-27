"""
Pascal AI Assistant - FIXED Online LLM Module
Resolves JSON parsing errors and improves Groq API reliability
"""

import asyncio
import json
import time
import re
from typing import Optional, AsyncGenerator, Dict, Any
from datetime import datetime, timezone

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from config.settings import settings

class OnlineLLM:
    """FIXED: Online LLM client with robust JSON parsing and error handling"""
    
    def __init__(self):
        self.session = None
        self.available = False
        self.last_error = None
        self.initialization_successful = False
        
        # Groq configuration
        self.api_key = settings.groq_api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_time = 0.0
        self.response_times = []
        
        # Current info cache
        self.current_info_cache = {}
        self.cache_timeout = 300
        
        # Enhanced current info patterns
        self.current_info_categories = {
            'datetime': {
                'patterns': [
                    r'\bwhat\s+(?:time|day|date)\s+(?:is\s+)?(?:it|today|now)\b',
                    r'\b(?:current|today\'?s?|now)\s+(?:time|date|day)\b',
                    r'\bwhat\s+day\s+(?:is\s+)?(?:it|today)\b',
                    r'\bwhat\s+(?:is\s+)?(?:the\s+)?(?:current\s+)?date\b'
                ],
                'confidence': 0.95
            },
            'weather': {
                'patterns': [
                    r'\b(?:weather|temperature|forecast|climate)\b',
                    r'\b(?:raining|snowing|sunny|cloudy|hot|cold)\b',
                    r'\bhow\s+(?:hot|cold|warm)\s+is\s+it\b',
                    r'\bweather\s+(?:forecast|conditions|update)\b'
                ],
                'confidence': 0.9
            },
            'news_events': {
                'patterns': [
                    r'\b(?:latest|recent|breaking|today\'?s?|current)\s+(?:news|headlines|events)\b',
                    r'\bwhat\'?s\s+(?:happening|going\s+on|new|in\s+the\s+news)\b',
                    r'\b(?:news|events)\s+(?:today|now|currently|recent|latest)\b',
                    r'\bbreaking\s+news\b',
                    r'\bcurrent\s+events\b',
                    r'\bin\s+the\s+news\b'
                ],
                'confidence': 0.95
            },
            'sports_results': {
                'patterns': [
                    r'\b(?:latest|recent|current|who\s+won)\s+(?:formula\s*1|f1|race|game|match|championship)\b',
                    r'\b(?:formula\s*1|f1)\s+(?:results|winner|standings|race|championship|latest)\b',
                    r'\bwho\s+(?:won|is\s+winning)\s+(?:the\s+)?(?:last|latest|recent|current|today\'?s?)\b',
                    r'\b(?:sports|game|match)\s+(?:results|scores|today|yesterday|recent|latest)\b'
                ],
                'confidence': 0.9
            },
            'politics': {
                'patterns': [
                    r'\b(?:current|who\s+is\s+(?:the\s+)?(?:current\s+)?)\s*(?:president|prime\s+minister|pm|leader)\b',
                    r'\bwho\s+(?:is\s+)?(?:the\s+)?(?:current\s+)?(?:us\s+)?president\b',
                    r'\b(?:election|political)\s+(?:results|news|updates|latest)\b'
                ],
                'confidence': 0.95
            }
        }
        
        # Temporal indicators
        self.strong_temporal_indicators = [
            'today', 'now', 'currently', 'right now', 'at the moment',
            'latest', 'recent', 'breaking', 'current', 'live', 'real-time'
        ]
        
        # Compile patterns
        self._compile_current_info_patterns()
    
    def _compile_current_info_patterns(self):
        """Compile regex patterns for performance"""
        self.compiled_patterns = {}
        
        for category, config in self.current_info_categories.items():
            self.compiled_patterns[category] = {
                'patterns': [re.compile(pattern, re.IGNORECASE) for pattern in config['patterns']],
                'confidence': config['confidence']
            }
    
    async def initialize(self) -> bool:
        """FIXED: Initialize Groq client with better error handling"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not installed - install with: pip install aiohttp"
            if settings.debug_mode:
                print("❌ [GROQ] aiohttp not available")
            return False
        
        if not self.api_key or not self._validate_api_key(self.api_key):
            self.last_error = "Invalid or missing Groq API key - get one free at console.groq.com"
            if settings.debug_mode:
                print("❌ [GROQ] Invalid/missing API key")
            return False
        
        try:
            # Create optimized session for speed
            timeout = aiohttp.ClientTimeout(total=25, connect=5, sock_read=20)
            connector = aiohttp.TCPConnector(
                limit=3,
                limit_per_host=2,
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
            
            # Test connection
            if await self._test_connection_fast():
                self.available = True
                self.initialization_successful = True
                if settings.debug_mode:
                    print("✅ [GROQ] Enhanced API initialized for current information")
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
        
        # Accept both gsk_ and gsk- formats
        if key.startswith('gsk_') or key.startswith('gsk-'):
            return len(key) > 20
        
        return False
    
    async def _test_connection_fast(self) -> bool:
        """FIXED: Quick connection test with robust JSON parsing"""
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
                    try:
                        response_text = await response.text()
                        data = self._safe_json_parse(response_text)
                        
                        if data and 'choices' in data and data['choices']:
                            if settings.debug_mode:
                                print("✅ [GROQ] Connection test successful")
                            return True
                        else:
                            if settings.debug_mode:
                                print("❌ [GROQ] Invalid response structure")
                            return False
                            
                    except Exception as e:
                        if settings.debug_mode:
                            print(f"❌ [GROQ] JSON parsing failed: {e}")
                        return False
                        
                elif response.status == 429:
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
    
    def _safe_json_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """FIXED: Safe JSON parsing with error handling"""
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            if settings.debug_mode:
                print(f"[GROQ] JSON parse error: {e}")
                print(f"[GROQ] Problematic text: {text[:200]}...")
            
            try:
                # Try to extract JSON from text
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                
                if start_idx >= 0 and end_idx >= 0 and end_idx > start_idx:
                    cleaned_text = text[start_idx:end_idx + 1]
                    return json.loads(cleaned_text)
                    
            except json.JSONDecodeError:
                pass
            
            return None
    
    def _safe_streaming_json_parse(self, line: str) -> Optional[Dict[str, Any]]:
        """FIXED: Safe JSON parsing for streaming responses"""
        if not line or not line.strip():
            return None
        
        line = line.strip()
        
        # Handle streaming format
        if line.startswith('data: '):
            line = line[6:].strip()
        
        # Skip [DONE] markers
        if line == '[DONE]':
            return {'done': True}
        
        return self._safe_json_parse(line)
    
    def detect_current_info_category(self, query: str) -> Optional[Dict[str, Any]]:
        """Enhanced current info category detection"""
        query_lower = query.lower().strip()
        
        # Check each category
        for category, config in self.compiled_patterns.items():
            for pattern in config['patterns']:
                if pattern.search(query_lower):
                    if settings.debug_mode:
                        print(f"[GROQ] 🎯 Current info detected: {category}")
                    return {
                        'category': category,
                        'confidence': config['confidence'],
                        'reason': f'{category}_pattern_match'
                    }
        
        # Check for temporal combinations
        has_strong_temporal = any(indicator in query_lower for indicator in self.strong_temporal_indicators)
        
        if has_strong_temporal:
            info_words = ['weather', 'news', 'president', 'results', 'scores', 'market', 'price']
            if any(word in query_lower for word in info_words):
                if settings.debug_mode:
                    print("[GROQ] 🎯 Current info detected: temporal+info combination")
                return {
                    'category': 'general_current',
                    'confidence': 0.8,
                    'reason': 'temporal_info_combination'
                }
        
        return None
    
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
        """Get current political information"""
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
    
    async def _gather_targeted_current_info(self, query: str) -> Dict[str, Any]:
        """Gather comprehensive current information"""
        current_info = {}
        
        # Always include datetime
        current_info['datetime'] = self._get_comprehensive_datetime_info()
        
        # Detect specific category
        category_info = self.detect_current_info_category(query)
        
        if category_info:
            category = category_info['category']
            
            if category in ['politics']:
                current_info['politics'] = self._get_current_political_info()
                
            elif category in ['weather']:
                location = self._extract_location_from_query(query) or "London"
                current_info['weather'] = {
                    'note': f'For real-time weather in {location}, I would need a weather API configured.',
                    'suggestion': 'For current weather conditions, I recommend checking weather.com, your local weather app, or asking a voice assistant.',
                    'location_requested': location
                }
                
            elif category in ['news_events']:
                current_info['news'] = {
                    'note': 'For the latest news headlines, I would need access to a news API.',
                    'suggestion': 'For current news, I recommend checking reliable sources like BBC News, Reuters, Associated Press, or NPR.',
                    'date': current_info['datetime']['current_date']
                }
                
            elif category in ['sports_results']:
                current_info['sports'] = {
                    'note': 'For current sports results and F1 information, I would need access to sports APIs.',
                    'suggestion': 'For latest sports results, check ESPN, BBC Sport, or official sport websites.',
                    'date': current_info['datetime']['current_date']
                }
        
        return current_info
    
    def _extract_location_from_query(self, query: str) -> Optional[str]:
        """Extract location from query"""
        import re
        
        # Look for location patterns
        location_patterns = [
            r'\b(?:weather|temperature|forecast)\s+(?:in|for|at)\s+([A-Za-z][A-Za-z\s]{1,25}?)(?:\s|$|[,.?!])',
            r'\bin\s+([A-Za-z][A-Za-z\s]{1,25}?)(?:\s+(?:today|tomorrow|now))?\b'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        # Common cities
        locations = [
            'london', 'paris', 'new york', 'tokyo', 'berlin', 'madrid', 'rome',
            'amsterdam', 'chicago', 'los angeles', 'sydney', 'melbourne'
        ]
        
        query_lower = query.lower()
        for location in locations:
            if location in query_lower:
                return location.title()
        
        return None
    
    def _build_enhanced_current_info_prompt(self, query: str, personality_context: str, 
                                          memory_context: str, current_info: Dict[str, Any]) -> list:
        """Build enhanced prompts with current information"""
        messages = []
        
        # Get datetime info
        datetime_info = current_info.get('datetime', {})
        
        # Enhanced system message
        system_content = f"""You are Pascal, a helpful AI assistant with access to current information.

CURRENT DATE & TIME:
Today is: {datetime_info.get('current_date', 'Unknown')}
Current time: {datetime_info.get('current_time', 'Unknown')}
Current day: {datetime_info.get('current_day', 'Unknown')}
Current year: {datetime_info.get('current_year', 'Unknown')}

INSTRUCTIONS:
- Use the EXACT information provided above for any date/time questions
- Be specific and helpful when providing current information
- If you don't have specific current data, acknowledge this and provide helpful alternatives
- Always be accurate about what information you have vs. what you don't have

{personality_context[:300] if personality_context else ''}"""

        # Add specific current information sections
        if 'politics' in current_info:
            politics_info = current_info['politics']
            system_content += f"""

CURRENT US POLITICAL INFORMATION (Accurate as of January 2025):
Current US President: {politics_info.get('us_president', 'Unknown')}
In office since: {politics_info.get('us_president_since', 'Unknown')}
Vice President: {politics_info.get('us_vice_president', 'Unknown')}
Context: {politics_info.get('context', '')}"""

        if 'weather' in current_info:
            weather_info = current_info['weather']
            system_content += f"""

WEATHER INFORMATION REQUEST:
Location requested: {weather_info.get('location_requested', 'Unknown')}
Note: {weather_info.get('note', '')}
Suggestion: {weather_info.get('suggestion', '')}"""

        if 'news' in current_info:
            news_info = current_info['news']
            system_content += f"""

NEWS INFORMATION REQUEST:
Note: {news_info.get('note', '')}
Suggestion: {news_info.get('suggestion', '')}"""

        if 'sports' in current_info:
            sports_info = current_info['sports']
            system_content += f"""

SPORTS INFORMATION REQUEST:
Note: {sports_info.get('note', '')}
Suggestion: {sports_info.get('suggestion', '')}"""

        messages.append({"role": "system", "content": system_content})
        
        # Add memory context if available
        if memory_context:
            messages.append({"role": "system", "content": f"Recent context: {memory_context[-200:]}"})
        
        # User query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    async def generate_response(self, query: str, personality_context: str, 
                               memory_context: str) -> str:
        """FIXED: Generate response with robust JSON parsing"""
        if not self.available:
            return "Online services are not available. Please check your Groq API key configuration and internet connection."
        
        # Detect current info category
        current_info_detection = self.detect_current_info_category(query)
        is_current_info = current_info_detection is not None
        
        if settings.debug_mode and current_info_detection:
            category = current_info_detection['category']
            confidence = current_info_detection['confidence']
            print(f"[GROQ] 🎯 Current info detected: {category} (confidence: {confidence:.2f})")
        
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
            
            # Optimize parameters
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": min(settings.max_response_tokens * 2, 500),
                "temperature": 0.1 if is_current_info else 0.3,
                "stream": False
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=20)
            ) as response:
                if response.status == 200:
                    try:
                        response_text = await response.text()
                        data = self._safe_json_parse(response_text)
                        
                        if data and 'choices' in data and data['choices']:
                            content = data['choices'][0]['message']['content']
                            
                            # Update performance stats
                            self.request_count += 1
                            self.success_count += 1
                            response_time = time.time() - start_time
                            self.total_time += response_time
                            self.response_times.append(response_time)
                            
                            if len(self.response_times) > 20:
                                self.response_times = self.response_times[-20:]
                            
                            if settings.debug_mode:
                                status = "⚡" if response_time < 4 else "✅" if response_time < 6 else "⚠️"
                                print(f"[GROQ] {status} Enhanced response in {response_time:.2f}s")
                            
                            return content.strip()
                        else:
                            self.failure_count += 1
                            return "I received an incomplete response from the online service."
                            
                    except Exception as json_error:
                        if settings.debug_mode:
                            print(f"[GROQ] JSON parsing error: {json_error}")
                        self.failure_count += 1
                        return "I had trouble parsing the response from the online service. Please try again."
                
                elif response.status == 429:
                    self.failure_count += 1
                    return "I'm being rate limited by the online service. Please try again in a moment."
                elif response.status in [401, 403]:
                    self.failure_count += 1
                    return "There's an authentication issue with the online service. Please check your Groq API key."
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
        """FIXED: Generate streaming response with robust JSON parsing"""
        if not self.available:
            yield "Online services are not available. Please check your Groq API key and internet connection."
            return
        
        # Detect current info category
        current_info_detection = self.detect_current_info_category(query)
        is_current_info = current_info_detection is not None
        
        if settings.debug_mode and current_info_detection:
            category = current_info_detection['category']
            confidence = current_info_detection['confidence']
            print(f"[GROQ] 🌊 Streaming enhanced info: {category} (confidence: {confidence:.2f})")
        
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
                "max_tokens": min(settings.max_response_tokens * 2, 500),
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
                    
                    try:
                        async for line in response.content:
                            if line:
                                line_str = line.decode('utf-8').strip()
                                if line_str:
                                    data = self._safe_streaming_json_parse(line_str)
                                    
                                    if data:
                                        if data.get('done'):
                                            break
                                        
                                        if 'choices' in data and data['choices']:
                                            delta = data['choices'][0].get('delta', {})
                                            if 'content' in delta and delta['content']:
                                                text_chunk = delta['content']
                                                if text_chunk:
                                                    yield text_chunk
                                                    response_received = True
                    
                    except Exception as stream_error:
                        if settings.debug_mode:
                            print(f"[GROQ] Streaming error: {stream_error}")
                        if not response_received:
                            yield "Error processing streaming response. Please try again."
                    
                    if response_received:
                        self.request_count += 1
                        self.success_count += 1
                        response_time = time.time() - start_time
                        self.total_time += response_time
                        self.response_times.append(response_time)
                        
                        if len(self.response_times) > 20:
                            self.response_times = self.response_times[-20:]
                            
                        if settings.debug_mode:
                            status = "⚡" if response_time < 5 else "✅" if response_time < 8 else "⚠️"
                            print(f"[GROQ] {status} Enhanced streaming complete in {response_time:.2f}s")
                    else:
                        self.failure_count += 1
                        yield "I didn't receive a proper response from the online service."
                        
                elif response.status == 429:
                    self.failure_count += 1
                    yield "I'm being rate limited. Please try again in a moment."
                elif response.status in [401, 403]:
                    self.failure_count += 1
                    yield "There's an authentication issue. Please check your Groq API key."
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
            'current_info_categories': list(self.current_info_categories.keys()),
            'json_parsing': 'robust_error_handling',
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
                    'optimization_level': 'enhanced_current_info_v2_fixed_json'
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
            'current_info_categories': list(self.current_info_categories.keys()),
            'json_parsing_fixed': True,
            'enhancements': [
                'FIXED: Robust JSON parsing with error handling',
                'FIXED: Safe streaming response processing',
                'FIXED: Python indentation and syntax errors',
                'Enhanced current info pattern matching',
                'Comprehensive datetime information',
                'Updated political information (2024-2025)',
                'Better weather/news/sports guidance',
                'Helpful alternative source suggestions',
                'More informative error messages',
                'Improved response quality validation'
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
                print(f"[GROQ] 📊 Enhanced session stats: {self.request_count} requests, {avg_time:.2f}s avg, {success_rate:.1f}% success")
            print("[GROQ] 🔌 Enhanced connection closed (JSON parsing and syntax fixed)")
