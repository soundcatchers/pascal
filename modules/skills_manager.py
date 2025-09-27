"""
Pascal AI Assistant - FIXED Enhanced Skills Manager
Only handles simple instant queries, allows router to handle current info
"""

import asyncio
import time
import re
import json
import math
import operator
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

@dataclass
class SkillResult:
    """Result from skill execution"""
    success: bool
    response: str
    execution_time: float
    skill_name: str
    confidence: float = 1.0
    data: Dict[str, Any] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}

class EnhancedSkillsManager:
    """FIXED: Skills manager that only handles simple instant responses"""
    
    def __init__(self):
        from config.settings import settings
        self.settings = settings
        self.session = None
        
        # API Keys
        self.weather_api_key = self._get_env_var('OPENWEATHER_API_KEY')
        self.news_api_key = self._get_env_var('NEWS_API_KEY')
        
        # API Status tracking
        self.api_status = {
            'weather': {'available': False, 'message': 'Not configured'},
            'news': {'available': False, 'message': 'Not configured'}
        }
        
        # Performance tracking
        self.skill_stats = {
            'datetime': {'executions': 0, 'total_time': 0.0, 'success_count': 0},
            'calculator': {'executions': 0, 'total_time': 0.0, 'success_count': 0},
            'weather': {'executions': 0, 'total_time': 0.0, 'success_count': 0},
            'news': {'executions': 0, 'total_time': 0.0, 'success_count': 0}
        }
        
        # FIXED: Only simple, instant patterns - NO current info patterns
        self.skill_patterns = {
            'datetime': [
                # ONLY simple time queries - NOT comprehensive current info
                r'^what time is it\??$',
                r'^time\??$',
                r'^current time\??$',
                # REMOVED: date queries that should go to online for current info
            ],
            'calculator': [
                r'\b\d+\s*[\+\-\*\/\%]\s*\d+',
                r'\b\d+\s*percent of\s*\d+',
                r'\b\d+%\s*of\s*\d+',
                r'\bcalculate\s+\d+',
                r'\bwhat is\s+\d+[\+\-\*\/]\d+',
                r'\bsquare root of\s+\d+',
                r'\b\d+\s*squared\b',
                r'\b\d+\s*to the power of\s*\d+'
            ],
            'weather': [
                r'\bweather in\b',
                r'\bweather today\b',
                r'\bcurrent weather\b',
                r'\bweather now\b',
                r'\btemperature in\b',
                r'\bhow hot is it\b',
                r'\bhow cold is it\b',
                r'\bis it raining\b',
                r'\bweather forecast\b'
            ],
            'news': [
                r'\blatest news\b',
                r'\brecent news\b',
                r'\bnews today\b',
                r'\bbreaking news\b',
                r'\bcurrent events\b',
                r'\bwhat\'?s happening\b',
                r'\bin the news\b',
                r'\bnews headlines\b',
                r'\btoday\'?s news\b'
            ]
        }
    
    def _get_env_var(self, var_name: str) -> Optional[str]:
        """Better environment variable validation"""
        import os
        value = os.getenv(var_name)
        
        if not value:
            return None
        
        value = value.strip()
        
        # Check for placeholder values
        invalid_values = [
            '', 'your_api_key_here', 'your_openweather_api_key_here', 
            'your_news_api_key_here', 'your_weather_api_key_here'
        ]
        
        if value.lower() in [v.lower() for v in invalid_values]:
            return None
            
        return value
    
    async def initialize(self) -> Dict[str, Dict[str, Any]]:
        """Initialize skills manager"""
        if self.settings.debug_mode:
            print("🚀 Initializing FIXED Enhanced Skills Manager...")
        
        # Initialize HTTP session for API calls
        if AIOHTTP_AVAILABLE:
            try:
                timeout = aiohttp.ClientTimeout(total=15, connect=5, sock_read=10)
                connector = aiohttp.TCPConnector(
                    limit=2,
                    limit_per_host=1,
                    enable_cleanup_closed=True
                )
                self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            except Exception as e:
                if self.settings.debug_mode:
                    print(f"[SKILLS] ⚠️ HTTP session creation failed: {e}")
                self.session = None
        
        # Test API connections
        await self._test_api_connections()
        
        return self.api_status
    
    async def _test_api_connections(self):
        """Test API connections"""
        # Test OpenWeatherMap API
        if self.weather_api_key and self.session:
            try:
                if self.settings.debug_mode:
                    print(f"[WEATHER] Testing OpenWeatherMap API...")
                
                test_url = f"https://api.openweathermap.org/data/2.5/weather"
                params = {
                    'q': 'London',
                    'appid': self.weather_api_key,
                    'units': 'metric'
                }
                
                async with self.session.get(test_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'main' in data and 'temp' in data['main']:
                            self.api_status['weather'] = {
                                'available': True,
                                'message': 'Connected and working'
                            }
                            if self.settings.debug_mode:
                                print("[WEATHER] ✅ OpenWeatherMap API test successful")
                        else:
                            self.api_status['weather'] = {
                                'available': False,
                                'message': 'API returned invalid data'
                            }
                    elif response.status == 401:
                        self.api_status['weather'] = {
                            'available': False,
                            'message': 'Invalid API key'
                        }
                    else:
                        self.api_status['weather'] = {
                            'available': False,
                            'message': f'API error: {response.status}'
                        }
            except Exception as e:
                self.api_status['weather'] = {
                    'available': False,
                    'message': f'Connection error: {str(e)[:50]}'
                }
        else:
            if not self.weather_api_key:
                self.api_status['weather'] = {
                    'available': False,
                    'message': 'API key not configured'
                }
        
        # Test News API
        if self.news_api_key and self.session:
            try:
                if self.settings.debug_mode:
                    print(f"[NEWS] Testing NewsAPI...")
                
                test_url = "https://newsapi.org/v2/top-headlines"
                params = {
                    'apiKey': self.news_api_key,
                    'pageSize': 3,
                    'page': 1,
                    'country': 'us'
                }
                
                async with self.session.get(test_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and 'articles' in data and data['articles']:
                            self.api_status['news'] = {
                                'available': True,
                                'message': 'Connected and working'
                            }
                            if self.settings.debug_mode:
                                print(f"[NEWS] ✅ NewsAPI test successful")
                        else:
                            self.api_status['news'] = {
                                'available': False,
                                'message': 'API returned no articles'
                            }
                    elif response.status == 401:
                        self.api_status['news'] = {
                            'available': False,
                            'message': 'Invalid API key'
                        }
                    else:
                        self.api_status['news'] = {
                            'available': False,
                            'message': f'API error: {response.status}'
                        }
            except Exception as e:
                self.api_status['news'] = {
                    'available': False,
                    'message': f'Connection error: {str(e)[:50]}'
                }
        else:
            if not self.news_api_key:
                self.api_status['news'] = {
                    'available': False,
                    'message': 'API key not configured'
                }
    
    def can_handle_directly(self, query: str) -> Optional[str]:
        """FIXED: Only handle simple instant queries - NOT current info"""
        query_lower = query.lower().strip()
        
        # CRITICAL FIX: Only handle SIMPLE datetime queries
        # Exclude any query that might need comprehensive current information
        for pattern in self.skill_patterns['datetime']:
            if re.search(pattern, query_lower):
                # ADDITIONAL CHECK: Exclude if it's asking for comprehensive info
                if any(word in query_lower for word in ['day', 'date', 'today', 'what day']):
                    # These should go to online for comprehensive current info
                    if self.settings.debug_mode:
                        print(f"[SKILLS] Skipping datetime - comprehensive info needed: '{query}'")
                    continue
                
                if self.settings.debug_mode:
                    print(f"[SKILLS] Simple datetime skill can handle: '{pattern}' matched")
                return 'datetime'
        
        # Calculator queries (these are always instant and don't need current info)
        for pattern in self.skill_patterns['calculator']:
            if re.search(pattern, query_lower):
                if self.settings.debug_mode:
                    print(f"[SKILLS] Calculator skill can handle: '{pattern}' matched")
                return 'calculator'
        
        # Weather queries (if API available) - but these should mostly go to online
        if self.api_status['weather']['available']:
            for pattern in self.skill_patterns['weather']:
                if re.search(pattern, query_lower):
                    if self.settings.debug_mode:
                        print(f"[SKILLS] Weather skill can handle: '{pattern}' matched")
                    return 'weather'
        
        # News queries (if API available) - but these should mostly go to online
        if self.api_status['news']['available']:
            for pattern in self.skill_patterns['news']:
                if re.search(pattern, query_lower):
                    if self.settings.debug_mode:
                        print(f"[SKILLS] News skill can handle: '{pattern}' matched")
                    return 'news'
        
        return None
    
    async def execute_skill(self, query: str, skill_name: str) -> SkillResult:
        """Execute skill with better validation"""
        start_time = time.time()
        
        try:
            # Update stats
            if skill_name in self.skill_stats:
                self.skill_stats[skill_name]['executions'] += 1
            
            # Execute skill
            if skill_name == 'datetime':
                result = await self._execute_datetime_skill(query)
            elif skill_name == 'calculator':
                result = await self._execute_calculator_skill(query)
            elif skill_name == 'weather':
                result = await self._execute_weather_skill(query)
            elif skill_name == 'news':
                result = await self._execute_news_skill(query)
            else:
                result = SkillResult(
                    success=False,
                    response=f"Unknown skill: {skill_name}",
                    execution_time=time.time() - start_time,
                    skill_name=skill_name
                )
            
            # Update execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Update stats
            if skill_name in self.skill_stats:
                self.skill_stats[skill_name]['total_time'] += execution_time
                if result.success:
                    self.skill_stats[skill_name]['success_count'] += 1
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            if self.settings.debug_mode:
                print(f"[SKILLS] Error executing {skill_name}: {e}")
            
            return SkillResult(
                success=False,
                response=f"Error executing {skill_name}: {str(e)[:100]}",
                execution_time=execution_time,
                skill_name=skill_name
            )
    
    async def _execute_datetime_skill(self, query: str) -> SkillResult:
        """FIXED: Only handle simple time queries"""
        now = datetime.now()
        query_lower = query.lower()
        
        # ONLY handle time queries - NOT date/day queries (those should go to online)
        if any(word in query_lower for word in ['time', 'what time']):
            response = f"The current time is {now.strftime('%I:%M %p')}."
        else:
            # Should not reach here with the fixed can_handle_directly
            response = f"It's currently {now.strftime('%I:%M %p')}."
        
        return SkillResult(
            success=True,
            response=response,
            execution_time=0.001,
            skill_name='datetime',
            confidence=1.0,
            data={
                'timestamp': now.timestamp(), 
                'formatted_time': now.strftime('%H:%M:%S'),
                'time_12h': now.strftime('%I:%M %p')
            }
        )
    
    async def _execute_calculator_skill(self, query: str) -> SkillResult:
        """Enhanced calculator with better error handling"""
        query_lower = query.lower().strip()
        
        try:
            # Handle percentage calculations
            if 'percent of' in query_lower or '% of' in query_lower:
                if 'percent of' in query_lower:
                    parts = query_lower.split('percent of')
                else:
                    parts = query_lower.split('% of')
                
                if len(parts) == 2:
                    try:
                        percent_match = re.search(r'\d+(?:\.\d+)?', parts[0])
                        number_match = re.search(r'\d+(?:\.\d+)?', parts[1])
                        
                        if percent_match and number_match:
                            percent = float(percent_match.group())
                            number = float(number_match.group())
                            result = (percent / 100) * number
                            
                            return SkillResult(
                                success=True,
                                response=f"{percent}% of {number} is {result}",
                                execution_time=0.001,
                                skill_name='calculator',
                                data={'operation': 'percentage', 'result': result}
                            )
                    except ValueError:
                        pass
            
            # Handle basic math operations
            math_match = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*\/\%])\s*(\d+(?:\.\d+)?)', query_lower)
            if math_match:
                try:
                    num1 = float(math_match.group(1))
                    operation = math_match.group(2)
                    num2 = float(math_match.group(3))
                    
                    operations = {
                        '+': operator.add,
                        '-': operator.sub,
                        '*': operator.mul,
                        '/': operator.truediv,
                        '%': operator.mod
                    }
                    
                    if operation in operations:
                        if operation == '/' and num2 == 0:
                            return SkillResult(
                                success=False,
                                response="Cannot divide by zero",
                                execution_time=0.001,
                                skill_name='calculator'
                            )
                        
                        result = operations[operation](num1, num2)
                        
                        # Format result nicely
                        if result == int(result):
                            result = int(result)
                        
                        return SkillResult(
                            success=True,
                            response=f"{num1} {operation} {num2} = {result}",
                            execution_time=0.001,
                            skill_name='calculator',
                            data={'operation': operation, 'result': result, 'operands': [num1, num2]}
                        )
                except ValueError:
                    pass
            
            # Handle square root
            sqrt_match = re.search(r'square root of\s+(\d+(?:\.\d+)?)', query_lower)
            if sqrt_match:
                try:
                    number = float(sqrt_match.group(1))
                    if number < 0:
                        return SkillResult(
                            success=False,
                            response="Cannot calculate square root of negative number",
                            execution_time=0.001,
                            skill_name='calculator'
                        )
                    
                    result = math.sqrt(number)
                    if result == int(result):
                        result = int(result)
                    
                    return SkillResult(
                        success=True,
                        response=f"The square root of {number} is {result}",
                        execution_time=0.001,
                        skill_name='calculator',
                        data={'operation': 'sqrt', 'result': result}
                    )
                except ValueError:
                    pass
            
            # Handle squared
            squared_match = re.search(r'(\d+(?:\.\d+)?)\s*squared', query_lower)
            if squared_match:
                try:
                    number = float(squared_match.group(1))
                    result = number ** 2
                    if result == int(result):
                        result = int(result)
                    
                    return SkillResult(
                        success=True,
                        response=f"{number} squared is {result}",
                        execution_time=0.001,
                        skill_name='calculator',
                        data={'operation': 'square', 'result': result}
                    )
                except ValueError:
                    pass
            
            # Handle power calculations
            power_match = re.search(r'(\d+(?:\.\d+)?)\s*to the power of\s*(\d+(?:\.\d+)?)', query_lower)
            if power_match:
                try:
                    base = float(power_match.group(1))
                    exponent = float(power_match.group(2))
                    result = base ** exponent
                    if result == int(result):
                        result = int(result)
                    
                    return SkillResult(
                        success=True,
                        response=f"{base} to the power of {exponent} is {result}",
                        execution_time=0.001,
                        skill_name='calculator',
                        data={'operation': 'power', 'result': result}
                    )
                except ValueError:
                    pass
            
            return SkillResult(
                success=False,
                response="I couldn't understand that calculation. Try something like '15 + 23' or '20% of 150'",
                execution_time=0.001,
                skill_name='calculator'
            )
            
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[CALCULATOR] Error: {e}")
            
            return SkillResult(
                success=False,
                response="I had trouble with that calculation. Please try a simpler format.",
                execution_time=0.001,
                skill_name='calculator'
            )
    
    async def _execute_weather_skill(self, query: str) -> SkillResult:
        """Weather skill with improved error handling"""
        if not self.api_status['weather']['available']:
            return SkillResult(
                success=False,
                response="Weather information is not available. Please configure the OpenWeatherMap API key in your .env file. Get a free key at openweathermap.org/api",
                execution_time=0.001,
                skill_name='weather'
            )
        
        # Extract location from query
        location = self._extract_location_from_query(query) or "London"
        
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': location,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse weather data
                    temp = data['main']['temp']
                    feels_like = data['main'].get('feels_like', temp)
                    description = data['weather'][0]['description'].title()
                    humidity = data['main']['humidity']
                    location_name = data['name']
                    country = data['sys'].get('country', '')
                    
                    # Build comprehensive response
                    response_text = f"Current weather in {location_name}"
                    if country:
                        response_text += f", {country}"
                    response_text += f": {description}, {temp}°C"
                    
                    if abs(feels_like - temp) > 2:
                        response_text += f" (feels like {feels_like}°C)"
                    
                    response_text += f", humidity {humidity}%"
                    
                    return SkillResult(
                        success=True,
                        response=response_text,
                        execution_time=0.5,
                        skill_name='weather',
                        data={
                            'location': location_name,
                            'country': country,
                            'temperature': temp,
                            'feels_like': feels_like,
                            'description': description,
                            'humidity': humidity
                        }
                    )
                elif response.status == 404:
                    return SkillResult(
                        success=False,
                        response=f"Could not find weather information for '{location}'. Please check the location name.",
                        execution_time=0.5,
                        skill_name='weather'
                    )
                elif response.status == 401:
                    return SkillResult(
                        success=False,
                        response="Weather API authentication failed. Please check your OpenWeatherMap API key.",
                        execution_time=0.5,
                        skill_name='weather'
                    )
                else:
                    return SkillResult(
                        success=False,
                        response=f"Weather service error (HTTP {response.status}). Please try again later.",
                        execution_time=0.5,
                        skill_name='weather'
                    )
                    
        except asyncio.TimeoutError:
            return SkillResult(
                success=False,
                response="Weather service request timed out. Please try again.",
                execution_time=5.0,
                skill_name='weather'
            )
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[WEATHER] API Error: {e}")
            
            return SkillResult(
                success=False,
                response=f"Weather service is temporarily unavailable. Please try again later.",
                execution_time=0.5,
                skill_name='weather'
            )
    
    async def _execute_news_skill(self, query: str) -> SkillResult:
        """News skill with improved error handling"""
        if not self.api_status['news']['available']:
            return SkillResult(
                success=False,
                response="News information is not available. Please configure the News API key in your .env file. Get a free key at newsapi.org",
                execution_time=0.001,
                skill_name='news'
            )
        
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                'apiKey': self.news_api_key,
                'pageSize': 5,
                'country': 'us',
                'category': 'general'
            }
            
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('articles') and len(data['articles']) > 0:
                        articles = data['articles'][:3]  # Top 3 articles
                        
                        news_response = "Here are today's top news headlines:\n\n"
                        for i, article in enumerate(articles, 1):
                            title = article.get('title', 'No title')
                            source = article.get('source', {}).get('name', 'Unknown source')
                            
                            # Truncate very long titles
                            if len(title) > 80:
                                title = title[:77] + "..."
                            
                            news_response += f"{i}. {title} (via {source})\n"
                        
                        return SkillResult(
                            success=True,
                            response=news_response.strip(),
                            execution_time=0.7,
                            skill_name='news',
                            data={
                                'articles_count': len(articles),
                                'articles': articles
                            }
                        )
                    else:
                        return SkillResult(
                            success=False,
                            response="No news articles found at the moment. Please try again later.",
                            execution_time=0.7,
                            skill_name='news'
                        )
                elif response.status == 401:
                    return SkillResult(
                        success=False,
                        response="News API authentication failed. Please check your News API key.",
                        execution_time=0.7,
                        skill_name='news'
                    )
                elif response.status == 429:
                    return SkillResult(
                        success=False,
                        response="News API rate limit exceeded. Please try again later.",
                        execution_time=0.7,
                        skill_name='news'
                    )
                else:
                    return SkillResult(
                        success=False,
                        response=f"News service error (HTTP {response.status}). Please try again later.",
                        execution_time=0.7,
                        skill_name='news'
                    )
                    
        except asyncio.TimeoutError:
            return SkillResult(
                success=False,
                response="News service request timed out. Please try again.",
                execution_time=5.0,
                skill_name='news'
            )
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[NEWS] API Error: {e}")
            
            return SkillResult(
                success=False,
                response="News service is temporarily unavailable. Please try again later.",
                execution_time=0.7,
                skill_name='news'
            )
    
    def _extract_location_from_query(self, query: str) -> Optional[str]:
        """Extract location from weather query"""
        import re
        
        # Look for location patterns
        location_patterns = [
            r'\b(?:weather|temperature|forecast)\s+(?:in|for|at)\s+([A-Za-z][A-Za-z\s]{1,25}?)(?:\s|$|[,.?!])',
            r'\bin\s+([A-Za-z][A-Za-z\s]{1,25}?)(?:\s+(?:today|tomorrow|now))?\b'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip().title()
                # Remove common trailing words
                trailing_words = ['Today', 'Tomorrow', 'Now', 'Weather', 'Temperature']
                for word in trailing_words:
                    if location.endswith(f' {word}'):
                        location = location[:-len(f' {word}')]
                return location
        
        # Enhanced city detection
        major_cities = [
            'london', 'paris', 'new york', 'tokyo', 'berlin', 'madrid', 'rome',
            'amsterdam', 'chicago', 'los angeles', 'sydney', 'melbourne',
            'toronto', 'vancouver', 'dubai', 'singapore', 'hong kong', 'miami',
            'boston', 'seattle', 'san francisco', 'mumbai', 'delhi', 'bangkok'
        ]
        
        query_lower = query.lower()
        for city in major_cities:
            if re.search(r'\b' + re.escape(city) + r'\b', query_lower):
                return city.title()
        
        return None
    
    def list_available_skills(self) -> List[Dict[str, Any]]:
        """Get list of available skills with their current status"""
        skills = [
            {
                'name': 'datetime',
                'description': 'SIMPLE time queries only (not date/day - those go to online)',
                'examples': ['What time is it?', 'Time?', 'Current time'],
                'speed': 'Instant (0.001s)',
                'confidence': 'Very High',
                'api_required': False,
                'api_configured': True,
                'priority': 1,
                'note': 'FIXED: Only handles simple time, not comprehensive current info'
            },
            {
                'name': 'calculator',
                'description': 'Mathematical calculations and operations',
                'examples': ['15 + 23', '20% of 150', 'square root of 64'],
                'speed': 'Instant (0.001s)', 
                'confidence': 'High',
                'api_required': False,
                'api_configured': True,
                'priority': 2
            },
            {
                'name': 'weather',
                'description': 'Current weather information for any location',
                'examples': ['Weather in London', 'Temperature today', 'Is it raining?'],
                'speed': 'Fast (0.5-2s)',
                'confidence': 'High' if self.api_status['weather']['available'] else 'Unavailable',
                'api_required': True,
                'api_configured': self.api_status['weather']['available'],
                'priority': 3,
                'status_message': self.api_status['weather']['message'],
                'note': 'NOTE: Most weather queries should go to online for better context'
            },
            {
                'name': 'news',
                'description': 'Latest news headlines from reliable sources',
                'examples': ['Latest news', 'Today\'s headlines', 'Breaking news'],
                'speed': 'Fast (0.5-2s)',
                'confidence': 'High' if self.api_status['news']['available'] else 'Unavailable',
                'api_required': True,
                'api_configured': self.api_status['news']['available'],
                'priority': 4,
                'status_message': self.api_status['news']['message'],
                'note': 'NOTE: Most news queries should go to online for better context'
            }
        ]
        
        return skills
    
    def get_skill_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all skills"""
        stats = {}
        
        for skill_name, raw_stats in self.skill_stats.items():
            executions = raw_stats['executions']
            total_time = raw_stats['total_time']
            success_count = raw_stats['success_count']
            
            if executions > 0:
                avg_time = total_time / executions
                success_rate = (success_count / executions) * 100
                time_saved = max(0, (executions * 3.0) - total_time)  # vs 3s LLM avg
            else:
                avg_time = 0.0
                success_rate = 0.0
                time_saved = 0.0
            
            stats[skill_name] = {
                'executions': executions,
                'success_count': success_count,
                'avg_execution_time': f"{avg_time:.3f}s",
                'success_rate': f"{success_rate:.1f}%",
                'total_time_saved': f"{time_saved:.1f}s",
                'status': 'Available' if skill_name in ['datetime', 'calculator'] else 
                         ('Available' if self.api_status.get(skill_name, {}).get('available') else 'API not configured'),
                'fixed_note': 'FIXED: Only handles simple instant queries' if skill_name == 'datetime' else None
            }
        
        return stats
    
    def get_skill_recommendations(self) -> List[Dict[str, str]]:
        """Get recommendations for improving skills functionality"""
        recommendations = []
        
        recommendations.append({
            'type': 'fix_applied',
            'skill': 'datetime',
            'message': 'FIXED: Skills now only handle simple time queries',
            'action': 'Current date/day queries now route to online for comprehensive info',
            'priority': 'info'
        })
        
        if not self.api_status['weather']['available']:
            recommendations.append({
                'type': 'api_setup',
                'skill': 'weather',
                'message': 'Configure OpenWeatherMap API key for weather queries',
                'action': 'Get free API key at openweathermap.org/api (1000 calls/day free)',
                'priority': 'medium'
            })
        
        if not self.api_status['news']['available']:
            recommendations.append({
                'type': 'api_setup',
                'skill': 'news',
                'message': 'Configure News API key for news queries',
                'action': 'Get free API key at newsapi.org (100 requests/day free)',
                'priority': 'medium'
            })
        
        # System recommendations
        if not AIOHTTP_AVAILABLE:
            recommendations.append({
                'type': 'system',
                'skill': 'all_api_skills',
                'message': 'aiohttp not available - API-based skills will not work',
                'action': 'Install aiohttp: pip install aiohttp==3.9.5',
                'priority': 'critical'
            })
        
        return recommendations
    
    async def close(self):
        """Close the skills manager and cleanup resources"""
        if self.session:
            try:
                await self.session.close()
            except Exception:
                pass
            self.session = None
        
        if self.settings.debug_mode:
            # Show session summary
            total_executions = sum(stats['executions'] for stats in self.skill_stats.values())
            successful_executions = sum(stats['success_count'] for stats in self.skill_stats.values())
            
            if total_executions > 0:
                success_rate = (successful_executions / total_executions) * 100
                print(f"[SKILLS] 📊 FIXED Session summary: {successful_executions}/{total_executions} successful ({success_rate:.1f}%)")
                print(f"[SKILLS] 🔧 FIXED: Only simple instant queries handled, current info routed to online")
            
            print("[SKILLS] 🔌 FIXED Enhanced Skills Manager closed")

# Maintain compatibility
SkillsManager = EnhancedSkillsManager
