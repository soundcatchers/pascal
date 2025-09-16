"""
Pascal AI Assistant - Enhanced Skills Manager
Provides instant responses for datetime, calculator, weather, and news queries
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
    """Enhanced skills manager with instant local skills and API integrations"""
    
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
        
        # Skill patterns for detection
        self.skill_patterns = {
            'datetime': [
                r'\bwhat time is it\b',
                r'\bwhat day is today\b',
                r'\bwhat date is today\b',
                r'\bcurrent time\b',
                r'\bcurrent date\b',
                r'\btoday\'?s date\b',
                r'\bwhat is the date\b',
                r'\bwhat is the time\b',
                r'\btime now\b',
                r'\bdate now\b',
                r'\bwhat day\b',
                r'\bwhat time\b'
            ],
            'calculator': [
                r'\b\d+\s*[\+\-\*\/\%]\s*\d+',  # Basic math operations
                r'\b\d+\s*percent of\s*\d+',     # Percentage calculations
                r'\b\d+%\s*of\s*\d+',            # Alternative percentage format
                r'\bcalculate\s+\d+',             # Calculate requests
                r'\bwhat is\s+\d+[\+\-\*\/]\d+', # "What is" math
                r'\bsquare root of\s+\d+',        # Square root
                r'\b\d+\s*squared\b',             # Squared numbers
                r'\b\d+\s*to the power of\s*\d+' # Power calculations
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
        """Get environment variable with validation"""
        import os
        value = os.getenv(var_name)
        
        if not value or value.strip() in ['', 'your_api_key_here', 'your_openweather_api_key_here', 'your_news_api_key_here']:
            return None
            
        return value.strip()
    
    async def initialize(self) -> Dict[str, Dict[str, Any]]:
        """Initialize skills manager and test API connections"""
        if self.settings.debug_mode:
            print("🚀 Initializing Enhanced Skills Manager...")
        
        # Initialize HTTP session for API calls
        if AIOHTTP_AVAILABLE:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Test API connections
        await self._test_api_connections()
        
        return self.api_status
    
    async def _test_api_connections(self):
        """Test API connections and update status"""
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
                
                async with self.session.get(test_url, params=params) as response:
                    if response.status == 200:
                        self.api_status['weather'] = {
                            'available': True,
                            'message': 'Connected and working'
                        }
                        if self.settings.debug_mode:
                            print("[WEATHER] ✅ OpenWeatherMap API test successful")
                    else:
                        error_text = await response.text()
                        self.api_status['weather'] = {
                            'available': False,
                            'message': f'API error: {response.status}'
                        }
                        if self.settings.debug_mode:
                            print(f"[WEATHER] ❌ API test failed: {response.status}")
            except Exception as e:
                self.api_status['weather'] = {
                    'available': False,
                    'message': f'Connection error: {str(e)[:50]}'
                }
                if self.settings.debug_mode:
                    print(f"[WEATHER] ❌ API test error: {e}")
        
        # Test News API
        if self.news_api_key and self.session:
            try:
                if self.settings.debug_mode:
                    print(f"[NEWS] Making request to https://newsapi.org/v2/top-headlines with params: {{'apiKey': '{self.news_api_key}', 'pageSize': 5, 'page': 1, 'country': 'us'}}")
                
                test_url = "https://newsapi.org/v2/top-headlines"
                params = {
                    'apiKey': self.news_api_key,
                    'pageSize': 5,
                    'page': 1,
                    'country': 'us'
                }
                
                async with self.session.get(test_url, params=params) as response:
                    if self.settings.debug_mode:
                        print(f"[NEWS] Response status: {response.status}")
                        
                    if response.status == 200:
                        data = await response.json()
                        if self.settings.debug_mode:
                            response_preview = json.dumps(data)[:200] + "..."
                            print(f"[NEWS] Response preview: {response_preview}")
                        
                        if data and 'articles' in data and data['articles']:
                            articles = data['articles'][:3]  # Get first 3 articles
                            news_summary = "Top headlines from US:\n\n"
                            for i, article in enumerate(articles, 1):
                                title = article.get('title', 'No title')[:80]
                                news_summary += f"{i}. {title}\n"
                            
                            self.api_status['news'] = {
                                'available': True,
                                'message': 'Connected and working'
                            }
                            
                            if self.settings.debug_mode:
                                print(f"[SKILLS] NewsAPI test result: True, {news_summary[:100]}...")
                        else:
                            self.api_status['news'] = {
                                'available': False,
                                'message': 'API returned no articles'
                            }
                    else:
                        error_text = await response.text()
                        self.api_status['news'] = {
                            'available': False,
                            'message': f'API error: {response.status}'
                        }
                        if self.settings.debug_mode:
                            print(f"[NEWS] API test failed: {response.status}")
                            
            except Exception as e:
                error_msg = f"News request failed: '{str(e)}'. Check your internet connection."
                self.api_status['news'] = {
                    'available': False,
                    'message': error_msg
                }
                if self.settings.debug_mode:
                    print(f"[NEWS] Unexpected error: {e}")
                    print(f"[SKILLS] NewsAPI test result: False, {error_msg}")
        
        # Update status for APIs without keys
        if not self.weather_api_key:
            self.api_status['weather'] = {
                'available': False,
                'message': 'API key not configured'
            }
        
        if not self.news_api_key:
            self.api_status['news'] = {
                'available': False,
                'message': 'API key not configured'
            }
    
    def can_handle_directly(self, query: str) -> Optional[str]:
        """Check if any skill can handle this query directly"""
        query_lower = query.lower().strip()
        
        # Check each skill's patterns
        for skill_name, patterns in self.skill_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    if self.settings.debug_mode:
                        print(f"[SKILLS] Skill '{skill_name}' can handle query: '{pattern}' matched")
                    return skill_name
        
        return None
    
    async def execute_skill(self, query: str, skill_name: str) -> SkillResult:
        """Execute a specific skill"""
        start_time = time.time()
        
        try:
            # Update stats
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
        """Execute datetime skill - instant response"""
        now = datetime.now()
        query_lower = query.lower()
        
        if 'time' in query_lower:
            response = f"The current time is {now.strftime('%I:%M %p')}."
        elif 'day' in query_lower:
            response = f"Today is {now.strftime('%A')}."
        elif 'date' in query_lower:
            response = f"Today's date is {now.strftime('%A, %B %d, %Y')}."
        else:
            response = f"It's currently {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."
        
        return SkillResult(
            success=True,
            response=response,
            execution_time=0.001,  # Virtually instant
            skill_name='datetime',
            data={'timestamp': now.timestamp(), 'formatted_date': now.isoformat()}
        )
    
    async def _execute_calculator_skill(self, query: str) -> SkillResult:
        """Execute calculator skill - instant math responses"""
        query_lower = query.lower().strip()
        
        try:
            # Handle percentage calculations
            if 'percent of' in query_lower or '% of' in query_lower:
                # Extract numbers for percentage calculation
                if 'percent of' in query_lower:
                    parts = query_lower.split('percent of')
                else:
                    parts = query_lower.split('% of')
                
                if len(parts) == 2:
                    percent = float(re.search(r'\d+(?:\.\d+)?', parts[0]).group())
                    number = float(re.search(r'\d+(?:\.\d+)?', parts[1]).group())
                    result = (percent / 100) * number
                    
                    return SkillResult(
                        success=True,
                        response=f"{percent}% of {number} is {result}",
                        execution_time=0.001,
                        skill_name='calculator',
                        data={'operation': 'percentage', 'result': result}
                    )
            
            # Handle basic math operations
            math_match = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*\/\%])\s*(\d+(?:\.\d+)?)', query_lower)
            if math_match:
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
            
            # Handle square root
            sqrt_match = re.search(r'square root of\s+(\d+(?:\.\d+)?)', query_lower)
            if sqrt_match:
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
            
            # Handle squared
            squared_match = re.search(r'(\d+(?:\.\d+)?)\s*squared', query_lower)
            if squared_match:
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
            
            # Handle power calculations
            power_match = re.search(r'(\d+(?:\.\d+)?)\s*to the power of\s*(\d+(?:\.\d+)?)', query_lower)
            if power_match:
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
            
            # If no specific pattern matched, try to evaluate simple expressions
            # Clean the query to extract just the mathematical expression
            expr = re.sub(r'[^0-9\+\-\*\/\.\(\)\s]', '', query)
            if expr.strip():
                # Use eval carefully for simple expressions
                result = eval(expr)
                return SkillResult(
                    success=True,
                    response=f"{expr.strip()} = {result}",
                    execution_time=0.001,
                    skill_name='calculator',
                    data={'operation': 'expression', 'result': result}
                )
            
            return SkillResult(
                success=False,
                response="I couldn't understand the calculation. Try something like '15 + 23' or '20% of 150'",
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
        """Execute weather skill using OpenWeatherMap API"""
        if not self.api_status['weather']['available']:
            return SkillResult(
                success=False,
                response="Weather information is not available. Please configure the OpenWeatherMap API key in your .env file.",
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
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse weather data
                    temp = data['main']['temp']
                    description = data['weather'][0]['description']
                    humidity = data['main']['humidity']
                    location_name = data['name']
                    
                    response_text = f"Current weather in {location_name}: {description}, {temp}°C, humidity {humidity}%"
                    
                    return SkillResult(
                        success=True,
                        response=response_text,
                        execution_time=0.5,  # API call time
                        skill_name='weather',
                        data={
                            'location': location_name,
                            'temperature': temp,
                            'description': description,
                            'humidity': humidity
                        }
                    )
                else:
                    return SkillResult(
                        success=False,
                        response=f"Could not get weather information for {location}. Please check the location name.",
                        execution_time=0.5,
                        skill_name='weather'
                    )
                    
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[WEATHER] API Error: {e}")
            
            return SkillResult(
                success=False,
                response=f"Weather service is temporarily unavailable: {str(e)[:50]}",
                execution_time=0.5,
                skill_name='weather'
            )
    
    async def _execute_news_skill(self, query: str) -> SkillResult:
        """Execute news skill using News API"""
        if not self.api_status['news']['available']:
            return SkillResult(
                success=False,
                response="News information is not available. Please configure the News API key in your .env file.",
                execution_time=0.001,
                skill_name='news'
            )
        
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                'apiKey': self.news_api_key,
                'pageSize': 5,
                'country': 'us',  # Can be made configurable
                'category': 'general'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data['articles']:
                        articles = data['articles'][:3]  # Top 3 articles
                        
                        news_response = "Here are today's top news headlines:\n\n"
                        for i, article in enumerate(articles, 1):
                            title = article['title'][:80]  # Truncate long titles
                            source = article['source']['name']
                            news_response += f"{i}. {title} (via {source})\n"
                        
                        return SkillResult(
                            success=True,
                            response=news_response.strip(),
                            execution_time=0.7,  # API call time
                            skill_name='news',
                            data={
                                'articles_count': len(articles),
                                'articles': articles
                            }
                        )
                    else:
                        return SkillResult(
                            success=False,
                            response="No news articles found at the moment.",
                            execution_time=0.7,
                            skill_name='news'
                        )
                else:
                    return SkillResult(
                        success=False,
                        response="News service is temporarily unavailable.",
                        execution_time=0.7,
                        skill_name='news'
                    )
                    
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[NEWS] API Error: {e}")
            
            return SkillResult(
                success=False,
                response=f"News service is temporarily unavailable: {str(e)[:50]}",
                execution_time=0.7,
                skill_name='news'
            )
    
    def _extract_location_from_query(self, query: str) -> Optional[str]:
        """Extract location from weather query"""
        # Look for common patterns like "weather in London"
        location_match = re.search(r'(?:weather in|temperature in)\s+([a-zA-Z\s]+)', query.lower())
        if location_match:
            return location_match.group(1).strip().title()
        
        # Look for city names (simple list - could be expanded)
        cities = ['london', 'paris', 'new york', 'tokyo', 'berlin', 'madrid', 'rome', 'amsterdam']
        query_lower = query.lower()
        
        for city in cities:
            if city in query_lower:
                return city.title()
        
        return None
    
    def list_available_skills(self) -> List[Dict[str, Any]]:
        """Get list of available skills with their status"""
        skills = [
            {
                'name': 'datetime',
                'description': 'Current date and time information',
                'examples': ['What time is it?', 'What day is today?', 'Current date'],
                'speed': 'Instant (0.001s)',
                'confidence': 'Very High',
                'api_required': False,
                'api_configured': True
            },
            {
                'name': 'calculator',
                'description': 'Mathematical calculations',
                'examples': ['15 + 23', '20% of 150', 'square root of 64'],
                'speed': 'Instant (0.001s)', 
                'confidence': 'High',
                'api_required': False,
                'api_configured': True
            },
            {
                'name': 'weather',
                'description': 'Current weather information',
                'examples': ['Weather in London', 'Temperature today', 'Is it raining?'],
                'speed': 'Fast (0.5-2s)',
                'confidence': 'High',
                'api_required': True,
                'api_configured': self.api_status['weather']['available']
            },
            {
                'name': 'news',
                'description': 'Latest news headlines',
                'examples': ['Latest news', 'Today\'s headlines', 'Breaking news'],
                'speed': 'Fast (0.5-2s)',
                'confidence': 'High', 
                'api_required': True,
                'api_configured': self.api_status['news']['available']
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
            else:
                avg_time = 0.0
                success_rate = 0.0
            
            stats[skill_name] = {
                'executions': executions,
                'success_count': success_count,
                'avg_execution_time': f"{avg_time:.3f}s",
                'success_rate': f"{success_rate:.1f}%",
                'total_time_saved': f"{max(0, (executions * 2.0) - total_time):.1f}s"  # vs 2s LLM avg
            }
        
        return stats
    
    def get_skill_recommendations(self) -> List[Dict[str, str]]:
        """Get recommendations for improving skills"""
        recommendations = []
        
        if not self.api_status['weather']['available']:
            recommendations.append({
                'type': 'api_setup',
                'message': 'Configure OpenWeatherMap API key for weather queries (free at openweathermap.org)'
            })
        
        if not self.api_status['news']['available']:
            recommendations.append({
                'type': 'api_setup', 
                'message': 'Configure News API key for news queries (free at newsapi.org)'
            })
        
        # Performance recommendations
        total_skill_executions = sum(stats['executions'] for stats in self.skill_stats.values())
        if total_skill_executions > 10:
            datetime_executions = self.skill_stats['datetime']['executions']
            calc_executions = self.skill_stats['calculator']['executions']
            
            if datetime_executions > total_skill_executions * 0.5:
                recommendations.append({
                    'type': 'performance',
                    'message': 'Datetime queries are very common - consider caching time zone info'
                })
            
            if calc_executions > total_skill_executions * 0.3:
                recommendations.append({
                    'type': 'performance',
                    'message': 'Calculator queries are frequent - all math operations are optimized'
                })
        
        return recommendations
    
    async def close(self):
        """Close the skills manager and cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
        
        if self.settings.debug_mode:
            print("[SKILLS] Enhanced Skills Manager closed")
