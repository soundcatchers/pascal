"""
Pascal AI Assistant - Current Information Module
Provides real current information for date, time, politics, weather, news
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from config.settings import settings

class CurrentInfoProvider:
    """Provides current information from various sources"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # API Keys (if configured)
        self.weather_api_key = None  # Could be added to settings
        self.news_api_key = None     # Could be added to settings
        
    async def initialize(self):
        """Initialize the current info provider"""
        if AIOHTTP_AVAILABLE:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self.session = aiohttp.ClientSession(timeout=timeout)
        
    async def close(self):
        """Close the provider"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get('timestamp', 0)
        return (datetime.now().timestamp() - cached_time) < self.cache_timeout
    
    def _cache_data(self, cache_key: str, data: Dict[str, Any]):
        """Cache data with timestamp"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now().timestamp()
        }
    
    def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if valid"""
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        return None
    
    async def get_datetime_info(self) -> Dict[str, Any]:
        """Get comprehensive current date/time information"""
        cache_key = 'datetime'
        
        # Check cache first (but with very short timeout for datetime)
        if cache_key in self.cache:
            cached_time = self.cache[cache_key].get('timestamp', 0)
            if (datetime.now().timestamp() - cached_time) < 60:  # 1 minute cache
                return self.cache[cache_key]['data']
        
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        
        info = {
            'current_date': now.strftime("%A, %B %d, %Y"),
            'current_time': now.strftime("%I:%M %P"),
            'current_day': now.strftime("%A"),
            'current_day_short': now.strftime("%a"),
            'current_day_number': now.day,
            'current_month': now.strftime("%B"),
            'current_month_short': now.strftime("%b"),
            'current_year': now.year,
            'current_hour': now.hour,
            'current_minute': now.minute,
            'utc_time': utc_now.strftime("%H:%M UTC"),
            'timestamp': now.timestamp(),
            'iso_date': now.isoformat(),
            'day_of_week': now.weekday() + 1,  # 1=Monday, 7=Sunday
            'day_of_year': now.timetuple().tm_yday,
            'week_number': now.isocalendar()[1],
            'quarter': (now.month - 1) // 3 + 1,
            'is_weekend': now.weekday() >= 5,
            'timezone': str(now.astimezone().tzinfo),
            'formatted_full': now.strftime("%A, %B %d, %Y at %I:%M %P"),
            'formatted_short': now.strftime("%a %b %d, %Y"),
            'formatted_time_12': now.strftime("%I:%M %P"),
            'formatted_time_24': now.strftime("%H:%M"),
        }
        
        self._cache_data(cache_key, info)
        return info
    
    async def get_political_info(self, country: str = "US") -> Dict[str, Any]:
        """Get current political information"""
        cache_key = f'politics_{country.lower()}'
        
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        if country.upper() == "US":
            # Updated for 2024 election results
            info = {
                'country': 'United States',
                'current_president': 'Donald Trump',
                'current_president_full': 'Donald J. Trump',
                'president_since': 'January 20, 2025',
                'inauguration_date': 'January 20, 2025',
                'current_vice_president': 'JD Vance',
                'current_party': 'Republican',
                'previous_president': 'Joe Biden',
                'previous_president_term': '2021-2025',
                'previous_party': 'Democratic',
                'election_year': 2024,
                'next_election': 2028,
                'term_number': '47th President',
                'note': 'Donald Trump won the 2024 presidential election, defeating Kamala Harris',
                'context': 'Trump previously served as the 45th President (2017-2021) and is now the 47th President',
                'transition': 'Inaugurated on January 20, 2025, after winning the November 2024 election'
            }
        elif country.upper() == "UK":
            # UK political info (as of knowledge cutoff - could be updated with real API)
            info = {
                'country': 'United Kingdom',
                'current_prime_minister': 'Check current UK government website',
                'note': 'For current UK political information, please check gov.uk or current news sources',
                'suggestion': 'UK political leadership can change frequently - recommend checking current sources'
            }
        else:
            info = {
                'country': country,
                'note': f'Political information for {country} not available',
                'suggestion': 'For current political information, please check reliable news sources'
            }
        
        self._cache_data(cache_key, info)
        return info
    
    async def get_weather_info(self, location: str = "London") -> Dict[str, Any]:
        """Get current weather information"""
        cache_key = f'weather_{location.lower()}'
        
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        # For now, return placeholder info
        # In a real implementation, this would call a weather API like OpenWeatherMap
        info = {
            'location': location,
            'status': 'API integration needed',
            'note': 'Real-time weather requires weather API integration',
            'suggestion': 'Configure weather API key in settings to get live weather data',
            'available_apis': [
                'OpenWeatherMap (free tier available)',
                'WeatherAPI (free tier available)', 
                'AccuWeather API'
            ],
            'implementation_note': 'Weather API can be easily added to this module'
        }
        
        # TODO: Implement real weather API call
        # if self.weather_api_key and self.session:
        #     try:
        #         weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.weather_api_key}&units=metric"
        #         async with self.session.get(weather_url) as response:
        #             if response.status == 200:
        #                 data = await response.json()
        #                 info = self._parse_weather_data(data)
        #     except Exception as e:
        #         if settings.debug_mode:
        #             print(f"Weather API error: {e}")
        
        self._cache_data(cache_key, info)
        return info
    
    async def get_news_info(self, category: str = "general") -> Dict[str, Any]:
        """Get current news information"""
        cache_key = f'news_{category}'
        
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        now = datetime.now()
        
        # For now, return placeholder info
        # In a real implementation, this would call news APIs
        info = {
            'category': category,
            'status': 'API integration needed',
            'date': now.strftime("%A, %B %d, %Y"),
            'note': 'Real-time news requires news API integration',
            'suggestion': 'For latest news, check reliable sources like BBC, Reuters, Associated Press',
            'available_apis': [
                'NewsAPI (free tier available)',
                'Guardian API',
                'Reuters API',
                'Associated Press API'
            ],
            'recommended_sources': [
                'BBC News',
                'Reuters',
                'Associated Press',
                'NPR',
                'Your local news sources'
            ],
            'implementation_note': 'News API can be easily added to this module'
        }
        
        # TODO: Implement real news API call
        # if self.news_api_key and self.session:
        #     try:
        #         news_url = f"https://newsapi.org/v2/top-headlines?category={category}&apiKey={self.news_api_key}"
        #         async with self.session.get(news_url) as response:
        #             if response.status == 200:
        #                 data = await response.json()
        #                 info = self._parse_news_data(data)
        #     except Exception as e:
        #         if settings.debug_mode:
        #             print(f"News API error: {e}")
        
        self._cache_data(cache_key, info)
        return info
    
    async def get_market_info(self) -> Dict[str, Any]:
        """Get current market/financial information"""
        cache_key = 'markets'
        
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        # Placeholder for market data
        info = {
            'status': 'API integration needed',
            'note': 'Real-time market data requires financial API integration',
            'suggestion': 'For current market data, check financial news sources',
            'available_apis': [
                'Alpha Vantage (free tier)',
                'Yahoo Finance API',
                'IEX Cloud',
                'Finnhub'
            ],
            'implementation_note': 'Financial APIs can be easily added'
        }
        
        self._cache_data(cache_key, info)
        return info
    
    async def get_comprehensive_current_info(self, query: str) -> Dict[str, Any]:
        """Get comprehensive current information based on query"""
        query_lower = query.lower()
        info = {}
        
        # Always include datetime for current info queries
        info['datetime'] = await self.get_datetime_info()
        
        # Add specific information based on query content
        if any(term in query_lower for term in ['president', 'politics', 'election', 'government', 'leader']):
            info['politics'] = await self.get_political_info("US")
        
        if any(term in query_lower for term in ['weather', 'temperature', 'forecast', 'rain', 'snow', 'climate']):
            location = self._extract_location_from_query(query) or "London"
            info['weather'] = await self.get_weather_info(location)
        
        if any(term in query_lower for term in ['news', 'happening', 'events', 'breaking', 'latest']):
            info['news'] = await self.get_news_info()
        
        if any(term in query_lower for term in ['market', 'stock', 'economy', 'financial', 'trading']):
            info['markets'] = await self.get_market_info()
        
        return info
    
    def _extract_location_from_query(self, query: str) -> Optional[str]:
        """Extract location from weather query (simple implementation)"""
        # Simple location extraction - could be enhanced with NLP
        common_cities = [
            'london', 'paris', 'new york', 'tokyo', 'berlin', 'madrid', 'rome',
            'amsterdam', 'chicago', 'los angeles', 'sydney', 'melbourne',
            'toronto', 'vancouver', 'dubai', 'singapore', 'hong kong'
        ]
        
        query_lower = query.lower()
        for city in common_cities:
            if city in query_lower:
                return city.title()
        
        return None
    
    def format_datetime_response(self, datetime_info: Dict[str, Any], query: str) -> str:
        """Format datetime information for response"""
        query_lower = query.lower()
        
        if 'what day' in query_lower:
            return f"Today is {datetime_info['current_day']}."
        elif 'what date' in query_lower or 'today\\'s date' in query_lower:
            return f"Today's date is {datetime_info['current_date']}."
        elif 'what time' in query_lower:
            return f"The current time is {datetime_info['current_time']}."
        else:
            return f"Today is {datetime_info['current_date']} and the time is {datetime_info['current_time']}."
    
    def format_political_response(self, political_info: Dict[str, Any], query: str) -> str:
        """Format political information for response"""
        if 'current_president' in political_info:
            president = political_info['current_president']
            since = political_info.get('president_since', '')
            context = political_info.get('context', '')
            
            response = f"The current President of the United States is {president}"
            if since:
                response += f", who has been in office since {since}"
            if context:
                response += f". {context}"
            response += "."
            return response
        else:
            return political_info.get('note', 'Political information not available.')

# Global instance
current_info_provider = CurrentInfoProvider()
