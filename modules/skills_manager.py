"""
Pascal AI Assistant - Enhanced Skills Manager with Full API Integration
Handles direct API skills for weather, news, time, calculations with real API calls
"""

import asyncio
import json
import time
import re
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from config.settings import settings

class SkillResult:
    """Result from a skill execution"""
    def __init__(self, success: bool, response: str, data: Dict[str, Any] = None, 
                 execution_time: float = 0.0, skill_used: str = "", confidence: float = 1.0):
        self.success = success
        self.response = response
        self.data = data or {}
        self.execution_time = execution_time
        self.skill_used = skill_used
        self.confidence = confidence
        self.timestamp = time.time()

class DateTimeSkill:
    """Enhanced date/time skill with timezone and calendar features"""
    
    @staticmethod
    def can_handle(query: str) -> bool:
        """Check if this skill can handle the query"""
        patterns = [
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
            r'\bwhat day of the week\b',
            r'\bwhat month is it\b',
            r'\bwhat year is it\b',
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in patterns)
    
    @staticmethod
    def execute(query: str) -> SkillResult:
        """Execute datetime query with enhanced features"""
        start_time = time.time()
        
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        query_lower = query.lower()
        
        # Determine response type based on query
        if 'time' in query_lower:
            response = f"The current time is {now.strftime('%I:%M %p')}."
            data_key = 'time'
        elif 'day' in query_lower and 'week' in query_lower:
            response = f"Today is {now.strftime('%A')}, day {now.weekday() + 1} of the week."
            data_key = 'day_of_week'
        elif 'day' in query_lower:
            response = f"Today is {now.strftime('%A')}."
            data_key = 'day'
        elif 'month' in query_lower:
            response = f"The current month is {now.strftime('%B')}."
            data_key = 'month'
        elif 'year' in query_lower:
            response = f"The current year is {now.year}."
            data_key = 'year'
        elif 'date' in query_lower:
            response = f"Today's date is {now.strftime('%A, %B %d, %Y')}."
            data_key = 'full_date'
        else:
            # Default comprehensive response
            response = f"Today is {now.strftime('%A, %B %d, %Y')} and the current time is {now.strftime('%I:%M %p')}."
            data_key = 'datetime'
        
        # Enhanced data structure
        data = {
            'current_datetime': now.isoformat(),
            'utc_datetime': utc_now.isoformat(),
            'formatted_date': now.strftime('%A, %B %d, %Y'),
            'formatted_time': now.strftime('%I:%M %p'),
            'day_name': now.strftime('%A'),
            'month_name': now.strftime('%B'),
            'year': now.year,
            'day_of_month': now.day,
            'day_of_week': now.weekday() + 1,  # Monday = 1
            'day_of_year': now.timetuple().tm_yday,
            'week_number': now.isocalendar()[1],
            'quarter': (now.month - 1) // 3 + 1,
            'is_weekend': now.weekday() >= 5,
            'timestamp': now.timestamp(),
            'timezone': str(now.astimezone().tzinfo),
            'query_type': data_key
        }
        
        execution_time = time.time() - start_time
        
        return SkillResult(
            success=True,
            response=response,
            data=data,
            execution_time=execution_time,
            skill_used='datetime',
            confidence=1.0
        )

class CalculatorSkill:
    """Enhanced calculation skill with more operations"""
    
    @staticmethod
    def can_handle(query: str) -> bool:
        """Check if this is a calculation query"""
        patterns = [
            r'\b\d+\s*[\+\-\*\/]\s*\d+',  # Basic math: 5+3, 10*2
            r'\bwhat is \d+',              # What is 15% of...
            r'\bcalculate',                # Calculate...
            r'\d+\s*percent of',           # 15 percent of
            r'\d+%\s*of',                  # 15% of
            r'\bsquare root of\b',         # Square root
            r'\bpower of\b',               # Power calculations
            r'\btip\b.*\bdollars?\b',      # Tip calculations
            r'\bconvert\b.*\bto\b',        # Unit conversions
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in patterns)
    
    @staticmethod
    def execute(query: str) -> SkillResult:
        """Execute enhanced calculation"""
        start_time = time.time()
        
        try:
            query_lower = query.lower()
            result = None
            operation_type = "unknown"
            
            # Handle percentage calculations
            percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:percent\s+)?of\s+(\d+(?:\.\d+)?)', query_lower)
            if percent_match:
                percent, number = float(percent_match.group(1)), float(percent_match.group(2))
                result = (percent / 100) * number
                response = f"{percent}% of {number} is {result:g}"
                operation_type = "percentage"
            
            # Handle tip calculations
            elif 'tip' in query_lower:
                tip_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*tip.*?(\d+(?:\.\d+)?)', query_lower)
                if tip_match:
                    tip_percent, bill_amount = float(tip_match.group(1)), float(tip_match.group(2))
                    if tip_percent <= 1:  # Handle decimal format
                        tip_percent *= 100
                    tip_amount = (tip_percent / 100) * bill_amount
                    total = bill_amount + tip_amount
                    response = f"A {tip_percent}% tip on ${bill_amount:.2f} is ${tip_amount:.2f}. Total: ${total:.2f}"
                    result = total
                    operation_type = "tip"
                else:
                    response = "Please specify tip percentage and bill amount (e.g., '20% tip on $50')"
                    result = None
            
            # Handle square root
            elif 'square root' in query_lower:
                sqrt_match = re.search(r'square root of\s+(\d+(?:\.\d+)?)', query_lower)
                if sqrt_match:
                    number = float(sqrt_match.group(1))
                    result = number ** 0.5
                    response = f"The square root of {number} is {result:.6g}"
                    operation_type = "square_root"
            
            # Handle power calculations
            elif 'power' in query_lower or '**' in query or '^' in query:
                power_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:\*\*|\^|to the power of|power of)\s*(\d+(?:\.\d+)?)', query_lower)
                if power_match:
                    base, exponent = float(power_match.group(1)), float(power_match.group(2))
                    result = base ** exponent
                    response = f"{base} to the power of {exponent} is {result:g}"
                    operation_type = "power"
            
            # Handle basic math expressions
            elif re.search(r'\d+\s*[\+\-\*\/]\s*\d+', query):
                # Extract the math expression - enhanced version
                math_expr = re.search(r'(\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)*)', query).group(1)
                
                # Safe evaluation - only allow basic operations
                allowed_chars = set('0123456789+-*/.() ')
                if all(c in allowed_chars for c in math_expr):
                    result = eval(math_expr)  # Safe for controlled input
                    response = f"{math_expr} = {result:g}"
                    operation_type = "arithmetic"
                else:
                    response = "I can only perform basic arithmetic operations (+, -, *, /)."
                    result = None
            
            # Handle "what is X" queries
            elif 'what is' in query_lower and any(op in query for op in ['+', '-', '*', '/']):
                math_part = query.split('what is')[1].strip()
                if re.match(r'^\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?$', math_part):
                    result = eval(math_part)
                    response = f"{math_part} = {result:g}"
                    operation_type = "arithmetic"
                else:
                    response = "I can help with basic math. Try something like 'what is 15 + 23'."
                    result = None
            
            # Unit conversions (basic)
            elif 'convert' in query_lower:
                # Temperature conversions
                if 'celsius' in query_lower and 'fahrenheit' in query_lower:
                    temp_match = re.search(r'(\d+(?:\.\d+)?)', query)
                    if temp_match:
                        temp = float(temp_match.group(1))
                        if 'celsius' in query_lower[:query_lower.find(str(temp)) + 10]:
                            result = (temp * 9/5) + 32
                            response = f"{temp}°C = {result:.1f}°F"
                        else:
                            result = (temp - 32) * 5/9
                            response = f"{temp}°F = {result:.1f}°C"
                        operation_type = "temperature_conversion"
                    else:
                        response = "Please specify the temperature to convert."
                        result = None
                else:
                    response = "I currently support temperature conversions (Celsius/Fahrenheit). More conversions coming soon!"
                    result = None
            
            else:
                response = "I can help with calculations like: '15 + 23', '20% of 150', 'square root of 144', or '20% tip on $45'."
                result = None
            
            execution_time = time.time() - start_time
            
            # Build data structure
            data = {
                'result': result,
                'operation_type': operation_type,
                'original_query': query,
                'calculation_successful': result is not None
            }
            
            return SkillResult(
                success=result is not None,
                response=response,
                data=data,
                execution_time=execution_time,
                skill_used='calculator',
                confidence=0.0
            )

class EnhancedSkillsManager:
    """Enhanced skills manager with full API integration and intelligent routing"""
    
    def __init__(self):
        self.datetime_skill = DateTimeSkill()
        self.calculator_skill = CalculatorSkill()
        self.weather_skill = WeatherSkill()
        self.news_skill = NewsSkill()
        
        # Skill order - fastest skills first, then by reliability
        self.skills = [
            (self.datetime_skill, 'datetime'),
            (self.calculator_skill, 'calculator'),
            (self.weather_skill, 'weather'),
            (self.news_skill, 'news')
        ]
        
        # Performance tracking with enhanced metrics
        self.skill_stats = {}
        self.query_history = []
        self.confidence_threshold = 0.7
        
        # Skill confidence patterns - learned over time
        self.skill_patterns = {
            'datetime': {'high_confidence_patterns': [], 'low_confidence_patterns': []},
            'calculator': {'high_confidence_patterns': [], 'low_confidence_patterns': []},
            'weather': {'high_confidence_patterns': [], 'low_confidence_patterns': []},
            'news': {'high_confidence_patterns': [], 'low_confidence_patterns': []}
        }
    
    async def initialize(self):
        """Initialize all skills with enhanced setup"""
        print("🚀 Initializing Enhanced Skills Manager...")
        
        # Initialize async skills
        await self.weather_skill.initialize()
        await self.news_skill.initialize()
        
        # Test API connections
        api_status = await self._test_api_connections()
        
        print("📊 API Status:")
        for service, status in api_status.items():
            status_icon = "✅" if status['available'] else "❌"
            print(f"   {status_icon} {service}: {status['message']}")
        
        return api_status
    
    async def _test_api_connections(self) -> Dict[str, Dict[str, Any]]:
        """Test all API connections"""
        status = {}
        
        # Test OpenWeatherMap
        if self.weather_skill.api_key:
            try:
                # Quick test with London
                test_result = await self.weather_skill.execute("weather in London")
                status['OpenWeatherMap'] = {
                    'available': test_result.success,
                    'message': 'Connected and working' if test_result.success else 'API key invalid or connection failed'
                }
            except Exception as e:
                status['OpenWeatherMap'] = {
                    'available': False,
                    'message': f'Connection error: {str(e)[:50]}'
                }
        else:
            status['OpenWeatherMap'] = {
                'available': False,
                'message': 'API key not configured (add OPENWEATHER_API_KEY to .env)'
            }
        
        # Test NewsAPI
        if self.news_skill.api_key:
            try:
                test_result = await self.news_skill.execute("latest news")
                status['NewsAPI'] = {
                    'available': test_result.success,
                    'message': 'Connected and working' if test_result.success else 'API key invalid or connection failed'
                }
            except Exception as e:
                status['NewsAPI'] = {
                    'available': False,
                    'message': f'Connection error: {str(e)[:50]}'
                }
        else:
            status['NewsAPI'] = {
                'available': False,
                'message': 'API key not configured (add NEWS_API_KEY to .env)'
            }
        
        return status
    
    async def close(self):
        """Close all skills"""
        await self.weather_skill.close()
        await self.news_skill.close()
    
    def _calculate_skill_confidence(self, query: str, skill_name: str) -> float:
        """Calculate confidence score for a skill handling a query"""
        base_confidence = 0.5
        
        # Check if skill can handle the query
        skill_map = {
            'datetime': self.datetime_skill,
            'calculator': self.calculator_skill,
            'weather': self.weather_skill,
            'news': self.news_skill
        }
        
        skill = skill_map.get(skill_name)
        if not skill or not skill.can_handle(query):
            return 0.0
        
        # Increase confidence based on pattern matching
        query_lower = query.lower()
        patterns = self.skill_patterns.get(skill_name, {})
        
        # Check high confidence patterns
        for pattern in patterns.get('high_confidence_patterns', []):
            if pattern in query_lower:
                base_confidence += 0.2
        
        # Check low confidence patterns
        for pattern in patterns.get('low_confidence_patterns', []):
            if pattern in query_lower:
                base_confidence -= 0.1
        
        # Adjust based on historical performance
        if skill_name in self.skill_stats:
            stats = self.skill_stats[skill_name]
            success_rate = stats['success_count'] / max(stats['count'], 1)
            base_confidence += (success_rate - 0.5) * 0.3
        
        return max(0.0, min(1.0, base_confidence))
    
    def can_handle_directly(self, query: str) -> Optional[str]:
        """Enhanced skill detection with confidence scoring"""
        best_skill = None
        best_confidence = 0.0
        
        for skill, skill_name in self.skills:
            if skill.can_handle(query):
                confidence = self._calculate_skill_confidence(query, skill_name)
                if confidence > best_confidence and confidence >= self.confidence_threshold:
                    best_confidence = confidence
                    best_skill = skill_name
        
        if settings.debug_mode and best_skill:
            print(f"[SKILLS] Best match: {best_skill} (confidence: {best_confidence:.2f})")
        
        return best_skill
    
    async def execute_skill(self, query: str, skill_name: str = None) -> Optional[SkillResult]:
        """Execute skill with enhanced error handling and learning"""
        start_time = time.time()
        
        if skill_name:
            # Execute specific skill
            skill_map = {
                'datetime': self.datetime_skill,
                'calculator': self.calculator_skill,
                'weather': self.weather_skill,
                'news': self.news_skill
            }
            
            if skill_name in skill_map:
                skill = skill_map[skill_name]
                
                try:
                    if hasattr(skill, 'execute'):
                        if asyncio.iscoroutinefunction(skill.execute):
                            result = await skill.execute(query)
                        else:
                            result = skill.execute(query)
                        
                        # Update stats and learning
                        self._update_skill_stats(skill_name, result, query)
                        self._learn_from_execution(skill_name, query, result)
                        
                        return result
                except Exception as e:
                    error_result = SkillResult(
                        success=False,
                        response=f"Skill execution error: {str(e)}",
                        data={'error': str(e), 'skill': skill_name},
                        execution_time=time.time() - start_time,
                        skill_used=skill_name,
                        confidence=0.0
                    )
                    self._update_skill_stats(skill_name, error_result, query)
                    return error_result
        
        else:
            # Auto-detect and execute best skill
            best_skills = []
            
            # Score all skills
            for skill, skill_name in self.skills:
                if skill.can_handle(query):
                    confidence = self._calculate_skill_confidence(query, skill_name)
                    if confidence >= self.confidence_threshold:
                        best_skills.append((skill, skill_name, confidence))
            
            # Sort by confidence
            best_skills.sort(key=lambda x: x[2], reverse=True)
            
            # Try skills in order of confidence
            for skill, skill_name, confidence in best_skills:
                try:
                    if hasattr(skill, 'execute'):
                        if asyncio.iscoroutinefunction(skill.execute):
                            result = await skill.execute(query)
                        else:
                            result = skill.execute(query)
                        
                        # Update stats and learning
                        self._update_skill_stats(skill_name, result, query)
                        self._learn_from_execution(skill_name, query, result)
                        
                        # If successful, return result
                        if result.success:
                            return result
                        
                        # If not successful but confidence was high, still return
                        if confidence > 0.8:
                            return result
                            
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[SKILLS] {skill_name} execution failed: {e}")
                    continue
        
        return None
    
    def _update_skill_stats(self, skill_name: str, result: SkillResult, query: str):
        """Update skill performance statistics"""
        if skill_name not in self.skill_stats:
            self.skill_stats[skill_name] = {
                'count': 0, 
                'total_time': 0.0, 
                'success_count': 0,
                'avg_confidence': 0.0,
                'total_confidence': 0.0
            }
        
        stats = self.skill_stats[skill_name]
        stats['count'] += 1
        stats['total_time'] += result.execution_time
        stats['total_confidence'] += result.confidence
        stats['avg_confidence'] = stats['total_confidence'] / stats['count']
        
        if result.success:
            stats['success_count'] += 1
        
        # Add to query history for learning
        self.query_history.append({
            'query': query,
            'skill': skill_name,
            'success': result.success,
            'confidence': result.confidence,
            'timestamp': time.time()
        })
        
        # Keep history manageable
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-50:]
    
    def _learn_from_execution(self, skill_name: str, query: str, result: SkillResult):
        """Learn patterns from skill execution results"""
        query_lower = query.lower()
        
        # Extract key phrases (simple approach)
        words = query_lower.split()
        phrases = []
        
        # Single words
        phrases.extend([word for word in words if len(word) > 2])
        
        # Two-word phrases
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        
        # Update patterns based on result
        if skill_name in self.skill_patterns:
            patterns = self.skill_patterns[skill_name]
            
            if result.success and result.confidence > 0.8:
                # Add to high confidence patterns
                for phrase in phrases[:3]:  # Limit to top 3 phrases
                    if phrase not in patterns['high_confidence_patterns']:
                        patterns['high_confidence_patterns'].append(phrase)
                        
                        # Keep lists manageable
                        if len(patterns['high_confidence_patterns']) > 20:
                            patterns['high_confidence_patterns'] = patterns['high_confidence_patterns'][-15:]
            
            elif not result.success or result.confidence < 0.3:
                # Add to low confidence patterns
                for phrase in phrases[:2]:
                    if phrase not in patterns['low_confidence_patterns']:
                        patterns['low_confidence_patterns'].append(phrase)
                        
                        if len(patterns['low_confidence_patterns']) > 10:
                            patterns['low_confidence_patterns'] = patterns['low_confidence_patterns'][-8:]
    
    def get_skill_stats(self) -> Dict[str, Any]:
        """Get comprehensive skill statistics"""
        stats = {}
        
        for skill_name, data in self.skill_stats.items():
            avg_time = data['total_time'] / data['count'] if data['count'] > 0 else 0
            success_rate = (data['success_count'] / data['count'] * 100) if data['count'] > 0 else 0
            
            stats[skill_name] = {
                'executions': data['count'],
                'success_rate': f"{success_rate:.1f}%",
                'avg_execution_time': f"{avg_time:.3f}s",
                'total_time': f"{data['total_time']:.3f}s",
                'avg_confidence': f"{data['avg_confidence']:.2f}",
                'high_confidence_patterns': len(self.skill_patterns.get(skill_name, {}).get('high_confidence_patterns', [])),
                'low_confidence_patterns': len(self.skill_patterns.get(skill_name, {}).get('low_confidence_patterns', []))
            }
        
        return stats
    
    def get_skill_recommendations(self) -> List[Dict[str, str]]:
        """Get recommendations for improving skill performance"""
        recommendations = []
        
        # Check API configurations
        if not self.weather_skill.api_key:
            recommendations.append({
                'type': 'configuration',
                'skill': 'weather',
                'message': 'Add OPENWEATHER_API_KEY to .env file for weather queries',
                'action': 'Get free API key from https://openweathermap.org/api'
            })
        
        if not self.news_skill.api_key:
            recommendations.append({
                'type': 'configuration', 
                'skill': 'news',
                'message': 'Add NEWS_API_KEY to .env file for news queries',
                'action': 'Get free API key from https://newsapi.org (100 requests/day)'
            })
        
        # Check skill performance
        for skill_name, stats in self.skill_stats.items():
            if stats['count'] > 5:  # Only analyze skills with sufficient data
                success_rate = stats['success_count'] / stats['count']
                
                if success_rate < 0.7:
                    recommendations.append({
                        'type': 'performance',
                        'skill': skill_name,
                        'message': f'{skill_name} success rate is low ({success_rate*100:.1f}%)',
                        'action': 'Check API keys and internet connection'
                    })
                
                avg_time = stats['total_time'] / stats['count']
                if avg_time > 5.0:
                    recommendations.append({
                        'type': 'performance',
                        'skill': skill_name,
                        'message': f'{skill_name} response time is slow ({avg_time:.1f}s)',
                        'action': 'Check internet connection speed'
                    })
        
        return recommendations
    
    def list_available_skills(self) -> List[Dict[str, Any]]:
        """List all available skills with enhanced metadata"""
        skills_info = []
        
        # DateTime skill
        skills_info.append({
            'name': 'datetime',
            'description': 'Current date, time, and calendar information',
            'examples': [
                'What time is it?', 
                'What day is today?', 
                'What\'s the date?',
                'What month is it?',
                'What year is it?'
            ],
            'speed': 'Instant (0.001s)',
            'requirements': 'None - always available',
            'confidence': 'Very High',
            'api_required': False
        })
        
        # Calculator skill
        skills_info.append({
            'name': 'calculator', 
            'description': 'Mathematical calculations, percentages, tips, and basic conversions',
            'examples': [
                '15 + 23', 
                '20% of 150', 
                'what is 8 * 7',
                'square root of 144',
                '20% tip on $45',
                'convert 25 celsius to fahrenheit'
            ],
            'speed': 'Instant (0.001s)',
            'requirements': 'None - always available',
            'confidence': 'High',
            'api_required': False
        })
        
        # Weather skill
        weather_available = bool(self.weather_skill.api_key)
        skills_info.append({
            'name': 'weather',
            'description': 'Current weather conditions and forecasts for any location',
            'examples': [
                'Weather in London', 
                'Is it hot in Paris?', 
                'Temperature in Tokyo',
                'Humidity in New York',
                'Wind speed in Berlin'
            ],
            'speed': 'Fast (0.5-2s)' if weather_available else 'N/A',
            'requirements': 'OpenWeatherMap API key',
            'confidence': 'High' if weather_available else 'Not Available',
            'api_required': True,
            'api_configured': weather_available
        })
        
        # News skill
        news_available = bool(self.news_skill.api_key)
        skills_info.append({
            'name': 'news',
            'description': 'Latest headlines and news by category and country',
            'examples': [
                'Latest news', 
                'Headlines from UK', 
                'Breaking news',
                'Technology news',
                'Sports news from US'
            ],
            'speed': 'Fast (0.5-2s)' if news_available else 'N/A',
            'requirements': 'NewsAPI key',
            'confidence': 'High' if news_available else 'Not Available',
            'api_required': True,
            'api_configured': news_available
        })
        
        return skills_info
    
    def export_skill_data(self) -> Dict[str, Any]:
        """Export skill statistics and learning data"""
        return {
            'skill_stats': self.skill_stats,
            'skill_patterns': self.skill_patterns,
            'recent_queries': self.query_history[-20:],  # Last 20 queries
            'total_queries': len(self.query_history),
            'confidence_threshold': self.confidence_threshold,
            'export_timestamp': time.time()
        }
    
    def import_skill_data(self, data: Dict[str, Any]):
        """Import skill statistics and learning data"""
        if 'skill_stats' in data:
            self.skill_stats.update(data['skill_stats'])
        
        if 'skill_patterns' in data:
            self.skill_patterns.update(data['skill_patterns'])
        
        if 'query_history' in data:
            self.query_history.extend(data['query_history'])
            # Keep manageable size
            if len(self.query_history) > 100:
                self.query_history = self.query_history[-100:]
        
        if 'confidence_threshold' in data:
            self.confidence_threshold = data['confidence_threshold']

# Global enhanced skills manager instance
skills_manager = EnhancedSkillsManager()0.9 if result is not None else 0.3
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return SkillResult(
                success=False,
                response=f"Calculation error: {str(e)}",
                data={'error': str(e), 'operation_type': 'error'},
                execution_time=execution_time,
                skill_used='calculator',
                confidence=0.0
            )

class WeatherSkill:
    """Enhanced weather skill using OpenWeatherMap API"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.session = None
        self.cache = {}
        self.cache_timeout = 600  # 10 minutes
    
    async def initialize(self):
        """Initialize HTTP session"""
        if AIOHTTP_AVAILABLE and not self.session:
            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def can_handle(self, query: str) -> bool:
        """Check if this is a weather query"""
        patterns = [
            r'\bweather\b',
            r'\btemperature\b', 
            r'\bhot\b.*\bis it\b',
            r'\bcold\b.*\bis it\b',
            r'\braining\b',
            r'\bsnowing\b',
            r'\bforecast\b',
            r'\bhumidity\b',
            r'\bwind\b.*\bspeed\b',
            r'\bfeels like\b',
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in patterns)
    
    def _extract_location(self, query: str) -> str:
        """Enhanced location extraction"""
        # Look for "in [location]" pattern
        in_match = re.search(r'\bin\s+([A-Za-z\s,]{2,50})', query, re.IGNORECASE)
        if in_match:
            return in_match.group(1).strip()
        
        # Look for "weather for [location]"
        for_match = re.search(r'\bfor\s+([A-Za-z\s,]{2,50})', query, re.IGNORECASE)
        if for_match:
            return for_match.group(1).strip()
        
        # Extended city list with countries/states
        cities = [
            'london', 'london uk', 'london england',
            'paris', 'paris france',
            'new york', 'new york city', 'nyc',
            'tokyo', 'tokyo japan',
            'berlin', 'berlin germany',
            'madrid', 'madrid spain',
            'rome', 'rome italy',
            'amsterdam', 'amsterdam netherlands',
            'chicago', 'chicago il',
            'los angeles', 'la', 'los angeles ca',
            'sydney', 'sydney australia',
            'melbourne', 'melbourne australia',
            'manchester', 'manchester uk',
            'birmingham', 'birmingham uk',
            'liverpool', 'liverpool uk',
            'edinburgh', 'edinburgh scotland',
            'glasgow', 'glasgow scotland',
            'dublin', 'dublin ireland',
            'toronto', 'toronto canada',
            'vancouver', 'vancouver canada',
            'miami', 'miami fl',
            'seattle', 'seattle wa',
            'san francisco', 'sf', 'san francisco ca',
        ]
        
        query_lower = query.lower()
        # Sort by length (longest first) to match more specific locations
        cities.sort(key=len, reverse=True)
        
        for city in cities:
            if city in query_lower:
                return city.title()
        
        return "London"  # Default
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get('timestamp', 0)
        return (time.time() - cached_time) < self.cache_timeout
    
    async def execute(self, query: str) -> SkillResult:
        """Execute enhanced weather query"""
        start_time = time.time()
        
        if not self.api_key:
            return SkillResult(
                success=False,
                response="Weather data requires OpenWeatherMap API key. Add OPENWEATHER_API_KEY to your .env file. Get a free key at https://openweathermap.org/api",
                execution_time=time.time() - start_time,
                skill_used='weather',
                confidence=0.0
            )
        
        location = self._extract_location(query)
        cache_key = f"weather_{location.lower()}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            cached_data = self.cache[cache_key]['data']
            return SkillResult(
                success=True,
                response=cached_data['response'],
                data=cached_data,
                execution_time=time.time() - start_time,
                skill_used='weather',
                confidence=0.8  # Slightly lower for cached
            )
        
        if not self.session:
            await self.initialize()
        
        try:
            # Current weather
            current_url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            async with self.session.get(current_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract comprehensive weather data
                    temp = data['main']['temp']
                    feels_like = data['main']['feels_like']
                    description = data['weather'][0]['description'].title()
                    humidity = data['main']['humidity']
                    pressure = data['main'].get('pressure', 0)
                    wind_speed = data.get('wind', {}).get('speed', 0)
                    wind_direction = data.get('wind', {}).get('deg', 0)
                    visibility = data.get('visibility', 0) / 1000  # Convert to km
                    location_name = data['name']
                    country = data['sys']['country']
                    
                    # Build response based on query specifics
                    query_lower = query.lower()
                    if 'temperature' in query_lower:
                        response_text = f"Temperature in {location_name}: {temp:.1f}°C (feels like {feels_like:.1f}°C)."
                    elif 'humidity' in query_lower:
                        response_text = f"Humidity in {location_name}: {humidity}%."
                    elif 'wind' in query_lower:
                        response_text = f"Wind in {location_name}: {wind_speed:.1f} m/s."
                    else:
                        # Comprehensive response
                        response_text = f"Weather in {location_name}, {country}: {temp:.1f}°C (feels like {feels_like:.1f}°C), {description}. Humidity: {humidity}%, Wind: {wind_speed:.1f} m/s."
                    
                    # Cache the result with comprehensive data
                    cache_data = {
                        'response': response_text,
                        'temperature': temp,
                        'feels_like': feels_like,
                        'description': description,
                        'humidity': humidity,
                        'pressure': pressure,
                        'wind_speed': wind_speed,
                        'wind_direction': wind_direction,
                        'visibility': visibility,
                        'location': location_name,
                        'country': country,
                        'query_type': 'current_weather',
                        'timestamp': time.time()
                    }
                    self.cache[cache_key] = {'data': cache_data, 'timestamp': time.time()}
                    
                    return SkillResult(
                        success=True,
                        response=response_text,
                        data=cache_data,
                        execution_time=time.time() - start_time,
                        skill_used='weather',
                        confidence=0.95
                    )
                
                elif response.status == 404:
                    return SkillResult(
                        success=False,
                        response=f"Weather location '{location}' not found. Try a different city name or include country (e.g., 'London, UK').",
                        execution_time=time.time() - start_time,
                        skill_used='weather',
                        confidence=0.2
                    )
                
                elif response.status == 401:
                    return SkillResult(
                        success=False,
                        response="Invalid OpenWeatherMap API key. Check your OPENWEATHER_API_KEY in .env file.",
                        execution_time=time.time() - start_time,
                        skill_used='weather',
                        confidence=0.0
                    )
                
                else:
                    return SkillResult(
                        success=False,
                        response=f"Weather service error (HTTP {response.status}). Please try again.",
                        execution_time=time.time() - start_time,
                        skill_used='weather',
                        confidence=0.1
                    )
                    
        except Exception as e:
            return SkillResult(
                success=False,
                response=f"Weather request failed: {str(e)}. Check your internet connection.",
                data={'error': str(e)},
                execution_time=time.time() - start_time,
                skill_used='weather',
                confidence=0.0
            )

class NewsSkill:
    """Enhanced news skill using NewsAPI"""
    
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.session = None
        self.cache = {}
        self.cache_timeout = 1800  # 30 minutes for news
    
    async def initialize(self):
        """Initialize HTTP session"""
        if AIOHTTP_AVAILABLE and not self.session:
            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def can_handle(self, query: str) -> bool:
        """Check if this is a news query"""
        patterns = [
            r'\bnews\b',
            r'\bheadlines\b',
            r'\bbreaking\b',
            r'\blatest\b.*\bevents\b',
            r'\bhappening\b.*\btoday\b',
            r'\bcurrent events\b',
            r'\bwhat\'?s happening\b',
            r'\bin the news\b',
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in patterns)
    
    def _extract_category_and_country(self, query: str) -> tuple:
        """Extract news category and country from query"""
        query_lower = query.lower()
        
        # Country mapping
        country_mapping = {
            'uk': 'gb', 'britain': 'gb', 'england': 'gb', 'united kingdom': 'gb',
            'usa': 'us', 'america': 'us', 'united states': 'us',
            'germany': 'de', 'france': 'fr', 'spain': 'es', 'italy': 'it',
            'canada': 'ca', 'australia': 'au', 'india': 'in', 'japan': 'jp'
        }
        
        # Category mapping
        categories = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
        
        # Extract country
        country = 'gb'  # Default
        for country_name, code in country_mapping.items():
            if country_name in query_lower:
                country = code
                break
        
        # Extract category
        category = 'general'  # Default
        for cat in categories:
            if cat in query_lower:
                category = cat
                break
        
        return category, country
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached news is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get('timestamp', 0)
        return (time.time() - cached_time) < self.cache_timeout
    
    async def execute(self, query: str) -> SkillResult:
        """Execute enhanced news query"""
        start_time = time.time()
        
        if not self.api_key:
            return SkillResult(
                success=False,
                response="News requires NewsAPI key. Add NEWS_API_KEY to your .env file. Get a free key at https://newsapi.org (100 requests/day free).",
                execution_time=time.time() - start_time,
                skill_used='news',
                confidence=0.0
            )
        
        category, country = self._extract_category_and_country(query)
        cache_key = f"news_{country}_{category}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            cached_data = self.cache[cache_key]['data']
            return SkillResult(
                success=True,
                response=cached_data['response'],
                data=cached_data,
                execution_time=time.time() - start_time,
                skill_used='news',
                confidence=0.8
            )
        
        if not self.session:
            await self.initialize()
        
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                'country': country,
                'category': category if category != 'general' else None,
                'apiKey': self.api_key,
                'pageSize': 7
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data['status'] == 'ok' and data['articles']:
                        headlines = []
                        response_parts = [f"Latest {category} news from {country.upper()}:\n"]
                        
                        for i, article in enumerate(data['articles'][:7], 1):
                            title = article['title']
                            source = article['source']['name']
                            published = article.get('publishedAt', '')
                            
                            # Format published time
                            if published:
                                try:
                                    pub_time = datetime.fromisoformat(published.replace('Z', '+00:00'))
                                    time_str = pub_time.strftime('%H:%M')
                                    headlines.append({
                                        'title': title, 
                                        'source': source,
                                        'published': time_str,
                                        'url': article.get('url', '')
                                    })
                                    response_parts.append(f"{i}. {title} - {source} ({time_str})")
                                except:
                                    headlines.append({
                                        'title': title, 
                                        'source': source,
                                        'published': 'Unknown',
                                        'url': article.get('url', '')
                                    })
                                    response_parts.append(f"{i}. {title} - {source}")
                            else:
                                headlines.append({
                                    'title': title, 
                                    'source': source,
                                    'published': 'Unknown',
                                    'url': article.get('url', '')
                                })
                                response_parts.append(f"{i}. {title} - {source}")
                        
                        response_text = "\n".join(response_parts)
                        
                        # Cache the result
                        cache_data = {
                            'response': response_text,
                            'headlines': headlines,
                            'category': category,
                            'country': country.upper(),
                            'total_articles': len(headlines),
                            'timestamp': time.time()
                        }
                        self.cache[cache_key] = {'data': cache_data, 'timestamp': time.time()}
                        
                        return SkillResult(
                            success=True,
                            response=response_text,
                            data=cache_data,
                            execution_time=time.time() - start_time,
                            skill_used='news',
                            confidence=0.9
                        )
                    
                    else:
                        return SkillResult(
                            success=False,
                            response=f"No {category} news articles found for {country.upper()}.",
                            execution_time=time.time() - start_time,
                            skill_used='news',
                            confidence=0.3
                        )
                
                elif response.status == 401:
                    return SkillResult(
                        success=False,
                        response="Invalid NewsAPI key. Check your NEWS_API_KEY in .env file.",
                        execution_time=time.time() - start_time,
                        skill_used='news',
                        confidence=0.0
                    )
                
                elif response.status == 429:
                    return SkillResult(
                        success=False,
                        response="News API rate limit exceeded. Free tier allows 100 requests/day.",
                        execution_time=time.time() - start_time,
                        skill_used='news',
                        confidence=0.1
                    )
                
                else:
                    return SkillResult(
                        success=False,
                        response=f"News service error (HTTP {response.status}). Please try again.",
                        execution_time=time.time() - start_time,
                        skill_used='news',
                        confidence=0.1
                    )
                    
        except Exception as e:
            return SkillResult(
                success=False,
                response=f"News request failed: {str(e)}. Check your internet connection.",
                data={'error': str(e)},
                execution_time=time.time() - start_time,
                skill_used='news',
                confidence
