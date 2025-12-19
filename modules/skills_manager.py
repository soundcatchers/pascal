"""
Pascal AI Assistant - Enhanced Skills Manager - COMPLETE

This file is an enhanced, backwards-compatible Skills Manager that:

- Keeps the existing built-in instant skills (datetime, calculator, weather, news).
- Performs API connectivity checks for weather/news as before.
- Dynamically registers external skills (e.g., modules.skills.sports) when available.
- Exposes initialize(), can_handle_directly(), execute_skill(), list_available_skills(),
  get_skill_stats(), and close() with the same SkillResult dataclass shape the rest
  of the project expects.

Save this file as modules/skills_manager.py (replacing the incomplete version).
"""

import importlib
import asyncio
import time
import re
import json
import math
import operator
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


@dataclass
class SkillResult:
    """Result from skill execution - matches existing project usage"""
    success: bool
    response: str
    execution_time: float
    skill_name: str
    confidence: float = 1.0
    data: Dict[str, Any] = None
    source: Optional[str] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


class EnhancedSkillsManager:
    """Skills manager that handles simple instant skills and registers pluggable skills."""

    def __init__(self):
        from config.settings import settings
        self.settings = settings
        self.session: Optional[aiohttp.ClientSession] = None

        # API Keys
        self.weather_api_key = self._get_env_var('OPENWEATHER_API_KEY')
        self.news_api_key = self._get_env_var('NEWS_API_KEY')

        # API Status tracking (will be populated on initialize)
        self.api_status = {
            'weather': {'available': False, 'message': 'Not configured'},
            'news': {'available': False, 'message': 'Not configured'},
            'sports': {'available': False, 'message': 'Not initialized'}
        }

        # Performance tracking
        self.skill_stats = {
            'datetime': {'executions': 0, 'total_time': 0.0, 'success_count': 0},
            'calculator': {'executions': 0, 'total_time': 0.0, 'success_count': 0},
            'weather': {'executions': 0, 'total_time': 0.0, 'success_count': 0},
            'news': {'executions': 0, 'total_time': 0.0, 'success_count': 0},
            'sports': {'executions': 0, 'total_time': 0.0, 'success_count': 0},
        }

        # FIXED: Only simple, instant patterns - NO comprehensive current info
        self.skill_patterns = {
            'datetime': [
                r'^what time is it\??$',
                r'^time\??$',
                r'^current time\??$',
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
                r'\btemperature in\b',
                r'\bhow hot is it\b',
                r'\bhow cold is it\b',
                r'\bis it raining\b',
            ],
            'news': [
                r'\blatest news\b',
                r'\brecent news\b',
                r'\bbreaking news\b',
                r'\bnews headlines\b',
            ]
        }

        # Container for external, pluggable skill instances (e.g., sports)
        self.external_skills: Dict[str, Any] = {}

    def _get_env_var(self, var_name: str) -> Optional[str]:
        """Better environment variable validation"""
        import os
        value = os.getenv(var_name)
        if not value:
            return None
        value = value.strip()
        invalid_values = [
            '', 'your_api_key_here', 'your_openweather_api_key_here',
            'your_news_api_key_here', 'your_weather_api_key_here'
        ]
        if value.lower() in [v.lower() for v in invalid_values]:
            return None
        return value

    async def initialize(self) -> Dict[str, Dict[str, Any]]:
        """Initialize skills manager: HTTP session, API tests, load external skills."""
        if self.settings.debug_mode:
            print("🚀 Initializing Enhanced Skills Manager...")

        # Initialize HTTP session for API calls
        if AIOHTTP_AVAILABLE:
            try:
                timeout = aiohttp.ClientTimeout(total=15, connect=5, sock_read=10)
                connector = aiohttp.TCPConnector(
                    limit=4,
                    limit_per_host=2,
                    enable_cleanup_closed=True
                )
                self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            except Exception as e:
                if self.settings.debug_mode:
                    print(f"[SKILLS] ⚠️ HTTP session creation failed: {e}")
                self.session = None

        # Test API connections (weather, news)
        await self._test_api_connections()

        # Attempt to load/register external skills (sports)
        await self._load_external_skills()

        return self.api_status

    async def _test_api_connections(self):
        """Test API connections for configured keys (weather, news)"""
        # Weather (OpenWeatherMap)
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
                    'message': f'Connection error: {str(e)[:100]}'
                }
        else:
            if not self.weather_api_key:
                self.api_status['weather'] = {
                    'available': False,
                    'message': 'API key not configured'
                }

        # News (NewsAPI.org)
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
                    'message': f'Connection error: {str(e)[:100]}'
                }
        else:
            if not self.news_api_key:
                self.api_status['news'] = {
                    'available': False,
                    'message': 'API key not configured'
                }

    async def _load_external_skills(self):
        """Try to dynamically import and initialize external skills (e.g., sports)."""
        # Try modules.skills.sports first (preferred)
        loaded = False
        try:
            mod = importlib.import_module('modules.skills.sports')
            cls = getattr(mod, 'SportsSkill', None)
            if cls:
                inst = cls(settings=self.settings)
                status = await inst.initialize()
                self.external_skills['sports'] = inst
                # status may be dict; normalize api_status entry
                self.api_status['sports'] = {'available': True, 'message': 'Sports skill initialized'}
                loaded = True
        except Exception:
            # Try legacy path skills.sports
            try:
                mod = importlib.import_module('skills.sports')
                cls = getattr(mod, 'SportsSkill', None)
                if cls:
                    inst = cls(settings=self.settings)
                    status = await inst.initialize()
                    self.external_skills['sports'] = inst
                    self.api_status['sports'] = {'available': True, 'message': 'Sports skill initialized'}
                    loaded = True
            except Exception:
                pass

        if not loaded:
            self.api_status['sports'] = {'available': False, 'message': 'Sports skill not found'}

    def can_handle_directly(self, query: str) -> Optional[str]:
        """Only handle very simple instant queries locally; return skill name or None"""
        query_lower = query.lower().strip()

        # Time queries (simple)
        for pattern in self.skill_patterns['datetime']:
            if re.search(pattern, query_lower, re.IGNORECASE):
                if self.settings.debug_mode:
                    print(f"[SKILLS] Simple time skill can handle: '{pattern}' matched")
                return 'datetime'

        # Calculator queries
        for pattern in self.skill_patterns['calculator']:
            if re.search(pattern, query_lower, re.IGNORECASE):
                if self.settings.debug_mode:
                    print(f"[SKILLS] Calculator skill can handle: '{pattern}' matched")
                return 'calculator'

        # Weather queries (only if weather API available)
        if self.api_status.get('weather', {}).get('available', False):
            for pattern in self.skill_patterns['weather']:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    if self.settings.debug_mode:
                        print(f"[SKILLS] Weather skill can handle: '{pattern}' matched")
                    return 'weather'

        # News queries (only if news API available)
        if self.api_status.get('news', {}).get('available', False):
            for pattern in self.skill_patterns['news']:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    if self.settings.debug_mode:
                        print(f"[SKILLS] News skill can handle: '{pattern}' matched")
                    return 'news'

        # External skills detection (e.g., sports) - prefer explicit analyzer to mark intent,
        # but attempt naive detection here if analyzer is not used.
        if 'f1' in query_lower or 'formula 1' in query_lower or 'grand prix' in query_lower or 'sprint' in query_lower:
            # Only claim if sports skill exists
            if 'sports' in self.external_skills:
                return 'sports'

        return None

    async def execute_skill(self, query: str, skill_name: str, entities: Dict[str, Any] = None) -> SkillResult:
        """Execute a registered or builtin skill and return SkillResult."""
        start_time = time.time()
        skill_name = (skill_name or '').lower()

        # Builtin instant skills
        try:
            if skill_name in self.skill_stats:
                self.skill_stats[skill_name]['executions'] += 1

            if skill_name in ('datetime', 'time'):
                res = await self._execute_datetime_skill(query)
                elapsed = time.time() - start_time
                res.execution_time = elapsed
                # update stats
                self._record_stats(skill_name, elapsed, res.success)
                return res

            if skill_name == 'calculator':
                res = await self._execute_calculator_skill(query)
                elapsed = time.time() - start_time
                res.execution_time = elapsed
                self._record_stats(skill_name, elapsed, res.success)
                return res

            if skill_name == 'weather':
                res = await self._execute_weather_skill(query)
                elapsed = time.time() - start_time
                res.execution_time = elapsed
                self._record_stats(skill_name, elapsed, res.success)
                return res

            if skill_name == 'news':
                res = await self._execute_news_skill(query)
                elapsed = time.time() - start_time
                res.execution_time = elapsed
                self._record_stats(skill_name, elapsed, res.success)
                return res

            # External, pluggable skills (e.g., sports)
            if skill_name in self.external_skills:
                skill = self.external_skills[skill_name]
                try:
                    if asyncio.iscoroutinefunction(getattr(skill, 'execute', None)):
                        raw = await skill.execute(query, entities or {})
                    else:
                        raw = skill.execute(query, entities or {})
                    elapsed = time.time() - start_time
                    # Normalize common return shapes to SkillResult
                    if hasattr(raw, 'success') and hasattr(raw, 'response'):
                        # external SkillResult-like object
                        result = SkillResult(bool(raw.success), str(raw.response), elapsed, skill_name,
                                             getattr(raw, 'confidence', 1.0),
                                             getattr(raw, 'data', None) or {},
                                             getattr(raw, 'source', None))
                        self._record_stats(skill_name, elapsed, result.success)
                        return result
                    if isinstance(raw, dict):
                        result = SkillResult(bool(raw.get('success')), raw.get('response', ''), elapsed, skill_name,
                                             float(raw.get('confidence', 1.0)),
                                             raw.get('data', {}), raw.get('source'))
                        self._record_stats(skill_name, elapsed, result.success)
                        return result
                    if isinstance(raw, tuple):
                        # try (success, response, source?)
                        success = bool(raw[0]) if len(raw) > 0 else False
                        response = raw[1] if len(raw) > 1 else ''
                        source = raw[2] if len(raw) > 2 else None
                        result = SkillResult(success, response, elapsed, skill_name, 1.0, {}, source)
                        self._record_stats(skill_name, elapsed, result.success)
                        return result

                    # Unknown format
                    result = SkillResult(False, "Unexpected skill result format", elapsed, skill_name)
                    self._record_stats(skill_name, elapsed, False)
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    if self.settings.debug_mode:
                        print(f"[SKILLS] External skill '{skill_name}' execution error: {e}")
                    self._record_stats(skill_name, elapsed, False)
                    return SkillResult(False, f"Skill execution error: {str(e)[:200]}", elapsed, skill_name)

            # Unknown skill
            elapsed = time.time() - start_time
            return SkillResult(False, f"Unknown or unavailable skill: {skill_name}", elapsed, skill_name)

        except Exception as e:
            elapsed = time.time() - start_time
            if self.settings.debug_mode:
                print(f"[SKILLS] Unexpected error in execute_skill: {e}")
            return SkillResult(False, f"Execution error: {str(e)[:200]}", elapsed, skill_name)

    def _record_stats(self, skill_name: str, elapsed: float, success: bool):
        """Helper to update stats safely"""
        try:
            if skill_name not in self.skill_stats:
                self.skill_stats[skill_name] = {'executions': 0, 'total_time': 0.0, 'success_count': 0}
            self.skill_stats[skill_name]['total_time'] += elapsed
            if success:
                self.skill_stats[skill_name]['success_count'] += 1
        except Exception:
            pass

    async def _execute_datetime_skill(self, query: str) -> SkillResult:
        """Only handle simple time queries - CLEAN RESPONSE"""
        now = datetime.now()
        response = f"The current time is {now.strftime('%I:%M %p')}."
        return SkillResult(
            success=True,
            response=response,
            execution_time=0.001,
            skill_name='datetime',
            confidence=1.0,
            data={'timestamp': now.timestamp(), 'formatted_time': now.strftime('%H:%M:%S'), 'time_12h': now.strftime('%I:%M %p')}
        )

    async def _execute_calculator_skill(self, query: str) -> SkillResult:
        """Enhanced calculator with better error handling"""
        query_lower = query.lower().strip()
        try:
            # Percentage calculations
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
                            return SkillResult(True, f"{percent}% of {number} is {result}", 0.001, 'calculator',
                                               data={'operation': 'percentage', 'result': result})
                    except ValueError:
                        pass

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
                            return SkillResult(False, "Cannot divide by zero", 0.001, 'calculator')
                        result = operations[operation](num1, num2)
                        if result == int(result):
                            result = int(result)
                        return SkillResult(True, f"{num1} {operation} {num2} = {result}", 0.001, 'calculator',
                                           data={'operation': operation, 'result': result, 'operands': [num1, num2]})
                except ValueError:
                    pass

            sqrt_match = re.search(r'square root of\s+(\d+(?:\.\d+)?)', query_lower)
            if sqrt_match:
                try:
                    number = float(sqrt_match.group(1))
                    if number < 0:
                        return SkillResult(False, "Cannot calculate square root of negative number", 0.001, 'calculator')
                    result = math.sqrt(number)
                    if result == int(result):
                        result = int(result)
                    return SkillResult(True, f"The square root of {number} is {result}", 0.001, 'calculator',
                                       data={'operation': 'sqrt', 'result': result})
                except ValueError:
                    pass

            squared_match = re.search(r'(\d+(?:\.\d+)?)\s*squared', query_lower)
            if squared_match:
                try:
                    number = float(squared_match.group(1))
                    result = number ** 2
                    if result == int(result):
                        result = int(result)
                    return SkillResult(True, f"{number} squared is {result}", 0.001, 'calculator',
                                       data={'operation': 'square', 'result': result})
                except ValueError:
                    pass

            power_match = re.search(r'(\d+(?:\.\d+)?)\s*to the power of\s*(\d+(?:\.\d+)?)', query_lower)
            if power_match:
                try:
                    base = float(power_match.group(1))
                    exponent = float(power_match.group(2))
                    result = base ** exponent
                    if result == int(result):
                        result = int(result)
                    return SkillResult(True, f"{base} to the power of {exponent} is {result}", 0.001, 'calculator',
                                       data={'operation': 'power', 'result': result})
                except ValueError:
                    pass

            return SkillResult(False, "I couldn't understand that calculation. Try something like '15 + 23' or '20% of 150'", 0.001, 'calculator')
        except Exception as e:
            if self.settings.debug_mode:
                print(f"[CALCULATOR] Error: {e}")
            return SkillResult(False, "I had trouble with that calculation. Please try a simpler format.", 0.001, 'calculator')

    async def _execute_weather_skill(self, query: str) -> SkillResult:
        """Weather skill - delegate to online for rich context (keeps compatibility)"""
        return SkillResult(False,
                           "For detailed weather information, I'll use my online connection for better context.",
                           0.001, 'weather')

    async def _execute_news_skill(self, query: str) -> SkillResult:
        """News skill - delegate to online for rich context (keeps compatibility)"""
        return SkillResult(False,
                           "For news and current events, I'll use my online connection for better context.",
                           0.001, 'news')

    def list_available_skills(self) -> List[Dict[str, Any]]:
        """Get list of available skills with their current status"""
        skills = [
            {
                'name': 'datetime',
                'description': 'SIMPLE time queries only (not date/day)',
                'examples': ['What time is it?', 'Time?', 'Current time'],
                'speed': 'Instant (0.001s)',
                'confidence': 'Very High',
                'api_required': False,
                'api_configured': True,
                'priority': 1,
                'note': 'Only simple time; comprehensive queries go to online'
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
        ]

        # Add info about weather/news/sports availability
        if 'weather' in self.api_status:
            skills.append({
                'name': 'weather',
                'description': 'Delegates to online weather provider',
                'api_required': True,
                'api_configured': self.api_status['weather'].get('available', False),
                'message': self.api_status['weather'].get('message')
            })
        if 'news' in self.api_status:
            skills.append({
                'name': 'news',
                'description': 'Delegates to online news provider',
                'api_required': True,
                'api_configured': self.api_status['news'].get('available', False),
                'message': self.api_status['news'].get('message')
            })
        if 'sports' in self.api_status:
            skills.append({
                'name': 'sports',
                'description': 'Domain-specific sports queries (F1) via Ergast or fallback',
                'api_required': False,
                'api_configured': self.api_status['sports'].get('available', False),
                'message': self.api_status['sports'].get('message')
            })

        return skills

    def get_skill_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all skills"""
        stats = {}
        for skill_name, raw_stats in self.skill_stats.items():
            executions = raw_stats.get('executions', 0)
            total_time = raw_stats.get('total_time', 0.0)
            success_count = raw_stats.get('success_count', 0)

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
                'status': 'Available' if skill_name in ['datetime', 'calculator'] else 'Delegated to online' if skill_name in ['weather', 'news'] else 'External'
            }

        return stats

    async def close(self):
        """Close the skills manager and cleanup resources, including external skills"""
        # Close session
        if self.session:
            try:
                await self.session.close()
                # Allow time for connections to close gracefully
                await asyncio.sleep(0.1)
            except Exception:
                pass
            self.session = None

        # Close external skills if they expose close()
        for name, inst in list(self.external_skills.items()):
            try:
                if asyncio.iscoroutinefunction(getattr(inst, 'close', None)):
                    await inst.close()
                elif callable(getattr(inst, 'close', None)):
                    inst.close()
            except Exception:
                pass

        if self.settings.debug_mode:
            total_executions = sum(stats['executions'] for stats in self.skill_stats.values())
            successful_executions = sum(stats['success_count'] for stats in self.skill_stats.values())

            if total_executions > 0:
                success_rate = (successful_executions / total_executions) * 100
                print(f"[SKILLS] 📊 Session: {successful_executions}/{total_executions} successful ({success_rate:.1f}%)")

            print("[SKILLS] 🔌 Enhanced Skills Manager closed")


# Maintain compatibility alias
SkillsManager = EnhancedSkillsManager
