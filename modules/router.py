"""
Pascal AI Assistant - ENHANCED Router Module  
Fixed routing logic with improved current info detection and better fallback handling
"""

import asyncio
import time
import re
from typing import Optional, AsyncGenerator, Dict, Any, List
from enum import Enum
from dataclasses import dataclass

from config.settings import settings

class RouteMode(Enum):
    """Enhanced routing modes"""
    BALANCED = "balanced"
    OFFLINE_ONLY = "offline_only"
    ONLINE_ONLY = "online_only"
    SKILLS_FIRST = "skills_first"
    FALLBACK = "fallback"

@dataclass
class RouteDecision:
    """Enhanced routing decision with better tracking"""
    route_type: str  # 'offline', 'online', 'skill', 'fallback'
    reason: str
    confidence: float = 0.8
    skill_name: Optional[str] = None
    is_current_info: bool = False
    expected_time: float = 2.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    @property
    def use_offline(self) -> bool:
        return self.route_type == 'offline'
    
    @property
    def use_online(self) -> bool:
        return self.route_type == 'online'
    
    @property
    def use_skill(self) -> bool:
        return self.route_type == 'skill'

class LightningRouter:
    """ENHANCED router with fixed current info detection and better error handling"""
    
    def __init__(self, personality_manager, memory_manager):
        self.personality_manager = personality_manager
        self.memory_manager = memory_manager
        
        # Initialize components
        self.offline_llm = None
        self.online_llm = None
        self.skills_manager = None
        
        # Router state
        self.mode = RouteMode.FALLBACK
        self.last_decision = None
        self.offline_available = False
        self.online_available = False
        self.skills_available = False
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'offline_requests': 0,
            'online_requests': 0,
            'skill_requests': 0,
            'fallback_requests': 0,
            'offline_total_time': 0.0,
            'online_total_time': 0.0,
            'skill_total_time': 0.0,
            'routing_decisions': 0,
            'correct_routes': 0,
            'current_info_detected': 0,
            'current_info_routed_online': 0
        }
        
        # ENHANCED: Compile regex patterns for better performance and accuracy
        self._compile_enhanced_patterns()
    
    def _compile_enhanced_patterns(self):
        """FIXED: Compile enhanced regex patterns for better current info detection"""
        
        # ENHANCED DATETIME PATTERNS - More comprehensive
        self.datetime_patterns = [
            re.compile(r'\bwhat\s+(?:time|day|date)\s+(?:is\s+)?(?:it|today|now)\b', re.IGNORECASE),
            re.compile(r'\b(?:current|today\'?s?|now|right\s+now)\s+(?:time|date|day)\b', re.IGNORECASE),
            re.compile(r'\bwhat\s+day\s+(?:is\s+)?(?:it|today)\b', re.IGNORECASE),
            re.compile(r'\bwhat\s+(?:is\s+)?(?:the\s+)?(?:current\s+)?date\b', re.IGNORECASE),
            re.compile(r'\btoday\'?s?\s+date\b', re.IGNORECASE),
        ]
        
        # ENHANCED CURRENT INFO PATTERNS - Much more aggressive
        self.current_info_patterns = [
            # Political current info
            re.compile(r'\b(?:current|who\s+is\s+(?:the\s+)?(?:current\s+)?)\s*(?:president|prime\s+minister|pm|leader)\b', re.IGNORECASE),
            re.compile(r'\bwho\s+(?:is\s+)?(?:the\s+)?(?:current\s+)?(?:us\s+)?president\b', re.IGNORECASE),
            
            # News and events - ENHANCED
            re.compile(r'\b(?:latest|recent|breaking|today\'?s?|current)\s+(?:news|headlines|events)\b', re.IGNORECASE),
            re.compile(r'\bwhat\'?s\s+(?:happening|going\s+on)(?:\s+(?:today|now|currently))?\b', re.IGNORECASE),
            re.compile(r'\b(?:news|events)\s+(?:today|now|currently|recent|latest)\b', re.IGNORECASE),
            re.compile(r'\bin\s+the\s+news\b', re.IGNORECASE),
            
            # Weather patterns - ENHANCED (treat ALL weather as current)
            re.compile(r'\bweather\b', re.IGNORECASE),  # ANY weather query is current
            re.compile(r'\btemperature\b', re.IGNORECASE),  # ANY temperature query
            re.compile(r'\bforecast\b', re.IGNORECASE),
            re.compile(r'\b(?:raining|snowing|sunny|cloudy|hot|cold)\b', re.IGNORECASE),
            re.compile(r'\bhow\s+(?:hot|cold|warm)\s+is\s+it\b', re.IGNORECASE),
            
            # Sports and current events - ENHANCED
            re.compile(r'\b(?:latest|recent|current|who\s+won)\s+(?:formula\s*1|f1|race|game|match|championship)\b', re.IGNORECASE),
            re.compile(r'\b(?:formula\s*1|f1)\s+(?:results|winner|standings|race|championship)\b', re.IGNORECASE),
            re.compile(r'\bwho\s+(?:won|is\s+winning)\s+(?:the\s+)?(?:last|latest|recent|current)\b', re.IGNORECASE),
            re.compile(r'\b(?:sports|game|match)\s+(?:results|scores|today|yesterday|recent)\b', re.IGNORECASE),
            
            # Market and financial current info
            re.compile(r'\b(?:current|today\'?s?|latest)\s+(?:stock|market|price|rates)\b', re.IGNORECASE),
            re.compile(r'\bstock\s+(?:market|prices)\s+(?:today|now|currently)\b', re.IGNORECASE),
        ]
        
        # Temporal indicators that strongly suggest current info needs
        self.strong_temporal_indicators = [
            'today', 'now', 'currently', 'right now', 'at the moment', 'these days',
            'recently', 'lately', 'this week', 'this month', 'this year',
            'tomorrow', 'tonight', 'current', 'latest', 'breaking', 'live',
            'real-time', 'up to date', 'fresh', 'new', 'just happened'
        ]
        
        # Information request indicators
        self.info_request_indicators = [
            'news', 'weather', 'temperature', 'forecast', 'president', 'leader',
            'events', 'happening', 'results', 'scores', 'standings', 'winner',
            'market', 'stocks', 'prices', 'rates', 'election', 'politics'
        ]
        
        # Skills patterns - For instant responses
        self.skill_patterns = {
            'datetime': [
                re.compile(r'\bwhat\s+time\s+is\s+it\b', re.IGNORECASE),
                re.compile(r'\bwhat\s+day\s+is\s+(?:it|today)\b', re.IGNORECASE),
                re.compile(r'\bcurrent\s+(?:time|date)\b', re.IGNORECASE),
                re.compile(r'\btoday\'?s?\s+date\b', re.IGNORECASE),
            ],
            'calculator': [
                re.compile(r'\b\d+\s*[\+\-\*\/\%]\s*\d+\b'),
                re.compile(r'\bwhat\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\b', re.IGNORECASE),
                re.compile(r'\bcalculate\s+\d+', re.IGNORECASE),
                re.compile(r'\b\d+\s*percent\s+of\s+\d+\b', re.IGNORECASE),
            ]
        }
        
        # Offline preferred patterns
        self.offline_patterns = [
            re.compile(r'\b(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))\b', re.IGNORECASE),
            re.compile(r'\bhow\s+are\s+you\b', re.IGNORECASE),
            re.compile(r'\bexplain\s+(?!.*(?:current|latest|today|now|recent))', re.IGNORECASE),
            re.compile(r'\bwrite\s+(?:a|some|code|function|program)', re.IGNORECASE),
            re.compile(r'\bwhat\s+is\s+(?!.*(?:current|today|now|latest))', re.IGNORECASE),
            re.compile(r'\bhow\s+(?:do|to)\s+(?!.*(?:current|today|now|latest))', re.IGNORECASE),
        ]
    
    async def _check_llm_availability(self):
        """FIXED: Check and initialize all available systems with better error handling"""
        try:
            if settings.debug_mode:
                print("[ROUTER] Checking system availability...")
            
            # Initialize offline LLM
            await self._init_offline_llm()
            
            # Initialize online LLM
            await self._init_online_llm()
            
            # Initialize skills manager
            await self._init_skills_manager()
            
            # Set optimal routing mode
            self._set_routing_mode()
            
        except Exception as e:
            if settings.debug_mode:
                print(f"[ROUTER] ❌ Availability check failed: {e}")
            self.mode = RouteMode.FALLBACK
    
    async def _init_offline_llm(self):
        """FIXED: Initialize offline LLM with better error handling"""
        try:
            from modules.offline_llm import LightningOfflineLLM
            self.offline_llm = LightningOfflineLLM()
            
            # Set to balanced profile for reliability
            self.offline_llm.set_performance_profile('balanced')
            
            self.offline_available = await self.offline_llm.initialize()
            
            if self.offline_available:
                if settings.debug_mode:
                    print("✅ [ROUTER] Offline LLM ready (Nemotron/Qwen)")
            else:
                if settings.debug_mode:
                    print("❌ [ROUTER] Offline LLM not available")
                    if hasattr(self.offline_llm, 'last_error') and self.offline_llm.last_error:
                        print(f"   Error: {self.offline_llm.last_error}")
                        
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ [ROUTER] Offline LLM initialization error: {e}")
            self.offline_available = False
            self.offline_llm = None
    
    async def _init_online_llm(self):
        """FIXED: Initialize online LLM with better validation"""
        if settings.is_online_available():
            try:
                from modules.online_llm import OnlineLLM
                self.online_llm = OnlineLLM()
                self.online_available = await self.online_llm.initialize()
                
                if self.online_available:
                    if settings.debug_mode:
                        print("✅ [ROUTER] Online LLM ready (Groq)")
                else:
                    if settings.debug_mode:
                        print("❌ [ROUTER] Online LLM not available")
                        if hasattr(self.online_llm, 'last_error'):
                            print(f"   Error: {self.online_llm.last_error}")
                        
            except Exception as e:
                if settings.debug_mode:
                    print(f"❌ [ROUTER] Online LLM initialization error: {e}")
                self.online_available = False
                self.online_llm = None
        else:
            self.online_available = False
            if settings.debug_mode:
                print("[ROUTER] No Groq API key - online features disabled")
    
    async def _init_skills_manager(self):
        """FIXED: Initialize skills manager with better error handling"""
        try:
            from modules.skills_manager import EnhancedSkillsManager
            self.skills_manager = EnhancedSkillsManager()
            
            # Initialize and check functionality
            api_status = await self.skills_manager.initialize()
            self.skills_available = True
            
            if settings.debug_mode:
                available_apis = sum(1 for status in api_status.values() if status['available'])
                print(f"✅ [ROUTER] Skills manager ready ({available_apis} APIs configured)")
                
        except Exception as e:
            if settings.debug_mode:
                print(f"⚠️ [ROUTER] Skills manager unavailable: {e}")
            self.skills_available = False
            self.skills_manager = None
    
    def _set_routing_mode(self):
        """Set optimal routing mode based on available systems"""
        if self.offline_available and self.online_available and self.skills_available:
            self.mode = RouteMode.BALANCED
            if settings.debug_mode:
                print("🚀 [ROUTER] ENHANCED MODE: Skills + Offline + Online")
        elif self.offline_available and self.online_available:
            self.mode = RouteMode.BALANCED
            if settings.debug_mode:
                print("🚀 [ROUTER] ENHANCED MODE: Offline + Online")
        elif self.skills_available and (self.offline_available or self.online_available):
            self.mode = RouteMode.SKILLS_FIRST
            if settings.debug_mode:
                print("🚀 [ROUTER] ENHANCED MODE: Skills + LLM")
        elif self.offline_available:
            self.mode = RouteMode.OFFLINE_ONLY
            if settings.debug_mode:
                print("🏠 [ROUTER] ENHANCED MODE: Offline only")
        elif self.online_available:
            self.mode = RouteMode.ONLINE_ONLY
            if settings.debug_mode:
                print("🌐 [ROUTER] ENHANCED MODE: Online only")
        else:
            self.mode = RouteMode.FALLBACK
            if settings.debug_mode:
                print("⚠️ [ROUTER] ENHANCED MODE: Fallback only")
    
    def _detect_skill(self, query: str) -> Optional[str]:
        """Detect if query can be handled by a skill"""
        if not self.skills_available or not self.skills_manager:
            return None
        
        try:
            return self.skills_manager.can_handle_directly(query)
        except Exception:
            return None
    
    def _detect_current_info_enhanced(self, query: str) -> bool:
        """ENHANCED: Much more aggressive current info detection"""
        query_lower = query.strip().lower()
        
        # First check: DateTime patterns (these should go to skills, not online)
        for pattern in self.datetime_patterns:
            if pattern.search(query_lower):
                # DateTime queries should go to skills, not online
                return False
        
        # Second check: Direct current info pattern matching
        for pattern in self.current_info_patterns:
            if pattern.search(query_lower):
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 Current info detected by pattern")
                self.stats['current_info_detected'] += 1
                return True
        
        # Third check: Weather is ALWAYS current info
        weather_keywords = ['weather', 'temperature', 'forecast', 'rain', 'snow', 'sunny', 'cloudy', 'hot', 'cold', 'humid', 'windy']
        if any(keyword in query_lower for keyword in weather_keywords):
            if settings.debug_mode:
                print(f"[ROUTER] 🎯 Current info detected: Weather query")
            self.stats['current_info_detected'] += 1
            return True
        
        # Fourth check: Temporal + Info combination
        query_words = set(query_lower.split())
        
        # Check for temporal indicators
        has_temporal = any(indicator in query_lower for indicator in self.strong_temporal_indicators)
        
        # Check for information request indicators
        has_info_request = any(indicator in query_lower for indicator in self.info_request_indicators)
        
        if has_temporal and has_info_request:
            if settings.debug_mode:
                print(f"[ROUTER] 🎯 Current info detected: Temporal + info combination")
            self.stats['current_info_detected'] += 1
            return True
        
        # Fifth check: Sports and events (F1, etc.)
        sports_indicators = ['formula', 'f1', 'race', 'championship', 'game', 'match', 'results', 'scores', 'winner', 'standings']
        temporal_sports = ['latest', 'recent', 'current', 'who won', 'today', 'yesterday']
        
        has_sports = any(indicator in query_lower for indicator in sports_indicators)
        has_temporal_sports = any(temporal in query_lower for temporal in temporal_sports)
        
        if has_sports and has_temporal_sports:
            if settings.debug_mode:
                print(f"[ROUTER] 🎯 Current info detected: Sports/events query")
            self.stats['current_info_detected'] += 1
            return True
        
        # Sixth check: Question words + strong temporal context
        question_words = ['what', 'who', 'where', 'when', 'how']
        strong_temporal = ['today', 'now', 'current', 'latest', 'recent', 'breaking']
        
        starts_with_question = any(query_lower.startswith(qw) for qw in question_words)
        has_strong_temporal = any(temporal in query_lower for temporal in strong_temporal)
        
        if starts_with_question and has_strong_temporal:
            if settings.debug_mode:
                print(f"[ROUTER] 🎯 Current info detected: Question + strong temporal")
            self.stats['current_info_detected'] += 1
            return True
        
        return False
    
    def _should_prefer_offline(self, query: str) -> bool:
        """ENHANCED: Check if query should prefer offline processing"""
        # Don't prefer offline for current info queries
        if self._detect_current_info_enhanced(query):
            return False
        
        # Check offline-preferred patterns
        for pattern in self.offline_patterns:
            if pattern.search(query):
                return True
        
        # Programming and technical queries (non-current)
        tech_indicators = ['code', 'function', 'program', 'algorithm', 'python', 'javascript', 'programming']
        query_lower = query.lower()
        if any(indicator in query_lower for indicator in tech_indicators):
            # Only if not asking for current/latest versions
            if not any(temporal in query_lower for temporal in ['current', 'latest', 'new', 'recent']):
                return True
        
        # Educational/explanatory queries (non-current)
        if query_lower.startswith(('explain', 'what is', 'how does', 'tell me about')):
            # Only if not asking for current information
            if not any(temporal in query_lower for temporal in ['current', 'latest', 'today', 'now', 'recent']):
                return True
        
        return False
    
    def _decide_route_enhanced(self, query: str) -> RouteDecision:
        """ENHANCED: Better routing decision logic"""
        self.stats['routing_decisions'] += 1
        
        # Priority 1: Check for instant skills (datetime, calculator)
        if self.skills_available:
            skill_name = self._detect_skill(query)
            if skill_name:
                return RouteDecision(
                    route_type='skill',
                    reason=f"Instant {skill_name} skill",
                    skill_name=skill_name,
                    confidence=0.95,
                    expected_time=0.1
                )
        
        # Priority 2: ENHANCED current information detection
        needs_current_info = self._detect_current_info_enhanced(query)
        if needs_current_info:
            self.stats['current_info_routed_online'] += 1
            if self.online_available:
                return RouteDecision(
                    route_type='online',
                    reason="Current information detected (enhanced)",
                    is_current_info=True,
                    confidence=0.95,
                    expected_time=4.0
                )
            elif self.offline_available:
                return RouteDecision(
                    route_type='offline',
                    reason="Current info but no online access (will note limitation)",
                    is_current_info=True,
                    confidence=0.3,
                    expected_time=3.0
                )
        
        # Priority 3: Check for offline-preferred queries
        if self._should_prefer_offline(query) and self.offline_available:
            return RouteDecision(
                route_type='offline',
                reason="Offline-preferred query type",
                confidence=0.8,
                expected_time=3.0
            )
        
        # Priority 4: Default routing based on mode and availability
        if self.mode == RouteMode.OFFLINE_ONLY or (not self.online_available and self.offline_available):
            return RouteDecision(
                route_type='offline',
                reason="Default to offline",
                confidence=0.7,
                expected_time=3.0
            )
        elif self.mode == RouteMode.ONLINE_ONLY or (not self.offline_available and self.online_available):
            return RouteDecision(
                route_type='online',
                reason="Online only available",
                confidence=0.7,
                expected_time=4.0
            )
        elif self.offline_available:
            # Prefer offline for non-current info
            return RouteDecision(
                route_type='offline',
                reason="Default to offline for general queries",
                confidence=0.6,
                expected_time=3.0
            )
        elif self.online_available:
            return RouteDecision(
                route_type='online',
                reason="Fallback to online",
                confidence=0.5,
                expected_time=4.0
            )
        
        # Fallback when no systems available
        return RouteDecision(
            route_type='fallback',
            reason="No systems available",
            confidence=0.0,
            expected_time=0.0
        )
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """ENHANCED: Get streaming response with better routing and error handling"""
        decision = self._decide_route_enhanced(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            route_display = decision.route_type.upper()
            if decision.skill_name:
                route_display = f"{decision.skill_name.upper()} SKILL"
            elif decision.is_current_info:
                route_display += " (CURRENT INFO)"
            print(f"[ROUTER] 🚀 Route: {route_display} - {decision.reason}")
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # SKILLS ROUTE
            if decision.use_skill and self.skills_manager:
                try:
                    result = await self.skills_manager.execute_skill(query, decision.skill_name)
                    if result and result.success:
                        yield result.response
                        self._update_stats('skill', time.time() - start_time, True)
                        return
                    else:
                        if settings.debug_mode:
                            print(f"[ROUTER] ⚠️ Skill failed, falling back")
                        # Continue to fallback
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ROUTER] ❌ Skill error: {e}")
                    # Continue to fallback
            
            # ONLINE ROUTE (prioritized for current info)
            if decision.use_online and self.online_llm and self.online_available:
                try:
                    personality_context = await self.personality_manager.get_system_prompt()
                    memory_context = await self.memory_manager.get_context()
                    
                    async for chunk in self.online_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                    
                    self._update_stats('online', time.time() - start_time, True)
                    return
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ROUTER] ❌ Online error: {e}")
                    
                    # If online fails for current info, fall back with warning
                    if decision.is_current_info and self.offline_available:
                        yield "⚠️ Online service unavailable. Using offline model (information may not be current)...\n\n"
            
            # OFFLINE ROUTE
            if decision.use_offline and self.offline_llm and self.offline_available:
                try:
                    personality_context = await self.personality_manager.get_system_prompt()
                    memory_context = await self.memory_manager.get_context()
                    
                    # Add current info warning for offline routing
                    if decision.is_current_info:
                        yield "ℹ️ Note: I don't have access to real-time information. For current data, please check online sources.\n\n"
                    
                    async for chunk in self.offline_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                    
                    self._update_stats('offline', time.time() - start_time, True)
                    return
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ROUTER] ❌ Offline error: {e}")
                    # Continue to fallback
            
            # FALLBACK RESPONSES
            fallback_response = self._generate_enhanced_fallback_response(query, decision)
            yield fallback_response
            self._update_stats('fallback', time.time() - start_time, False)
            
        except Exception as e:
            if settings.debug_mode:
                print(f"[ROUTER] ❌ Critical error: {e}")
            yield "I'm experiencing technical difficulties. Please try again."
            self._update_stats('fallback', time.time() - start_time, False)
    
    def _generate_enhanced_fallback_response(self, query: str, decision: RouteDecision) -> str:
        """ENHANCED: Generate more helpful fallback responses"""
        query_lower = query.lower().strip()
        
        # Handle common queries with better responses
        if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey']):
            return "Hello! I'm Pascal, but I'm having trouble accessing my AI systems right now. Please check that Ollama is running or your internet connection is working."
        
        # Current info fallbacks - More helpful
        if decision.is_current_info or self._detect_current_info_enhanced(query):
            if any(word in query_lower for word in ['time', 'what time']):
                from datetime import datetime
                now = datetime.now()
                return f"The current time is {now.strftime('%I:%M %p')}. (My AI systems are currently unavailable for more complex queries)"
            
            if any(word in query_lower for word in ['date', 'day', 'today']):
                from datetime import datetime
                now = datetime.now()
                return f"Today is {now.strftime('%A, %B %d, %Y')}. (My AI systems are currently unavailable for more complex queries)"
            
            if 'weather' in query_lower:
                return ("I can't access current weather information right now as my systems are unavailable. "
                       "Please check weather.com, your local weather app, or ask a voice assistant.")
            
            if any(term in query_lower for term in ['news', 'events', 'happening']):
                return ("I can't access current news right now as my systems are unavailable. "
                       "Please check BBC News, Reuters, AP News, or your preferred news source.")
            
            if any(term in query_lower for term in ['formula', 'f1', 'race', 'sports']):
                return ("I can't access current sports results right now as my systems are unavailable. "
                       "Please check ESPN, BBC Sport, or the official Formula 1 website for latest results.")
            
            if any(term in query_lower for term in ['president', 'politics', 'election']):
                return ("I can't access current political information right now. As of my last update, Donald Trump is the current US President (inaugurated January 20, 2025), but please check current news sources for the latest information.")
            
            return ("I can't access current information right now as my AI systems are unavailable. "
                   "Please check reliable online sources for the latest information.")
        
        # Math calculations - Still try to help
        import re
        math_match = re.search(r'(\d+)\s*[\+\-\*\/]\s*(\d+)', query_lower)
        if math_match:
            try:
                num1, op, num2 = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', query_lower).groups()
                num1, num2 = int(num1), int(num2)
                if op == '+':
                    result = num1 + num2
                elif op == '-':
                    result = num1 - num2
                elif op == '*':
                    result = num1 * num2
                elif op == '/' and num2 != 0:
                    result = num1 / num2
                else:
                    result = "undefined"
                return f"{num1} {op} {num2} = {result}. (My AI systems are currently unavailable for more complex queries)"
            except:
                pass
        
        # Generic fallback based on system availability - More helpful
        if not self.offline_available and not self.online_available:
            return ("I'm sorry, but both my offline and online AI systems are currently unavailable. "
                   "To fix this:\n"
                   "• For offline: Run 'sudo systemctl start ollama' and ensure you have models downloaded\n"
                   "• For online: Check your internet connection and Groq API key in .env file\n"
                   "• Run diagnostics: python quick_fix.py")
        elif not self.offline_available:
            return ("My offline AI system is currently unavailable. "
                   "To fix this, run: sudo systemctl start ollama\n"
                   "Or run diagnostics: python quick_fix.py")
        elif not self.online_available:
            return ("My online AI system is currently unavailable. "
                   "To fix this, check your internet connection and add your Groq API key to the .env file.\n"
                   "Get a free key at: https://console.groq.com/")
        else:
            return "I'm having trouble processing your request right now. Please try again in a moment, or run: python quick_fix.py"
    
    async def get_response(self, query: str) -> str:
        """Get non-streaming response"""
        response_parts = []
        async for chunk in self.get_streaming_response(query):
            response_parts.append(chunk)
        return ''.join(response_parts)
    
    def _update_stats(self, route_type: str, response_time: float, success: bool):
        """Update routing statistics"""
        if route_type == 'offline':
            self.stats['offline_requests'] += 1
            if success:
                self.stats['offline_total_time'] += response_time
                self.stats['correct_routes'] += 1
        elif route_type == 'online':
            self.stats['online_requests'] += 1
            if success:
                self.stats['online_total_time'] += response_time
                self.stats['correct_routes'] += 1
        elif route_type == 'skill':
            self.stats['skill_requests'] += 1
            if success:
                self.stats['skill_total_time'] += response_time
                self.stats['correct_routes'] += 1
        elif route_type == 'fallback':
            self.stats['fallback_requests'] += 1
    
    # Legacy compatibility methods
    def _needs_current_information(self, query: str) -> bool:
        """Legacy alias for current info detection"""
        return self._detect_current
