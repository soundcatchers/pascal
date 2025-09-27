"""
Pascal AI Assistant - ENHANCED Router Module
Fixed routing logic with aggressive current info detection for better online routing
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
    """Standardized routing decision"""
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
    """Enhanced router with aggressive current info detection and optimized routing"""
    
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
        
        # Compiled patterns for performance
        self._compile_enhanced_patterns()
    
    def _compile_enhanced_patterns(self):
        """Compile enhanced regex patterns for better current info detection"""
        
        # ENHANCED CURRENT INFO PATTERNS - More aggressive detection
        self.current_info_patterns = [
            # Time and date queries
            re.compile(r'\b(?:what\s+)?(?:time|date|day)\s+(?:is\s+)?(?:it|today|now)\b', re.IGNORECASE),
            re.compile(r'\b(?:current|today\'?s?|now|right\s+now)\s+(?:time|date|day)\b', re.IGNORECASE),
            re.compile(r'\bwhat\s+day\s+(?:is\s+)?(?:it|today)\b', re.IGNORECASE),
            re.compile(r'\bwhat\s+(?:is\s+)?(?:the\s+)?(?:current\s+)?date\b', re.IGNORECASE),
            
            # Political current info
            re.compile(r'\b(?:current|who\s+is\s+(?:the\s+)?(?:current\s+)?)\s*(?:president|prime\s+minister|pm|leader)\b', re.IGNORECASE),
            re.compile(r'\bwho\s+(?:is\s+)?(?:the\s+)?(?:current\s+)?(?:us\s+)?president\b', re.IGNORECASE),
            
            # News and events
            re.compile(r'\b(?:latest|recent|breaking|today\'?s?|current)\s+(?:news|headlines|events)\b', re.IGNORECASE),
            re.compile(r'\bwhat\'?s\s+(?:happening|going\s+on)(?:\s+(?:today|now|currently))?\b', re.IGNORECASE),
            re.compile(r'\b(?:news|events)\s+(?:today|now|currently|recent)\b', re.IGNORECASE),
            
            # Weather patterns - ENHANCED
            re.compile(r'\b(?:weather|temperature|forecast|climate)\s+(?:today|tomorrow|now|currently)\b', re.IGNORECASE),
            re.compile(r'\b(?:weather|temperature)\s+(?:in|for)\s+[a-zA-Z\s]{2,}(?:\s+(?:today|tomorrow|now))?\b', re.IGNORECASE),
            re.compile(r'\b(?:current|today\'?s?|tomorrow\'?s?)\s+(?:weather|temperature|forecast)\b', re.IGNORECASE),
            re.compile(r'\bwhat\'?s\s+the\s+weather\s+(?:like\s+)?(?:today|tomorrow|now|currently)?\b', re.IGNORECASE),
            re.compile(r'\bis\s+it\s+(?:raining|snowing|sunny|cloudy|hot|cold)(?:\s+(?:today|now))?\b', re.IGNORECASE),
            
            # Sports and current events - ENHANCED
            re.compile(r'\b(?:latest|recent|current|who\s+won)\s+(?:formula\s*1|f1|race|game|match|championship)\b', re.IGNORECASE),
            re.compile(r'\b(?:formula\s*1|f1)\s+(?:results|winner|standings|race|championship)\b', re.IGNORECASE),
            re.compile(r'\bwho\s+(?:won|is\s+winning)\s+(?:the\s+)?(?:last|latest|recent|current|today\'?s?)\b', re.IGNORECASE),
            re.compile(r'\b(?:sports|game|match)\s+(?:results|scores|today|yesterday|recent)\b', re.IGNORECASE),
            
            # Market and financial current info
            re.compile(r'\b(?:current|today\'?s?|latest)\s+(?:stock|market|price|rates)\b', re.IGNORECASE),
            re.compile(r'\bstock\s+(?:market|prices)\s+(?:today|now|currently)\b', re.IGNORECASE),
            
            # General current patterns
            re.compile(r'\bwhat\'?s\s+new\b', re.IGNORECASE),
            re.compile(r'\banything\s+(?:new|happening|recent)\b', re.IGNORECASE),
            re.compile(r'\btoday\'?s?\s+(?!date|time)\w+', re.IGNORECASE),  # "today's anything except date/time"
        ]
        
        # Temporal indicators that suggest current info needs
        self.temporal_indicators = [
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
        
        # Skills patterns
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
        """Check and initialize all available systems"""
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
        """Initialize offline LLM with speed optimizations"""
        try:
            from modules.offline_llm import LightningOfflineLLM
            self.offline_llm = LightningOfflineLLM()
            
            # Set to speed profile for fastest responses
            self.offline_llm.set_performance_profile('speed')
            
            self.offline_available = await self.offline_llm.initialize()
            
            if self.offline_available:
                if settings.debug_mode:
                    print("✅ [ROUTER] Offline LLM ready (Nemotron)")
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
        """Initialize online LLM"""
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
        """Initialize skills manager"""
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
                print("🚀 [ROUTER] BALANCED MODE: Skills + Offline + Online")
        elif self.offline_available and self.online_available:
            self.mode = RouteMode.BALANCED
            if settings.debug_mode:
                print("🚀 [ROUTER] BALANCED MODE: Offline + Online")
        elif self.skills_available and (self.offline_available or self.online_available):
            self.mode = RouteMode.SKILLS_FIRST
            if settings.debug_mode:
                print("🚀 [ROUTER] SKILLS_FIRST MODE: Skills + LLM")
        elif self.offline_available:
            self.mode = RouteMode.OFFLINE_ONLY
            if settings.debug_mode:
                print("🏠 [ROUTER] OFFLINE_ONLY MODE: Nemotron only")
        elif self.online_available:
            self.mode = RouteMode.ONLINE_ONLY
            if settings.debug_mode:
                print("🌐 [ROUTER] ONLINE_ONLY MODE: Groq only")
        else:
            self.mode = RouteMode.FALLBACK
            if settings.debug_mode:
                print("⚠️ [ROUTER] FALLBACK MODE: No systems available")
    
    def _detect_skill(self, query: str) -> Optional[str]:
        """Detect if query can be handled by a skill"""
        if not self.skills_available or not self.skills_manager:
            return None
        
        try:
            return self.skills_manager.can_handle_directly(query)
        except Exception:
            return None
    
    def _detect_current_info_enhanced(self, query: str) -> bool:
        """ENHANCED current info detection - much more aggressive"""
        query_lower = query.strip().lower()
        
        # First check: Direct pattern matching
        for pattern in self.current_info_patterns:
            if pattern.search(query_lower):
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 Current info detected by pattern: {pattern.pattern}")
                self.stats['current_info_detected'] += 1
                return True
        
        # Second check: Temporal + Info combination (more sensitive)
        query_words = set(query_lower.split())
        
        # Check for temporal indicators
        has_temporal = any(indicator in query_lower for indicator in self.temporal_indicators)
        
        # Check for information request indicators
        has_info_request = any(indicator in query_lower for indicator in self.info_request_indicators)
        
        if has_temporal and has_info_request:
            if settings.debug_mode:
                print(f"[ROUTER] 🎯 Current info detected by temporal+info combination")
            self.stats['current_info_detected'] += 1
            return True
        
        # Third check: Weather-specific enhanced detection
        weather_indicators = ['weather', 'temperature', 'forecast', 'rain', 'snow', 'sunny', 'cloudy', 'hot', 'cold']
        location_pattern = r'\b(?:weather|temperature|forecast)\s+(?:in|for|at)\s+[a-zA-Z\s]{2,}\b'
        
        if any(indicator in query_lower for indicator in weather_indicators):
            # Any weather query is considered current info
            if settings.debug_mode:
                print(f"[ROUTER] 🎯 Current info detected: Weather query")
            self.stats['current_info_detected'] += 1
            return True
        
        if re.search(location_pattern, query_lower):
            if settings.debug_mode:
                print(f"[ROUTER] 🎯 Current info detected: Location-specific query")
            self.stats['current_info_detected'] += 1
            return True
        
        # Fourth check: Sports and events (F1, etc.)
        sports_indicators = ['formula', 'f1', 'race', 'championship', 'game', 'match', 'results', 'scores', 'winner', 'standings']
        if any(indicator in query_lower for indicator in sports_indicators):
            # Check if it's asking about recent/current events
            if any(temporal in query_lower for temporal in ['latest', 'recent', 'current', 'who won', 'winner', 'results']):
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 Current info detected: Sports/events query")
                self.stats['current_info_detected'] += 1
                return True
        
        # Fifth check: Question words + temporal context
        question_words = ['what', 'who', 'where', 'when', 'how']
        if any(query_lower.startswith(qw) for qw in question_words):
            # If it's a question AND has any temporal context, treat as current
            if any(temporal in query_lower for temporal in ['today', 'now', 'current', 'latest', 'recent']):
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 Current info detected: Question + temporal context")
                self.stats['current_info_detected'] += 1
                return True
        
        return False
    
    def _should_prefer_offline(self, query: str) -> bool:
        """Check if query should prefer offline processing"""
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
        """Enhanced routing decision with aggressive current info routing"""
        self.stats['routing_decisions'] += 1
        
        # Priority 1: Check for instant skills
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
                    reason="Current information query (enhanced detection)",
                    is_current_info=True,
                    confidence=0.95,  # Higher confidence
                    expected_time=3.0
                )
            elif self.offline_available:
                return RouteDecision(
                    route_type='offline',
                    reason="Current info but no online access",
                    is_current_info=True,
                    confidence=0.3,  # Low confidence for offline current info
                    expected_time=2.0
                )
        
        # Priority 3: Check for offline-preferred queries
        if self._should_prefer_offline(query) and self.offline_available:
            return RouteDecision(
                route_type='offline',
                reason="Offline-preferred query type",
                confidence=0.8,
                expected_time=1.5
            )
        
        # Priority 4: Default routing based on mode and availability
        if self.mode == RouteMode.OFFLINE_ONLY or (not self.online_available and self.offline_available):
            return RouteDecision(
                route_type='offline',
                reason="Default to offline (speed optimized)",
                confidence=0.7,
                expected_time=1.5
            )
        elif self.mode == RouteMode.ONLINE_ONLY or (not self.offline_available and self.online_available):
            return RouteDecision(
                route_type='online',
                reason="Online only available",
                confidence=0.7,
                expected_time=3.0
            )
        elif self.offline_available:
            # Prefer offline for speed unless it's clearly current info
            return RouteDecision(
                route_type='offline',
                reason="Default to offline for speed",
                confidence=0.6,
                expected_time=1.5
            )
        elif self.online_available:
            return RouteDecision(
                route_type='online',
                reason="Fallback to online",
                confidence=0.5,
                expected_time=3.0
            )
        
        # Fallback when no systems available
        return RouteDecision(
            route_type='fallback',
            reason="No systems available",
            confidence=0.0,
            expected_time=0.0
        )
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """Get streaming response with enhanced routing"""
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
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ROUTER] ❌ Skill error: {e}")
            
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
                    
                    # If online fails for current info, fall back to offline with warning
                    if decision.is_current_info and self.offline_available:
                        yield "⚠️ Online service unavailable. Getting best available information...\n\n"
            
            # OFFLINE ROUTE
            if decision.use_offline and self.offline_llm and self.offline_available:
                try:
                    # Minimize context for speed unless it's a complex query
                    personality_context = ""
                    memory_context = ""
                    
                    if decision.confidence < 0.8 or len(query.split()) > 15:
                        personality_context = await self.personality_manager.get_system_prompt()
                        memory_context = await self.memory_manager.get_context()
                    
                    # Add current info warning for offline routing
                    if decision.is_current_info:
                        yield "ℹ️ Note: This information may not be current. For latest info, configure online access.\n\n"
                    
                    async for chunk in self.offline_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                    
                    self._update_stats('offline', time.time() - start_time, True)
                    return
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ROUTER] ❌ Offline error: {e}")
            
            # FALLBACK RESPONSES
            fallback_response = self._generate_fallback_response(query, decision)
            yield fallback_response
            self._update_stats('fallback', time.time() - start_time, False)
            
        except Exception as e:
            if settings.debug_mode:
                print(f"[ROUTER] ❌ Critical error: {e}")
            yield "I'm experiencing technical difficulties. Please try again."
            self._update_stats('fallback', time.time() - start_time, False)
    
    def _generate_fallback_response(self, query: str, decision: RouteDecision) -> str:
        """Generate appropriate fallback response"""
        query_lower = query.lower().strip()
        
        # Handle common queries with static responses
        if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey']):
            return "Hello! I'm Pascal, but I'm having trouble accessing my AI systems right now. Please check the configuration."
        
        # Current info fallbacks
        if decision.is_current_info or self._detect_current_info_enhanced(query):
            if 'time' in query_lower and ('what' in query_lower or 'current' in query_lower):
                from datetime import datetime
                now = datetime.now()
                return f"The current time is {now.strftime('%I:%M %p')}. (Note: My AI systems are currently unavailable)"
            
            if 'date' in query_lower or 'day' in query_lower:
                from datetime import datetime
                now = datetime.now()
                return f"Today is {now.strftime('%A, %B %d, %Y')}. (Note: My AI systems are currently unavailable)"
            
            if 'weather' in query_lower:
                return ("I can't access current weather information right now. "
                       "Please check a weather service like weather.com or your local forecast.")
            
            if any(term in query_lower for term in ['news', 'events', 'happening']):
                return ("I can't access current news or events right now. "
                       "Please check reliable news sources like BBC, Reuters, or AP News.")
            
            if any(term in query_lower for term in ['formula', 'f1', 'race', 'sports']):
                return ("I can't access current sports results right now. "
                       "Please check ESPN, BBC Sport, or official F1 websites for latest results.")
            
            return ("I can't access current information right now. "
                   "Please check online sources for the latest information.")
        
        # Math calculations
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
                return f"{num1} {op} {num2} = {result}. (Note: My AI systems are currently unavailable for complex queries)"
            except:
                pass
        
        # Generic fallback based on system availability
        if not self.offline_available and not self.online_available:
            return ("I'm sorry, but both my offline and online AI systems are currently unavailable. "
                   "Please check that Ollama is running and your internet connection is working.")
        elif not self.offline_available:
            return ("My offline AI system (Nemotron) is currently unavailable. "
                   "Please check that Ollama is running: sudo systemctl start ollama")
        elif not self.online_available:
            return ("My online AI system (Groq) is currently unavailable. "
                   "Please check your internet connection and API key configuration.")
        else:
            return "I'm having trouble processing your request right now. Please try again in a moment."
    
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
        return self._detect_current_info_enhanced(query)
    
    def _detect_current_info(self, query: str) -> bool:
        """Legacy alias for current info detection"""
        return self._detect_current_info_enhanced(query)
    
    def _decide_route(self, query: str) -> RouteDecision:
        """Legacy alias for route decision"""
        return self._decide_route_enhanced(query)
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get comprehensive router statistics"""
        total_requests = self.stats['total_requests']
        
        if total_requests > 0:
            offline_percentage = (self.stats['offline_requests'] / total_requests) * 100
            online_percentage = (self.stats['online_requests'] / total_requests) * 100
            skill_percentage = (self.stats['skill_requests'] / total_requests) * 100
            fallback_percentage = (self.stats['fallback_requests'] / total_requests) * 100
            routing_accuracy = (self.stats['correct_routes'] / total_requests) * 100
            current_info_accuracy = (self.stats['current_info_routed_online'] / max(self.stats['current_info_detected'], 1)) * 100
        else:
            offline_percentage = online_percentage = skill_percentage = fallback_percentage = routing_accuracy = current_info_accuracy = 0
        
        # Calculate average times
        offline_avg = (self.stats['offline_total_time'] / max(self.stats['offline_requests'], 1))
        online_avg = (self.stats['online_total_time'] / max(self.stats['online_requests'], 1))
        skill_avg = (self.stats['skill_total_time'] / max(self.stats['skill_requests'], 1))
        
        return {
            'mode': self.mode.value,
            'system_status': {
                'offline_llm': self.offline_available,
                'online_llm': self.online_available,
                'skills_manager': self.skills_available,
            },
            'routing_strategy': 'enhanced_current_info_detection',
            'current_info_stats': {
                'detected': self.stats['current_info_detected'],
                'routed_online': self.stats['current_info_routed_online'],
                'accuracy': f"{current_info_accuracy:.1f}%"
            },
            'last_decision': {
                'route_type': self.last_decision.route_type,
                'reason': self.last_decision.reason,
                'confidence': self.last_decision.confidence,
                'is_current_info': self.last_decision.is_current_info,
                'skill_name': self.last_decision.skill_name,
                'expected_time': self.last_decision.expected_time
            } if self.last_decision else None,
            'performance_stats': {
                'total_requests': total_requests,
                'routing_decisions': self.stats['routing_decisions'],
                'routing_accuracy': f"{routing_accuracy:.1f}%",
                'offline_requests': self.stats['offline_requests'],
                'online_requests': self.stats['online_requests'],
                'skill_requests': self.stats['skill_requests'],
                'fallback_requests': self.stats['fallback_requests'],
                'offline_percentage': f"{offline_percentage:.1f}%",
                'online_percentage': f"{online_percentage:.1f}%",
                'skill_percentage': f"{skill_percentage:.1f}%",
                'fallback_percentage': f"{fallback_percentage:.1f}%",
                'offline_avg_time': f"{offline_avg:.2f}s",
                'online_avg_time': f"{online_avg:.2f}s",
                'skill_avg_time': f"{skill_avg:.3f}s"
            },
            'optimizations': [
                'Enhanced current info detection patterns',
                'Aggressive weather/events routing',
                'Sports and F1 current info detection',
                'Temporal+context analysis',
                'Compiled regex patterns for speed',
                'Fallback handling with context'
            ],
            'recommendations': self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get performance recommendations based on enhanced routing"""
        recommendations = []
        
        if not self.offline_available:
            recommendations.append("Install Ollama and Nemotron for fast offline responses")
        if not self.online_available:
            recommendations.append("Configure Groq API key for current information queries")
        if not self.skills_available:
            recommendations.append("Skills manager optimization available")
        
        # Current info routing recommendations
        if self.stats['current_info_detected'] > 0:
            online_ratio = self.stats['current_info_routed_online'] / self.stats['current_info_detected']
            if online_ratio < 0.8 and self.online_available:
                recommendations.append("Some current info queries may be routed offline - check detection patterns")
        
        if self.stats['fallback_requests'] > 0:
            recommendations.append("Some requests required fallback responses - check system health")
        
        # Performance recommendations
        if self.stats['total_requests'] > 10:
            offline_ratio = self.stats['offline_requests'] / self.stats['total_requests']
            if offline_ratio < 0.3 and self.offline_available:
                recommendations.append("Consider using offline mode more for better performance")
        
        return recommendations
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        health_score = 0
        components = {}
        
        # Offline LLM (35% of health)
        if self.offline_available:
            health_score += 35
            components['offline_llm'] = 'Available and speed-optimized'
        else:
            components['offline_llm'] = 'Unavailable'
        
        # Online LLM (35% of health - more important for current info)
        if self.online_available:
            health_score += 35
            components['online_llm'] = 'Available with enhanced current info'
        else:
            components['online_llm'] = 'Unavailable (limited current info)'
        
        # Skills Manager (20% of health)
        if self.skills_available:
            health_score += 20
            components['skills_manager'] = 'Available for instant responses'
        else:
            components['skills_manager'] = 'Unavailable'
        
        # Routing system (10% of health)
        health_score += 10  # Always available
        components['routing_system'] = 'Active with enhanced current info detection'
        
        # Determine health status
        if health_score >= 90:
            status = 'Excellent'
        elif health_score >= 70:
            status = 'Good'
        elif health_score >= 50:
            status = 'Fair'
        else:
            status = 'Poor'
        
        return {
            'overall_health_score': health_score,
            'system_status': status,
            'components': components,
            'fallback_available': True,
            'current_info_capability': 'Enhanced' if self.online_available else 'Limited',
            'recommendations': self._get_recommendations()
        }
    
    async def close(self):
        """Clean shutdown of all router components"""
        if self.offline_llm:
            try:
                await self.offline_llm.close()
            except Exception:
                pass
        if self.online_llm:
            try:
                await self.online_llm.close()
            except Exception:
                pass
        if self.skills_manager:
            try:
                await self.skills_manager.close()
            except Exception:
                pass

# Maintain compatibility
EnhancedRouter = LightningRouter
