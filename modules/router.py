"""
Pascal AI Assistant - FIXED Router Module
Improved system availability checking and current info routing
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
    """FIXED router with improved system initialization"""
    
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
        
        # Compile patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for performance"""
        
        # Simple datetime patterns (should go to skills for instant response)
        self.simple_datetime_patterns = [
            re.compile(r'^what time is it\??$', re.IGNORECASE),
            re.compile(r'^time\??$', re.IGNORECASE),
            re.compile(r'^current time\??$', re.IGNORECASE),
        ]
        
        # Current info datetime patterns (should go online)
        self.current_info_datetime_patterns = [
            re.compile(r'\bwhat day is (?:it |today)\b', re.IGNORECASE),
            re.compile(r'\bwhat (?:is )?(?:the )?(?:current )?date\b', re.IGNORECASE),
            re.compile(r'\btoday\'?s? date\b', re.IGNORECASE),
            re.compile(r'\bwhat\'?s (?:the )?date today\b', re.IGNORECASE),
        ]
        
        # Comprehensive current info patterns
        self.current_info_patterns = [
            # Political info
            re.compile(r'\b(?:current|who\s+is\s+(?:the\s+)?(?:current\s+)?)\s*(?:president|prime\s+minister|leader)\b', re.IGNORECASE),
            
            # News and events
            re.compile(r'\b(?:latest|recent|breaking|today\'?s?|current)\s+(?:news|headlines|events)\b', re.IGNORECASE),
            re.compile(r'\bwhat\'?s\s+(?:happening|going\s+on)(?:\s+(?:today|now|currently))?\b', re.IGNORECASE),
            
            # Weather
            re.compile(r'\bweather\b', re.IGNORECASE),
            re.compile(r'\btemperature\b', re.IGNORECASE),
            re.compile(r'\bforecast\b', re.IGNORECASE),
            
            # Sports results
            re.compile(r'\b(?:latest|recent|current|who\s+won)\s+(?:formula\s*1|f1|race|game|match)\b', re.IGNORECASE),
        ]
        
        # Skills patterns - only for simple instant responses
        self.instant_skill_patterns = {
            'datetime': [
                re.compile(r'^what time is it\??$', re.IGNORECASE),
                re.compile(r'^time\??$', re.IGNORECASE),
                re.compile(r'^current time\??$', re.IGNORECASE),
            ],
            'calculator': [
                re.compile(r'\b\d+\s*[\+\-\*\/\%]\s*\d+\b'),
                re.compile(r'\bwhat\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\b', re.IGNORECASE),
            ]
        }
    
    async def _check_llm_availability(self):
        """FIXED: Better system availability checking"""
        try:
            if settings.debug_mode:
                print("[ROUTER] Checking system availability...")
            
            # Initialize offline LLM with better error handling
            await self._init_offline_llm()
            
            # Initialize online LLM
            await self._init_online_llm()
            
            # Initialize skills manager
            await self._init_skills_manager()
            
            # Set routing mode
            self._set_routing_mode()
            
            if settings.debug_mode:
                print(f"[ROUTER] Systems available: offline={self.offline_available}, online={self.online_available}, skills={self.skills_available}")
            
        except Exception as e:
            if settings.debug_mode:
                print(f"[ROUTER] ❌ Availability check failed: {e}")
            self.mode = RouteMode.FALLBACK
    
    async def _init_offline_llm(self):
        """Initialize offline LLM with better error handling"""
        try:
            from modules.offline_llm import LightningOfflineLLM
            
            if settings.debug_mode:
                print("[ROUTER] Initializing offline LLM...")
            
            self.offline_llm = LightningOfflineLLM()
            self.offline_llm.set_performance_profile('balanced')
            
            # Use shorter timeout for initialization check
            try:
                self.offline_available = await asyncio.wait_for(
                    self.offline_llm.initialize(), 
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                if settings.debug_mode:
                    print("[ROUTER] ⚠️ Offline LLM initialization timed out")
                self.offline_available = False
            
            if self.offline_available:
                if settings.debug_mode:
                    print("✅ [ROUTER] Offline LLM ready")
            else:
                if settings.debug_mode:
                    error = getattr(self.offline_llm, 'last_error', 'Unknown error')
                    print(f"❌ [ROUTER] Offline LLM not available: {error}")
                        
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
                
                if settings.debug_mode:
                    print("[ROUTER] Initializing online LLM...")
                
                self.online_llm = OnlineLLM()
                
                try:
                    self.online_available = await asyncio.wait_for(
                        self.online_llm.initialize(), 
                        timeout=15.0  # 15 second timeout for online
                    )
                except asyncio.TimeoutError:
                    if settings.debug_mode:
                        print("[ROUTER] ⚠️ Online LLM initialization timed out")
                    self.online_available = False
                
                if self.online_available:
                    if settings.debug_mode:
                        print("✅ [ROUTER] Online LLM ready")
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
            
            if settings.debug_mode:
                print("[ROUTER] Initializing skills manager...")
            
            self.skills_manager = EnhancedSkillsManager()
            
            try:
                api_status = await asyncio.wait_for(
                    self.skills_manager.initialize(), 
                    timeout=10.0  # 10 second timeout for skills
                )
                self.skills_available = True
                
                if settings.debug_mode:
                    available_apis = sum(1 for status in api_status.values() if status['available'])
                    print(f"✅ [ROUTER] Skills manager ready ({available_apis} APIs configured)")
                    
            except asyncio.TimeoutError:
                if settings.debug_mode:
                    print("[ROUTER] ⚠️ Skills manager initialization timed out")
                self.skills_available = False
                
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
    
    def _detect_simple_datetime_query(self, query: str) -> bool:
        """Detect simple datetime queries that should go to skills"""
        query_clean = query.strip()
        
        for pattern in self.simple_datetime_patterns:
            if pattern.match(query_clean):
                if settings.debug_mode:
                    print(f"[ROUTER] 🕐 Simple datetime detected: {query}")
                return True
        
        return False
    
    def _detect_current_info_enhanced(self, query: str) -> bool:
        """Enhanced current info detection"""
        query_lower = query.strip().lower()
        
        # Check for current info datetime patterns first
        for pattern in self.current_info_datetime_patterns:
            if pattern.search(query_lower):
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 Current info datetime detected: {query}")
                self.stats['current_info_detected'] += 1
                return True
        
        # Check for other current info patterns
        for pattern in self.current_info_patterns:
            if pattern.search(query_lower):
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 Current info detected by pattern: {query}")
                self.stats['current_info_detected'] += 1
                return True
        
        # Enhanced temporal + info combination detection
        temporal_indicators = ['today', 'now', 'currently', 'latest', 'recent', 'breaking', 'current']
        info_indicators = ['news', 'weather', 'temperature', 'president', 'events', 'happening']
        
        has_temporal = any(indicator in query_lower for indicator in temporal_indicators)
        has_info_request = any(indicator in query_lower for indicator in info_indicators)
        
        if has_temporal and has_info_request:
            if settings.debug_mode:
                print(f"[ROUTER] 🎯 Current info detected: Temporal + info combination")
            self.stats['current_info_detected'] += 1
            return True
        
        return False
    
    def _detect_instant_skill(self, query: str) -> Optional[str]:
        """Detect instant skill queries"""
        if not self.skills_available or not self.skills_manager:
            return None
        
        query_lower = query.lower().strip()
        
        for skill_name, patterns in self.instant_skill_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    if settings.debug_mode:
                        print(f"[ROUTER] ⚡ Instant {skill_name} skill detected")
                    return skill_name
        
        return None
    
    def _should_prefer_offline(self, query: str) -> bool:
        """Check if query should prefer offline processing"""
        if self._detect_current_info_enhanced(query):
            return False
        
        # General patterns that work well offline
        offline_patterns = [
            r'\b(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))\b',
            r'\bhow\s+are\s+you\b',
            r'\bexplain\s+(?!.*(?:current|latest|today|now|recent))',
            r'\bwrite\s+(?:a|some|code|function|program)',
            r'\bwhat\s+is\s+(?!.*(?:current|today|now|latest))',
        ]
        
        query_lower = query.lower()
        for pattern in offline_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _decide_route_enhanced(self, query: str) -> RouteDecision:
        """Enhanced routing decision"""
        self.stats['routing_decisions'] += 1
        
        # Priority 1: Current information detection
        is_current_info = self._detect_current_info_enhanced(query)
        if is_current_info:
            self.stats['current_info_routed_online'] += 1
            if self.online_available:
                return RouteDecision(
                    route_type='online',
                    reason="Current information detected - routing to Groq",
                    is_current_info=True,
                    confidence=0.95,
                    expected_time=4.0
                )
            elif self.offline_available:
                return RouteDecision(
                    route_type='offline',
                    reason="Current info needed but no online access",
                    is_current_info=True,
                    confidence=0.3,
                    expected_time=3.0
                )
        
        # Priority 2: Simple instant skills
        instant_skill = self._detect_instant_skill(query)
        if instant_skill and not is_current_info:
            return RouteDecision(
                route_type='skill',
                reason=f"Instant {instant_skill} skill",
                skill_name=instant_skill,
                confidence=0.95,
                expected_time=0.1
            )
        
        # Priority 3: Offline-preferred queries
        if self._should_prefer_offline(query) and self.offline_available:
            return RouteDecision(
                route_type='offline',
                reason="Offline-preferred query type",
                confidence=0.8,
                expected_time=3.0
            )
        
        # Priority 4: Default routing
        if self.offline_available:
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
        
        # Fallback
        return RouteDecision(
            route_type='fallback',
            reason="No systems available",
            confidence=0.0,
            expected_time=0.0
        )
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """Get streaming response with proper routing"""
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
            if decision.use_skill and self.skills_manager and not decision.is_current_info:
                try:
                    result = await self.skills_manager.execute_skill(query, decision.skill_name)
                    if result and result.success:
                        yield result.response
                        self._update_stats('skill', time.time() - start_time, True)
                        return
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ROUTER] ❌ Skill error: {e}")
            
            # ONLINE ROUTE
            if decision.use_online and self.online_llm and self.online_available:
                try:
                    personality_context = await self.personality_manager.get_system_prompt()
                    memory_context = await self.memory_manager.get_context()
                    
                    if decision.is_current_info:
                        yield "🌐 Getting current information... "
                    
                    async for chunk in self.online_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                    
                    self._update_stats('online', time.time() - start_time, True)
                    return
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ROUTER] ❌ Online error: {e}")
                    
                    if decision.is_current_info and self.offline_available:
                        yield "⚠️ Online service unavailable. Using offline model...\n\n"
            
            # OFFLINE ROUTE
            if decision.use_offline and self.offline_llm and self.offline_available:
                try:
                    personality_context = await self.personality_manager.get_system_prompt()
                    memory_context = await self.memory_manager.get_context()
                    
                    if decision.is_current_info:
                        yield "ℹ️ Note: Using offline model - information may not be current.\n\n"
                    
                    async for chunk in self.offline_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                    
                    self._update_stats('offline', time.time() - start_time, True)
                    return
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ROUTER] ❌ Offline error: {e}")
            
            # FALLBACK
            fallback_response = self._generate_fallback_response(query, decision)
            yield fallback_response
            self._update_stats('fallback', time.time() - start_time, False)
            
        except Exception as e:
            if settings.debug_mode:
                print(f"[ROUTER] ❌ Critical error: {e}")
            yield "I'm experiencing technical difficulties. Please try again."
            self._update_stats('fallback', time.time() - start_time, False)
    
    def _generate_fallback_response(self, query: str, decision: RouteDecision) -> str:
        """Generate fallback responses"""
        query_lower = query.lower().strip()
        
        # Handle greetings
        if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey']):
            return "Hello! I'm Pascal, but I'm having trouble accessing my AI systems right now. Please check that Ollama is running or your internet connection is working."
        
        # Handle current info requests
        if decision.is_current_info or self._detect_current_info_enhanced(query):
            if any(word in query_lower for word in ['time', 'what time']):
                from datetime import datetime
                now = datetime.now()
                return f"The current time is {now.strftime('%I:%M %p')}. (My AI systems are currently unavailable)"
            
            if any(word in query_lower for word in ['date', 'day', 'today']):
                from datetime import datetime
                now = datetime.now()
                return f"Today is {now.strftime('%A, %B %d, %Y')}. (My AI systems are currently unavailable)"
            
            return ("I can't access current information right now as my AI systems are unavailable. "
                   "Please check reliable online sources for the latest information.")
        
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
                return f"{num1} {op} {num2} = {result}. (My AI systems are currently unavailable)"
            except:
                pass
        
        # Generic fallback
        if not self.offline_available and not self.online_available:
            return ("I'm sorry, but both my offline and online AI systems are currently unavailable. "
                   "To fix this:\n"
                   "• For offline: Run 'sudo systemctl start ollama'\n"
                   "• For online: Check your Groq API key in .env file\n"
                   "• Run diagnostics: python quick_fix.py")
        elif not self.offline_available:
            return ("My offline AI system is unavailable. Run: sudo systemctl start ollama")
        elif not self.online_available:
            return ("My online AI system is unavailable. Check your Groq API key in .env file.")
        else:
            return "I'm having trouble processing your request. Please try again."
    
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
            } if self.last_decision else None,
            'performance_stats': {
                'total_requests': total_requests,
                'routing_decisions': self.stats['routing_decisions'],
                'routing_accuracy': f"{routing_accuracy:.1f}%",
                'offline_percentage': f"{offline_percentage:.1f}%",
                'online_percentage': f"{online_percentage:.1f}%",
                'skill_percentage': f"{skill_percentage:.1f}%",
                'fallback_percentage': f"{fallback_percentage:.1f}%",
                'offline_avg_time': f"{offline_avg:.2f}s",
                'online_avg_time': f"{online_avg:.2f}s",
                'skill_avg_time': f"{skill_avg:.3f}s"
            },
            'improvements': [
                'FIXED: Better system initialization with timeouts',
                'FIXED: Improved error handling during startup',
                'FIXED: Current info detection prioritized',
                'FIXED: Fallback responses for system unavailability'
            ]
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health report"""
        health_score = 0
        components = {}
        
        if self.offline_available:
            health_score += 30
            components['offline_llm'] = 'Available and ready'
        else:
            components['offline_llm'] = 'Unavailable'
        
        if self.online_available:
            health_score += 40
            components['online_llm'] = 'Available for current info'
        else:
            components['online_llm'] = 'Unavailable'
        
        if self.skills_available:
            health_score += 20
            components['skills_manager'] = 'Available for instant responses'
        else:
            components['skills_manager'] = 'Unavailable'
        
        health_score += 10  # Always available routing
        components['routing_system'] = 'Active with current info priority'
        
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
            'current_info_capability': 'Available' if self.online_available else 'Limited'
        }
    
    async def close(self):
        """Clean shutdown"""
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
