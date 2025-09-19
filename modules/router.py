"""
Pascal AI Assistant - ENHANCED Fast Router (Fixed Version)
Improved routing with better current info detection and speed optimization
FOCUS: Accurate routing decisions and <2s offline, <4s online responses
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
    FAST_OFFLINE = "fast_offline"     # Nemotron-first for speed
    BALANCED = "balanced"             # Smart routing with both systems
    ONLINE_ONLY = "online_only"       # Groq only
    OFFLINE_ONLY = "offline_only"     # Nemotron only

@dataclass
class EnhancedRouteDecision:
    """Enhanced routing decision with detailed reasoning"""
    route_type: str  # 'offline', 'online', 'skill'
    reason: str
    is_current_info: bool = False
    confidence: float = 0.8
    skill_name: Optional[str] = None
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
    """Enhanced fast router with improved routing logic and performance"""
    
    def __init__(self, personality_manager, memory_manager):
        self.personality_manager = personality_manager
        self.memory_manager = memory_manager
        
        # Initialize LLM components
        self.offline_llm = None
        self.online_llm = None
        self.skills_manager = None
        
        # Router state
        self.mode = RouteMode.BALANCED  # Default to balanced routing
        self.last_decision = None
        self.offline_available = False
        self.online_available = False
        self.skills_available = False
        
        # Performance tracking
        self.stats = {
            'offline_requests': 0,
            'online_requests': 0,
            'skill_requests': 0,
            'offline_total_time': 0.0,
            'online_total_time': 0.0,
            'skill_total_time': 0.0,
            'routing_decisions': 0,
            'correct_routes': 0,
        }
        
        # Enhanced current info patterns (compiled regex for speed)
        self.current_info_patterns = [
            # Date and time queries
            re.compile(r'\b(?:what\s+)?(?:time|date|day)\s+(?:is\s+)?(?:it|today)\b', re.IGNORECASE),
            re.compile(r'\b(?:current|today\'?s?|now)\s+(?:time|date|day)\b', re.IGNORECASE),
            re.compile(r'\bwhat\s+day\s+(?:is\s+)?(?:it|today)\b', re.IGNORECASE),
            re.compile(r'\bwhat\s+(?:is\s+)?(?:the\s+)?date\b', re.IGNORECASE),
            re.compile(r'\btell\s+me\s+the\s+(?:time|date|day)\b', re.IGNORECASE),
            
            # Political/leadership queries
            re.compile(r'\b(?:current|who\s+is\s+(?:the\s+)?(?:current\s+)?)\s*(?:president|prime\s+minister|pm|leader)\b', re.IGNORECASE),
            re.compile(r'\bwho\s+is\s+president\b', re.IGNORECASE),
            re.compile(r'\bwho\s+runs\s+(?:the\s+)?(?:country|government|us|uk)\b', re.IGNORECASE),
            
            # News and events
            re.compile(r'\b(?:latest|recent|breaking|today\'?s?)\s+news\b', re.IGNORECASE),
            re.compile(r'\bwhat\'?s\s+(?:happening|going\s+on)\b', re.IGNORECASE),
            re.compile(r'\b(?:news|events)\s+today\b', re.IGNORECASE),
            re.compile(r'\bcurrent\s+events\b', re.IGNORECASE),
            re.compile(r'\bin\s+the\s+news\b', re.IGNORECASE),
            
            # Weather queries
            re.compile(r'\b(?:weather|temperature)\s+(?:today|now|currently)\b', re.IGNORECASE),
            re.compile(r'\bcurrent\s+weather\b', re.IGNORECASE),
            re.compile(r'\bweather\s+(?:in|for)\s+\w+\s+today\b', re.IGNORECASE),
            
            # Market/financial current info
            re.compile(r'\b(?:current|latest)\s+(?:stock|market|price)\b', re.IGNORECASE),
            re.compile(r'\bstock\s+market\s+today\b', re.IGNORECASE),
        ]
        
        # Instant skill patterns
        self.instant_skill_patterns = [
            # DateTime patterns
            re.compile(r'\bwhat\s+time\s+is\s+it\b', re.IGNORECASE),
            re.compile(r'\bwhat\s+day\s+is\s+(?:it|today)\b', re.IGNORECASE),
            re.compile(r'\bwhat\s+(?:is\s+)?(?:the\s+)?date\b', re.IGNORECASE),
            re.compile(r'\bcurrent\s+(?:time|date|day)\b', re.IGNORECASE),
            
            # Math patterns
            re.compile(r'\b\d+\s*[\+\-\*\/\%]\s*\d+\b'),
            re.compile(r'\b\d+\s+(?:plus|minus|times|divided\s+by)\s+\d+\b', re.IGNORECASE),
            re.compile(r'\bwhat\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\b', re.IGNORECASE),
            re.compile(r'\bcalculate\s+\d+', re.IGNORECASE),
        ]
        
        # General patterns that should stay offline
        self.offline_preferred_patterns = [
            re.compile(r'\b(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))\b', re.IGNORECASE),
            re.compile(r'\bhow\s+are\s+you\b', re.IGNORECASE),
            re.compile(r'\bexplain\s+(?!.*(?:current|latest|today|now))', re.IGNORECASE),
            re.compile(r'\bwhat\s+is\s+(?!.*(?:current|latest|today|now))', re.IGNORECASE),
            re.compile(r'\bwrite\s+(?:a|some|code|function|program)', re.IGNORECASE),
            re.compile(r'\bhelp\s+(?:me\s+)?(?:with|understand)', re.IGNORECASE),
        ]
    
    async def _check_llm_availability(self):
        """Enhanced availability check with skills support"""
        try:
            if settings.debug_mode:
                print("[ROUTER] Enhanced availability check...")
            
            # Initialize offline LLM (Nemotron) - PRIORITY for speed
            await self._initialize_offline_llm()
            
            # Initialize online LLM (Groq) for current info
            await self._initialize_online_llm()
            
            # Initialize skills manager for instant responses
            await self._initialize_skills_manager()
            
            # Set optimal routing mode
            self._set_optimal_mode()
            
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Router availability check failed: {e}")
    
    async def _initialize_offline_llm(self):
        """Initialize offline LLM with error handling"""
        try:
            from modules.offline_llm import LightningOfflineLLM
            self.offline_llm = LightningOfflineLLM()
            
            # Set to speed profile for maximum performance
            self.offline_llm.set_performance_profile('speed')
            
            self.offline_available = await self.offline_llm.initialize()
            
            if self.offline_available:
                if settings.debug_mode:
                    print("✅ Offline LLM ready (Nemotron - Ultra-fast mode)")
            else:
                if settings.debug_mode:
                    print("❌ Offline LLM not available")
                    print(f"   Error: {self.offline_llm.last_error}")
                    
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Offline LLM initialization error: {e}")
            self.offline_available = False
            self.offline_llm = None
    
    async def _initialize_online_llm(self):
        """Initialize online LLM with error handling"""
        if settings.is_online_available():
            try:
                from modules.online_llm import OnlineLLM
                self.online_llm = OnlineLLM()
                self.online_available = await self.online_llm.initialize()
                
                if self.online_available:
                    if settings.debug_mode:
                        print("✅ Online LLM ready (Groq - Current info enabled)")
                else:
                    if settings.debug_mode:
                        print("❌ Online LLM not available")
                        
            except Exception as e:
                if settings.debug_mode:
                    print(f"❌ Online LLM initialization error: {e}")
                self.online_available = False
                self.online_llm = None
        else:
            self.online_available = False
            if settings.debug_mode:
                print("[ROUTER] No Groq API key - current info limited")
    
    async def _initialize_skills_manager(self):
        """Initialize skills manager for instant responses"""
        try:
            from modules.skills_manager import EnhancedSkillsManager
            self.skills_manager = EnhancedSkillsManager()
            
            # Initialize and check availability
            api_status = await self.skills_manager.initialize()
            self.skills_available = True  # Basic skills always available
            
            if settings.debug_mode:
                available_apis = sum(1 for status in api_status.values() if status['available'])
                print(f"✅ Skills manager ready ({available_apis} APIs configured)")
                
        except Exception as e:
            if settings.debug_mode:
                print(f"⚠️ Skills manager unavailable: {e}")
            self.skills_available = False
            self.skills_manager = None
    
    def _set_optimal_mode(self):
        """Set optimal routing mode based on available systems"""
        if self.offline_available and self.online_available and self.skills_available:
            self.mode = RouteMode.BALANCED
            if settings.debug_mode:
                print("🚀 BALANCED MODE: Skills + Nemotron + Groq (optimal)")
        elif self.offline_available and self.online_available:
            self.mode = RouteMode.BALANCED
            if settings.debug_mode:
                print("🚀 BALANCED MODE: Nemotron + Groq")
        elif self.offline_available:
            self.mode = RouteMode.OFFLINE_ONLY
            if settings.debug_mode:
                print("🏠 OFFLINE MODE: Nemotron only")
        elif self.online_available:
            self.mode = RouteMode.ONLINE_ONLY
            if settings.debug_mode:
                print("🌐 ONLINE MODE: Groq only")
        else:
            if settings.debug_mode:
                print("❌ NO SYSTEMS AVAILABLE")
    
    def _detect_instant_skill(self, query: str) -> Optional[str]:
        """Detect if query can be handled by instant skills"""
        if not self.skills_available or not self.skills_manager:
            return None
        
        # Use skills manager's detection
        return self.skills_manager.can_handle_directly(query)
    
    def _detect_current_info_enhanced(self, query: str) -> bool:
        """Enhanced current info detection with better accuracy"""
        query_lower = query.strip().lower()
        
        # Check against compiled patterns
        for pattern in self.current_info_patterns:
            if pattern.search(query_lower):
                return True
        
        # Check for temporal indicators combined with information requests
        temporal_words = ['today', 'now', 'current', 'latest', 'recent', 'this']
        info_words = ['news', 'weather', 'time', 'date', 'president', 'events', 'happening']
        
        query_words = set(query_lower.split())
        has_temporal = any(word in query_words for word in temporal_words)
        has_info_request = any(word in query_words for word in info_words)
        
        if has_temporal and has_info_request:
            return True
        
        # Avoid false positives for educational/explanatory queries
        educational_indicators = ['explain', 'definition', 'what is', 'how does', 'why', 'example']
        if any(indicator in query_lower for indicator in educational_indicators):
            # Only consider current info if explicitly mentioned
            explicit_current = ['current', 'now', 'today', 'latest']
            return any(word in query_words for word in explicit_current)
        
        return False
    
    def _should_prefer_offline(self, query: str) -> bool:
        """Check if query should prefer offline processing"""
        query_lower = query.lower()
        
        # Check offline-preferred patterns
        for pattern in self.offline_preferred_patterns:
            if pattern.search(query_lower):
                return True
        
        # Programming and technical queries
        tech_indicators = ['code', 'function', 'program', 'algorithm', 'python', 'javascript', 'sql']
        if any(indicator in query_lower for indicator in tech_indicators):
            return True
        
        # Educational content
        educational_indicators = ['explain', 'how to', 'tutorial', 'example', 'concept']
        if any(indicator in query_lower for indicator in educational_indicators):
            # Unless it's asking for current information
            if not self._detect_current_info_enhanced(query):
                return True
        
        return False
    
    def _decide_route_enhanced(self, query: str) -> EnhancedRouteDecision:
        """Enhanced routing decision with improved logic"""
        self.stats['routing_decisions'] += 1
        
        # Priority 1: Check for instant skills (fastest response)
        skill_name = self._detect_instant_skill(query)
        if skill_name:
            return EnhancedRouteDecision(
                route_type='skill',
                reason=f"Instant {skill_name} skill",
                skill_name=skill_name,
                confidence=0.95,
                expected_time=0.1
            )
        
        # Priority 2: Check for current information needs
        needs_current_info = self._detect_current_info_enhanced(query)
        if needs_current_info:
            if self.online_available:
                return EnhancedRouteDecision(
                    route_type='online',
                    reason="Current information query",
                    is_current_info=True,
                    confidence=0.9,
                    expected_time=3.0
                )
            elif self.offline_available:
                # Fallback to offline with note about limitations
                return EnhancedRouteDecision(
                    route_type='offline',
                    reason="Current info (limited - no online access)",
                    is_current_info=True,
                    confidence=0.4,
                    expected_time=2.0
                )
        
        # Priority 3: Check for offline-preferred queries
        if self._should_prefer_offline(query) and self.offline_available:
            return EnhancedRouteDecision(
                route_type='offline',
                reason="Offline-preferred query type",
                confidence=0.8,
                expected_time=1.5
            )
        
        # Priority 4: Default routing based on available systems
        if self.mode == RouteMode.OFFLINE_ONLY or (not self.online_available and self.offline_available):
            return EnhancedRouteDecision(
                route_type='offline',
                reason="Default to offline (fastest)",
                confidence=0.7,
                expected_time=1.5
            )
        elif self.mode == RouteMode.ONLINE_ONLY or (not self.offline_available and self.online_available):
            return EnhancedRouteDecision(
                route_type='online',
                reason="Online only available",
                confidence=0.7,
                expected_time=3.0
            )
        elif self.offline_available:
            # Prefer offline for speed in balanced mode
            return EnhancedRouteDecision(
                route_type='offline',
                reason="Default to offline for speed",
                confidence=0.6,
                expected_time=1.5
            )
        elif self.online_available:
            return EnhancedRouteDecision(
                route_type='online',
                reason="Fallback to online",
                confidence=0.5,
                expected_time=3.0
            )
        
        # No systems available
        return EnhancedRouteDecision(
            route_type='none',
            reason="No systems available",
            confidence=0.0,
            expected_time=0.0
        )
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """Enhanced streaming response with improved routing"""
        decision = self._decide_route_enhanced(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            route_display = decision.route_type.upper()
            if decision.skill_name:
                route_display = f"{decision.skill_name.upper()} SKILL"
            print(f"[ROUTER] 🚀 Route: {route_display} - {decision.reason} (confidence: {decision.confidence:.2f})")
        
        start_time = time.time()
        
        try:
            # SKILLS ROUTE (Instant responses)
            if decision.use_skill and self.skills_manager:
                try:
                    result = await self.skills_manager.execute_skill(query, decision.skill_name)
                    if result and result.success:
                        yield result.response
                        self._update_stats('skill', time.time() - start_time, True)
                        return
                    else:
                        # Skill failed, fall back to offline
                        if settings.debug_mode:
                            print(f"[ROUTER] ⚠️ Skill failed, falling back to offline")
                        if self.offline_available:
                            decision = EnhancedRouteDecision('offline', 'Skill fallback', confidence=0.6)
                        elif self.online_available:
                            decision = EnhancedRouteDecision('online', 'Skill fallback', confidence=0.6)
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ROUTER] ❌ Skill error: {e}")
                    # Continue to LLM fallback
            
            # OFFLINE ROUTE (Fast local responses)
            if decision.use_offline and self.offline_llm:
                # Get optimized context
                personality_context = ""
                memory_context = ""
                
                # Only add context for complex queries to maintain speed
                if decision.confidence < 0.8 or len(query.split()) > 10:
                    personality_context = await self.personality_manager.get_system_prompt()
                    memory_context = await self.memory_manager.get_context()
                
                async for chunk in self.offline_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    yield chunk
                
                self._update_stats('offline', time.time() - start_time, True)
                return
            
            # ONLINE ROUTE (Current information)
            if decision.use_online and self.online_llm:
                # Get full context for online queries (they need it for current info)
                personality_context = await self.personality_manager.get_system_prompt()
                memory_context = await self.memory_manager.get_context()
                
                async for chunk in self.online_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    yield chunk
                
                self._update_stats('online', time.time() - start_time, True)
                return
            
            # No systems available
            yield "I'm sorry, but I'm unable to process your request right now. Please check that Pascal's systems are properly configured."
            self._update_stats('none', time.time() - start_time, False)
            
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Streaming error in {decision.route_type}: {e}")
            yield f"I'm experiencing technical difficulties: {str(e)[:100]}"
            self._update_stats(decision.route_type, time.time() - start_time, False)
    
    async def get_response(self, query: str) -> str:
        """Enhanced non-streaming response"""
        response_parts = []
        async for chunk in self.get_streaming_response(query):
            response_parts.append(chunk)
        return ''.join(response_parts)
    
    def _update_stats(self, route_type: str, response_time: float, success: bool):
        """Enhanced stats tracking"""
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
    
    # Legacy method aliases for compatibility
    def _needs_current_information(self, query: str) -> bool:
        """Legacy alias for current info detection"""
        return self._detect_current_info_enhanced(query)
    
    def _decide_route(self, query: str) -> EnhancedRouteDecision:
        """Legacy alias for route decision"""
        return self._decide_route_enhanced(query)
    
    def _decide_route_fast(self, query: str) -> EnhancedRouteDecision:
        """Legacy alias for fast route decision"""
        return self._decide_route_enhanced(query)
    
    def _detect_current_info_fast(self, query: str) -> bool:
        """Legacy alias for current info detection"""
        return self._detect_current_info_enhanced(query)
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get comprehensive router statistics"""
        total_requests = self.stats['offline_requests'] + self.stats['online_requests'] + self.stats['skill_requests']
        
        if total_requests > 0:
            offline_percentage = (self.stats['offline_requests'] / total_requests) * 100
            online_percentage = (self.stats['online_requests'] / total_requests) * 100
            skill_percentage = (self.stats['skill_requests'] / total_requests) * 100
            routing_accuracy = (self.stats['correct_routes'] / total_requests) * 100
        else:
            offline_percentage = online_percentage = skill_percentage = routing_accuracy = 0
        
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
            'routing_strategy': 'enhanced_smart_routing',
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
                'offline_percentage': f"{offline_percentage:.1f}%",
                'online_percentage': f"{online_percentage:.1f}%",
                'skill_percentage': f"{skill_percentage:.1f}%",
                'offline_avg_time': f"{offline_avg:.2f}s",
                'online_avg_time': f"{online_avg:.2f}s",
                'skill_avg_time': f"{skill_avg:.3f}s"
            },
            'optimizations': [
                'Enhanced current info detection',
                'Instant skills routing',
                'Context-aware prompt optimization',
                'Fallback handling',
                'Speed-prioritized offline routing'
            ],
            'recommendations': self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get performance recommendations"""
        recommendations = []
        
        if not self.offline_available:
            recommendations.append("Install Ollama and Nemotron for fast offline responses")
        if not self.online_available:
            recommendations.append("Configure Groq API key for current information queries")
        if not self.skills_available:
            recommendations.append("Skills manager could be optimized")
        
        total_requests = self.stats['offline_requests'] + self.stats['online_requests'] + self.stats['skill_requests']
        if total_requests > 10:
            if self.stats['skill_requests'] < total_requests * 0.1:
                recommendations.append("Consider using instant skills for common queries")
            if self.stats['offline_requests'] < total_requests * 0.6:
                recommendations.append("More queries could use fast offline processing")
        
        return recommendations
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        health_score = 0
        components = {}
        
        # Offline LLM (40% of health)
        if self.offline_available:
            health_score += 40
            components['offline_llm'] = 'Available and optimized'
        else:
            components['offline_llm'] = 'Unavailable'
        
        # Online LLM (30% of health)
        if self.online_available:
            health_score += 30
            components['online_llm'] = 'Available with current info'
        else:
            components['online_llm'] = 'Unavailable (limited current info)'
        
        # Skills Manager (20% of health)
        if self.skills_available:
            health_score += 20
            components['skills_manager'] = 'Available for instant responses'
        else:
            components['skills_manager'] = 'Unavailable'
        
        # Routing Performance (10% of health)
        total_requests = self.stats['offline_requests'] + self.stats['online_requests'] + self.stats['skill_requests']
        if total_requests > 0:
            routing_accuracy = (self.stats['correct_routes'] / total_requests) * 100
            if routing_accuracy > 80:
                health_score += 10
                components['routing_performance'] = f'Excellent ({routing_accuracy:.1f}% accuracy)'
            elif routing_accuracy > 60:
                health_score += 5
                components['routing_performance'] = f'Good ({routing_accuracy:.1f}% accuracy)'
            else:
                components['routing_performance'] = f'Needs improvement ({routing_accuracy:.1f}% accuracy)'
        else:
            health_score += 5  # Default score for new system
            components['routing_performance'] = 'Not yet measured'
        
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
            'performance_summary': {
                'total_requests': total_requests,
                'routing_decisions': self.stats['routing_decisions'],
                'systems_available': sum([self.offline_available, self.online_available, self.skills_available]),
                'routing_efficiency': 'High (enhanced algorithms)'
            },
            'recommendations': self._get_recommendations()
        }
    
    async def close(self):
        """Enhanced cleanup of all systems"""
        if self.offline_llm:
            await self.offline_llm.close()
        if self.online_llm:
            await self.online_llm.close()
        if self.skills_manager:
            await self.skills_manager.close()

# Maintain compatibility
EnhancedRouter = LightningRouter
