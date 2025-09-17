"""
Pascal AI Assistant - STREAMLINED Fast Router
Minimal overhead routing optimized for <3 second responses
FOCUS: Fast offline routing with minimal processing delays
"""

import asyncio
import time
import re
from typing import Optional, AsyncGenerator, Dict, Any, List  # Added List import
from enum import Enum

from config.settings import settings

class RouteMode(Enum):
    """Simplified routing modes"""
    FAST_OFFLINE = "fast_offline"     # Nemotron-first for speed
    BALANCED = "balanced"             # Smart routing
    ONLINE_ONLY = "online_only"       # Groq only

class FastRouteDecision:
    """Lightweight routing decision"""
    def __init__(self, route_type: str, reason: str, is_current_info: bool = False, 
                 confidence: float = 0.8):
        self.route_type = route_type  # 'offline', 'online'
        self.reason = reason
        self.is_current_info = is_current_info
        self.confidence = confidence
        self.timestamp = time.time()
    
    @property
    def use_offline(self) -> bool:
        return self.route_type == 'offline'
    
    @property
    def use_online(self) -> bool:
        return self.route_type == 'online'

class LightningRouter:
    """STREAMLINED fast router with minimal overhead"""
    
    def __init__(self, personality_manager, memory_manager):
        self.personality_manager = personality_manager
        self.memory_manager = memory_manager
        
        # Initialize LLM components
        self.offline_llm = None
        self.online_llm = None
        
        # Router state
        self.mode = RouteMode.FAST_OFFLINE  # Default to fast offline
        self.last_decision = None
        self.offline_available = False
        self.online_available = False
        
        # Minimal performance tracking
        self.stats = {
            'offline_requests': 0,
            'online_requests': 0,
            'offline_avg_time': 0.0,
            'online_avg_time': 0.0,
        }
        
        # Fast current info patterns (compiled regex for speed)
        self.current_info_patterns = [
            re.compile(r'\bwhat time is it\b', re.IGNORECASE),
            re.compile(r'\bwhat day is today\b', re.IGNORECASE),
            re.compile(r'\bwhat date is today\b', re.IGNORECASE),
            re.compile(r'\bcurrent time\b', re.IGNORECASE),
            re.compile(r'\bcurrent date\b', re.IGNORECASE),
            re.compile(r'\btoday\'?s date\b', re.IGNORECASE),
            re.compile(r'\bcurrent president\b', re.IGNORECASE),
            re.compile(r'\bwho is president\b', re.IGNORECASE),
            re.compile(r'\blatest news\b', re.IGNORECASE),
            re.compile(r'\brecent news\b', re.IGNORECASE),
            re.compile(r'\bnews today\b', re.IGNORECASE),
            re.compile(r'\bbreaking news\b', re.IGNORECASE),
            re.compile(r'\bwhat\'?s happening\b', re.IGNORECASE),
            re.compile(r'\bweather today\b', re.IGNORECASE),
            re.compile(r'\bcurrent weather\b', re.IGNORECASE),
        ]
    
    async def _check_llm_availability(self):
        """FAST availability check with minimal overhead"""
        try:
            if settings.debug_mode:
                print("[ROUTER] Fast availability check...")
            
            # Initialize offline LLM (Nemotron) - PRIORITY for speed
            try:
                from modules.offline_llm import LightningOfflineLLM
                self.offline_llm = LightningOfflineLLM()
                
                # Set to speed profile immediately for fastest responses
                self.offline_llm.set_performance_profile('speed')
                
                self.offline_available = await self.offline_llm.initialize()
                
                if self.offline_available:
                    if settings.debug_mode:
                        print("✅ Offline LLM ready (Nemotron - SPEED mode)")
                else:
                    if settings.debug_mode:
                        print("❌ Offline LLM not available")
                        
            except Exception as e:
                if settings.debug_mode:
                    print(f"❌ Offline LLM error: {e}")
                self.offline_available = False
                self.offline_llm = None
            
            # Initialize online LLM (Groq) - only if offline fails or for current info
            if settings.is_online_available():
                try:
                    from modules.online_llm import OnlineLLM
                    self.online_llm = OnlineLLM()
                    self.online_available = await self.online_llm.initialize()
                    
                    if self.online_available:
                        if settings.debug_mode:
                            print("✅ Online LLM ready (Groq - for current info)")
                    else:
                        if settings.debug_mode:
                            print("❌ Online LLM not available")
                            
                except Exception as e:
                    if settings.debug_mode:
                        print(f"❌ Online LLM error: {e}")
                    self.online_available = False
                    self.online_llm = None
            else:
                self.online_available = False
                if settings.debug_mode:
                    print("[ROUTER] No Groq API key - limited current info")
            
            # Set optimal mode
            self._set_fast_mode()
            
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Router availability check failed: {e}")
    
    def _set_fast_mode(self):
        """Set routing mode for maximum speed"""
        if self.offline_available and self.online_available:
            self.mode = RouteMode.BALANCED
            if settings.debug_mode:
                print("🚀 FAST MODE: Nemotron-first with Groq fallback")
        elif self.offline_available:
            self.mode = RouteMode.FAST_OFFLINE
            if settings.debug_mode:
                print("🏠 OFFLINE MODE: Nemotron only (ultra-fast)")
        elif self.online_available:
            self.mode = RouteMode.ONLINE_ONLY
            if settings.debug_mode:
                print("🌐 ONLINE MODE: Groq only")
        else:
            if settings.debug_mode:
                print("❌ NO LLMs AVAILABLE")
    
    def _detect_current_info_fast(self, query: str) -> bool:
        """FAST current info detection using compiled regex"""
        # Check against compiled patterns for speed
        for pattern in self.current_info_patterns:
            if pattern.search(query):
                return True
        
        # Quick word check for additional triggers
        query_lower = query.lower()
        quick_triggers = ['today', 'now', 'current', 'latest', 'recent']
        
        # Avoid false positives
        if any(avoid in query_lower for avoid in ['explain', 'definition', 'what is', 'how does']):
            return False
        
        return any(trigger in query_lower for trigger in quick_triggers)
    
    def _decide_route_fast(self, query: str) -> FastRouteDecision:
        """FAST routing decision with minimal overhead"""
        
        # Fast current info detection
        is_current_info = self._detect_current_info_fast(query)
        
        if is_current_info and self.online_available:
            # Route current info to online LLM
            return FastRouteDecision(
                'online',
                "Current info query",
                is_current_info=True,
                confidence=0.9
            )
        
        # Default to offline for maximum speed
        if self.offline_available:
            return FastRouteDecision(
                'offline',
                "General query - fast offline",
                confidence=0.8
            )
        
        # Fallback to online if no offline
        if self.online_available:
            return FastRouteDecision(
                'online',
                "Fallback to online",
                confidence=0.7
            )
        
        # No systems available
        return FastRouteDecision(
            'none',
            "No systems available",
            confidence=0.0
        )
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """FAST streaming response with minimal routing overhead"""
        decision = self._decide_route_fast(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            route_name = "NEMOTRON" if decision.use_offline else "GROQ" 
            print(f"[ROUTER] 🚀 Fast route: {route_name} - {decision.reason}")
        
        start_time = time.time()
        
        try:
            # FAST OFFLINE ROUTE (PRIORITY)
            if decision.use_offline and self.offline_llm:
                # Get minimal context for speed
                personality_context = ""  # Skip for speed
                memory_context = ""       # Skip for speed
                
                # For current info queries with offline, add minimal note
                if decision.is_current_info:
                    personality_context = "Be helpful with current information."
                
                async for chunk in self.offline_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    yield chunk
                    
                self._update_stats('offline', time.time() - start_time)
                return
            
            # ONLINE ROUTE
            if decision.use_online and self.online_llm:
                # Get context only for online (current info needs context)
                personality_context = await self.personality_manager.get_system_prompt()
                memory_context = await self.memory_manager.get_context()
                
                async for chunk in self.online_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    yield chunk
                    
                self._update_stats('online', time.time() - start_time)
                return
            
            # No systems available
            yield "I'm sorry, but I'm unable to process your request right now. Please check that Pascal's systems are properly configured."
            
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Streaming error in {decision.route_type}: {e}")
            yield f"I'm experiencing technical difficulties: {str(e)[:100]}"
    
    async def get_response(self, query: str) -> str:
        """Fast non-streaming response"""
        response_parts = []
        async for chunk in self.get_streaming_response(query):
            response_parts.append(chunk)
        return ''.join(response_parts)
    
    def _needs_current_information(self, query: str) -> bool:
        """Check if query needs current information (alias for compatibility)"""
        return self._detect_current_info_fast(query)
    
    def _decide_route(self, query: str) -> FastRouteDecision:
        """Route decision (alias for compatibility)"""
        return self._decide_route_fast(query)
    
    def _update_stats(self, route_type: str, response_time: float):
        """Minimal stats tracking"""
        if route_type == 'offline':
            self.stats['offline_requests'] += 1
            current_avg = self.stats['offline_avg_time']
            count = self.stats['offline_requests']
            self.stats['offline_avg_time'] = ((current_avg * (count - 1)) + response_time) / count
            
        elif route_type == 'online':
            self.stats['online_requests'] += 1
            current_avg = self.stats['online_avg_time']
            count = self.stats['online_requests']
            self.stats['online_avg_time'] = ((current_avg * (count - 1)) + response_time) / count
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get minimal router stats"""
        total_requests = self.stats['offline_requests'] + self.stats['online_requests']
        
        if total_requests > 0:
            offline_percentage = (self.stats['offline_requests'] / total_requests) * 100
            online_percentage = (self.stats['online_requests'] / total_requests) * 100
        else:
            offline_percentage = online_percentage = 0
        
        return {
            'mode': self.mode.value,
            'system_status': {
                'offline_llm': self.offline_available,
                'online_llm': self.online_available,
            },
            'routing_strategy': 'fast_offline_first',
            'last_decision': {
                'route_type': self.last_decision.route_type,
                'reason': self.last_decision.reason,
                'confidence': self.last_decision.confidence
            } if self.last_decision else None,
            'performance_stats': {
                'total_requests': total_requests,
                'offline_requests': self.stats['offline_requests'],
                'online_requests': self.stats['online_requests'],
                'offline_percentage': f"{offline_percentage:.1f}%",
                'online_percentage': f"{online_percentage:.1f}%",
                'offline_avg_time': f"{self.stats['offline_avg_time']:.2f}s",
                'online_avg_time': f"{self.stats['online_avg_time']:.2f}s"
            },
            'optimizations': [
                'Compiled regex patterns',
                'Minimal context for speed',
                'Fast offline prioritization',
                'Reduced routing overhead'
            ]
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health (simplified)"""
        health_score = 0
        
        if self.offline_available:
            health_score += 70  # Offline is primary
        if self.online_available:
            health_score += 30  # Online is secondary
        
        return {
            'overall_health_score': health_score,
            'system_status': 'Excellent' if health_score >= 90 else 'Good' if health_score >= 70 else 'Fair' if health_score >= 50 else 'Poor',
            'components': {
                'offline_llm': 'Available' if self.offline_available else 'Unavailable',
                'online_llm': 'Available' if self.online_available else 'Unavailable'
            },
            'performance_summary': {
                'total_requests': self.stats['offline_requests'] + self.stats['online_requests'],
                'avg_offline_time': f"{self.stats['offline_avg_time']:.2f}s",
                'routing_efficiency': 'High (minimal overhead)'
            },
            'recommendations': [
                'System optimized for speed',
                'Offline-first routing active',
                'Minimal processing overhead'
            ]
        }
    
    async def close(self):
        """Close all systems"""
        if self.offline_llm:
            await self.offline_llm.close()
        if self.online_llm:
            await self.online_llm.close()

# Maintain compatibility - Alias for EnhancedRouter
EnhancedRouter = LightningRouter
