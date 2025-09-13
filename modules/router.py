"""
Pascal AI Assistant - FIXED Router
Binary routing logic: Current info = Online (Groq), Everything else = Offline (Ollama)
FIXED: Correct import and class names
"""

import asyncio
import time
import re
from typing import Optional, AsyncGenerator
from enum import Enum
from datetime import datetime

from config.settings import settings

class RouteMode(Enum):
    """Routing modes"""
    OFFLINE_ONLY = "offline_only"
    ONLINE_ONLY = "online_only"
    AUTO = "auto"

class RouteDecision:
    """Simple routing decision"""
    def __init__(self, use_offline: bool, reason: str, is_current_info: bool = False):
        self.use_offline = use_offline
        self.use_online = not use_offline
        self.reason = reason
        self.is_current_info = is_current_info
        self.timestamp = time.time()

class LightningRouter:
    """FIXED: Simplified router with correct imports"""
    
    def __init__(self, personality_manager, memory_manager):
        self.personality_manager = personality_manager
        self.memory_manager = memory_manager
        
        # Initialize LLM modules
        self.offline_llm = None
        self.online_llm = None
        
        # Router state
        self.mode = RouteMode.AUTO
        self.last_decision = None
        self.offline_available = False
        self.online_available = False
        
        # Performance tracking
        self.response_times = {'offline': [], 'online': []}
        self.stats = {
            'offline_requests': 0,
            'online_requests': 0,
            'offline_avg_time': 0.0,
            'online_avg_time': 0.0,
            'current_info_requests': 0
        }
        
        # Current info patterns (comprehensive but focused)
        self.current_info_patterns = [
            # Date/time queries
            'what day is today', 'what date is today', 'what time is it',
            'current date', 'current time', 'todays date', "today's date",
            'what is today', 'tell me the date', 'what day is it',
            
            # Current status queries  
            'current president', 'current prime minister', 'current pm',
            'who is the current', 'current leader', 'current government',
            
            # News and events
            'latest news', 'recent news', 'news today', 'breaking news',
            'current events', "what's happening", 'in the news',
            
            # Weather
            'weather today', 'current weather', 'weather now',
        ]
        
        # Compile regex patterns for efficiency
        self.current_info_regex = [
            re.compile(r'\bwhat\s+(?:day|date|time)\s+(?:is\s+)?(?:it|today)\b', re.IGNORECASE),
            re.compile(r'\bcurrent\s+(?:date|time|president|pm|prime\s+minister|weather)\b', re.IGNORECASE),
            re.compile(r'\btoday\'?s?\s+(?:date|news|weather)\b', re.IGNORECASE),
            re.compile(r'\b(?:latest|recent|breaking)\s+news\b', re.IGNORECASE),
        ]
    
    async def _check_llm_availability(self):
        """FIXED: Check LLM availability with correct imports"""
        try:
            if settings.debug_mode:
                print("[ROUTER] Checking LLM availability...")
            
            # Initialize offline LLM first (for fallback)
            try:
                # FIXED: Import the correct class
                from modules.offline_llm import LightningOfflineLLM
                self.offline_llm = LightningOfflineLLM()
                self.offline_available = await self.offline_llm.initialize()
                
                if self.offline_available:
                    print("✅ Offline LLM ready (Ollama)")
                else:
                    print("❌ Offline LLM not available")
                    
            except Exception as e:
                print(f"❌ Offline LLM initialization failed: {e}")
                if settings.debug_mode:
                    import traceback
                    traceback.print_exc()
                self.offline_available = False
            
            # Initialize online LLM - CRITICAL FOR CURRENT INFO (Groq only)
            self.online_available = False
            
            if settings.is_online_available():
                try:
                    from modules.online_llm import OnlineLLM
                    self.online_llm = OnlineLLM()
                    self.online_available = await self.online_llm.initialize()
                    
                    if self.online_available:
                        print("✅ Online LLM ready (Groq - CURRENT INFO ENABLED)")
                    else:
                        print("❌ Online LLM not available")
                        print("⚠️ CURRENT INFO QUERIES WILL NOT WORK PROPERLY")
                        
                except Exception as e:
                    print(f"⚠️ Online LLM initialization failed: {e}")
                    if settings.debug_mode:
                        import traceback
                        traceback.print_exc()
                    self.online_available = False
                    print("⚠️ CURRENT INFO QUERIES WILL NOT WORK PROPERLY")
            else:
                if settings.debug_mode:
                    print("[ROUTER] No Groq API key configured - CURRENT INFO DISABLED")
                    print("   For current info queries, add API key to .env:")
                    print("   GROQ_API_KEY=gsk_your-actual-key")
                self.online_available = False
            
            # Adjust routing mode based on availability
            if self.online_available and not self.offline_available:
                self.mode = RouteMode.ONLINE_ONLY
                print("ℹ️ Running in online-only mode (Groq)")
            elif self.offline_available and not self.online_available:
                self.mode = RouteMode.OFFLINE_ONLY
                print("⚠️ Running in offline-only mode (Ollama) - CURRENT INFO QUERIES DISABLED")
                print("   Add Groq API key to enable current information queries")
            elif self.offline_available and self.online_available:
                # Both available - prefer online for current info, offline for general
                self.mode = RouteMode.AUTO
                print("✅ Both offline and online LLMs available")
                print("🎯 Current info queries will automatically use online")
            else:
                print("❌ ERROR: No LLMs available!")
                print("Solutions:")
                print("1. For offline: sudo systemctl start ollama && ./download_models.sh")
                print("2. For current info: Configure GROQ_API_KEY in .env file")
            
            if settings.debug_mode:
                print(f"[ROUTER] Final status - Offline: {self.offline_available}, Online: {self.online_available}")
                print(f"[ROUTER] Routing mode: {self.mode.value}")
            
        except Exception as e:
            print(f"❌ Critical error in LLM availability check: {e}")
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
    
    def _needs_current_information(self, query: str) -> bool:
        """Enhanced detection of queries requiring current/recent information"""
        query_lower = query.lower().strip()
        
        # Remove punctuation for cleaner matching
        query_clean = re.sub(r'[^\w\s]', '', query_lower)
        
        # PRIORITY 1: Exact phrase matching (most reliable)
        for pattern in self.current_info_patterns:
            if pattern in query_lower:
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 CURRENT INFO DETECTED - exact pattern: '{pattern}' in '{query}'")
                return True
        
        # PRIORITY 2: Flexible regex patterns
        for pattern in self.current_info_regex:
            if pattern.search(query_lower):
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 CURRENT INFO DETECTED - regex match")
                return True
        
        # PRIORITY 3: Word-based detection for edge cases
        current_info_words = [
            (['what', 'is'], ['today', 'date', 'day', 'time', 'todays']),
            (['what'], ['day', 'date', 'time', 'today', 'todays']),
            (['tell', 'me'], ['today', 'date', 'day', 'time', 'todays']),
            (['current'], ['president', 'pm', 'minister', 'leader', 'date', 'time', 'day']),
            (['news'], ['today', 'latest', 'recent', 'breaking', 'current']),
            (['weather'], ['today', 'current', 'now']),
        ]
        
        query_words = query_clean.split()
        for primary_words, context_words in current_info_words:
            # Check if all primary words are present
            if all(word in query_words for word in primary_words):
                # Check if any context words are present
                if any(word in query_words for word in context_words):
                    if settings.debug_mode:
                        print(f"[ROUTER] 🎯 CURRENT INFO DETECTED - word combination")
                    return True
        
        return False
    
    def _decide_route(self, query: str) -> RouteDecision:
        """Binary routing decision"""
        # CRITICAL CHECK: Current information queries ALWAYS go online (highest priority)
        is_current_info = self._needs_current_information(query)
        
        if is_current_info:
            if not self.online_available:
                return RouteDecision(
                    False,  # Still try online even if not available - will trigger error message
                    "CURRENT INFO REQUIRED - no online LLM available (CRITICAL ERROR)",
                    is_current_info=True
                )
            else:
                return RouteDecision(
                    False,  # Use online
                    "CURRENT INFO REQUIRED - mandatory online routing",
                    is_current_info=True
                )
        
        # Force offline if no online available (for non-current-info queries)
        if not self.online_available:
            return RouteDecision(True, "No online LLM available", is_current_info=False)
        
        # Force online if no offline available
        if not self.offline_available:
            return RouteDecision(False, "No offline LLM available", is_current_info=False)
        
        # Handle different routing modes
        if self.mode == RouteMode.OFFLINE_ONLY:
            return RouteDecision(True, "Offline-only mode", is_current_info=False)
        
        elif self.mode == RouteMode.ONLINE_ONLY:
            return RouteDecision(False, "Online-only mode", is_current_info=False)
        
        else:  # AUTO mode - intelligent routing
            # Simple queries go offline for speed
            simple_words = ['hello', 'hi', 'thanks', 'bye', 'yes', 'no']
            if any(word in query.lower() for word in simple_words) and len(query.split()) <= 5:
                return RouteDecision(True, "Simple greeting - offline faster", is_current_info=False)
            
            # Complex queries benefit from online
            if len(query.split()) > 25 or any(word in query.lower() for word in ['analyze', 'compare', 'research', 'detailed']):
                return RouteDecision(False, "Complex query - online better", is_current_info=False)
            
            # DEFAULT: Use offline for general queries (faster locally)
            return RouteDecision(True, "General query - using offline", is_current_info=False)
    
    async def get_response(self, query: str) -> str:
        """Get response with enhanced current info handling"""
        decision = self._decide_route(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            route_type = "Offline" if decision.use_offline else "Online"
            current_info_flag = " [CURRENT INFO]" if decision.is_current_info else ""
            print(f"[ROUTER] Decision: {route_type}{current_info_flag} - {decision.reason}")
        
        # CRITICAL: Handle current info queries that require online but online is unavailable
        if decision.is_current_info and not self.online_available:
            return ("I'm unable to provide current information right now because online services are not available. "
                   "Please configure GROQ_API_KEY in your .env file to enable current date, time, news, "
                   "and other real-time information queries.")
        
        # Get context
        personality_context = await self.personality_manager.get_system_prompt()
        memory_context = await self.memory_manager.get_context()
        
        start_time = time.time()
        
        try:
            if decision.use_offline and self.offline_llm:
                response = await self.offline_llm.generate_response(
                    query, personality_context, memory_context
                )
                route_used = 'offline'
                self._update_stats('offline', time.time() - start_time)
            elif decision.use_online and self.online_llm:
                response = await self.online_llm.generate_response(
                    query, personality_context, memory_context
                )
                route_used = 'online'
                self._update_stats('online', time.time() - start_time)
            else:
                # Fallback logic
                if self.offline_llm and not decision.is_current_info:
                    response = await self.offline_llm.generate_response(
                        query, personality_context, memory_context
                    )
                    route_used = 'offline'
                    if settings.debug_mode:
                        print("[ROUTER] Using offline fallback")
                elif self.online_llm:
                    response = await self.online_llm.generate_response(
                        query, personality_context, memory_context
                    )
                    route_used = 'online'
                    if settings.debug_mode:
                        print("[ROUTER] Using online fallback")
                else:
                    return "I'm sorry, but I'm unable to process your request right now. Please check that either Ollama is running or API keys are configured."
            
            # Track current info requests
            if decision.is_current_info:
                self.stats['current_info_requests'] += 1
            
            return response
            
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Error in {route_used if 'route_used' in locals() else 'unknown'} LLM: {e}")
            
            # Try fallback only for non-current-info queries
            if not decision.is_current_info:
                try:
                    if decision.use_offline and self.online_llm:
                        # Offline failed, try online
                        if settings.debug_mode:
                            print("[ROUTER] Offline failed, trying online fallback")
                        response = await self.online_llm.generate_response(
                            query, personality_context, memory_context
                        )
                        return response
                    elif decision.use_online and self.offline_llm:
                        # Online failed, try offline
                        if settings.debug_mode:
                            print("[ROUTER] Online failed, trying offline fallback")
                        response = await self.offline_llm.generate_response(
                            query, personality_context, memory_context
                        )
                        return response
                except Exception as fallback_error:
                    if settings.debug_mode:
                        print(f"❌ Fallback also failed: {fallback_error}")
            
            # For current info queries, provide specific error message
            if decision.is_current_info:
                return ("I'm having trouble accessing current information right now. "
                       "Please check your internet connection and Groq API key configuration, then try again.")
            
            return f"I'm having trouble processing your request right now. Error: {str(e)}"
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """Get streaming response with enhanced current info handling"""
        decision = self._decide_route(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            route_type = "Offline" if decision.use_offline else "Online"
            current_info_flag = " [CURRENT INFO]" if decision.is_current_info else ""
            print(f"[ROUTER] Decision: {route_type}{current_info_flag} - {decision.reason}")
        
        # CRITICAL: Handle current info queries that require online but online is unavailable
        if decision.is_current_info and not self.online_available:
            yield ("I'm unable to provide current information right now because online services are not available. "
                   "Please configure GROQ_API_KEY in your .env file to enable current date, time, news, "
                   "and other real-time information queries.")
            return
        
        # Get context
        personality_context = await self.personality_manager.get_system_prompt()
        memory_context = await self.memory_manager.get_context()
        
        start_time = time.time()
        response_generated = False
        
        try:
            if decision.use_offline and self.offline_llm and not decision.is_current_info:
                route_used = 'offline'
                async for chunk in self.offline_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    yield chunk
                    response_generated = True
                self._update_stats('offline', time.time() - start_time)
                    
            elif decision.use_online and self.online_llm:
                route_used = 'online'
                async for chunk in self.online_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    yield chunk
                    response_generated = True
                self._update_stats('online', time.time() - start_time)
                    
            else:
                # Fallback logic - but never fallback for current info queries
                if self.offline_llm and not decision.is_current_info:
                    route_used = 'offline'
                    if settings.debug_mode:
                        print("[ROUTER] Using offline fallback for streaming")
                    async for chunk in self.offline_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                        response_generated = True
                elif self.online_llm:
                    route_used = 'online'
                    if settings.debug_mode:
                        print("[ROUTER] Using online fallback for streaming")
                    async for chunk in self.online_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                        response_generated = True
                else:
                    yield "I'm sorry, but I'm unable to process your request right now."
                    return
            
            # Track current info requests
            if decision.is_current_info:
                self.stats['current_info_requests'] += 1
                
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Streaming error: {e}")
            
            # Try fallback (non-streaming) only for non-current-info queries
            if not decision.is_current_info:
                try:
                    if decision.use_offline and self.online_llm:
                        if settings.debug_mode:
                            print("[ROUTER] Streaming offline failed, trying online fallback")
                        response = await self.online_llm.generate_response(
                            query, personality_context, memory_context
                        )
                        yield response
                    elif decision.use_online and self.offline_llm:
                        if settings.debug_mode:
                            print("[ROUTER] Streaming online failed, trying offline fallback")
                        response = await self.offline_llm.generate_response(
                            query, personality_context, memory_context
                        )
                        yield response
                    else:
                        yield f"I'm having trouble processing your request. Error: {str(e)}"
                except Exception as fallback_error:
                    if settings.debug_mode:
                        print(f"❌ Fallback streaming failed: {fallback_error}")
                    yield f"I'm experiencing technical difficulties right now."
            else:
                # For current info queries, provide specific error message
                yield ("I'm having trouble accessing current information right now. "
                       "Please check your internet connection and Groq API key configuration, then try again.")
    
    def _update_stats(self, route_type: str, response_time: float):
        """Update performance statistics"""
        if route_type == 'online':
            self.stats['online_requests'] += 1
            # Running average
            current_avg = self.stats['online_avg_time']
            count = self.stats['online_requests']
            self.stats['online_avg_time'] = ((current_avg * (count - 1)) + response_time) / count
        else:
            self.stats['offline_requests'] += 1
            current_avg = self.stats['offline_avg_time']
            count = self.stats['offline_requests']
            self.stats['offline_avg_time'] = ((current_avg * (count - 1)) + response_time) / count
    
    def set_performance_preference(self, preference: str):
        """Set performance preference (speed/balanced/quality)"""
        if preference == 'speed':
            self.mode = RouteMode.AUTO  # Keep auto for current info detection
        elif preference == 'balanced':
            self.mode = RouteMode.AUTO
        elif preference == 'quality':
            self.mode = RouteMode.AUTO
        
        # Also pass to LLM modules if available
        if self.offline_llm and hasattr(self.offline_llm, 'set_performance_profile'):
            self.offline_llm.set_performance_profile(preference)
    
    def get_router_stats(self) -> dict:
        """Get router stats"""
        return {
            'mode': self.mode.value,
            'offline_available': self.offline_available,
            'online_available': self.online_available,
            'current_info_enabled': self.online_available,  # Current info requires online
            'supported_providers': ['Groq'],  # Only Groq now
            'last_decision': {
                'use_offline': self.last_decision.use_offline,
                'reason': self.last_decision.reason,
                'is_current_info': self.last_decision.is_current_info
            } if self.last_decision else None,
            'stats': self.stats
        }
    
    async def close(self):
        """Close all LLM connections"""
        if self.offline_llm:
            await self.offline_llm.close()
        if self.online_llm:
            await self.online_llm.close()
