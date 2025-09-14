"""
Pascal AI Assistant - Router (FIXED)
Proper routing: Current info = Groq, Everything else = Nemotron
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
    """FIXED router for Nemotron + Groq with proper current info routing"""
    
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
        self.stats = {
            'offline_requests': 0,
            'online_requests': 0,
            'offline_avg_time': 0.0,
            'online_avg_time': 0.0,
            'current_info_requests': 0
        }
        
        # ENHANCED current info patterns - more aggressive detection
        self.current_info_patterns = [
            # Date/time queries (PRIORITY)
            'what day is today', 'what date is today', 'what time is it',
            'current date', 'current time', 'todays date', "today's date",
            'what is today', 'tell me the date', 'what day is it',
            'what is the date', 'what is the time', 'date today',
            'time now', 'current day', 'what day', 'what date',
            
            # Current status queries  
            'current president', 'current prime minister', 'current pm',
            'who is the current', 'current leader', 'current government',
            'who is president now', 'current us president',
            
            # News and events (PRIORITY for Groq)
            'latest news', 'recent news', 'news today', 'breaking news',
            'current events', "what's happening", 'in the news',
            'news now', 'today news', 'current news',
            
            # Weather (PRIORITY for Groq)
            'weather today', 'current weather', 'weather now',
            'what is the weather', 'todays weather',
            
            # Other current info
            'latest', 'recent', 'current', 'now', 'today',
        ]
        
        # CRITICAL: Compile regex patterns for aggressive current info detection
        self.current_info_regex = [
            # Date/time patterns
            re.compile(r'\b(what|tell me|give me)\s+(day|date|time)\s+(is\s+)?(it|today|now)\b', re.IGNORECASE),
            re.compile(r'\bcurrent\s+(date|time|day|president|pm|prime\s+minister|weather|news)\b', re.IGNORECASE),
            re.compile(r'\btoday\'?s?\s+(date|news|weather|time)\b', re.IGNORECASE),
            re.compile(r'\b(latest|recent|breaking|current)\s+(news|events|weather|information)\b', re.IGNORECASE),
            re.compile(r'\b(what|who)\s+is\s+(the\s+)?(current|today)\b', re.IGNORECASE),
            re.compile(r'\bweather\s+(today|now|currently)\b', re.IGNORECASE),
            re.compile(r'\bnews\s+(today|now|latest|current)\b', re.IGNORECASE),
            # Time-sensitive phrases
            re.compile(r'\bright now\b', re.IGNORECASE),
            re.compile(r'\bat the moment\b', re.IGNORECASE),
            re.compile(r'\bcurrently\b', re.IGNORECASE),
        ]
    
    async def _check_llm_availability(self):
        """Check LLM availability - simplified for Nemotron + Groq"""
        try:
            if settings.debug_mode:
                print("[ROUTER] Checking LLM availability...")
            
            # Initialize offline LLM (Nemotron via Ollama)
            try:
                from modules.offline_llm import LightningOfflineLLM
                self.offline_llm = LightningOfflineLLM()
                self.offline_available = await self.offline_llm.initialize()
                
                if self.offline_available:
                    print("✅ Offline LLM ready (Nemotron via Ollama)")
                else:
                    print("❌ Offline LLM not available (check Ollama)")
                    
            except Exception as e:
                print(f"❌ Offline LLM initialization failed: {e}")
                if settings.debug_mode:
                    import traceback
                    traceback.print_exc()
                self.offline_available = False
            
            # Initialize online LLM (Groq only) - CRITICAL FOR CURRENT INFO
            self.online_available = False
            
            if settings.is_online_available():
                try:
                    from modules.online_llm import OnlineLLM
                    self.online_llm = OnlineLLM()
                    self.online_available = await self.online_llm.initialize()
                    
                    if self.online_available:
                        print("✅ Online LLM ready (Groq - CURRENT INFO ENABLED)")
                    else:
                        print("❌ Online LLM not available (check Groq API key)")
                        print("⚠️ CURRENT INFO QUERIES WILL NOT WORK")
                        
                except Exception as e:
                    print(f"⚠️ Online LLM initialization failed: {e}")
                    if settings.debug_mode:
                        import traceback
                        traceback.print_exc()
                    self.online_available = False
                    print("⚠️ CURRENT INFO QUERIES WILL NOT WORK")
            else:
                if settings.debug_mode:
                    print("[ROUTER] No Groq API key configured - CURRENT INFO DISABLED")
                    print("   For current info queries, add API key to .env:")
                    print("   GROQ_API_KEY=gsk_your-actual-key")
                self.online_available = False
            
            # Set routing mode based on availability
            if self.online_available and not self.offline_available:
                self.mode = RouteMode.ONLINE_ONLY
                print("ℹ️ Running in online-only mode (Groq)")
            elif self.offline_available and not self.online_available:
                self.mode = RouteMode.OFFLINE_ONLY
                print("⚠️ Running in offline-only mode (Nemotron) - CURRENT INFO DISABLED")
                print("   Add Groq API key to enable current information queries")
            elif self.offline_available and self.online_available:
                self.mode = RouteMode.AUTO
                print("✅ Both Nemotron and Groq available")
                print("🎯 Current info queries will automatically use Groq")
            else:
                print("❌ ERROR: No LLMs available!")
                print("Solutions:")
                print("1. For offline: sudo systemctl start ollama && ollama pull nemotron-mini:4b-instruct-q4_K_M")
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
        """ENHANCED detection of current information queries - more aggressive"""
        query_lower = query.lower().strip()
        
        # Remove punctuation for cleaner matching
        query_clean = re.sub(r'[^\w\s]', '', query_lower)
        
        # PRIORITY 1: Exact phrase matching (enhanced)
        for pattern in self.current_info_patterns:
            if pattern in query_lower:
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 CURRENT INFO DETECTED - pattern: '{pattern}'")
                return True
        
        # PRIORITY 2: Regex patterns (enhanced)
        for pattern in self.current_info_regex:
            if pattern.search(query_lower):
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 CURRENT INFO DETECTED - regex match")
                return True
        
        # PRIORITY 3: Single word triggers for aggressive routing
        single_word_triggers = ['today', 'now', 'current', 'latest', 'recent']
        words = query_lower.split()
        
        for word in words:
            if word in single_word_triggers:
                # Additional context check - avoid false positives
                if any(context in query_lower for context in ['explain', 'definition', 'what is', 'how does', 'why']):
                    continue  # Skip if it's an explanation request
                
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 CURRENT INFO DETECTED - single word trigger: '{word}'")
                return True
        
        return False
    
    def _decide_route(self, query: str) -> RouteDecision:
        """FIXED routing decision: Current info = Groq, Rest = Nemotron"""
        
        # CRITICAL: Current information queries ALWAYS go to Groq
        is_current_info = self._needs_current_information(query)
        
        if is_current_info:
            if not self.online_available:
                return RouteDecision(
                    False,  # Still route to online to show proper error
                    "CURRENT INFO REQUIRED - Groq not available (ERROR)",
                    is_current_info=True
                )
            else:
                return RouteDecision(
                    False,  # Use Groq (online)
                    "CURRENT INFO DETECTED - routing to Groq",
                    is_current_info=True
                )
        
        # Handle different routing modes for non-current-info queries
        if self.mode == RouteMode.ONLINE_ONLY:
            return RouteDecision(False, "Online-only mode", is_current_info=False)
        
        elif self.mode == RouteMode.OFFLINE_ONLY:
            return RouteDecision(True, "Offline-only mode", is_current_info=False)
        
        else:  # AUTO mode
            # Force online if no offline available
            if not self.offline_available:
                return RouteDecision(False, "No Nemotron available - using Groq", is_current_info=False)
            
            # Force offline if no online available (for non-current-info)
            if not self.online_available:
                return RouteDecision(True, "No Groq available - using Nemotron", is_current_info=False)
            
            # DEFAULT: Use Nemotron for general queries (faster locally)
            return RouteDecision(True, "General query - using Nemotron for speed", is_current_info=False)
    
    async def get_response(self, query: str) -> str:
        """Get response with FIXED current info handling"""
        decision = self._decide_route(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            route_type = "GROQ" if decision.use_online else "NEMOTRON"
            current_info_flag = " [CURRENT INFO]" if decision.is_current_info else ""
            print(f"[ROUTER] 🚦 Decision: {route_type}{current_info_flag} - {decision.reason}")
        
        # CRITICAL: Handle current info queries that require Groq but Groq unavailable
        if decision.is_current_info and not self.online_available:
            return ("I'm unable to provide current information right now because Groq is not available. "
                   "Please configure GROQ_API_KEY in your .env file to enable current date, time, news, "
                   "and other real-time information queries.")
        
        # Get context
        personality_context = await self.personality_manager.get_system_prompt()
        memory_context = await self.memory_manager.get_context()
        
        start_time = time.time()
        
        try:
            if decision.use_online and self.online_llm:
                # GROQ route
                response = await self.online_llm.generate_response(
                    query, personality_context, memory_context
                )
                route_used = 'online'
                self._update_stats('online', time.time() - start_time)
                
                if settings.debug_mode:
                    print(f"[ROUTER] ✅ Used GROQ successfully")
                    
            elif decision.use_offline and self.offline_llm:
                # NEMOTRON route
                response = await self.offline_llm.generate_response(
                    query, personality_context, memory_context
                )
                route_used = 'offline'
                self._update_stats('offline', time.time() - start_time)
                
                if settings.debug_mode:
                    print(f"[ROUTER] ✅ Used NEMOTRON successfully")
                    
            else:
                # Fallback logic - try any available LLM
                if self.online_llm and decision.is_current_info:
                    # Force Groq for current info
                    response = await self.online_llm.generate_response(
                        query, personality_context, memory_context
                    )
                    route_used = 'online'
                    if settings.debug_mode:
                        print("[ROUTER] 🔄 Using Groq fallback for current info")
                        
                elif self.offline_llm:
                    # Use Nemotron as fallback
                    response = await self.offline_llm.generate_response(
                        query, personality_context, memory_context
                    )
                    route_used = 'offline'
                    if settings.debug_mode:
                        print("[ROUTER] 🔄 Using Nemotron fallback")
                        
                elif self.online_llm:
                    # Use Groq as last resort
                    response = await self.online_llm.generate_response(
                        query, personality_context, memory_context
                    )
                    route_used = 'online'
                    if settings.debug_mode:
                        print("[ROUTER] 🔄 Using Groq as last resort")
                        
                else:
                    return "I'm sorry, but I'm unable to process your request right now. Please check that either Ollama is running (for Nemotron) or Groq API key is configured."
            
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
                        if settings.debug_mode:
                            print("[ROUTER] 🔄 Nemotron failed, trying Groq fallback")
                        response = await self.online_llm.generate_response(
                            query, personality_context, memory_context
                        )
                        return response
                    elif decision.use_online and self.offline_llm:
                        if settings.debug_mode:
                            print("[ROUTER] 🔄 Groq failed, trying Nemotron fallback")
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
        """Get streaming response with FIXED current info handling"""
        decision = self._decide_route(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            route_type = "GROQ" if decision.use_online else "NEMOTRON"
            current_info_flag = " [CURRENT INFO]" if decision.is_current_info else ""
            print(f"[ROUTER] 🚦 Streaming Decision: {route_type}{current_info_flag} - {decision.reason}")
        
        # CRITICAL: Handle current info queries that require Groq but Groq unavailable
        if decision.is_current_info and not self.online_available:
            yield ("I'm unable to provide current information right now because Groq is not available. "
                   "Please configure GROQ_API_KEY in your .env file to enable current date, time, news, "
                   "and other real-time information queries.")
            return
        
        # Get context
        personality_context = await self.personality_manager.get_system_prompt()
        memory_context = await self.memory_manager.get_context()
        
        start_time = time.time()
        
        try:
            if decision.use_online and self.online_llm:
                # GROQ route
                route_used = 'online'
                if settings.debug_mode:
                    print("[ROUTER] 🌊 Streaming via GROQ")
                async for chunk in self.online_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    yield chunk
                self._update_stats('online', time.time() - start_time)
                    
            elif decision.use_offline and self.offline_llm:
                # NEMOTRON route
                route_used = 'offline'
                if settings.debug_mode:
                    print("[ROUTER] 🌊 Streaming via NEMOTRON")
                async for chunk in self.offline_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    yield chunk
                self._update_stats('offline', time.time() - start_time)
                    
            else:
                # Fallback logic
                if self.online_llm and decision.is_current_info:
                    route_used = 'online'
                    if settings.debug_mode:
                        print("[ROUTER] 🌊 Streaming via Groq fallback for current info")
                    async for chunk in self.online_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                elif self.offline_llm:
                    route_used = 'offline'
                    if settings.debug_mode:
                        print("[ROUTER] 🌊 Streaming via Nemotron fallback")
                    async for chunk in self.offline_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                elif self.online_llm:
                    route_used = 'online'
                    if settings.debug_mode:
                        print("[ROUTER] 🌊 Streaming via Groq as last resort")
                    async for chunk in self.online_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                else:
                    yield "I'm sorry, but I'm unable to process your request right now."
                    return
            
            # Track current info requests
            if decision.is_current_info:
                self.stats['current_info_requests'] += 1
                
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Streaming error: {e}")
            
            # For current info queries, provide specific error message
            if decision.is_current_info:
                yield ("I'm having trouble accessing current information right now. "
                       "Please check your internet connection and Groq API key configuration, then try again.")
            else:
                yield f"I'm experiencing technical difficulties right now."
    
    def _update_stats(self, route_type: str, response_time: float):
        """Update performance statistics"""
        if route_type == 'online':
            self.stats['online_requests'] += 1
            current_avg = self.stats['online_avg_time']
            count = self.stats['online_requests']
            self.stats['online_avg_time'] = ((current_avg * (count - 1)) + response_time) / count
        else:
            self.stats['offline_requests'] += 1
            current_avg = self.stats['offline_avg_time']
            count = self.stats['offline_requests']
            self.stats['offline_avg_time'] = ((current_avg * (count - 1)) + response_time) / count
    
    def get_router_stats(self) -> dict:
        """Get router stats"""
        return {
            'mode': self.mode.value,
            'offline_available': self.offline_available,
            'online_available': self.online_available,
            'current_info_enabled': self.online_available,  # Current info requires Groq
            'supported_providers': ['Groq'],  # Only Groq
            'last_decision': {
                'use_offline': self.last_decision.use_offline,
                'use_online': self.last_decision.use_online,
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
