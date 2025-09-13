"""
Pascal AI Assistant - FIXED Lightning Router with Bulletproof Current Info Detection
Intelligently routes requests between offline and online LLMs with mandatory online for current info
FIXED: Enhanced pattern matching for all current info query variations - Groq + Gemini only
"""

import asyncio
import time
import re
from typing import Optional, Dict, Any, AsyncGenerator
from enum import Enum
from datetime import datetime

from config.settings import settings

class RouteMode(Enum):
    """Routing modes"""
    OFFLINE_ONLY = "offline_only"
    ONLINE_ONLY = "online_only"
    AUTO = "auto"
    OFFLINE_PREFERRED = "offline_preferred"
    ONLINE_PREFERRED = "online_preferred"

class RouteDecision:
    """Represents a routing decision for lightning-fast responses"""
    
    def __init__(self, use_offline: bool, reason: str, confidence: float = 1.0, is_current_info: bool = False):
        self.use_offline = use_offline
        self.reason = reason
        self.confidence = confidence
        self.is_current_info = is_current_info
        self.timestamp = time.time()
    
    @property
    def use_online(self) -> bool:
        return not self.use_offline

class LightningRouter:
    """FIXED: Lightning-fast router with bulletproof current info detection - Groq + Gemini"""
    
    def __init__(self, personality_manager, memory_manager):
        self.personality_manager = personality_manager
        self.memory_manager = memory_manager
        
        # Initialize LLM modules
        self.offline_llm = None
        self.online_llm = None
        
        # Router state
        self.mode = RouteMode.AUTO  # Default to AUTO for intelligent routing
        self.last_decision = None
        self.offline_available = False
        self.online_available = False
        
        # Performance tracking
        self.response_times = {'offline': [], 'online': []}
        self.first_token_times = {'offline': [], 'online': []}
        
        # FIXED: Enhanced current information detection patterns - COMPREHENSIVE LIST
        # These patterns MUST route to online - no exceptions
        self.mandatory_current_info_patterns = [
            # Date/time queries (HIGHEST PRIORITY) - FIXED with all variations
            'what day is today', 'what day is it', 'what day today',
            'what date is today', 'what date is it', 'what\'s the date',
            'what\'s today\'s date', 'what is todays date', 'what is today\'s date',
            'what is the date today', 'what is the date', 'todays date',
            'today\'s date', 'current date', 'current day',
            'tell me todays date', 'tell me today\'s date', 'tell me the date',
            'give me todays date', 'give me today\'s date', 'give me the date',
            'show me todays date', 'show me today\'s date',
            'what time is it', 'current time', 'what\'s the time',
            'tell me the time', 'what year is it', 'current year',
            'what month is it', 'current month', 'what is today',
            
            # Current events and status - FIXED with variations  
            'current president', 'current prime minister', 'current pm',
            'who is the current', 'who is current', 'what\'s the current',
            'current leader', 'current government', 'who\'s the current',
            'latest news', 'recent news', 'breaking news', 'news today',
            'today\'s news', 'what\'s happening today', 'current events',
            'what\'s in the news', 'in the news today',
            
            # Weather and conditions
            'weather today', 'today\'s weather', 'current weather',
            'current temperature', 'weather now', 'what\'s the weather',
        ]
        
        # FIXED: Enhanced regex patterns for current info (more comprehensive)
        self.current_info_regex_patterns = [
            # Flexible date/time patterns - FIXED for maximum coverage
            r'\bwhat\s+(?:day|date|time)\s+(?:is\s+)?(?:it|today|now)\s*\??\s*$',
            r'\bwhat\s+is\s+(?:todays?|today\'?s|the)\s+(?:day|date|time)\s*\??\s*$',
            r'\bwhat\'?s\s+(?:the\s+)?(?:date|day|time)(?:\s+today)?\s*\??\s*$',
            r'\b(?:tell|give|show)\s+me\s+(?:todays?|today\'?s|the)\s+(?:date|day|time)\s*\??\s*$',
            r'\btoday\'?s\s+(?:date|day)\s*\??\s*$',
            r'\bcurrent\s+(?:date|day|time|year|month)\s*\??\s*$',
            
            # Current status patterns - FIXED
            r'\bwho\s+is\s+(?:the\s+)?current\s+\w+\s*\??\s*$',
            r'\bwho\'?s\s+(?:the\s+)?current\s+\w+\s*\??\s*$',
            r'\bcurrent\s+(?:president|pm|prime\s+minister|leader)\s*\??\s*$',
            
            # News and events - FIXED
            r'\b(?:latest|recent|breaking|today\'?s)\s+news\s*\??\s*$',
            r'\bwhat\'?s\s+happening\s+(?:today|now)\s*\??\s*$',
            r'\bin\s+the\s+news\s+today\s*\??\s*$',
            r'\bnews\s+today\s*\??\s*$',
        ]
        
        # Compile regex patterns for efficiency
        self.current_info_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.current_info_regex_patterns]
        
        # Simple query patterns (always offline for speed) - NEVER current info
        self.simple_query_patterns = [
            r'^(?:hi|hello|hey|thanks|thank\s+you|bye|goodbye)\s*\.?\s*$',
            r'^(?:yes|no|ok|okay|sure)\s*\.?\s*$',
            r'^\d+\s*[\+\-\*\/]\s*\d+\s*=?\s*\??\s*$',  # Simple math
        ]
        
        # Complex query patterns (prefer online for quality)
        self.complex_query_patterns = [
            r'\b(?:analyze|compare|evaluate|research|detailed|comprehensive|explain\s+in\s+detail)\b',
            r'\b(?:write\s+code|debug|create|design|develop|implement)\b',
            r'\b(?:strategy|plan|algorithm|step\s+by\s+step)\b',
        ]
        
        # Compile for efficiency
        self.simple_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.simple_query_patterns]
        self.complex_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.complex_query_patterns]
    
    async def _check_llm_availability(self):
        """FIXED: Check LLM availability with Groq + Gemini focus"""
        try:
            if settings.debug_mode:
                print("[ROUTER] ⚡ FIXED Lightning Router (Groq + Gemini) checking LLM availability...")
            
            # Initialize offline LLM first (for fallback)
            try:
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
            
            # Initialize online LLM - CRITICAL FOR CURRENT INFO (Groq + Gemini only)
            self.online_available = False
            
            if settings.is_online_available():
                try:
                    from modules.online_llm import OnlineLLM
                    self.online_llm = OnlineLLM()
                    self.online_available = await self.online_llm.initialize()
                    
                    if self.online_available:
                        # Show which provider is primary (Groq + Gemini only)
                        if self.online_llm.preferred_provider:
                            provider_name = self.online_llm.preferred_provider.value.title()
                            if provider_name.lower() == 'groq':
                                print(f"✅ Online LLM ready (Primary: {provider_name} ⚡ - CURRENT INFO ENABLED)")
                            else:
                                print(f"✅ Online LLM ready (Primary: {provider_name} - CURRENT INFO ENABLED)")
                        else:
                            print("✅ Online LLM ready - CURRENT INFO ENABLED")
                    else:
                        # Get detailed error info
                        if self.online_llm and hasattr(self.online_llm, 'get_provider_stats'):
                            stats = self.online_llm.get_provider_stats()
                            error_msg = stats.get('last_error', 'Configuration or connectivity issue')
                            print(f"❌ Online LLM failed: {error_msg}")
                            print("⚠️ CURRENT INFO QUERIES WILL NOT WORK PROPERLY")
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
                    print("[ROUTER] ❌ No online API keys configured - CURRENT INFO DISABLED")
                    print("   For current info queries, add API key to .env:")
                    print("   GROQ_API_KEY=gsk_your-actual-key    # Primary - fastest")
                    print("   GEMINI_API_KEY=your-actual-key      # Secondary - free")
                self.online_available = False
            
            # Adjust routing mode based on availability
            if self.online_available and not self.offline_available:
                self.mode = RouteMode.ONLINE_ONLY
                provider_info = ""
                if self.online_llm and hasattr(self.online_llm, 'preferred_provider'):
                    if self.online_llm.preferred_provider:
                        provider_name = self.online_llm.preferred_provider.value.title()
                        provider_info = f" ({provider_name})"
                print(f"ℹ️ Running in online-only mode{provider_info}")
            elif self.offline_available and not self.online_available:
                self.mode = RouteMode.OFFLINE_ONLY
                print("⚠️ Running in offline-only mode (Ollama) - CURRENT INFO QUERIES DISABLED")
                print("   Add online API key to enable current information queries")
            elif self.offline_available and self.online_available:
                # Both available - prefer online for current info, offline for general
                self.mode = RouteMode.AUTO
                provider_info = ""
                if self.online_llm and hasattr(self.online_llm, 'preferred_provider'):
                    if self.online_llm.preferred_provider:
                        provider_name = self.online_llm.preferred_provider.value.title()
                        if provider_name.lower() == 'groq':
                            provider_info = f" (Online Primary: {provider_name} ⚡)"
                        else:
                            provider_info = f" (Online Primary: {provider_name})"
                print(f"✅ Both offline and online LLMs available{provider_info}")
                print("🎯 Current info queries will automatically use online")
            else:
                print("❌ ERROR: No LLMs available!")
                print("Solutions:")
                print("1. For offline: sudo systemctl start ollama && ./download_models.sh")
                print("2. For current info: Configure API keys in .env file")
                print("   - Groq (fastest): GROQ_API_KEY=gsk_your-key")
                print("   - Gemini (free): GEMINI_API_KEY=your-key")
            
            if settings.debug_mode:
                print(f"[ROUTER] Final status - Offline: {self.offline_available}, Online: {self.online_available}")
                print(f"[ROUTER] Routing mode: {self.mode.value}")
                print(f"[ROUTER] Current info priority: {settings.force_online_current_info}")
            
        except Exception as e:
            print(f"❌ Critical error in LLM availability check: {e}")
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
    
    def _needs_current_information(self, query: str) -> bool:
        """FIXED: Enhanced detection of queries requiring current/recent information"""
        query_lower = query.lower().strip()
        
        # Remove punctuation for cleaner matching
        query_clean = re.sub(r'[^\w\s]', '', query_lower)
        
        # PRIORITY 1: Exact phrase matching (most reliable) - FIXED with comprehensive patterns
        for pattern in self.mandatory_current_info_patterns:
            if pattern in query_lower:
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 CURRENT INFO DETECTED - exact pattern: '{pattern}' in '{query}'")
                return True
        
        # PRIORITY 2: Flexible regex patterns - FIXED for better coverage
        for pattern in self.current_info_regex:
            if pattern.search(query_lower):
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 CURRENT INFO DETECTED - regex match: {pattern.pattern}")
                return True
        
        # PRIORITY 3: Word-based detection for edge cases - FIXED and enhanced
        current_info_words = [
            (['what', 'is'], ['today', 'date', 'day', 'time', 'todays']),    # "what is" + date words
            (['what'], ['day', 'date', 'time', 'today', 'todays']),          # "what" + time words
            (['tell', 'me'], ['today', 'date', 'day', 'time', 'todays']),    # "tell me" variations  
            (['give', 'me'], ['today', 'date', 'day', 'time', 'todays']),    # "give me" variations
            (['show', 'me'], ['today', 'date', 'day', 'time', 'todays']),    # "show me" variations
            (['current'], ['president', 'pm', 'minister', 'leader', 'date', 'time', 'day']),
            (['news'], ['today', 'latest', 'recent', 'breaking', 'current']),
            (['weather'], ['today', 'current', 'now']),
            (['who', 'is'], ['current']),                                    # "who is current"
            (['who'], ['current']),                                          # "who current"
            (['whats'], ['today', 'date', 'time', 'current']),               # "whats today"
        ]
        
        query_words = query_clean.split()
        for primary_words, context_words in current_info_words:
            # Check if all primary words are present
            if all(word in query_words for word in primary_words):
                # Check if any context words are present
                if any(word in query_words for word in context_words):
                    if settings.debug_mode:
                        print(f"[ROUTER] 🎯 CURRENT INFO DETECTED - word combination: {primary_words} + {context_words}")
                    return True
        
        # PRIORITY 4: Additional pattern detection for missed cases
        # Check for questions about "today" specifically
        if 'today' in query_words and any(word in query_words for word in ['what', 'is', 'whats']):
            if settings.debug_mode:
                print(f"[ROUTER] 🎯 CURRENT INFO DETECTED - 'today' question pattern")
            return True
        
        return False
    
    def _is_simple_query(self, query: str) -> bool:
        """Check if query is simple and can be handled offline quickly"""
        query_lower = query.lower().strip()
        
        # NEVER treat current info queries as simple - they MUST go online
        if self._needs_current_information(query):
            return False
        
        # Check compiled simple patterns
        for pattern in self.simple_regex:
            if pattern.match(query_lower):
                return True
        
        # Very short queries that are likely greetings
        if len(query.split()) <= 3:
            simple_words = ['hi', 'hello', 'thanks', 'bye', 'yes', 'no', 'ok', 'sure']
            if any(word in query_lower for word in simple_words):
                return True
        
        return False
    
    def _is_complex_query(self, query: str) -> bool:
        """Check if query is complex and benefits from online processing"""
        query_lower = query.lower()
        
        # Check compiled complex patterns
        for pattern in self.complex_regex:
            if pattern.search(query_lower):
                return True
        
        # Long queries are often complex
        if len(query.split()) > 25:
            return True
        
        # Multiple questions
        if query.count('?') > 1:
            return True
        
        # Code-related queries
        code_indicators = ['function', 'class', 'method', 'algorithm', 'implement', 'debug', 'syntax']
        if any(indicator in query_lower for indicator in code_indicators):
            return True
        
        return False
    
    def _decide_route(self, query: str) -> RouteDecision:
        """FIXED: Decide routing with mandatory current info online routing"""
        
        # CRITICAL CHECK: Current information queries ALWAYS go online (highest priority)
        is_current_info = self._needs_current_information(query)
        
        if is_current_info:
            if not self.online_available:
                return RouteDecision(
                    False,  # Still try online even if not available - will trigger error message
                    "CURRENT INFO REQUIRED - no online LLM available (CRITICAL ERROR)",
                    1.0,
                    is_current_info=True
                )
            else:
                return RouteDecision(
                    False,  # Use online
                    "CURRENT INFO REQUIRED - mandatory online routing",
                    1.0,
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
        
        # Handle preference modes
        if self.mode == RouteMode.OFFLINE_PREFERRED:
            # Use offline unless complex
            if self._is_complex_query(query):
                return RouteDecision(False, "Complex query needs online quality", is_current_info=False)
            return RouteDecision(True, "Offline preferred mode", is_current_info=False)
        
        elif self.mode == RouteMode.ONLINE_PREFERRED:
            # Only use offline for very simple queries
            if self._is_simple_query(query):
                return RouteDecision(True, "Simple query - offline faster", is_current_info=False)
            # Everything else goes online
            return RouteDecision(False, "Online preferred mode", is_current_info=False)
        
        else:  # AUTO mode - intelligent routing
            
            # PRIORITY 2: Very simple queries go offline for speed
            if self._is_simple_query(query):
                return RouteDecision(True, "Simple query - offline faster", is_current_info=False)
            
            # PRIORITY 3: Complex analysis/research queries benefit from online
            if self._is_complex_query(query):
                return RouteDecision(False, "Complex query - online better", is_current_info=False)
            
            # PRIORITY 4: Code generation/debugging often better online
            code_patterns = ['write', 'code', 'function', 'script', 'program', 'debug', 'fix']
            if any(pattern in query.lower() for pattern in code_patterns):
                return RouteDecision(False, "Code-related query - online better", is_current_info=False)
            
            # DEFAULT: Use offline for general queries (faster locally)
            return RouteDecision(True, "General query - using offline", is_current_info=False)
    
    async def get_response(self, query: str) -> str:
        """FIXED: Get response with enhanced current info handling"""
        decision = self._decide_route(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            route_type = "Offline" if decision.use_offline else "Online"
            current_info_flag = " [CURRENT INFO]" if decision.is_current_info else ""
            print(f"[ROUTER] Decision: {route_type}{current_info_flag} - {decision.reason}")
        
        # CRITICAL: Handle current info queries that require online but online is unavailable
        if decision.is_current_info and not self.online_available:
            return ("I'm unable to provide current information right now because online services are not available. "
                   "Please configure an API key (GROQ_API_KEY or GEMINI_API_KEY) in your .env file "
                   "to enable current date, time, news, and other real-time information queries.")
        
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
            elif decision.use_online and self.online_llm:
                response = await self.online_llm.generate_response(
                    query, personality_context, memory_context
                )
                route_used = 'online'
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
            
            # Track response time
            response_time = time.time() - start_time
            self.response_times[route_used].append(response_time)
            
            # Keep only last 20 measurements
            if len(self.response_times[route_used]) > 20:
                self.response_times[route_used] = self.response_times[route_used][-20:]
            
            # VALIDATION: Check if current info query got proper response
            if decision.is_current_info and route_used == 'offline':
                if settings.debug_mode:
                    print("[ROUTER] ⚠️ WARNING: Current info query was handled by offline model")
            
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
                       "Please check your internet connection and API key configuration, then try again.")
            
            return f"I'm having trouble processing your request right now. Error: {str(e)}"
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """FIXED: Get streaming response with enhanced current info handling"""
        decision = self._decide_route(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            route_type = "Offline" if decision.use_offline else "Online"
            current_info_flag = " [CURRENT INFO]" if decision.is_current_info else ""
            print(f"[ROUTER] Decision: {route_type}{current_info_flag} - {decision.reason}")
        
        # CRITICAL: Handle current info queries that require online but online is unavailable
        if decision.is_current_info and not self.online_available:
            yield ("I'm unable to provide current information right now because online services are not available. "
                   "Please configure an API key (GROQ_API_KEY or GEMINI_API_KEY) in your .env file "
                   "to enable current date, time, news, and other real-time information queries.")
            return
        
        # Get context
        personality_context = await self.personality_manager.get_system_prompt()
        memory_context = await self.memory_manager.get_context()
        
        start_time = time.time()
        first_token_received = False
        response_generated = False
        
        try:
            if decision.use_offline and self.offline_llm and not decision.is_current_info:
                route_used = 'offline'
                async for chunk in self.offline_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    if not first_token_received:
                        first_token_time = time.time() - start_time
                        self.first_token_times[route_used].append(first_token_time)
                        first_token_received = True
                    yield chunk
                    response_generated = True
                    
            elif decision.use_online and self.online_llm:
                route_used = 'online'
                async for chunk in self.online_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    if not first_token_received:
                        first_token_time = time.time() - start_time
                        self.first_token_times[route_used].append(first_token_time)
                        first_token_received = True
                    yield chunk
                    response_generated = True
                    
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
            
            # Track total response time
            if response_generated:
                total_time = time.time() - start_time
                self.response_times[route_used].append(total_time)
                
                # Keep only last 20 measurements
                if len(self.response_times[route_used]) > 20:
                    self.response_times[route_used] = self.response_times[route_used][-20:]
                
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
                       "Please check your internet connection and API key configuration, then try again.")
    
    def set_mode(self, mode: RouteMode):
        """Set routing mode"""
        self.mode = mode
        if settings.debug_mode:
            print(f"Router mode set to: {mode.value}")
    
    def set_performance_preference(self, preference: str):
        """Set performance preference (speed/balanced/quality)"""
        if preference == 'speed':
            self.mode = RouteMode.OFFLINE_PREFERRED
        elif preference == 'balanced':
            self.mode = RouteMode.AUTO
        elif preference == 'quality':
            self.mode = RouteMode.ONLINE_PREFERRED
        
        # Also pass to LLM modules if available
        if self.offline_llm:
            self.offline_llm.set_performance_profile(preference)
    
    def get_router_stats(self) -> Dict[str, Any]:
        """FIXED: Get router stats with Groq + Gemini info"""
        stats = {
            'mode': self.mode.value,
            'offline_available': self.offline_available,
            'online_available': self.online_available,
            'current_info_enabled': self.online_available,  # Current info requires online
            'force_online_current_info': settings.force_online_current_info,
            'supported_providers': ['Groq', 'Gemini'],  # Only these two
            'last_decision': {
                'use_offline': self.last_decision.use_offline,
                'reason': self.last_decision.reason,
                'confidence': self.last_decision.confidence,
                'is_current_info': self.last_decision.is_current_info
            } if self.last_decision else None
        }
        
        # Add performance metrics
        for route_type in ['offline', 'online']:
            if self.response_times[route_type]:
                avg_time = sum(self.response_times[route_type]) / len(self.response_times[route_type])
                stats[f'{route_type}_avg_response_time'] = f"{avg_time:.2f}s"
                stats[f'{route_type}_total_requests'] = len(self.response_times[route_type])
            
            if self.first_token_times[route_type]:
                avg_first_token = sum(self.first_token_times[route_type]) / len(self.first_token_times[route_type])
                stats[f'{route_type}_avg_first_token_time'] = f"{avg_first_token:.2f}s"
        
        return stats
    
    async def close(self):
        """Close all LLM connections"""
        if self.offline_llm:
            await self.offline_llm.close()
        if self.online_llm:
            await self.online_llm.close()
