"""
Pascal AI Assistant - FIXED: Lightning-Fast Router with Enhanced Current Info Detection
Intelligently routes requests between offline and online LLMs with Groq priority
FIXED: Better routing for current information queries and API key handling
"""

import asyncio
import time
import re
from typing import Optional, Dict, Any, AsyncGenerator
from enum import Enum

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
    
    def __init__(self, use_offline: bool, reason: str, confidence: float = 1.0):
        self.use_offline = use_offline
        self.reason = reason
        self.confidence = confidence
        self.timestamp = time.time()
    
    @property
    def use_online(self) -> bool:
        return not self.use_offline

class LightningRouter:
    """FIXED: Lightning-fast router for offline/online LLM selection"""
    
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
        
        # FIXED: Enhanced patterns for detecting current information needs
        self.current_info_direct_patterns = [
            # Exact phrases that MUST route online
            'what day is today',
            'what day is it', 
            'what date is today',
            'what date is it',
            'what\'s the date',
            'what\'s the day',
            'what\'s today\'s date',
            'today\'s date',
            'current date',
            'current day',
            'what time is it',
            'current time',
            'what\'s the time',
        ]
        
        # Regex patterns for current info
        self.current_info_patterns = [
            # Date/time patterns
            r'\b(?:what\s+)?(?:day|date|time)\s+(?:is\s+)?(?:it|today|now)\b',
            r'\bwhat\'?s\s+(?:the\s+)?(?:date|day|time)(?:\s+today)?\b',
            r'\btoday\'?s\s+(?:date|day)\b',
            r'\bcurrent\s+(?:date|day|time)\b',
            r'\bwhat\s+(?:day|date|time|year|month)\s+(?:is\s+)?(?:it|today)\b',
            
            # Current events and news
            r'\b(?:today\'?s|latest|recent|current|breaking)\s+(?:news|events?)\b',
            r'\bwhat\'?s\s+happening\s+(?:today|now|currently)\b',
            r'\bnews\s+(?:today|now|currently)\b',
            r'\bin\s+the\s+news\b',
            
            # Current status queries
            r'\bwho\s+is\s+(?:the\s+)?current\s+(?:president|pm|prime\s+minister)\b',
            r'\bcurrent\s+(?:president|prime\s+minister|government|leader)\b',
            r'\bwho\s+is\s+(?:president|pm)\s+(?:now|currently|today)\b',
            
            # Weather and conditions
            r'\bweather\s+(?:today|now|currently)\b',
            r'\btoday\'?s\s+weather\b',
            r'\bcurrent\s+(?:weather|temperature|conditions)\b',
        ]
        
        # Compile patterns for efficiency
        self.current_info_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.current_info_patterns]
        
        # Query complexity patterns for smart routing
        self.simple_query_patterns = [
            r'^(?:hi|hello|hey|thanks|thank\s+you|bye|goodbye)\.?$',
            r'^(?:yes|no|ok|okay|sure)\.?$',
            r'^\d+\s*[\+\-\*\/]\s*\d+\s*=?\s*\??$',  # Simple math
            r'^what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\??$',
        ]
        
        self.complex_query_patterns = [
            r'\b(?:analyze|compare|evaluate|research|detailed|comprehensive)\b',
            r'\b(?:explain\s+in\s+detail|write\s+code|debug|create|design|develop)\b',
            r'\b(?:strategy|plan|implementation|algorithm)\b',
            r'\bstep\s+by\s+step\b',
        ]
        
        # Compile for efficiency
        self.simple_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.simple_query_patterns]
        self.complex_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.complex_query_patterns]
    
    async def _check_llm_availability(self):
        """Check which LLM options are available"""
        try:
            if settings.debug_mode:
                print("[ROUTER] ⚡ Lightning Router checking LLM availability...")
            
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
            
            # Initialize online LLM if API keys are available
            self.online_available = False
            
            # Check if online APIs are configured
            if settings.is_online_available():
                try:
                    from modules.online_llm import OnlineLLM
                    self.online_llm = OnlineLLM()
                    self.online_available = await self.online_llm.initialize()
                    
                    if self.online_available:
                        # Show which provider is primary
                        if self.online_llm.preferred_provider:
                            provider_name = self.online_llm.preferred_provider.value.title()
                            if provider_name.lower() == 'groq':
                                print(f"✅ Online LLM ready (Primary: {provider_name} - Lightning Fast!)")
                            else:
                                print(f"✅ Online LLM ready (Primary: {provider_name})")
                        else:
                            print("✅ Online LLM ready")
                    else:
                        # Get detailed error info
                        if self.online_llm and hasattr(self.online_llm, 'get_provider_stats'):
                            stats = self.online_llm.get_provider_stats()
                            error_msg = stats.get('last_error', 'Configuration or connectivity issue')
                            print(f"❌ Online LLM failed: {error_msg}")
                        else:
                            print("❌ Online LLM not available")
                        
                except Exception as e:
                    print(f"⚠️ Online LLM initialization failed: {e}")
                    if settings.debug_mode:
                        import traceback
                        traceback.print_exc()
                    self.online_available = False
            else:
                if settings.debug_mode:
                    print("[ROUTER] ℹ️ No online API keys configured - running offline only")
                    print("   For best performance, consider adding Groq API key (fastest)")
                    print("   GROQ_API_KEY=gsk_... in your .env file")
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
                print("ℹ️ Running in offline-only mode (Ollama)")
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
            else:
                print("❌ ERROR: No LLMs available!")
                print("Solutions:")
                print("1. For offline: sudo systemctl start ollama && ./download_models.sh")
                print("2. For online: Configure API keys in .env file")
                print("   - Groq (fastest): GROQ_API_KEY=gsk_your-key")
                print("   - Gemini (free): GEMINI_API_KEY=your-key")
                print("   - OpenAI (reliable): OPENAI_API_KEY=sk-your-key")
            
            if settings.debug_mode:
                print(f"[ROUTER] Final status - Offline: {self.offline_available}, Online: {self.online_available}")
                print(f"[ROUTER] Routing mode: {self.mode.value}")
            
        except Exception as e:
            print(f"❌ Critical error in LLM availability check: {e}")
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
    
    def _needs_current_information(self, query: str) -> bool:
        """FIXED: Enhanced detection of queries requiring current/recent information"""
        query_lower = query.lower().strip()
        
        # PRIORITY CHECK: Direct exact phrase matching
        for pattern in self.current_info_direct_patterns:
            if pattern in query_lower:
                if settings.debug_mode:
                    print(f"[ROUTER] CURRENT INFO REQUIRED - direct pattern: {pattern}")
                return True
        
        # Check compiled regex patterns
        for pattern in self.current_info_regex:
            if pattern.search(query_lower):
                if settings.debug_mode:
                    print(f"[ROUTER] CURRENT INFO REQUIRED - regex pattern: {pattern.pattern}")
                return True
        
        # Additional high-priority current info indicators
        current_indicators = [
            'who is the current',
            'current prime minister', 
            'current president',
            'who is the president',
            'latest news',
            'recent news', 
            'breaking news',
            'what happened today',
            'happening today',
            'in the news',
            'weather today',
            'today\'s weather',
            'stock price today',
            'market today',
        ]
        
        for indicator in current_indicators:
            if indicator in query_lower:
                if settings.debug_mode:
                    print(f"[ROUTER] CURRENT INFO REQUIRED - indicator: {indicator}")
                return True
        
        return False
    
    def _is_simple_query(self, query: str) -> bool:
        """Check if query is simple and can be handled offline quickly"""
        query_lower = query.lower().strip()
        
        # NEVER treat date/time questions as simple - they MUST go online
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
        """FIXED: Decide whether to use offline or online LLM with enhanced logic"""
        
        # Force offline if no online available
        if not self.online_available:
            return RouteDecision(True, "No online LLM available")
        
        # Force online if no offline available
        if not self.offline_available:
            return RouteDecision(False, "No offline LLM available")
        
        # Handle different routing modes
        if self.mode == RouteMode.OFFLINE_ONLY:
            return RouteDecision(True, "Offline-only mode")
        
        elif self.mode == RouteMode.ONLINE_ONLY:
            return RouteDecision(False, "Online-only mode")
        
        # CRITICAL: Current information queries ALWAYS go online (highest priority)
        if self._needs_current_information(query):
            return RouteDecision(False, "Query requires current information (PRIORITY)")
        
        # Handle preference modes
        if self.mode == RouteMode.OFFLINE_PREFERRED:
            # Use offline unless complex
            if self._is_complex_query(query):
                return RouteDecision(False, "Complex query needs online quality")
            return RouteDecision(True, "Offline preferred mode")
        
        elif self.mode == RouteMode.ONLINE_PREFERRED:
            # Only use offline for very simple queries
            if self._is_simple_query(query):
                return RouteDecision(True, "Simple query - offline faster")
            # Everything else goes online
            return RouteDecision(False, "Online preferred mode")
        
        else:  # AUTO mode - intelligent routing with enhanced logic
            
            # PRIORITY 1: Current information queries ALWAYS go online
            if self._needs_current_information(query):
                return RouteDecision(False, "Current information required (AUTO)")
            
            # PRIORITY 2: Very simple queries go offline for speed
            if self._is_simple_query(query):
                return RouteDecision(True, "Simple query - offline faster")
            
            # PRIORITY 3: Complex analysis/research queries benefit from online
            if self._is_complex_query(query):
                return RouteDecision(False, "Complex query - online better")
            
            # PRIORITY 4: Code generation/debugging often better online
            code_patterns = ['write', 'code', 'function', 'script', 'program', 'debug', 'fix']
            if any(pattern in query.lower() for pattern in code_patterns):
                return RouteDecision(False, "Code-related query - online better")
            
            # DEFAULT: Use offline for general queries (faster locally)
            return RouteDecision(True, "General query - using offline")
    
    async def get_response(self, query: str) -> str:
        """Get response using appropriate LLM (non-streaming)"""
        decision = self._decide_route(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            route_type = "Offline" if decision.use_offline else "Online"
            print(f"[ROUTER] Decision: {route_type} - {decision.reason}")
        
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
                if self.offline_llm:
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
            
            return response
            
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Error in {route_used if 'route_used' in locals() else 'unknown'} LLM: {e}")
            
            # Try fallback
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
            
            return f"I'm having trouble processing your request right now. Error: {str(e)}"
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """Get streaming response using appropriate LLM"""
        decision = self._decide_route(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            route_type = "Offline" if decision.use_offline else "Online"
            print(f"[ROUTER] Decision: {route_type} - {decision.reason}")
        
        # Get context
        personality_context = await self.personality_manager.get_system_prompt()
        memory_context = await self.memory_manager.get_context()
        
        start_time = time.time()
        first_token_received = False
        response_generated = False
        
        try:
            if decision.use_offline and self.offline_llm:
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
                # Fallback logic
                if self.offline_llm:
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
            
            # Try fallback (non-streaming)
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
        """Get router performance statistics"""
        stats = {
            'mode': self.mode.value,
            'offline_available': self.offline_available,
            'online_available': self.online_available,
            'last_decision': {
                'use_offline': self.last_decision.use_offline,
                'reason': self.last_decision.reason,
                'confidence': self.last_decision.confidence
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
