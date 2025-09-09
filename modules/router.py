"""
Pascal AI Assistant - Lightning-Fast Router with Streaming and Groq Priority
Intelligently routes requests between offline and online LLMs with Groq as primary provider
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
    """Lightning-fast router for offline/online LLM selection with streaming and Groq priority"""
    
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
        
        # Enhanced patterns for detecting current information needs
        self.current_info_patterns = [
            # Direct date/time questions - HIGHEST PRIORITY
            r'\bwhat\s+(day|date|time|year|month)\s+(is\s+)?(it|today|tomorrow|yesterday)\b',
            r'\bwhat\'s\s+the\s+(date|time|day|year|month)\b',
            r'\bwhat\s+is\s+(the\s+)?(current|today\'s)\s+(date|day|time|year|month)\b',
            r'\btoday\'s\s+(date|day|weather|news)\b',
            r'\bcurrent\s+(date|day|time|year|month)\b',
            
            # Temporal indicators
            r'\b(today|tonight|tomorrow|yesterday|now|current|currently|latest|recent|recently)\b',
            r'\b(this\s+(year|month|week|day|morning|afternoon|evening))\b',
            r'\b(last\s+(year|month|week|day|night))\b',
            r'\b(next\s+(year|month|week|day))\b',
            
            # Date patterns - current years
            r'\b(202[4-9]|203\d)\b',  # Years 2024-2039
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+202[4-9]\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]202[4-9]\b',  # Date formats
            
            # Event/news patterns
            r'\b(news|headlines|happening|events?|announcement|breaking)\b',
            r'\b(weather|forecast|temperature|climate)\b',
            r'\b(stock|market|price|trading|economy)\b',
            r'\b(score|game|match|championship|election|results)\b',
            r'\bwho\s+(won|lost|is\s+winning|leads)\b',
            
            # Current affairs and people
            r'\b(president|prime\s+minister|government|politician)\b',
            r'\b(who\s+is\s+the\s+current|current\s+.*(president|pm|minister))\b',
            r'\bwho\s+is\s+(now|currently|presently)\b',
            
            # Status and updates
            r'\b(update|status|situation|condition)\b',
            r'\b(live|streaming|broadcast)\b'
        ]
        
        # Compile patterns for efficiency
        self.current_info_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.current_info_patterns]
        
        # Query complexity patterns for fast routing
        self.complexity_patterns = {
            'simple_queries': [
                'hi', 'hello', 'hey', 'thanks', 'bye', 'yes', 'no', 'ok',
                'what is 2+2', 'calculate', 'count', 'add', 'subtract', 'multiply', 'divide'
            ],
            'complex_queries': [
                'analyze', 'compare', 'evaluate', 'research', 'detailed',
                'comprehensive', 'explain in detail', 'write code', 'debug',
                'create', 'design', 'develop', 'implement', 'strategy', 'plan'
            ]
        }
    
    async def _check_llm_availability(self):
        """Check which LLM options are available"""
        try:
            if settings.debug_mode:
                print("⚡ Lightning Router checking LLM availability...")
            
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
                    print("ℹ️ No online API keys configured - running offline only")
                    print("   For best performance, consider adding Groq API key (fastest)")
                    print("   GROQ_API_KEY=gsk-... in your .env file")
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
                # Both available - PREFER ONLINE for current info
                self.mode = RouteMode.ONLINE_PREFERRED  # Changed from AUTO
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
                print("   - Groq (fastest): GROQ_API_KEY=gsk-your-key")
                print("   - Gemini (free): GEMINI_API_KEY=your-key")
                print("   - OpenAI (reliable): OPENAI_API_KEY=sk-your-key")
            
            if settings.debug_mode:
                print(f"Final status - Offline: {self.offline_available}, Online: {self.online_available}")
                print(f"Routing mode: {self.mode.value}")
            
        except Exception as e:
            print(f"❌ Critical error in LLM availability check: {e}")
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
    
    def _needs_current_information(self, query: str) -> bool:
        """Check if query requires current/recent information - IMPROVED"""
        query_lower = query.lower()
        
        # PRIORITY CHECK: Direct date/time questions ALWAYS need current info
        direct_date_patterns = [
            'what day is',
            'what date is',
            'what time is',
            'what\'s the date',
            'what\'s the day',
            'what\'s the time',
            'current date',
            'current day',
            'current time',
            'today\'s date',
            'today\'s day'
        ]
        
        for pattern in direct_date_patterns:
            if pattern in query_lower:
                if settings.debug_mode:
                    print(f"[DEBUG] Query needs current info - direct date/time question: {pattern}")
                return True
        
        # Check all current info patterns
        for pattern in self.current_info_regex:
            if pattern.search(query_lower):
                if settings.debug_mode:
                    print(f"[DEBUG] Query needs current info - matched pattern: {pattern.pattern}")
                return True
        
        # Additional specific checks for common current info queries
        current_info_indicators = [
            # Direct questions about today/current time
            ('who is the current', True),
            ('current prime minister', True),
            ('current president', True),
            ('who is the president', True),  # Added
            
            # Weather queries
            ('weather', True),
            ('temperature', True),
            ('forecast', True),
            
            # News and events
            ('latest news', True),
            ('recent news', True),
            ('what happened', True),
            ('breaking news', True),
            ('in the news', True),  # Added
            ('happening today', True),  # Added
            
            # Market/financial data
            ('stock price', True),
            ('market today', True),
            ('exchange rate', True),
        ]
        
        for indicator, needs_current in current_info_indicators:
            if indicator in query_lower and needs_current:
                if settings.debug_mode:
                    print(f"[DEBUG] Query needs current info - matched indicator: {indicator}")
                return True
        
        return False
    
    def _is_simple_query(self, query: str) -> bool:
        """Check if query is simple and can be handled offline quickly"""
        query_lower = query.lower().strip()
        
        # NEVER treat date/time questions as simple
        if self._needs_current_information(query):
            return False
        
        # Very short queries that are likely greetings or simple responses
        if len(query.split()) <= 3:
            simple_patterns = ['hi', 'hello', 'thanks', 'bye', 'yes', 'no', 'ok', 'sure']
            if any(pattern in query_lower for pattern in simple_patterns):
                return True
        
        # Simple math
        if 'what is 2+2' in query_lower or 'what is 2 + 2' in query_lower:
            return True
        
        # Simple factual questions that don't need current info
        simple_question_patterns = [
            r'^define \w+$',
            r'^calculate .+$',
            r'^what.*\d+.*\d+.*$',  # Math questions
            r'^how do you.*$',
            r'^can you explain.*$'
        ]
        
        for pattern in simple_question_patterns:
            if re.match(pattern, query_lower):
                # But not if it needs current info
                if not self._needs_current_information(query):
                    return True
        
        return False
    
    def _is_complex_query(self, query: str) -> bool:
        """Check if query is complex and benefits from online processing"""
        query_lower = query.lower()
        
        # Check for complexity indicators
        complex_keywords = self.complexity_patterns['complex_queries']
        
        for keyword in complex_keywords:
            if keyword in query_lower:
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
        """Decide whether to use offline or online LLM - FIXED LOGIC"""
        
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
        
        elif self.mode == RouteMode.OFFLINE_PREFERRED:
            # Use offline unless it absolutely needs current info
            if self._needs_current_information(query):
                return RouteDecision(False, "Query requires current information")
            return RouteDecision(True, "Offline preferred")
        
        elif self.mode == RouteMode.ONLINE_PREFERRED:
            # PRIORITY: Current information MUST go online
            if self._needs_current_information(query):
                return RouteDecision(False, "Query requires current information")
            
            # Only use offline for very simple queries
            if self._is_simple_query(query):
                return RouteDecision(True, "Simple query - offline faster")
            
            # Everything else goes online when in ONLINE_PREFERRED mode
            return RouteDecision(False, "Online preferred mode")
        
        else:  # AUTO mode - intelligent routing with priority rules
            
            # PRIORITY 1: Current information queries ALWAYS go online
            if self._needs_current_information(query):
                return RouteDecision(False, "Query requires current information")
            
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
