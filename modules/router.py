"""
Pascal AI Assistant - Lightning-Fast Router with Streaming
Intelligently routes requests between offline and online LLMs with streaming support
"""

import asyncio
import time
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
    """Lightning-fast router for offline/online LLM selection with streaming"""
    
    def __init__(self, personality_manager, memory_manager):
        self.personality_manager = personality_manager
        self.memory_manager = memory_manager
        
        # Initialize LLM modules
        self.offline_llm = None
        self.online_llm = None
        
        # Router state
        self.mode = RouteMode.OFFLINE_PREFERRED  # Default to offline for speed
        self.last_decision = None
        self.offline_available = False
        self.online_available = False
        
        # Performance tracking
        self.response_times = {'offline': [], 'online': []}
        self.first_token_times = {'offline': [], 'online': []}
        
        # Query complexity patterns for fast routing
        self.complexity_patterns = {
            'needs_current_info': [
                'current', 'latest', 'recent', 'today', 'now', 'news', 'update', 
                'weather', '2024', '2025', 'this year', 'this month', 'what day',
                'what date', 'current date', 'current time'
            ],
            'simple_queries': [
                'hi', 'hello', 'hey', 'thanks', 'bye', 'yes', 'no', 'ok', 
                'what is', 'who is', 'how far', 'distance'
            ],
            'complex_queries': [
                'analyze', 'compare', 'evaluate', 'research', 'detailed', 
                'comprehensive', 'explain in detail'
            ]
        }
    
    async def _check_llm_availability(self):
        """Check which LLM options are available"""
        try:
            if settings.debug_mode:
                print("⚡ Lightning Router checking LLM availability...")
            
            # Initialize offline LLM first (priority for offline-first)
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
            if getattr(settings, 'grok_api_key', None) or getattr(settings, 'openai_api_key', None) or getattr(settings, 'anthropic_api_key', None):
                try:
                    from modules.online_llm import OnlineLLM
                    self.online_llm = OnlineLLM()
                    self.online_available = await self.online_llm.initialize()
                    
                    if self.online_available:
                        print("✅ Online LLM ready (Grok/OpenAI/Anthropic)")
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
                self.online_available = False
            
            # Final status
            if not self.offline_available and not self.online_available:
                print("❌ ERROR: No LLMs available!")
                print("Solutions:")
                print("1. For offline: sudo systemctl start ollama && ./download_models.sh")
                print("2. For online: Configure API keys in .env file")
            elif self.offline_available and not self.online_available:
                print("ℹ️ Running in offline-only mode")
            elif not self.offline_available and self.online_available:
                print("ℹ️ Running in online-only mode")
            else:
                print("✅ Both offline and online LLMs available")
            
            if settings.debug_mode:
                print(f"Final status - Offline: {self.offline_available}, Online: {self.online_available}")
            
        except Exception as e:
            print(f"❌ Critical error in LLM availability check: {e}")
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
    
    def _analyze_query_speed(self, query: str) -> Dict[str, Any]:
        """Fast query analysis for immediate routing decision"""
        query_lower = query.lower()
        
        # Quick pattern matching
        needs_current = any(pattern in query_lower for pattern in self.complexity_patterns['needs_current_info'])
        is_simple = any(pattern in query_lower for pattern in self.complexity_patterns['simple_queries'])
        is_complex = any(pattern in query_lower for pattern in self.complexity_patterns['complex_queries'])
        
        word_count = len(query.split())
        
        return {
            'needs_current_info': needs_current,
            'is_simple': is_simple,
            'is_complex': is_complex,
            'word_count': word_count,
            'estimated_offline_time': self._estimate_response_time(is_simple, is_complex, word_count)
        }
    
    def _estimate_response_time(self, is_simple: bool, is_complex: bool, word_count: int) -> float:
        """Estimate response time for offline processing"""
        if is_simple:
            return 1.0
        elif is_complex:
            return 3.0
        elif word_count <= 10:
            return 1.5
        elif word_count <= 30:
            return 2.0
        else:
            return 2.5
    
    def _make_lightning_decision(self, query: str, analysis: Dict[str, Any]) -> RouteDecision:
        """Make ultra-fast routing decision"""
        
        # Handle forced modes
        if self.mode == RouteMode.OFFLINE_ONLY:
            return RouteDecision(True, "Forced offline mode", 1.0)
        elif self.mode == RouteMode.ONLINE_ONLY:
            return RouteDecision(False, "Forced online mode", 1.0)
        
        # Handle availability constraints
        if not self.offline_available and not self.online_available:
            return RouteDecision(True, "No LLMs available (will show error)", 0.1)
        elif self.offline_available and not self.online_available:
            return RouteDecision(True, "Only offline available", 1.0)
        elif not self.offline_available and self.online_available:
            return RouteDecision(False, "Only online available", 1.0)
        
        # Both available - intelligent routing
        # Only use online for queries that explicitly need current information
        if analysis['needs_current_info']:
            return RouteDecision(False, "Query needs current information", 0.9)
        
        # Everything else goes to offline for speed
        return RouteDecision(True, "Offline preferred for speed", 0.95)
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """Get streaming response for instant feedback"""
        start_time = time.time()
        first_token_time = None
        response_generated = False
        
        try:
            # Fast analysis and routing
            analysis = self._analyze_query_speed(query)
            decision = self._make_lightning_decision(query, analysis)
            self.last_decision = decision
            
            if settings.debug_mode:
                print(f"⚡ Query: '{query[:50]}...'")
                print(f"⚡ Routing: {'Offline' if decision.use_offline else 'Online'} - {decision.reason}")
            
            # Get context (minimal for speed)
            personality_context = await self.personality_manager.get_system_prompt()
            memory_context = ""
            if not analysis['is_simple']:
                memory_context = await self.memory_manager.get_context()
            
            # Route to appropriate LLM
            if decision.use_offline and self.offline_available and self.offline_llm:
                try:
                    async for chunk in self.offline_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        if not first_token_time:
                            first_token_time = time.time() - start_time
                            self.first_token_times['offline'].append(first_token_time)
                        yield chunk
                        response_generated = True
                except Exception as e:
                    if settings.debug_mode:
                        print(f"❌ Offline error: {e}")
                    # Try online fallback
                    if self.online_available and self.online_llm:
                        try:
                            async for chunk in self.online_llm.generate_response_stream(
                                query, personality_context, memory_context
                            ):
                                yield chunk
                                response_generated = True
                        except Exception as online_e:
                            yield f"I'm having trouble processing your request. Error: {str(e)[:100]}"
                            response_generated = True
                    else:
                        yield f"I'm having trouble processing your request: {str(e)}"
                        response_generated = True
            
            elif decision.use_online and self.online_available and self.online_llm:
                try:
                    async for chunk in self.online_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        if not first_token_time:
                            first_token_time = time.time() - start_time
                            self.first_token_times['online'].append(first_token_time)
                        yield chunk
                        response_generated = True
                except Exception as e:
                    if settings.debug_mode:
                        print(f"❌ Online error: {e}")
                    # Try offline fallback
                    if self.offline_available and self.offline_llm:
                        try:
                            async for chunk in self.offline_llm.generate_response_stream(
                                query, personality_context, memory_context
                            ):
                                yield chunk
                                response_generated = True
                        except Exception as offline_e:
                            yield f"I'm having trouble with both services. Error: {str(e)[:100]}"
                            response_generated = True
                    else:
                        yield f"I'm having trouble connecting to online services: {str(e)[:100]}"
                        response_generated = True
            
            else:
                # Final fallback - try whatever is available
                if self.offline_available and self.offline_llm:
                    try:
                        async for chunk in self.offline_llm.generate_response_stream(
                            query, personality_context, memory_context
                        ):
                            yield chunk
                            response_generated = True
                    except Exception as e:
                        yield f"I'm having trouble processing your request: {str(e)}"
                        response_generated = True
                elif self.online_available and self.online_llm:
                    try:
                        async for chunk in self.online_llm.generate_response_stream(
                            query, personality_context, memory_context
                        ):
                            yield chunk
                            response_generated = True
                    except Exception as e:
                        yield f"I'm having trouble processing your request: {str(e)}"
                        response_generated = True
                else:
                    yield "I'm sorry, but no AI services are currently available. Please check that Ollama is running or configure API keys."
                    response_generated = True
            
            # Track performance
            if response_generated:
                total_time = time.time() - start_time
                source = 'offline' if decision.use_offline else 'online'
                self.response_times[source].append(total_time)
                
                # Keep recent measurements
                if len(self.response_times[source]) > 20:
                    self.response_times[source] = self.response_times[source][-20:]
                if first_token_time and len(self.first_token_times[source]) > 20:
                    self.first_token_times[source] = self.first_token_times[source][-20:]
                
                if settings.debug_mode and first_token_time:
                    print(f"⚡ Performance: {first_token_time:.2f}s first token, {total_time:.2f}s total")
            
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Router streaming error: {e}")
                import traceback
                traceback.print_exc()
            yield f"I encountered an error: {str(e)}"
    
    async def get_response(self, query: str) -> str:
        """Get complete response (non-streaming fallback)"""
        try:
            # Collect streaming response
            response_parts = []
            async for chunk in self.get_streaming_response(query):
                response_parts.append(chunk)
            
            response = ''.join(response_parts)
            
            # Store in memory if valid response
            if (response and 
                not response.startswith("I'm sorry") and 
                not response.startswith("Error") and 
                not response.startswith("I'm having trouble") and
                not response.startswith("I encountered an error")):
                await self.memory_manager.add_interaction(query, response)
            
            return response
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Router error: {e}")
                import traceback
                traceback.print_exc()
            
            return f"I encountered an error processing your request: {str(e)}"
    
    def set_mode(self, mode: RouteMode):
        """Set routing mode"""
        self.mode = mode
        print(f"Routing mode set to: {mode.value}")
    
    def set_performance_preference(self, preference: str):
        """Set performance preference"""
        if preference in ['speed', 'balanced', 'quality']:
            settings.set_performance_mode(preference)
            if self.offline_llm and hasattr(self.offline_llm, 'set_performance_profile'):
                self.offline_llm.set_performance_profile(preference)
            print(f"Performance preference set to: {preference}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get router status with performance metrics"""
        status = {
            'mode': self.mode.value,
            'offline_available': self.offline_available,
            'online_available': self.online_available,
            'hardware_info': settings.get_hardware_info(),
            'performance_mode': getattr(settings, 'performance_mode', 'balanced'),
            'streaming_enabled': getattr(settings, 'streaming_enabled', True)
        }
        
        # Add performance metrics
        if self.response_times['offline']:
            status['avg_offline_time'] = f"{sum(self.response_times['offline']) / len(self.response_times['offline']):.2f}s"
        if self.response_times['online']:
            status['avg_online_time'] = f"{sum(self.response_times['online']) / len(self.response_times['online']):.2f}s"
        
        if self.first_token_times['offline']:
            status['avg_offline_first_token'] = f"{sum(self.first_token_times['offline']) / len(self.first_token_times['offline']):.2f}s"
        if self.first_token_times['online']:
            status['avg_online_first_token'] = f"{sum(self.first_token_times['online']) / len(self.first_token_times['online']):.2f}s"
        
        if self.last_decision:
            status['last_decision'] = {
                'use_offline': self.last_decision.use_offline,
                'reason': self.last_decision.reason,
                'confidence': self.last_decision.confidence
            }
        
        # Get component stats
        if self.offline_llm and hasattr(self.offline_llm, 'get_performance_stats'):
            status['offline_model_info'] = self.offline_llm.get_performance_stats()
        
        if self.online_llm and hasattr(self.online_llm, 'get_provider_stats'):
            status['online_provider_info'] = self.online_llm.get_provider_stats()
        
        return status

# For backwards compatibility
Router = LightningRouter
