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
            'needs_current_info': ['current', 'latest', 'recent', 'today', 'now', 'news', 'update', 'weather', '2024', '2025', 'this year', 'this month'],
            'simple_queries': ['hi', 'hello', 'hey', 'thanks', 'bye', 'yes', 'no', 'ok', 'what is', 'who is', 'how far', 'distance'],
            'complex_queries': ['analyze', 'compare', 'evaluate', 'research', 'detailed', 'comprehensive', 'explain in detail']
        }
    
    async def _check_llm_availability(self):
        """Check which LLM options are available"""
        try:
            if settings.debug_mode:
                print("⚡ Lightning Router checking LLM availability...")
            
            # Always check offline LLM first (priority for offline-first approach)
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
            
            # Check online LLM only if API keys are configured
            self.online_available = False
            if settings.is_online_available():
                try:
                    from modules.online_llm import OnlineLLM
                    self.online_llm = OnlineLLM()
                    self.online_available = await self.online_llm.initialize()
                    
                    if self.online_available:
                        print("✅ Online LLM ready (Grok/OpenAI/Anthropic)")
                    else:
                        # Get more detailed error info
                        if self.online_llm and hasattr(self.online_llm, 'get_provider_stats'):
                            stats = self.online_llm.get_provider_stats()
                            error_msg = stats.get('last_error', 'Unknown error')
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
            
            # Final status reporting
            if not self.offline_available and not self.online_available:
                print("❌ ERROR: No LLMs available!")
                print("Please ensure:")
                print("1. Ollama is running: sudo systemctl start ollama")
                print("2. Models are downloaded: ./download_models.sh")
                print("3. Or configure API keys in .env for online fallback")
            elif self.offline_available and not self.online_available:
                print("ℹ️ Running in offline-only mode")
            elif not self.offline_available and self.online_available:
                print("ℹ️ Running in online-only mode (no offline models)")
            else:
                print("✅ Both offline and online LLMs available")
            
            if settings.debug_mode:
                print(f"Final availability - Offline: {self.offline_available}, Online: {self.online_available}")
            
        except Exception as e:
            print(f"❌ Critical error in LLM availability check: {e}")
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
    
    def _analyze_query_speed(self, query: str) -> Dict[str, Any]:
        """Fast query analysis for immediate routing decision"""
        query_lower = query.lower()
        
        # Quick checks for routing decision
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
            return 1.0  # 1 second for simple queries
        elif is_complex:
            return 3.0  # 3 seconds for complex
        elif word_count <= 10:
            return 1.5  # Quick for short queries
        elif word_count <= 30:
            return 2.0  # Medium queries
        else:
            return 2.5  # Longer queries
    
    def _make_lightning_decision(self, query: str, analysis: Dict[str, Any]) -> RouteDecision:
        """Make ultra-fast routing decision with improved logic"""
        
        # Force modes
        if self.mode == RouteMode.OFFLINE_ONLY:
            return RouteDecision(True, "Forced offline mode", 1.0)
        elif self.mode == RouteMode.ONLINE_ONLY:
            return RouteDecision(False, "Forced online mode", 1.0)
        
        # Check availability - handle cases where neither is available
        if not self.offline_available and not self.online_available:
            return RouteDecision(True, "No LLMs available (will show error)", 0.1)
        elif self.offline_available and not self.online_available:
            return RouteDecision(True, "Only offline LLM available", 1.0)
        elif not self.offline_available and self.online_available:
            return RouteDecision(False, "Only online LLM available", 1.0)
        
        # Both available - make intelligent decision
        
        # Check if query explicitly needs current information
        if analysis['needs_current_info']:
            return RouteDecision(False, "Query requires current information", 0.9)
        
        # For all other queries, prefer offline for speed
        # This includes general knowledge, calculations, coding help, etc.
        return RouteDecision(True, "Offline preferred for speed", 0.95)
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """Get streaming response for instant feedback"""
        start_time = time.time()
        first_token_time = None
        
        try:
            # Fast query analysis
            analysis = self._analyze_query_speed(query)
            
            # Make routing decision
            decision = self._make_lightning_decision(query, analysis)
            self.last_decision = decision
            
            if settings.debug_mode:
                print(f"⚡ Query: '{query[:50]}...'")
                print(f"⚡ Analysis: Simple={analysis['is_simple']}, Current={analysis['needs_current_info']}")
                print(f"⚡ Routing: {'Offline' if decision.use_offline else 'Online'} - {decision.reason}")
            
            # Get personality and memory context (minimal for speed)
            personality_context = await self.personality_manager.get_system_prompt()
            memory_context = ""  # Skip memory for simple queries
            
            if not analysis['is_simple']:
                memory_context = await self.memory_manager.get_context()
            
            # Stream from appropriate source
            response_generated = False
            
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
                        print(f"❌ Offline streaming error: {e}")
                    # Try online fallback if available
                    if self.online_available and self.online_llm:
                        try:
                            async for chunk in self.online_llm.generate_response_stream(
                                query, personality_context, memory_context
                            ):
                                yield chunk
                                response_generated = True
                        except Exception as online_e:
                            yield f"I'm having trouble generating a response. Offline error: {str(e)[:50]}. Online error: {str(online_e)[:50]}"
                            response_generated = True
                    else:
                        yield f"I'm having trouble generating a response: {str(e)}"
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
                        print(f"❌ Online streaming error: {e}")
                    # Try offline fallback if available
                    if self.offline_available and self.offline_llm:
                        try:
                            async for chunk in self.offline_llm.generate_response_stream(
                                query, personality_context, memory_context
                            ):
                                yield chunk
                                response_generated = True
                        except Exception as offline_e:
                            yield f"I'm having trouble with both online and offline services. Online error: {str(e)[:50]}. Offline error: {str(offline_e)[:50]}"
                            response_generated = True
                    else:
                        yield f"Online service error: {str(e)}"
                        response_generated = True
            
            else:
                # Final fallback - try anything available
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
                    yield "I'm sorry, but I'm currently unable to process your request. Please check that Ollama is running and models are installed, or configure API keys for online services."
                    response_generated = True
            
            # Track total response time
            if response_generated:
                total_time = time.time() - start_time
                source = 'offline' if decision.use_offline else 'online'
                self.response_times[source].append(total_time)
                
                # Keep only last 20 measurements
                if len(self.response_times[source]) > 20:
                    self.response_times[source] = self.response_times[source][-20:]
                if first_token_time and len(self.first_token_times[source]) > 20:
                    self.first_token_times[source] = self.first_token_times[source][-20:]
                
                if settings.debug_mode and first_token_time:
                    print(f"⚡ First token: {first_token_time:.2f}s, Total: {total_time:.2f}s")
            
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
            
            # Store in memory if we got a valid response
            if response and not response.startswith("I'm sorry") and not response.startswith("Error") and not response.startswith("I'm having trouble"):
                await self.memory_manager.add_interaction(query, response)
            
            return response
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Router error: {e}")
                import traceback
                traceback.print_exc()
            
            # Try direct non-streaming fallback
            try:
                analysis = self._analyze_query_speed(query)
                decision = self._make_lightning_decision(query, analysis)
                
                personality_context = await self.personality_manager.get_system_prompt()
                memory_context = await self.memory_manager.get_context() if not analysis['is_simple'] else ""
                
                if decision.use_offline and self.offline_available and self.offline_llm:
                    response = await self.offline_llm.generate_response(
                        query, personality_context, memory_context
                    )
                elif decision.use_online and self.online_available and self.online_llm:
                    response = await self.online_llm.generate_response(
                        query, personality_context, memory_context
                    )
                else:
                    response = "I'm having trouble processing your request. Please ensure Ollama is running or configure online API keys."
                
                if response and not response.startswith("I'm sorry") and not response.startswith("Error"):
                    await self.memory_manager.add_interaction(query, response)
                
                return response
                
            except Exception as fallback_error:
                return f"I encountered an error processing your request: {fallback_error}"
    
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
            'performance_mode': settings.performance_mode,
            'streaming_enabled': settings.streaming_enabled
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
        
        # Get offline model info
        if self.offline_llm and hasattr(self.offline_llm, 'get_performance_stats'):
            status['offline_model_info'] = self.offline_llm.get_performance_stats()
        
        # Get online provider info
        if self.online_llm and hasattr(self.online_llm, 'get_provider_stats'):
            status['online_provider_info'] = self.online_llm.get_provider_stats()
        
        return status

# For backwards compatibility
Router = LightningRouter
