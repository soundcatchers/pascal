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
        
        # Initialize LLM modules (will be imported when needed)
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
            'needs_current_info': ['current', 'latest', 'recent', 'today', 'now', 'news', 'update', 'weather'],
            'simple_queries': ['hi', 'hello', 'hey', 'thanks', 'bye', 'yes', 'no', 'ok', 'what is', 'who is'],
            'complex_queries': ['analyze', 'compare', 'evaluate', 'research', 'detailed', 'comprehensive', 'explain']
        }
    
    async def _check_llm_availability(self):
        """Check which LLM options are available"""
        try:
            if settings.debug_mode:
                print("⚡ Lightning Router checking LLM availability...")
            
            # Check offline LLM with lightning module
            if settings.is_local_model_available():
                try:
                    from modules.offline_llm import LightningOfflineLLM
                    self.offline_llm = LightningOfflineLLM()
                    self.offline_available = await self.offline_llm.initialize()
                    
                    if self.offline_available:
                        print("⚡ Offline LLM ready with lightning speed")
                        
                except Exception as e:
                    print(f"❌ Offline LLM initialization failed: {e}")
                    self.offline_available = False
            
            # Check online LLM (Grok priority)
            if settings.is_online_available():
                try:
                    from modules.online_llm import OnlineLLM
                    self.online_llm = OnlineLLM()
                    self.online_available = await self.online_llm.initialize()
                    
                    if self.online_available:
                        print("✅ Online LLM ready (Grok/OpenAI/Anthropic)")
                        
                except Exception as e:
                    print(f"❌ Online LLM initialization failed: {e}")
                    self.online_available = False
            
            if settings.debug_mode:
                print(f"Final availability - Offline: {self.offline_available}, Online: {self.online_available}")
            
        except Exception as e:
            print(f"❌ Error in LLM availability check: {e}")
    
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
        """Make ultra-fast routing decision"""
        
        # Force modes
        if self.mode == RouteMode.OFFLINE_ONLY:
            return RouteDecision(True, "Forced offline mode", 1.0)
        elif self.mode == RouteMode.ONLINE_ONLY:
            return RouteDecision(False, "Forced online mode", 1.0)
        
        # Check availability
        if not self.offline_available and not self.online_available:
            return RouteDecision(True, "No LLMs available (fallback)", 0.1)
        elif not self.offline_available:
            return RouteDecision(False, "Offline LLM not available", 0.9)
        elif not self.online_available:
            return RouteDecision(True, "Online LLM not available", 0.9)
        
        # Lightning routing logic
        
        # Always use online for current information
        if analysis['needs_current_info']:
            return RouteDecision(False, "Query requires current information", 0.95)
        
        # Prefer offline for everything else (speed priority)
        if self.mode == RouteMode.OFFLINE_PREFERRED or settings.prefer_offline:
            return RouteDecision(True, "Offline preferred for speed", 0.9)
        
        # Simple queries always offline for speed
        if analysis['is_simple']:
            return RouteDecision(True, "Simple query - lightning offline", 0.95)
        
        # Complex queries - still prefer offline unless estimated time > 3s
        if analysis['estimated_offline_time'] <= settings.target_response_time:
            return RouteDecision(True, "Within target response time", 0.8)
        
        # Fallback to online for very complex queries
        return RouteDecision(False, "Complex query exceeding time target", 0.7)
    
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
                print(f"⚡ Routing: {'Offline' if decision.use_offline else 'Online'} - {decision.reason}")
            
            # Get personality and memory context (minimal for speed)
            personality_context = await self.personality_manager.get_system_prompt()
            memory_context = ""  # Skip memory for speed unless necessary
            
            if not analysis['is_simple']:
                memory_context = await self.memory_manager.get_context()
            
            # Stream from appropriate source
            streamed_any = False
            
            if decision.use_offline and self.offline_available:
                async for chunk in self.offline_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    if not first_token_time:
                        first_token_time = time.time() - start_time
                        self.first_token_times['offline'].append(first_token_time)
                    yield chunk
                    streamed_any = True
            
            elif decision.use_online and self.online_available:
                async for chunk in self.online_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    if not first_token_time:
                        first_token_time = time.time() - start_time
                        self.first_token_times['online'].append(first_token_time)
                    yield chunk
                    streamed_any = True
            
            else:
                # Fallback
                if self.offline_available:
                    async for chunk in self.offline_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                        streamed_any = True
                elif self.online_available:
                    async for chunk in self.online_llm.generate_response_stream(
                        query, personality_context, memory_context
                    ):
                        yield chunk
                        streamed_any = True
                else:
                    yield "I'm sorry, but I'm currently unable to process your request. Please check my configuration."
            
            # Track total response time
            total_time = time.time() - start_time
            source = 'offline' if decision.use_offline else 'online'
            self.response_times[source].append(total_time)
            
            # Keep only last 20 measurements
            if len(self.response_times[source]) > 20:
                self.response_times[source] = self.response_times[source][-20:]
            if len(self.first_token_times[source]) > 20:
                self.first_token_times[source] = self.first_token_times[source][-20:]
            
            if settings.debug_mode and first_token_time:
                print(f"⚡ First token: {first_token_time:.2f}s, Total: {total_time:.2f}s")
            
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Streaming error: {e}")
            yield f"I encountered an error: {str(e)}"
    
    async def get_response(self, query: str) -> str:
        """Get complete response (non-streaming fallback)"""
        start_time = time.time()
        
        try:
            # Collect streaming response
            response_parts = []
            async for chunk in self.get_streaming_response(query):
                response_parts.append(chunk)
            
            response = ''.join(response_parts)
            
            # Store in memory
            if response:
                await self.memory_manager.add_interaction(query, response)
            
            return response
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Router error: {e}")
            
            # Try non-streaming fallback
            try:
                analysis = self._analyze_query_speed(query)
                decision = self._make_lightning_decision(query, analysis)
                
                personality_context = await self.personality_manager.get_system_prompt()
                memory_context = await self.memory_manager.get_context()
                
                if decision.use_offline and self.offline_available:
                    response = await self.offline_llm.generate_response(
                        query, personality_context, memory_context
                    )
                elif decision.use_online and self.online_available:
                    response = await self.online_llm.generate_response(
                        query, personality_context, memory_context
                    )
                else:
                    response = "I'm having trouble processing your request."
                
                if response:
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
            if self.offline_llm:
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
            status['avg_offline_time'] = sum(self.response_times['offline']) / len(self.response_times['offline'])
        if self.response_times['online']:
            status['avg_online_time'] = sum(self.response_times['online']) / len(self.response_times['online'])
        
        if self.first_token_times['offline']:
            status['avg_offline_first_token'] = sum(self.first_token_times['offline']) / len(self.first_token_times['offline'])
        if self.first_token_times['online']:
            status['avg_online_first_token'] = sum(self.first_token_times['online']) / len(self.first_token_times['online'])
        
        if self.last_decision:
            status['last_decision'] = {
                'use_offline': self.last_decision.use_offline,
                'reason': self.last_decision.reason,
                'confidence': self.last_decision.confidence
            }
        
        if self.offline_llm:
            status['offline_model_info'] = self.offline_llm.get_performance_stats()
        
        return status

# For backwards compatibility
Router = LightningRouter
