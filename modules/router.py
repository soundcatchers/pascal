"""
Pascal AI Assistant - Smart Router
Intelligently routes requests between offline and online LLMs
"""

import asyncio
import time
from typing import Optional, Dict, Any
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
    """Represents a routing decision"""
    
    def __init__(self, use_offline: bool, reason: str, confidence: float = 1.0):
        self.use_offline = use_offline
        self.reason = reason
        self.confidence = confidence
        self.timestamp = time.time()
    
    @property
    def use_online(self) -> bool:
        return not self.use_offline

class Router:
    """Smart router for offline/online LLM selection"""
    
    def __init__(self, personality_manager, memory_manager):
        self.personality_manager = personality_manager
        self.memory_manager = memory_manager
        
        # Initialize LLM modules (will be imported when needed)
        self.offline_llm = None
        self.online_llm = None
        
        # Router state
        self.mode = RouteMode.AUTO
        self.last_decision = None
        self.offline_available = False
        self.online_available = False
        
        # Performance tracking
        self.offline_response_times = []
        self.online_response_times = []
        self.offline_failures = 0
        self.online_failures = 0
        
        # Initialize LLM availability
        asyncio.create_task(self._check_llm_availability())
    
    async def _check_llm_availability(self):
        """Check which LLM options are available"""
        try:
            print("DEBUG: Router checking LLM availability...")
            
            # Check offline LLM
            print(f"DEBUG: settings.is_local_model_available() = {settings.is_local_model_available()}")
            if settings.is_local_model_available():
                try:
                    print("DEBUG: Importing OfflineLLM...")
                    from modules.offline_llm import OfflineLLM
                    print("DEBUG: Creating OfflineLLM instance...")
                    self.offline_llm = OfflineLLM()
                    print("DEBUG: Initializing OfflineLLM...")
                    init_result = await self.offline_llm.initialize()
                    print(f"DEBUG: OfflineLLM initialize returned: {init_result}")
                    print(f"DEBUG: OfflineLLM is_available: {self.offline_llm.is_available()}")
                    self.offline_available = init_result and self.offline_llm.is_available()
                    print(f"DEBUG: Final offline_available: {self.offline_available}")
                except Exception as e:
                    print(f"DEBUG: OfflineLLM initialization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self.offline_available = False
            else:
                print("DEBUG: Local model not available according to settings")
            
            # Check online LLM
            print(f"DEBUG: settings.is_online_available() = {settings.is_online_available()}")
            if settings.is_online_available():
                try:
                    print("DEBUG: Importing OnlineLLM...")
                    from modules.online_llm import OnlineLLM
                    print("DEBUG: Creating OnlineLLM instance...")
                    self.online_llm = OnlineLLM()
                    print("DEBUG: Initializing OnlineLLM...")
                    init_result = await self.online_llm.initialize()
                    print(f"DEBUG: OnlineLLM initialize returned: {init_result}")
                    self.online_available = init_result
                    print(f"DEBUG: Final online_available: {self.online_available}")
                except Exception as e:
                    print(f"DEBUG: OnlineLLM initialization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self.online_available = False
            else:
                print("DEBUG: Online APIs not available according to settings")
            
            print(f"DEBUG: Final availability - Offline: {self.offline_available}, Online: {self.online_available}")
            
        except Exception as e:
            print(f"DEBUG: Error in _check_llm_availability: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine complexity and requirements"""
        query_lower = query.lower()
        
        # Simple heuristics for complexity analysis
        complexity_indicators = {
            'simple': ['hi', 'hello', 'thanks', 'yes', 'no', 'ok'],
            'medium': ['how', 'what', 'where', 'when', 'why', 'explain'],
            'complex': ['analyze', 'compare', 'research', 'detailed', 'comprehensive'],
            'current': ['current', 'latest', 'recent', 'today', 'now', 'update'],
            'creative': ['write', 'create', 'generate', 'story', 'poem', 'creative'],
            'technical': ['code', 'program', 'debug', 'algorithm', 'technical']
        }
        
        scores = {}
        for category, keywords in complexity_indicators.items():
            scores[category] = sum(1 for keyword in keywords if keyword in query_lower)
        
        # Calculate overall complexity
        word_count = len(query.split())
        char_count = len(query)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'complexity_scores': scores,
            'needs_current_info': scores.get('current', 0) > 0,
            'is_creative': scores.get('creative', 0) > 0,
            'is_technical': scores.get('technical', 0) > 0,
            'estimated_complexity': self._calculate_complexity_score(scores, word_count)
        }
    
    def _calculate_complexity_score(self, scores: Dict[str, int], word_count: int) -> str:
        """Calculate overall complexity score"""
        if scores.get('simple', 0) > 0 and word_count < 5:
            return 'simple'
        elif scores.get('complex', 0) > 0 or word_count > 50:
            return 'complex'
        else:
            return 'medium'
    
    def _make_routing_decision(self, query: str, analysis: Dict[str, Any]) -> RouteDecision:
        """Make intelligent routing decision"""
        
        # Force modes
        if self.mode == RouteMode.OFFLINE_ONLY:
            return RouteDecision(True, "Forced offline mode")
        elif self.mode == RouteMode.ONLINE_ONLY:
            return RouteDecision(False, "Forced online mode")
        
        # Check availability
        if not self.offline_available and not self.online_available:
            return RouteDecision(True, "No LLMs available (fallback to offline)")
        elif not self.offline_available:
            return RouteDecision(False, "Offline LLM not available")
        elif not self.online_available:
            return RouteDecision(True, "Online LLM not available")
        
        # Decision factors
        factors = {
            'prefer_offline': settings.prefer_offline,
            'needs_current': analysis['needs_current_info'],
            'is_complex': analysis['estimated_complexity'] == 'complex',
            'offline_faster': self._is_offline_faster(),
            'offline_reliable': self._is_offline_reliable(),
        }
        
        # Decision logic
        if factors['needs_current']:
            return RouteDecision(False, "Query requires current information", 0.9)
        
        if factors['prefer_offline'] and not factors['is_complex']:
            return RouteDecision(True, "Simple query + offline preference", 0.8)
        
        if factors['is_complex'] and self.online_available:
            return RouteDecision(False, "Complex query better handled online", 0.7)
        
        if factors['offline_faster'] and factors['offline_reliable']:
            return RouteDecision(True, "Offline is faster and reliable", 0.8)
        
        # Default to user preference
        if settings.prefer_offline:
            return RouteDecision(True, "Default offline preference", 0.6)
        else:
            return RouteDecision(False, "Default online preference", 0.6)
    
    def _is_offline_faster(self) -> bool:
        """Check if offline LLM is faster based on history"""
        if not self.offline_response_times or not self.online_response_times:
            return True  # Assume offline is faster initially
        
        avg_offline = sum(self.offline_response_times[-10:]) / len(self.offline_response_times[-10:])
        avg_online = sum(self.online_response_times[-10:]) / len(self.online_response_times[-10:])
        
        return avg_offline < avg_online
    
    def _is_offline_reliable(self) -> bool:
        """Check if offline LLM is reliable based on failure rate"""
        total_offline_attempts = len(self.offline_response_times) + self.offline_failures
        if total_offline_attempts == 0:
            return True
        
        failure_rate = self.offline_failures / total_offline_attempts
        return failure_rate < 0.1  # Less than 10% failure rate
    
    async def get_response(self, query: str) -> str:
        """Get response from appropriate LLM"""
        start_time = time.time()
        
        try:
            # Analyze query
            analysis = self._analyze_query_complexity(query)
            
            # Make routing decision
            decision = self._make_routing_decision(query, analysis)
            self.last_decision = decision
            
            if settings.debug_mode:
                print(f"Routing decision: {'Offline' if decision.use_offline else 'Online'} - {decision.reason}")
            
            # Get personality context
            personality_context = await self.personality_manager.get_system_prompt()
            
            # Get memory context
            memory_context = await self.memory_manager.get_context()
            
            # Route to appropriate LLM
            if decision.use_offline and self.offline_available:
                response = await self._get_offline_response(query, personality_context, memory_context)
                response_time = time.time() - start_time
                self.offline_response_times.append(response_time)
            
            elif decision.use_online and self.online_available:
                response = await self._get_online_response(query, personality_context, memory_context)
                response_time = time.time() - start_time
                self.online_response_times.append(response_time)
            
            else:
                # Fallback logic
                if self.offline_available:
                    response = await self._get_offline_response(query, personality_context, memory_context)
                    response_time = time.time() - start_time
                    self.offline_response_times.append(response_time)
                elif self.online_available:
                    response = await self._get_online_response(query, personality_context, memory_context)
                    response_time = time.time() - start_time
                    self.online_response_times.append(response_time)
                else:
                    response = "I'm sorry, but I'm currently unable to process your request. Please check my configuration."
            
            # Store in memory
            await self.memory_manager.add_interaction(query, response)
            
            return response
            
        except Exception as e:
            # Track failures
            if self.last_decision and self.last_decision.use_offline:
                self.offline_failures += 1
            else:
                self.online_failures += 1
            
            if settings.debug_mode:
                print(f"Router error: {e}")
            
            return f"I encountered an error processing your request: {e}"
    
    async def _get_offline_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Get response from offline LLM"""
        if not self.offline_llm:
            raise Exception("Offline LLM not initialized")
        
        return await self.offline_llm.generate_response(query, personality_context, memory_context)
    
    async def _get_online_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Get response from online LLM"""
        if not self.online_llm:
            raise Exception("Online LLM not initialized")
        
        return await self.online_llm.generate_response(query, personality_context, memory_context)
    
    def set_mode(self, mode: RouteMode):
        """Set routing mode"""
        self.mode = mode
    
    def get_status(self) -> Dict[str, Any]:
        """Get router status information"""
        return {
            'mode': self.mode.value,
            'offline_available': self.offline_available,
            'online_available': self.online_available,
            'last_decision': {
                'use_offline': self.last_decision.use_offline if self.last_decision else None,
                'reason': self.last_decision.reason if self.last_decision else None,
                'confidence': self.last_decision.confidence if self.last_decision else None
            } if self.last_decision else None,
            'performance': {
                'avg_offline_time': sum(self.offline_response_times[-10:]) / len(self.offline_response_times[-10:]) if self.offline_response_times else 0,
                'avg_online_time': sum(self.online_response_times[-10:]) / len(self.online_response_times[-10:]) if self.online_response_times else 0,
                'offline_failures': self.offline_failures,
                'online_failures': self.online_failures
            }
        }
