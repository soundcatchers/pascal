"""
Pascal AI Assistant - Smart Router with Performance Optimization
Intelligently routes requests between offline and online LLMs with Pi 5 optimizations
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
    """Represents a routing decision with performance optimization"""
    
    def __init__(self, use_offline: bool, reason: str, confidence: float = 1.0, 
                 suggested_profile: str = "balanced"):
        self.use_offline = use_offline
        self.reason = reason
        self.confidence = confidence
        self.suggested_profile = suggested_profile  # speed, balanced, quality
        self.timestamp = time.time()
    
    @property
    def use_online(self) -> bool:
        return not self.use_offline

class Router:
    """Smart router for offline/online LLM selection with Pi 5 optimizations"""
    
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
        
        # Performance tracking with profiles
        self.performance_stats = {
            'speed': {'offline_times': [], 'online_times': [], 'failures': 0},
            'balanced': {'offline_times': [], 'online_times': [], 'failures': 0},
            'quality': {'offline_times': [], 'online_times': [], 'failures': 0}
        }
        
        # Pi 5 specific optimizations
        self.pi5_optimizations = {
            'adaptive_profiling': True,
            'thermal_aware': True,
            'memory_aware': True,
            'context_switching': True
        }
        
        # Query complexity patterns for Pi 5
        self.complexity_patterns = {
            'simple_greetings': ['hi', 'hello', 'hey', 'thanks', 'bye', 'yes', 'no', 'ok'],
            'quick_questions': ['what is', 'who is', 'when is', 'where is'],
            'code_requests': ['code', 'function', 'script', 'program', 'debug'],
            'analysis_requests': ['analyze', 'compare', 'evaluate', 'research', 'detailed'],
            'creative_requests': ['write', 'create', 'story', 'poem', 'creative', 'generate'],
            'current_info': ['current', 'latest', 'recent', 'today', 'now', 'news', 'update']
        }
    
    async def _check_llm_availability(self):
        """Check which LLM options are available with Pi 5 optimizations"""
        try:
            print("DEBUG: Router checking LLM availability with Pi 5 optimizations...")
            
            # Check offline LLM with optimized module
            if settings.is_local_model_available():
                try:
                    print("DEBUG: Importing OptimizedOfflineLLM...")
                    from modules.offline_llm import OptimizedOfflineLLM
                    print("DEBUG: Creating OptimizedOfflineLLM instance...")
                    self.offline_llm = OptimizedOfflineLLM()
                    print("DEBUG: Initializing OptimizedOfflineLLM...")
                    init_result = await self.offline_llm.initialize()
                    print(f"DEBUG: OptimizedOfflineLLM initialize returned: {init_result}")
                    self.offline_available = init_result and self.offline_llm.is_available()
                    print(f"DEBUG: Final offline_available: {self.offline_available}")
                    
                    if self.offline_available:
                        # Set initial performance profile based on hardware
                        self._set_optimal_initial_profile()
                        
                except Exception as e:
                    print(f"DEBUG: OptimizedOfflineLLM initialization failed: {e}")
                    self.offline_available = False
            
            # Check online LLM
            if settings.is_online_available():
                try:
                    print("DEBUG: Importing OnlineLLM...")
                    from modules.online_llm import OnlineLLM
                    print("DEBUG: Creating OnlineLLM instance...")
                    self.online_llm = OnlineLLM()
                    print("DEBUG: Initializing OnlineLLM...")
                    init_result = await self.online_llm.initialize()
                    self.online_available = init_result
                    print(f"DEBUG: Final online_available: {self.online_available}")
                except Exception as e:
                    print(f"DEBUG: OnlineLLM initialization failed: {e}")
                    self.online_available = False
            
            print(f"DEBUG: Final availability - Offline: {self.offline_available}, Online: {self.online_available}")
            
        except Exception as e:
            print(f"DEBUG: Error in _check_llm_availability: {e}")
    
    def _set_optimal_initial_profile(self):
        """Set optimal initial performance profile based on Pi 5 hardware"""
        if not self.offline_llm:
            return
        
        # Get hardware info
        hw_info = settings.get_hardware_info()
        
        # Set profile based on available RAM and performance mode
        if hw_info['available_ram_gb'] >= 12:
            # Plenty of RAM - can use quality profile
            initial_profile = settings.performance_mode
        elif hw_info['available_ram_gb'] >= 8:
            # Moderate RAM - balanced is best
            initial_profile = 'balanced'
        else:
            # Limited RAM - prioritize speed
            initial_profile = 'speed'
        
        self.offline_llm.set_performance_profile(initial_profile)
        print(f"Set initial performance profile: {initial_profile}")
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Enhanced query analysis with Pi 5 specific optimizations"""
        query_lower = query.lower()
        
        # Analyze against complexity patterns
        complexity_scores = {}
        for category, patterns in self.complexity_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            complexity_scores[category] = score
        
        # Calculate metrics
        word_count = len(query.split())
        char_count = len(query)
        question_marks = query.count('?')
        
        # Determine complexity level
        complexity_level = self._calculate_complexity_level(complexity_scores, word_count)
        
        # Determine optimal profile
        optimal_profile = self._determine_optimal_profile(complexity_scores, complexity_level, word_count)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'question_marks': question_marks,
            'complexity_scores': complexity_scores,
            'complexity_level': complexity_level,
            'optimal_profile': optimal_profile,
            'needs_current_info': complexity_scores.get('current_info', 0) > 0,
            'is_creative': complexity_scores.get('creative_requests', 0) > 0,
            'is_code_related': complexity_scores.get('code_requests', 0) > 0,
            'is_analysis': complexity_scores.get('analysis_requests', 0) > 0,
            'estimated_response_time': self._estimate_response_time(complexity_level, optimal_profile)
        }
    
    def _calculate_complexity_level(self, scores: Dict[str, int], word_count: int) -> str:
        """Calculate complexity level with Pi 5 optimizations"""
        # Simple patterns
        if scores.get('simple_greetings', 0) > 0 and word_count <= 5:
            return 'simple'
        
        # Quick questions
        if scores.get('quick_questions', 0) > 0 and word_count <= 15:
            return 'simple'
        
        # Complex analysis
        if scores.get('analysis_requests', 0) > 0 or word_count > 50:
            return 'complex'
        
        # Code requests are medium complexity
        if scores.get('code_requests', 0) > 0:
            return 'medium'
        
        # Creative requests can be medium to complex
        if scores.get('creative_requests', 0) > 0:
            return 'medium' if word_count <= 30 else 'complex'
        
        # Default based on length
        if word_count <= 10:
            return 'simple'
        elif word_count <= 30:
            return 'medium'
        else:
            return 'complex'
    
    def _determine_optimal_profile(self, scores: Dict[str, int], complexity: str, word_count: int) -> str:
        """Determine optimal performance profile for Pi 5"""
        # Current info always needs online (no profile optimization needed)
        if scores.get('current_info', 0) > 0:
            return 'balanced'  # Will use online anyway
        
        # Simple queries - optimize for speed
        if complexity == 'simple':
            return 'speed'
        
        # Code requests - balanced is usually best
        if scores.get('code_requests', 0) > 0:
            return 'balanced'
        
        # Analysis requests - use quality for better reasoning
        if scores.get('analysis_requests', 0) > 0:
            return 'quality'
        
        # Creative requests - quality for better creativity
        if scores.get('creative_requests', 0) > 0:
            return 'quality' if word_count > 20 else 'balanced'
        
        # Default based on complexity
        profile_map = {
            'simple': 'speed',
            'medium': 'balanced', 
            'complex': 'quality'
        }
        
        return profile_map.get(complexity, 'balanced')
    
    def _estimate_response_time(self, complexity: str, profile: str) -> float:
        """Estimate response time for Pi 5"""
        base_times = {
            'speed': {'simple': 1.5, 'medium': 2.0, 'complex': 3.0},
            'balanced': {'simple': 2.0, 'medium': 3.0, 'complex': 4.5},
            'quality': {'simple': 3.0, 'medium': 4.5, 'complex': 6.0}
        }
        
        return base_times.get(profile, base_times['balanced']).get(complexity, 3.0)
    
    def _make_routing_decision(self, query: str, analysis: Dict[str, Any]) -> RouteDecision:
        """Make intelligent routing decision with performance optimization"""
        
        # Force modes
        if self.mode == RouteMode.OFFLINE_ONLY:
            return RouteDecision(True, "Forced offline mode", 1.0, analysis['optimal_profile'])
        elif self.mode == RouteMode.ONLINE_ONLY:
            return RouteDecision(False, "Forced online mode", 1.0, 'balanced')
        
        # Check availability
        if not self.offline_available and not self.online_available:
            return RouteDecision(True, "No LLMs available (fallback)", 0.1, 'speed')
        elif not self.offline_available:
            return RouteDecision(False, "Offline LLM not available", 0.9, 'balanced')
        elif not self.online_available:
            return RouteDecision(True, "Online LLM not available", 0.9, analysis['optimal_profile'])
        
        # Decision factors
        needs_current = analysis['needs_current_info']
        complexity_level = analysis['complexity_level']
        optimal_profile = analysis['optimal_profile']
        
        # High confidence decisions
        if needs_current:
            return RouteDecision(False, "Query requires current information", 0.95, 'balanced')
        
        # Pi 5 optimized decisions
        if complexity_level == 'simple' and settings.prefer_offline:
            return RouteDecision(True, "Simple query - offline speed optimized", 0.9, 'speed')
        
        if complexity_level == 'complex' and analysis['is_analysis']:
            # For complex analysis, consider both options
            offline_perf = self._get_offline_performance_score(optimal_profile)
            if offline_perf > 0.7:  # Good offline performance
                return RouteDecision(True, "Complex analysis - offline capable", 0.8, 'quality')
            else:
                return RouteDecision(False, "Complex analysis - online preferred", 0.8, 'balanced')
        
        # Code requests - often better offline for privacy
        if analysis['is_code_related'] and settings.prefer_offline:
            return RouteDecision(True, "Code request - offline for privacy", 0.8, 'balanced')
        
        # Creative requests - depends on quality needs
        if analysis['is_creative']:
            if optimal_profile == 'quality' and self.offline_available:
                return RouteDecision(True, "Creative request - offline quality mode", 0.7, 'quality')
            else:
                return RouteDecision(False, "Creative request - online variety", 0.7, 'balanced')
        
        # Default decision based on preferences and performance
        if settings.prefer_offline:
            offline_perf = self._get_offline_performance_score(optimal_profile)
            if offline_perf > 0.6:
                return RouteDecision(True, f"Offline preferred - {optimal_profile} profile", 0.7, optimal_profile)
        
        # Fallback to online for uncertain cases
        return RouteDecision(False, "Default online for reliability", 0.6, 'balanced')
    
    def _get_offline_performance_score(self, profile: str) -> float:
        """Get performance score for offline LLM with specific profile"""
        if not self.offline_llm:
            return 0.0
        
        stats = self.performance_stats.get(profile, {})
        offline_times = stats.get('offline_times', [])
        failures = stats.get('failures', 0)
        
        if not offline_times:
            return 0.8  # Default optimistic score
        
        # Calculate score based on average response time and failure rate
        avg_time = sum(offline_times[-10:]) / len(offline_times[-10:])  # Last 10 responses
        total_attempts = len(offline_times) + failures
        failure_rate = failures / total_attempts if total_attempts > 0 else 0
        
        # Good performance: < 5s response time, < 10% failure rate
        time_score = max(0, 1 - (avg_time - 2) / 8)  # Score decreases after 2s
        reliability_score = max(0, 1 - failure_rate * 10)  # Penalty for failures
        
        return (time_score + reliability_score) / 2
    
    async def get_response(self, query: str) -> str:
        """Get response from appropriate LLM with performance optimization"""
        start_time = time.time()
        
        try:
            # Analyze query with Pi 5 optimizations
            analysis = self._analyze_query_complexity(query)
            
            # Make routing decision
            decision = self._make_routing_decision(query, analysis)
            self.last_decision = decision
            
            if settings.debug_mode:
                print(f"Routing decision: {'Offline' if decision.use_offline else 'Online'}")
                print(f"Reason: {decision.reason}")
                print(f"Suggested profile: {decision.suggested_profile}")
                print(f"Estimated time: {analysis['estimated_response_time']:.1f}s")
            
            # Get personality and memory context
            personality_context = await self.personality_manager.get_system_prompt()
            memory_context = await self.memory_manager.get_context()
            
            # Route to appropriate LLM with profile optimization
            response = None
            if decision.use_offline and self.offline_available:
                response = await self._get_offline_response(
                    query, personality_context, memory_context, decision.suggested_profile
                )
                
                # Track performance
                response_time = time.time() - start_time
                profile_stats = self.performance_stats.get(decision.suggested_profile, {})
                if 'offline_times' not in profile_stats:
                    profile_stats['offline_times'] = []
                profile_stats['offline_times'].append(response_time)
                
                # Keep only last 20 measurements
                if len(profile_stats['offline_times']) > 20:
                    profile_stats['offline_times'] = profile_stats['offline_times'][-20:]
                
                self.performance_stats[decision.suggested_profile] = profile_stats
            
            elif decision.use_online and self.online_available:
                response = await self._get_online_response(query, personality_context, memory_context)
                
                # Track online performance
                response_time = time.time() - start_time
                profile_stats = self.performance_stats.get('balanced', {})
                if 'online_times' not in profile_stats:
                    profile_stats['online_times'] = []
                profile_stats['online_times'].append(response_time)
                
                if len(profile_stats['online_times']) > 20:
                    profile_stats['online_times'] = profile_stats['online_times'][-20:]
                
                self.performance_stats['balanced'] = profile_stats
            
            else:
                # Fallback logic
                if self.offline_available:
                    response = await self._get_offline_response(
                        query, personality_context, memory_context, 'speed'
                    )
                elif self.online_available:
                    response = await self._get_online_response(query, personality_context, memory_context)
                else:
                    response = "I'm sorry, but I'm currently unable to process your request. Please check my configuration."
            
            # Store in memory
            if response:
                await self.memory_manager.add_interaction(query, response)
            
            return response or "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            # Track failures
            if self.last_decision and self.last_decision.use_offline:
                profile = self.last_decision.suggested_profile
                if profile not in self.performance_stats:
                    self.performance_stats[profile] = {}
                self.performance_stats[profile]['failures'] = self.performance_stats[profile].get('failures', 0) + 1
            
            if settings.debug_mode:
                print(f"Router error: {e}")
            
            return f"I encountered an error processing your request: {e}"
    
    async def _get_offline_response(self, query: str, personality_context: str, 
                                  memory_context: str, profile: str = 'balanced') -> str:
        """Get response from offline LLM with performance profile"""
        if not self.offline_llm:
            raise Exception("Offline LLM not initialized")
        
        # Set the performance profile
        self.offline_llm.set_performance_profile(profile)
        
        return await self.offline_llm.generate_response(query, personality_context, memory_context, profile)
    
    async def _get_online_response(self, query: str, personality_context: str, memory_context: str) -> str:
        """Get response from online LLM"""
        if not self.online_llm:
            raise Exception("Online LLM not initialized")
        
        return await self.online_llm.generate_response(query, personality_context, memory_context)
    
    def set_mode(self, mode: RouteMode):
        """Set routing mode"""
        self.mode = mode
    
    def set_performance_preference(self, preference: str):
        """Set performance preference (speed/balanced/quality)"""
        if preference in ['speed', 'balanced', 'quality']:
            settings.set_performance_mode(preference)
            if self.offline_llm:
                self.offline_llm.set_performance_profile(preference)
            print(f"Performance preference set to: {preference}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive router status with Pi 5 specific info"""
        return {
            'mode': self.mode.value,
            'offline_available': self.offline_available,
            'online_available': self.online_available,
            'hardware_info': settings.get_hardware_info(),
            'performance_mode': settings.performance_mode,
            'last_decision': {
                'use_offline': self.last_decision.use_offline if self.last_decision else None,
                'reason': self.last_decision.reason if self.last_decision else None,
                'confidence': self.last_decision.confidence if self.last_decision else None,
                'suggested_profile': self.last_decision.suggested_profile if self.last_decision else None
            } if self.last_decision else None,
            'performance_stats': self._get_performance_summary(),
            'offline_model_info': self.offline_llm.get_performance_stats() if self.offline_llm else None
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance across all profiles"""
        summary = {}
        
        for profile, stats in self.performance_stats.items():
            offline_times = stats.get('offline_times', [])
            online_times = stats.get('online_times', [])
            failures = stats.get('failures', 0)
            
            summary[profile] = {
                'avg_offline_time': sum(offline_times[-10:]) / len(offline_times[-10:]) if offline_times else 0,
                'avg_online_time': sum(online_times[-10:]) / len(online_times[-10:]) if online_times else 0,
                'total_offline_responses': len(offline_times),
                'total_online_responses': len(online_times),
                'failures': failures,
                'reliability_score': self._get_offline_performance_score(profile)
            }
        
        return summary
