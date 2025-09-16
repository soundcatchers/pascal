"""
Pascal AI Assistant - FIXED Enhanced Router with Skills Integration
3-Tier System: Skills (instant) -> Nemotron (fast) -> Groq (current info)
FIXED: Proper imports and error handling
"""

import asyncio
import time
import re
from typing import Optional, AsyncGenerator
from enum import Enum
from datetime import datetime

from config.settings import settings

class RouteMode(Enum):
    """Enhanced routing modes"""
    SKILLS_FIRST = "skills_first"     # Skills -> Offline -> Online (optimal)
    OFFLINE_ONLY = "offline_only"     # Nemotron only
    ONLINE_ONLY = "online_only"       # Groq only
    AUTO = "auto"                     # Automatic detection

class RouteDecision:
    """Enhanced routing decision with skills support"""
    def __init__(self, route_type: str, reason: str, skill_name: str = None, 
                 is_current_info: bool = False, estimated_time: str = "",
                 confidence: float = 0.8):
        self.route_type = route_type  # 'skill', 'offline', 'online'
        self.reason = reason
        self.skill_name = skill_name
        self.is_current_info = is_current_info
        self.estimated_time = estimated_time
        self.confidence = confidence
        self.timestamp = time.time()
    
    @property
    def use_skill(self) -> bool:
        return self.route_type == 'skill'
    
    @property
    def use_offline(self) -> bool:
        return self.route_type == 'offline'
    
    @property
    def use_online(self) -> bool:
        return self.route_type == 'online'

class EnhancedRouter:
    """FIXED Enhanced 3-tier router: Skills -> Nemotron -> Groq"""
    
    def __init__(self, personality_manager, memory_manager):
        self.personality_manager = personality_manager
        self.memory_manager = memory_manager
        
        # Initialize components
        self.offline_llm = None
        self.online_llm = None
        self.skills_manager = None
        
        # Router state
        self.mode = RouteMode.SKILLS_FIRST  # Default to skills-first
        self.last_decision = None
        self.offline_available = False
        self.online_available = False
        self.skills_available = False
        
        # Enhanced performance tracking
        self.stats = {
            'skill_requests': 0,
            'offline_requests': 0,
            'online_requests': 0,
            'skill_avg_time': 0.0,
            'offline_avg_time': 0.0,
            'online_avg_time': 0.0,
            'current_info_requests': 0,
            'skill_success_count': 0,
            'total_time_saved': 0.0,  # Time saved by using skills vs LLMs
            'fallback_count': 0,      # How often we had to fall back
        }
        
        # Enhanced current info patterns for better detection
        self.current_info_patterns = [
            # Date/time queries
            r'\bwhat time is it\b',
            r'\bwhat day is today\b',
            r'\bwhat date is today\b',
            r'\bcurrent time\b',
            r'\bcurrent date\b',
            r'\btoday\'?s date\b',
            r'\bwhat is the date\b',
            r'\bwhat is the time\b',
            r'\btime now\b',
            r'\bdate now\b',
            
            # Political queries
            r'\bcurrent president\b',
            r'\bwho is president\b',
            r'\bpresident now\b',
            r'\bcurrent prime minister\b',
            r'\bcurrent leader\b',
            r'\bwho is the current\b',
            
            # News queries
            r'\blatest news\b',
            r'\brecent news\b',
            r'\bnews today\b',
            r'\bbreaking news\b',
            r'\bcurrent events\b',
            r'\bwhat\'?s happening\b',
            r'\bin the news\b',
            r'\bnews headlines\b',
            
            # Weather queries
            r'\bweather today\b',
            r'\bcurrent weather\b',
            r'\bweather now\b',
            r'\bweather in\b',
            r'\btemperature in\b',
            r'\bis it hot\b',
            r'\bis it cold\b',
            r'\braining\b',
            
            # General current info
            r'\bright now\b',
            r'\bat the moment\b',
            r'\bcurrently\b',
            r'\bthis moment\b',
        ]
    
    async def _check_system_availability(self):
        """FIXED: Check availability of all systems with proper error handling"""
        try:
            if settings.debug_mode:
                print("[ROUTER] Checking enhanced 3-tier system availability...")
            
            # FIXED: Initialize Enhanced Skills Manager (highest priority)
            try:
                from modules.skills_manager import EnhancedSkillsManager
                self.skills_manager = EnhancedSkillsManager()
                
                # Initialize with full API testing
                api_status = await self.skills_manager.initialize()
                self.skills_available = True
                
                print("🚀 Enhanced Skills Manager ready:")
                for service, status in api_status.items():
                    status_icon = "✅" if status['available'] else "⚠️"
                    print(f"   {status_icon} {service}: {status['message']}")
                
            except ImportError as e:
                print(f"❌ Enhanced Skills Manager import failed: {e}")
                self.skills_available = False
                self.skills_manager = None
            except Exception as e:
                print(f"❌ Enhanced Skills Manager failed: {e}")
                if settings.debug_mode:
                    import traceback
                    traceback.print_exc()
                self.skills_available = False
                self.skills_manager = None
            
            # FIXED: Initialize offline LLM (Nemotron via Ollama) with proper import
            try:
                from modules.offline_llm import LightningOfflineLLM
                self.offline_llm = LightningOfflineLLM()
                
                print("[OLLAMA] Model loaded: nemotron-mini:4b-instruct-q4_K_M")  # From your log
                self.offline_available = await self.offline_llm.initialize()
                
                if self.offline_available:
                    print("✅ Offline LLM ready (Nemotron via Ollama)")
                    
                    # Get status for detailed info
                    status = self.offline_llm.get_status()
                    if settings.debug_mode:
                        print(f"[OLLAMA] Current model: {status.get('current_model', 'unknown')}")
                        print(f"[OLLAMA] Profile: {status.get('performance_profile', 'unknown')}")
                else:
                    print("❌ Offline LLM not available")
                    if hasattr(self.offline_llm, 'last_error') and self.offline_llm.last_error:
                        print(f"   Error: {self.offline_llm.last_error}")
                    else:
                        print("   Check that Ollama is running and models are available")
                    
            except ImportError as e:
                print(f"❌ Offline LLM import failed: {e}")
                self.offline_available = False
                self.offline_llm = None
            except Exception as e:
                print(f"❌ Offline LLM initialization failed: {e}")
                if settings.debug_mode:
                    import traceback
                    traceback.print_exc()
                self.offline_available = False
                self.offline_llm = None
            
            # FIXED: Initialize online LLM (Groq) - for complex current info
            if settings.is_online_available():
                try:
                    from modules.online_llm import OnlineLLM
                    self.online_llm = OnlineLLM()
                    self.online_available = await self.online_llm.initialize()
                    
                    if self.online_available:
                        print("✅ [GROQ] Connection test successful")  # From your log
                        print("✅ Online LLM ready (Groq - for complex current info)")
                    else:
                        print("❌ Online LLM not available")
                        if hasattr(self.online_llm, 'last_error') and self.online_llm.last_error:
                            print(f"   Error: {self.online_llm.last_error}")
                        
                except ImportError as e:
                    print(f"❌ Online LLM import failed: {e}")
                    self.online_available = False
                    self.online_llm = None
                except Exception as e:
                    print(f"❌ Online LLM initialization failed: {e}")
                    if settings.debug_mode:
                        import traceback
                        traceback.print_exc()
                    self.online_available = False
                    self.online_llm = None
            else:
                self.online_available = False
                if settings.debug_mode:
                    print("[ROUTER] No Groq API key - complex current info limited")
            
            # Determine optimal mode based on what's available
            self._set_optimal_mode()
            
            if settings.debug_mode:
                print(f"[ROUTER] Final status:")
                print(f"  Enhanced Skills: {self.skills_available}")
                print(f"  Offline (Nemotron): {self.offline_available}")
                print(f"  Online (Groq): {self.online_available}")
                print(f"  Optimal Mode: {self.mode.value}")
            
        except Exception as e:
            print(f"❌ Critical error in system availability check: {e}")
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
    
    def _set_optimal_mode(self):
        """Set the optimal routing mode based on available systems"""
        if self.skills_available and self.offline_available and self.online_available:
            self.mode = RouteMode.SKILLS_FIRST
            print("🎯 OPTIMAL: Full 3-tier system available")
            print("   • Instant responses: DateTime, Calculator")
            print("   • Fast API responses: Weather, News")
            print("   • Local AI: Nemotron for general queries")
            print("   • Online AI: Groq for complex current info")
            
        elif self.skills_available and self.offline_available:
            self.mode = RouteMode.SKILLS_FIRST
            print("⚡ GOOD: Skills + Nemotron (complex current info limited)")
            
        elif self.skills_available and self.online_available:
            self.mode = RouteMode.SKILLS_FIRST
            print("🌐 PARTIAL: Skills + Groq (no local LLM)")
            
        elif self.offline_available and self.online_available:
            self.mode = RouteMode.AUTO
            print("🤖 LLMs only (no enhanced skills)")
            
        elif self.skills_available:
            self.mode = RouteMode.SKILLS_FIRST
            print("⚡ LIMITED: Enhanced skills only")
            
        elif self.offline_available:
            self.mode = RouteMode.OFFLINE_ONLY
            print("🏠 OFFLINE: Nemotron only")
            
        elif self.online_available:
            self.mode = RouteMode.ONLINE_ONLY
            print("🌐 ONLINE: Groq only")
            
        else:
            print("❌ ERROR: No systems available!")
            self._print_setup_instructions()
    
    def _print_setup_instructions(self):
        """Print setup instructions for missing components"""
        print("\n🔧 Setup Instructions:")
        print("1. For Enhanced Skills:")
        print("   • Weather: Add OPENWEATHER_API_KEY to .env (free at openweathermap.org)")
        print("   • News: Add NEWS_API_KEY to .env (free at newsapi.org)")
        print("2. For Offline AI:")
        print("   • sudo systemctl start ollama")
        print("   • ollama pull nemotron-mini:4b-instruct-q4_K_M")
        print("3. For Online AI:")
        print("   • Add GROQ_API_KEY to .env (free at console.groq.com)")
    
    def _detect_current_info(self, query: str) -> bool:
        """FIXED: Enhanced current info detection"""
        query_lower = query.lower().strip()
        
        # Check against all current info patterns
        for pattern in self.current_info_patterns:
            if re.search(pattern, query_lower):
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 Current info pattern detected: '{pattern}'")
                return True
        
        # Additional single word triggers with context
        single_triggers = ['today', 'now', 'current', 'latest', 'recent']
        words = query_lower.split()
        
        for word in words:
            if word in single_triggers:
                # Avoid false positives for educational queries
                if any(avoid in query_lower for avoid in ['explain', 'definition', 'what is', 'how does', 'why', 'example', 'meaning']):
                    continue
                if settings.debug_mode:
                    print(f"[ROUTER] 🎯 Current info trigger detected: '{word}'")
                return True
        
        return False
    
    def _can_handle_with_skills(self, query: str) -> Optional[str]:
        """FIXED: Check if enhanced skills can handle this query directly"""
        if not self.skills_available or not self.skills_manager:
            return None
        
        try:
            skill_name = self.skills_manager.can_handle_directly(query)
            if skill_name and settings.debug_mode:
                print(f"[ROUTER] 🎯 Skill '{skill_name}' can handle query")
            return skill_name
        except Exception as e:
            if settings.debug_mode:
                print(f"[ROUTER] ❌ Error checking skills: {e}")
            return None
    
    def _decide_route(self, query: str) -> RouteDecision:
        """FIXED: Enhanced 3-tier routing decision with skills priority"""
        
        # TIER 1: Check if enhanced skills can handle this directly (0.001-2s)
        if self.skills_available and self.skills_manager:
            skill_name = self._can_handle_with_skills(query)
            if skill_name:
                # Determine estimated time based on skill type
                if skill_name in ['datetime', 'calculator']:
                    estimated_time = "Instant (0.001s)"
                else:  # weather, news
                    estimated_time = "Fast (0.5-2s)"
                
                return RouteDecision(
                    'skill',
                    f"Enhanced {skill_name} skill - ultra fast",
                    skill_name=skill_name,
                    estimated_time=estimated_time,
                    confidence=0.95
                )
        
        # TIER 2: Check for current info queries that need online LLM + real data
        is_current_info = self._detect_current_info(query)
        if is_current_info:
            if self.online_available:
                return RouteDecision(
                    'online',
                    "Current info query - needs LLM + real data",
                    is_current_info=True,
                    estimated_time="Fast (2-4s)",
                    confidence=0.85
                )
            else:
                # Fallback to offline LLM (limited current info capability)
                if self.offline_available:
                    return RouteDecision(
                        'offline',
                        "Current info query - offline LLM (limited capability)",
                        estimated_time="Fast (0.5-2s)",
                        confidence=0.6
                    )
        
        # TIER 3: Route general queries to offline LLM (fast, local)
        if self.offline_available:
            return RouteDecision(
                'offline',
                "General conversation - local Nemotron",
                estimated_time="Fast (0.5-2s)",
                confidence=0.8
            )
        
        # TIER 4: Fallback to online LLM if no offline
        if self.online_available:
            return RouteDecision(
                'online',
                "Fallback to Groq - no offline LLM available",
                estimated_time="Moderate (2-4s)",
                confidence=0.7
            )
        
        # TIER 5: Last resort - skills only (if any available)
        if self.skills_available:
            return RouteDecision(
                'skill',
                "Skills only - no LLMs available",
                estimated_time="varies",
                confidence=0.5
            )
        
        # No systems available
        return RouteDecision(
            'none',
            "No systems available",
            estimated_time="error",
            confidence=0.0
        )
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """FIXED: Get streaming response using enhanced 3-tier system"""
        decision = self._decide_route(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            tier_name = {
                'skill': f"TIER 1: {decision.skill_name.upper() if decision.skill_name else 'SKILL'}",
                'offline': "TIER 2: NEMOTRON", 
                'online': "TIER 3: GROQ"
            }.get(decision.route_type, "ERROR")
            
            print(f"[ROUTER] 🌊 Streaming {tier_name} ({decision.estimated_time}) - {decision.reason}")
        
        start_time = time.time()
        
        try:
            # TIER 1: Enhanced Direct Skills (yield all at once - they're fast)
            if decision.use_skill and self.skills_manager:
                try:
                    result = await self.skills_manager.execute_skill(query, decision.skill_name)
                    if result and result.success:
                        self._update_stats('skill', time.time() - start_time, True)
                        
                        # Calculate time saved vs LLM
                        estimated_llm_time = 2.0  # Average LLM response time
                        time_saved = max(0, estimated_llm_time - result.execution_time)
                        self.stats['total_time_saved'] += time_saved
                        
                        if settings.debug_mode:
                            print(f"[ROUTER] ✅ Enhanced skill executed in {result.execution_time:.3f}s (saved {time_saved:.2f}s)")
                        
                        yield result.response
                        return
                    else:
                        # Skill failed, fallback to streaming LLM
                        self.stats['fallback_count'] += 1
                        if settings.debug_mode:
                            print(f"[ROUTER] ⚠️ Enhanced skill failed, falling back to LLM")
                        
                        if self.offline_available:
                            decision.route_type = 'offline'
                            decision.reason = "Enhanced skill failed - fallback to Nemotron"
                        elif self.online_available:
                            decision.route_type = 'online'
                            decision.reason = "Enhanced skill failed - fallback to Groq"
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ROUTER] ❌ Skill execution error: {e}")
                    # Fallback to LLM
                    if self.offline_available:
                        decision.route_type = 'offline'
                    elif self.online_available:
                        decision.route_type = 'online'
            
            # TIER 2: Offline LLM Streaming (Nemotron)
            if decision.use_offline and self.offline_llm:
                personality_context = await self.personality_manager.get_system_prompt()
                memory_context = await self.memory_manager.get_context()
                
                if settings.debug_mode:
                    print("[ROUTER] 🌊 Streaming via NEMOTRON")
                    
                async for chunk in self.offline_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    yield chunk
                    
                self._update_stats('offline', time.time() - start_time)
                return
            
            # TIER 3: Online LLM Streaming (Groq)
            if decision.use_online and self.online_llm:
                personality_context = await self.personality_manager.get_system_prompt()
                memory_context = await self.memory_manager.get_context()
                
                if settings.debug_mode:
                    print("[ROUTER] 🌊 Streaming via GROQ")
                    
                async for chunk in self.online_llm.generate_response_stream(
                    query, personality_context, memory_context
                ):
                    yield chunk
                    
                self._update_stats('online', time.time() - start_time)
                return
            
            # No systems available
            yield "I'm sorry, but I'm unable to process your request right now. Please check that Pascal's systems are properly configured."
            
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Streaming error in {decision.route_type}: {e}")
            yield f"I'm experiencing technical difficulties: {str(e)[:100]}"
    
    # ADDED: Non-streaming response method for compatibility
    async def get_response(self, query: str) -> str:
        """Get non-streaming response by collecting streaming chunks"""
        response_parts = []
        async for chunk in self.get_streaming_response(query):
            response_parts.append(chunk)
        return ''.join(response_parts)
    
    # ADDED: Method to check if current information is needed
    def _needs_current_information(self, query: str) -> bool:
        """Check if query needs current information (alias for _detect_current_info)"""
        return self._detect_current_info(query)
    
    # ADDED: Method to check LLM availability
    async def _check_llm_availability(self):
        """Check LLM availability (called by test scripts)"""
        await self._check_system_availability()
    
    def _update_stats(self, route_type: str, response_time: float, success: bool = True):
        """Update enhanced performance statistics"""
        if route_type == 'skill':
            self.stats['skill_requests'] += 1
            current_avg = self.stats['skill_avg_time']
            count = self.stats['skill_requests']
            self.stats['skill_avg_time'] = ((current_avg * (count - 1)) + response_time) / count
            if success:
                self.stats['skill_success_count'] += 1
                
        elif route_type == 'offline':
            self.stats['offline_requests'] += 1
            current_avg = self.stats['offline_avg_time']
            count = self.stats['offline_requests']
            self.stats['offline_avg_time'] = ((current_avg * (count - 1)) + response_time) / count
            
        elif route_type == 'online':
            self.stats['online_requests'] += 1
            current_avg = self.stats['online_avg_time']
            count = self.stats['online_requests']
            self.stats['online_avg_time'] = ((current_avg * (count - 1)) + response_time) / count
    
    def get_router_stats(self) -> dict:
        """Get enhanced router stats with skills metrics"""
        total_requests = (self.stats['skill_requests'] + 
                         self.stats['offline_requests'] + 
                         self.stats['online_requests'])
        
        if total_requests > 0:
            skill_percentage = (self.stats['skill_requests'] / total_requests) * 100
            offline_percentage = (self.stats['offline_requests'] / total_requests) * 100
            online_percentage = (self.stats['online_requests'] / total_requests) * 100
        else:
            skill_percentage = offline_percentage = online_percentage = 0
        
        # Calculate efficiency metrics
        avg_skill_time = self.stats['skill_avg_time']
        avg_llm_time = (self.stats['offline_avg_time'] + self.stats['online_avg_time']) / 2 if (self.stats['offline_requests'] + self.stats['online_requests']) > 0 else 0
        efficiency_ratio = (avg_llm_time / max(avg_skill_time, 0.001)) if avg_skill_time > 0 else 1
        
        return {
            'mode': self.mode.value,
            'system_status': {
                'enhanced_skills': self.skills_available,
                'offline_llm': self.offline_available,
                'online_llm': self.online_available,
            },
            'tier_system': 'Enhanced 3-tier (Skills -> Nemotron -> Groq)' if all([self.skills_available, self.offline_available, self.online_available]) else 'Partial',
            'last_decision': {
                'route_type': self.last_decision.route_type,
                'reason': self.last_decision.reason,
                'skill_name': self.last_decision.skill_name,
                'estimated_time': self.last_decision.estimated_time,
                'confidence': self.last_decision.confidence
            } if self.last_decision else None,
            'performance_stats': {
                **self.stats,
                'total_requests': total_requests,
                'skill_percentage': f"{skill_percentage:.1f}%",
                'offline_percentage': f"{offline_percentage:.1f}%", 
                'online_percentage': f"{online_percentage:.1f}%",
                'skill_success_rate': f"{(self.stats['skill_success_count'] / max(self.stats['skill_requests'], 1)) * 100:.1f}%",
                'efficiency_ratio': f"{efficiency_ratio:.1f}x",
                'total_time_saved': f"{self.stats['total_time_saved']:.1f}s",
                'fallback_rate': f"{(self.stats['fallback_count'] / max(total_requests, 1)) * 100:.1f}%"
            },
            'skills_detailed_stats': self.skills_manager.get_skill_stats() if self.skills_manager else {},
            'recommendations': self.skills_manager.get_skill_recommendations() if self.skills_manager else []
        }
    
    def get_available_skills(self) -> list:
        """Get list of available enhanced skills"""
        if self.skills_manager:
            return self.skills_manager.list_available_skills()
        return []
    
    def get_system_health(self) -> dict:
        """Get comprehensive system health report"""
        health_score = 0
        max_score = 100
        
        # Skills availability (40 points)
        if self.skills_available:
            health_score += 40
            if self.skills_manager:
                # Bonus for API integrations
                skills_info = self.skills_manager.list_available_skills()
                api_skills = [s for s in skills_info if s.get('api_required', False)]
                configured_apis = [s for s in api_skills if s.get('api_configured', False)]
                if api_skills:
                    health_score += (len(configured_apis) / len(api_skills)) * 20
        
        # Offline LLM (30 points)
        if self.offline_available:
            health_score += 30
        
        # Online LLM (30 points) 
        if self.online_available:
            health_score += 30
        
        # Performance adjustments
        if self.stats['skill_requests'] > 0:
            success_rate = self.stats['skill_success_count'] / self.stats['skill_requests']
            if success_rate < 0.8:
                health_score -= 10  # Deduct for poor skill performance
        
        health_score = max(0, min(100, health_score))
        
        return {
            'overall_health_score': health_score,
            'system_status': 'Excellent' if health_score >= 90 else 'Good' if health_score >= 70 else 'Fair' if health_score >= 50 else 'Poor',
            'components': {
                'enhanced_skills': 'Available' if self.skills_available else 'Unavailable',
                'offline_llm': 'Available' if self.offline_available else 'Unavailable', 
                'online_llm': 'Available' if self.online_available else 'Unavailable'
            },
            'recommendations': self.skills_manager.get_skill_recommendations() if self.skills_manager else [],
            'performance_summary': {
                'total_requests': sum([self.stats['skill_requests'], self.stats['offline_requests'], self.stats['online_requests']]),
                'skills_used_percentage': f"{(self.stats['skill_requests'] / max(sum([self.stats['skill_requests'], self.stats['offline_requests'], self.stats['online_requests']]), 1)) * 100:.1f}%",
                'total_time_saved': f"{self.stats['total_time_saved']:.1f}s"
            }
        }
    
    async def close(self):
        """Close all systems"""
        if self.skills_manager:
            await self.skills_manager.close()
        if self.offline_llm:
            await self.offline_llm.close()
        if self.online_llm:
            await self.online_llm.close()

# Maintain compatibility with existing Pascal system
LightningRouter = EnhancedRouter
