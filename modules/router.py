"""
Pascal AI Assistant - Enhanced Router with Full Skills Integration
3-Tier System: Skills (instant) -> Nemotron (fast) -> Groq (current info)
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
    """Enhanced 3-tier router: Skills -> Nemotron -> Groq"""
    
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
        
        # Complex current info patterns (that need LLM interpretation)
        self.complex_current_info_patterns = [
            r'\bis it good weather for\b',          # "Is it good weather for running?"
            r'\bshould i wear\b.*\bweather\b',      # "Should I wear a jacket today?"
            r'\bcompare\b.*\bweather\b',            # "Compare weather in London and Paris"
            r'\bweather\b.*\band\b.*\bnews\b',      # "Weather and news today"
            r'\bhow does.*relate to\b',             # Complex correlations
            r'\banalyze\b.*\bcurrent\b',            # Analysis requests
            r'\bwhat should i do\b.*\btoday\b',     # Decision requests
            r'\bsummarize\b.*\b(news|weather)\b',   # Summary requests
        ]
        
        # Simple current info patterns (direct skills can handle)
        self.simple_current_info_patterns = [
            r'\bwhat time is it\b',
            r'\bwhat day is today\b',
            r'\bwhat date is today\b',
            r'\bcurrent time\b',
            r'\bcurrent date\b',
            r'\bweather in\b',
            r'\btemperature in\b',
            r'\blatest news\b',
            r'\bcalculate\b',
            r'\bwhat is \d+',
        ]
    
    async def _check_system_availability(self):
        """Check availability of all systems: Skills, Offline LLM, Online LLM"""
        try:
            if settings.debug_mode:
                print("[ROUTER] Checking enhanced 3-tier system availability...")
            
            # Initialize Enhanced Skills Manager (highest priority)
            try:
                from modules.skills_manager import skills_manager
                self.skills_manager = skills_manager
                
                # Initialize with full API testing
                api_status = await self.skills_manager.initialize()
                self.skills_available = True
                
                print("✅ Enhanced Skills Manager ready:")
                for service, status in api_status.items():
                    status_icon = "✅" if status['available'] else "⚠️"
                    print(f"   {status_icon} {service}: {status['message']}")
                
                # Show skill recommendations
                recommendations = self.skills_manager.get_skill_recommendations()
                if recommendations:
                    print("💡 Recommendations:")
                    for rec in recommendations[:3]:  # Show top 3
                        print(f"   • {rec['message']}")
                        print(f"     Action: {rec['action']}")
                
            except Exception as e:
                print(f"❌ Enhanced Skills Manager failed: {e}")
                if settings.debug_mode:
                    import traceback
                    traceback.print_exc()
                self.skills_available = False
            
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
                self.offline_available = False
            
            # Initialize online LLM (Groq) - for complex current info
            if settings.is_online_available():
                try:
                    from modules.online_llm import OnlineLLM
                    self.online_llm = OnlineLLM()
                    self.online_available = await self.online_llm.initialize()
                    
                    if self.online_available:
                        print("✅ Online LLM ready (Groq - for complex current info)")
                    else:
                        print("❌ Online LLM not available (check Groq API key)")
                        
                except Exception as e:
                    print(f"❌ Online LLM initialization failed: {e}")
                    self.online_available = False
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
    
    def _needs_complex_current_info(self, query: str) -> bool:
        """Check if query needs complex current info interpretation (LLM required)"""
        query_lower = query.lower().strip()
        
        # Check for complex patterns that need LLM interpretation
        for pattern in self.complex_current_info_patterns:
            if re.search(pattern, query_lower):
                if settings.debug_mode:
                    print(f"[ROUTER] 🧠 Complex current info detected: needs LLM")
                return True
        
        return False
    
    def _can_handle_with_skills(self, query: str) -> bool:
        """Check if enhanced skills can handle this query directly"""
        if not self.skills_available or not self.skills_manager:
            return False
        
        skill_name = self.skills_manager.can_handle_directly(query)
        return skill_name is not None
    
    def _decide_route(self, query: str) -> RouteDecision:
        """Enhanced 3-tier routing decision with skills priority"""
        
        # TIER 1: Check if enhanced skills can handle this directly (0.001-2s)
        if self.skills_available and self.skills_manager:
            skill_name = self.skills_manager.can_handle_directly(query)
            if skill_name:
                # Determine estimated time based on skill type
                if skill_name in ['datetime', 'calculator']:
                    estimated_time = "0.001s"
                else:  # weather, news
                    estimated_time = "0.5-2s"
                
                return RouteDecision(
                    'skill',
                    f"Enhanced {skill_name} skill - ultra fast",
                    skill_name=skill_name,
                    estimated_time=estimated_time,
                    confidence=0.95
                )
        
        # TIER 2: Check for complex current info that needs LLM + real data
        is_complex_current_info = self._needs_complex_current_info(query)
        if is_complex_current_info:
            if self.online_available:
                return RouteDecision(
                    'online',
                    "Complex current info query - needs LLM + real data",
                    is_current_info=True,
                    estimated_time="2-4s",
                    confidence=0.85
                )
            else:
                # Fallback to offline LLM (limited current info capability)
                if self.offline_available:
                    return RouteDecision(
                        'offline',
                        "Complex query - offline LLM (limited current info)",
                        estimated_time="0.5-2s",
                        confidence=0.6
                    )
        
        # TIER 3: Route general queries to offline LLM (fast, local)
        if self.offline_available:
            return RouteDecision(
                'offline',
                "General conversation - local Nemotron",
                estimated_time="0.5-2s",
                confidence=0.8
            )
        
        # TIER 4: Fallback to online LLM if no offline
        if self.online_available:
            return RouteDecision(
                'online',
                "Fallback to Groq - no offline LLM available",
                estimated_time="2-4s",
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
    
    async def get_response(self, query: str) -> str:
        """Get response using enhanced 3-tier system"""
        decision = self._decide_route(query)
        self.last_decision = decision
        
        if settings.debug_mode:
            tier_name = {
                'skill': f"TIER 1: {decision.skill_name.upper() if decision.skill_name else 'SKILL'}",
                'offline': "TIER 2: NEMOTRON", 
                'online': "TIER 3: GROQ"
            }.get(decision.route_type, "ERROR")
            
            print(f"[ROUTER] 🎯 {tier_name} ({decision.estimated_time}) - {decision.reason}")
        
        start_time = time.time()
        
        try:
            # TIER 1: Enhanced Direct Skills
            if decision.use_skill and self.skills_manager:
                result = await self.skills_manager.execute_skill(query, decision.skill_name)
                if result and result.success:
                    execution_time = time.time() - start_time
                    self._update_stats('skill', execution_time, True)
                    
                    # Calculate time saved vs LLM
                    estimated_llm_time = 2.0  # Average LLM response time
                    time_saved = max(0, estimated_llm_time - execution_time)
                    self.stats['total_time_saved'] += time_saved
                    
                    if settings.debug_mode:
                        print(f"[ROUTER] ✅ Enhanced skill executed in {result.execution_time:.3f}s (saved {time_saved:.2f}s)")
                    
                    return result.response
                else:
                    # Skill failed, fallback to LLM
                    self.stats['fallback_count'] += 1
                    if settings.debug_mode:
                        print(f"[ROUTER] ⚠️ Enhanced skill failed, falling back to LLM")
                    
                    if self.offline_available:
                        decision.route_type = 'offline'
                        decision.reason = "Enhanced skill failed - fallback to Nemotron"
                    elif self.online_available:
                        decision.route_type = 'online'
                        decision.reason = "Enhanced skill failed - fallback to Groq"
            
            # TIER 2: Offline LLM (Nemotron)
            if decision.use_offline and self.offline_llm:
                personality_context = await self.personality_manager.get_system_prompt()
                memory_context = await self.memory_manager.get_context()
                
                response = await self.offline_llm.generate_response(
                    query, personality_context, memory_context
                )
                self._update_stats('offline', time.time() - start_time)
                
                if settings.debug_mode:
                    elapsed = time.time() - start_time
                    print(f"[ROUTER] ✅ Used NEMOTRON in {elapsed:.2f}s")
                
                return response
            
            # TIER 3: Online LLM (Groq)
            if decision.use_online and self.online_llm:
                personality_context = await self.personality_manager.get_system_prompt()
                memory_context = await self.memory_manager.get_context()
                
                response = await self.online_llm.generate_response(
                    query, personality_context, memory_context
                )
                self._update_stats('online', time.time() - start_time)
                
                if settings.debug_mode:
                    elapsed = time.time() - start_time
                    print(f"[ROUTER] ✅ Used GROQ in {elapsed:.2f}s")
                
                return response
            
            # No systems available
            return ("I'm sorry, but I'm unable to process your request right now. "
                   "Please check that Pascal's systems are properly configured.")
                   
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Error in {decision.route_type} system: {e}")
            
            # Try fallback
            try:
                if decision.route_type == 'skill' or decision.route_type == 'offline':
                    if self.online_llm:
                        if settings.debug_mode:
                            print("[ROUTER] 🔄 Trying Groq fallback")
                        personality_context = await self.personality_manager.get_system_prompt()
                        memory_context = await self.memory_manager.get_context()
                        return await self.online_llm.generate_response(query, personality_context, memory_context)
                        
                elif decision.route_type == 'online':
                    if self.offline_llm:
                        if settings.debug_mode:
                            print("[ROUTER] 🔄 Trying Nemotron fallback")
                        personality_context = await self.personality_manager.get_system_prompt()
                        memory_context = await self.memory_manager.get_context()
                        return await self.offline_llm.generate_response(query, personality_context, memory_context)
                    elif self.skills_manager:
                        # Try skills as last resort
                        result = await self.skills_manager.execute_skill(query)
                        if result and result.success:
                            return result.response
                        
            except Exception as fallback_error:
                if settings.debug_mode:
                    print(f"❌ Fallback also failed: {fallback_error}")
            
            return f"I'm having trouble processing your request right now."
    
    async def get_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """Get streaming response using enhanced 3-tier system"""
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
                result = await self.skills_manager.execute_skill(query, decision.skill_name)
                if result and result.success:
                    self._update_stats('skill', time.time() - start_time, True)
                    yield result.response
                    return
                else:
                    # Skill failed, fallback to streaming LLM
                    self.stats['fallback_count'] += 1
                    decision.route_type = 'offline' if self.offline_available else 'online'
            
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
            yield "I'm sorry, but I'm unable to process your request right now."
            
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ Streaming error in {decision.route_type}: {e}")
            yield f"I'm experiencing technical difficulties right now."
    
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
