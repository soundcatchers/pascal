"""
Pascal AI Assistant - FIXED Router Module
Standardized routing logic and improved error handling
"""

import asyncio
import time
import re
from typing import Optional, AsyncGenerator, Dict, Any, List
from enum import Enum
from dataclasses import dataclass

from config.settings import settings

class RouteMode(Enum):
    """Enhanced routing modes"""
    BALANCED = "balanced"
    OFFLINE_ONLY = "offline_only"
    ONLINE_ONLY = "online_only"
    SKILLS_FIRST = "skills_first"
    FALLBACK = "fallback"

@dataclass
class RouteDecision:
    """Standardized routing decision"""
    route_type: str  # 'offline', 'online', 'skill', 'fallback'
    reason: str
    confidence: float = 0.8
    skill_name: Optional[str] = None
    is_current_info: bool = False
    expected_time: float = 2.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    @property
    def use_offline(self) -> bool:
        return self.route_type == 'offline'
    
    @property
    def use_online(self) -> bool:
        return self.route_type == 'online'
    
    @property
    def use_skill(self) -> bool:
        return self.route_type == 'skill'

class LightningRouter:
    """Fixed router with standardized logic and improved error handling"""
    
    def __init__(self, personality_manager, memory_manager):
        self.personality_manager = personality_manager
        self.memory_manager = memory_manager
        
        # Initialize components
        self.offline_llm = None
        self.online_llm = None
        self.skills_manager = None
        
        # Router state
        self.mode = RouteMode.FALLBACK
        self.last_decision = None
        self.offline_available = False
        self.online_available = False
        self.skills_available = False
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'offline_requests': 0,
            'online_requests': 0,
            'skill_requests': 0,
            'fallback_requests': 0,
            'offline_total_time': 0.0,
            'online_total_time': 0.0,
            'skill_total_time': 0.0,
            'routing_decisions': 0,
            'correct_routes': 0,
        }
        
        # Compiled patterns for performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance"""
        # Current info patterns
        self.current_info_patterns = [
            re.compile(r'\b(?:what\s+)?(?:time|date|day)\s+(?:is\s+)?(?:it|today)\b', re.IGNORECASE),
            re.compile(r'\b(?:current|today\'?s?|now)\s+(?:time|date|day)\b', re.IGNORECASE),
            re.compile(r'\bwhat\s+day\s+(?:is\s+)?(?:it|today)\b', re.IGNORECASE),
            re.compile(r'\bwhat\s+(?:is\s+)?(?:the\s+)?date\b', re.IGNORECASE),
            re.compile(r'\b(?:current|who\s+is\s+(?:the\s+)?(?:current\s+)?)\s*(?:president|prime\s+minister|pm|leader)\b', re.IGNORECASE),
            re.compile(r'\b(?:latest|recent|breaking|today\'?s?)\s+news\b', re.IGNORECASE),
            re.compile(r'\bwhat\'?s\s+(?:happening|going\s+on)\b', re.IGNORECASE),
            re.compile(r'\bcurrent\s+(?:weather|events)\b', re.IGNORECASE),
        ]
        
        # Skills patterns
        self.skill_patterns = {
            'datetime': [
                re.compile(r'\bwhat\s+time\s+is\s+it\b', re.IGNORECASE),
                re.compile(r'\bwhat\s+day\s+is\s+(?:it|today)\b', re.IGNORECASE),
                re.compile(r'\bcurrent\s+(?:time|date)\b', re.IGNORECASE),
            ],
            'calculator': [
                re.compile(r'\b\d+\s*[\+\-\*\/\%]\s*\d+\b'),
                re.compile(r'\bwhat\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\b', re.IGNORECASE),
            ]
        }
        
        # Offline preferred patterns
        self.offline_patterns = [
            re.compile(r'\b(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))\b', re.IGNORECASE),
            re.compile(r'\bhow\s+are\s+you\b', re.IGNORECASE),
            re.compile(r'\bexplain\s+(?!.*(?:current|latest|today|now))', re.IGNORECASE),
            re.compile(r'\bwrite\s+(?:a|some|code|function|program)', re.IGNORECASE),
        ]
    
    async def _check_llm_availability(self):
        """Check and initialize all available systems"""
        try:
            if settings.debug_mode:
                print("[ROUTER] Checking system availability...")
            
            # Initialize offline LLM
            await self._init_offline_llm()
            
            # Initialize online LLM
            await self._init_online_llm()
            
            # Initialize skills manager
            await self._init_skills_manager()
            
            # Set optimal routing mode
            self._set_routing_mode()
            
        except Exception as e:
            if settings.debug_mode:
                print(f"[ROUTER] ❌ Availability check failed: {e}")
            self.mode = RouteMode.FALLBACK
    
    async def _init_offline_llm(self):
        """Initialize offline LLM"""
        try:
            from modules.offline_llm import LightningOfflineLLM
            self.offline_llm = LightningOfflineLLM()
            self.offline_llm.set_performance_profile('speed')
            
            self.offline_available = await self.offline_llm.initialize()
            
            if self.offline_available:
                if settings.debug_mode:
                    print("✅ [ROUTER] Offline LLM ready (Nemotron)")
            else:
                if settings.debug_mode:
                    print("❌ [ROUTER] Offline LLM not available")
                    if hasattr(self.offline_llm, 'last_error') and self.offline_llm.last_error:
                        print(f"   Error: {self.offline_llm.last_error}")
                        
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ [ROUTER] Offline
