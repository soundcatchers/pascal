"""
Pascal AI Assistant - Lightning-Fast Router with Streaming and Groq Priority
Intelligently routes requests between offline and online LLMs with Groq as primary provider
FIXED: Better routing for current information queries
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
        self.mode = RouteMode.AUTO  # Changed to AUTO to use best available
        self.last_decision = None
        self.offline_available = False
        self.online_available = False
        
        # Performance tracking
        self.response_times = {'offline': [], 'online': []}
        self.first_token_times = {'offline': [], 'online': []}
        
        # Enhanced patterns for detecting current information needs
        self.current_info_patterns = [
            # Temporal indicators
            r'\b(today|tonight|tomorrow|yesterday|now|current|currently|latest|recent|recently)\b',
            r'\b(this\s+(year|month|week|morning|afternoon|evening))\b',
            r'\b(last\s+(year|month|week|night))\b',
            r'\b(next\s+(year|month|week))\b',
            
            # Date patterns
            r'\b(202[4-9]|203\d)\b',  # Years 2024-2039
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+202[4-9]\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]202[4-9]\b',  # Date formats
            
            # Question patterns about time/date
            r'\bwhat\s+(day|date|time|year|month)\s+(is\s+)?it\b',
            r'\bwhat\'s\s+the\s+(date|time|day)\b',
            r'\btoday\'s\s+(date|day|weather|news)\b',
            
            # Event/news patterns
            r'\b(news|headlines|happening|event|announcement)\b',
            r'\b(weather|forecast|temperature)\b',
            r'\b(stock|market|price|trading)\b',
            r'\b(score|game|match|championship|election|results)\b',
            r'\bwho\s+(won|lost|is\s+winning)\b',
            
            # Current affairs
            r'\b(president|prime\s+minister|government|election)\b',
            r'\b(covid|pandemic|outbreak)\b',
            r'\b(update|status|situation)\b'
        ]
        
        # Compile patterns for efficiency
        self.current_info_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.current_info_patterns]
        
        # Query complexity patterns for fast routing
        self.complexity_patterns = {
            'simple_queries': [
                'hi', 'hello', 'hey', 'thanks', 'bye', 'yes', 'no', 'ok',
                'what is', 'who is', 'how far', 'distance', 'define', 'meaning',
                'calculate', 'count', 'add', 'subtract', 'multiply', 'divide'
            ],
            'complex_queries': [
                'analyze', 'compare', 'evaluate', 'research', 'detailed',
                'comprehensive', 'explain in detail', 'write code', 'debug',
                'create', 'design', 'develop', 'implement'
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
                # Both available - use auto mode for intelligent routing
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
        """Check if query requires current/recent information"""
        query_lower = query.lower()
        
        # Check all current info patterns
        for pattern in self.current_info_regex:
            if pattern.search(query_lower):
                if settings.debug_mode:
                    print(f"[DEBUG] Query needs current info - matched pattern: {pattern.pattern}")
                return True
        
        # Additional checks for implicit current info needs
        # Questions about specific recent events or people in news
        recent_event_keywords = [
            'election', 'olympics', 'world cup', 'championship',
            'award', 'announcement', 'launch', 'release'
        ]
        
        for keyword in recent_event_keywords:
            if keyword in query_lower and any(year in query_lower for year in
