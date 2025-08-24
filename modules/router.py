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
            'needs_current_info': ['current', 'latest', 'recent', 'today', 'now', 'news', 'update', 'weather', '2024', '2025'],
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
            if settings.is_online_available():
                try:
                    from modules.online_llm import OnlineLLM
                    self.online_llm = OnlineLLM()
                    self.online_available = await self.online_llm.initialize()
                    
                    if self.online_available:
                        print("✅ Online LLM ready (Grok/OpenAI/Anthropic)")
                        
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
            return 1.5  # Quick for
