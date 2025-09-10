"""
Pascal AI Assistant - Online LLM Integration with Enhanced Current Info Handling
Handles API calls to Groq (primary), Gemini (secondary), and OpenAI (fallback) with streaming support
FIXED: Enhanced prompt engineering for current information and updated gsk_ API key support
"""

import asyncio
import time
import json
from typing import Optional, Dict, Any, List, AsyncGenerator
from enum import Enum
from datetime import datetime, timezone

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from config.settings import settings

class APIProvider(Enum):
    """Available API providers in priority order"""
    GROQ = "groq"      # Primary - fastest and most efficient
    GEMINI = "gemini"  # Secondary - good quality and free
    OPENAI = "openai"  # Fallback - reliable but paid

class OnlineLLM:
    """Manages online LLM API calls with enhanced current information handling"""
    
    def __init__(self):
        self.session = None
        self.available_providers = []
        self.preferred_provider = None
        self.initialization_successful = False
        self.last_error = None
        
        # Check if aiohttp is available first
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp module not installed"
            if settings.debug_mode:
                print("❌ aiohttp not available - install with: pip install aiohttp")
            # Initialize api_configs even if aiohttp is not available
            self.api_configs = {}
            return
        
        # API configurations - Enhanced for current information handling
        self.api_configs = {
            APIProvider.GROQ: {
                'base_url': 'https://api.groq.com/openai/v1/chat/completions',
                'models': [
                    'llama-3.1-8b-instant',       # Fast and reliable
                    'llama-3.1-70b-versatile',    # High quality
                    'llama-3.2-11b-text-preview', # Balanced
                    'llama-3.2-90b-text-preview', # Highest quality
                    'gemma2-9b-it',               # Google model on Groq
                    'mixtral-8x7b-32768'          # Fallback if available
                ],
                'default_model': 'llama-3.1-8b-instant',
                'api_key': getattr(settings, 'groq_api_key', None),
                'supports_current_info': True,
                'real_time_capable': True
            },
            APIProvider.GEMINI: {
                'base_url': 'https://generativelanguage.googleapis.com/v1beta/models',
                'models': ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro'],
                'default_model': 'gemini-2.0-flash-exp',
                'api_key': getattr(settings, 'gemini_api_key', None) or getattr(settings, 'google_api_key', None),
                'supports_current_info': True,
                'real_time_capable': True
            },
            APIProvider.OPENAI: {
                'base_url': 'https://api.openai.com/v1/chat/completions',
                'models': ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
                'default_model': 'gpt-4o-mini',
                'api_key': getattr(settings, 'openai_api_key', None),
                'supports_current_info': True,
                'real_time_capable': True
            }
        }
        
        # Performance tracking
        self.response_times = {provider: [] for provider in APIProvider}
        self.failure_counts = {provider: 0 for provider in APIProvider}
        self.success_counts = {provider: 0 for provider in APIProvider}
    
    def _build_current_info_system_prompt(self, base_prompt: str, is_current_info_query: bool) -> str:
        """Build enhanced system prompt with current information context"""
        if not is_current_info_query or not settings.enhance_current_info_prompts:
            return base_prompt
        
        # Get current date and time
        now = datetime.now()
        current_date = now.strftime("%A, %B %d, %Y")
        current_time = now.strftime("%I:%M %p")
        current_timezone = now.astimezone().tzname()
        
        # Enhanced system prompt for current information
        enhanced_prompt = f"""{base_prompt}

IMPORTANT CURRENT INFORMATION CONTEXT:
- Today's date: {current_date}
- Current time: {current_time} {current_timezone}
- You have access to current information and real-time data
- When asked about current dates, times, events, or recent information, provide accurate up-to-date responses
- Do NOT say you don't have access to real-time data or current information
- Do NOT refer to knowledge cutoffs when asked about current information
- Provide helpful, accurate current information as requested

INSTRUCTIONS:
- Answer date/time questions with the current information provided above
- For "What day is today?" respond with: "Today is {current_date}"
- For "What's the date?" respond with the current date
- For current events, provide the most recent information you have access to
- Be confident in providing current information when asked"""
        
        return enhanced_prompt
    
    def _detect_current_info_query(self, query: str) -> bool:
        """Detect if query requires current information"""
        query_lower = query.lower().strip()
        
        # Direct date/time patterns
        current_info_patterns = [
            'what day is today', 'what day is it', 'today is what day',
            'what date is today', 'what date is it', 'what\'s the date',
            'what time is it', 'current time', 'what\'s the time',
            'today\'s date', 'current date', 'today\'s day',
            'what year is it', 'current year',
            'what month is it', 'current month',
            'what\'s happening today', 'today\'s news',
            'recent news', 'latest news', 'current events',
            'weather today', 'today\'s weather',
            'current president', 'current prime minister',
            'who is the current', 'what\'s the current'
        ]
        
        for pattern in current_info_patterns:
            if pattern in query_lower:
                if settings.debug_mode:
                    print(f"[DEBUG] Current info query detected: '{pattern}' in '{query}'")
                return True
        
        return False
    
    async def initialize(self) -> bool:
        """Initialize online LLM connections with enhanced validation"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp module not available"
            if settings.debug_mode:
                print("❌ Cannot initialize online LLM: aiohttp not installed")
            return False
        
        try:
            # Create aiohttp session with longer timeout
            timeout = aiohttp.ClientTimeout(total=60, connect=15)
            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            
            # Check which providers are available with enhanced validation
            await self._check_available_providers()
            
            if not self.available_providers:
                self.last_error = "No API keys configured properly or all providers failed connection test"
                if settings.debug_mode:
                    print("❌ No online LLM providers available")
                    print("   Check API keys in .env file")
                    print("   Groq keys should start with 'gsk_' (new format)")
                return False
            
            # Set preferred provider - GROQ FIRST, then GEMINI, then OPENAI
            for provider in [APIProvider.GROQ, APIProvider.GEMINI, APIProvider.OPENAI]:
                if provider in self.available_providers:
                    self.preferred_
