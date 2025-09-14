"""
Pascal AI Assistant - COMPLETE Online LLM (Groq Only)
Enhanced for better current information and real-time queries
"""

import asyncio
import json
import time
from typing import Optional, AsyncGenerator
from datetime import datetime

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from config.settings import settings

class OnlineLLM:
    """Complete enhanced online LLM client using Groq with better current info handling"""
    
    def __init__(self):
        self.session = None
        self.available = False
        self.last_error = None
        self.initialization_successful = False
        
        # Groq configuration
        self.api_key = settings.groq_api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"  # Fastest Groq model
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_time = 0.0
        self.response_times = []
    
    async def initialize(self) -> bool:
        """Initialize Groq client"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not installed - install with: pip install aiohttp"
            if settings.debug_mode:
                print("❌ [GROQ] aiohttp not available - install with: pip install aiohttp")
            return False
        
        if not self.api_key:
            self.last_error = "No Groq API key configured"
            if settings.debug_mode:
                print("❌ [GROQ] No Groq API key configured")
            return False
        
        try:
            # Create optimized session
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=5, force_close=True)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            
            # Test connection
            if await self._test_connection():
                self.available = True
                self.initialization_successful = True
                if settings.debug_mode:
                    print(f"✅ [GROQ] API initialized with model: {self.model}")
                return True
            else:
                return False
                
        except Exception as e:
            self.last_error = f"Initialization failed: {str(e)}"
            if settings.debug_mode:
                print(f"❌ [GROQ] Initialization failed: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test Groq API connection"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
                "temperature": 0.1
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'choices' in data and data['choices']:
                        if settings.debug_mode:
                            print("✅ [GROQ] Connection test successful")
                        return True
                elif response.status == 429:
                    # Rate limited but API key works
                    if settings.debug_mode:
                        print("⚠️ [GROQ] API rate limited but functional")
                    return True
                elif response.status in [401, 403]:
                    self.last_error = "Invalid Groq API key"
                    if settings.debug_mode:
                        print("❌ [GROQ] Invalid API key")
                    return False
                else:
                    error_text = await response.text()
                    self.last_error = f"Groq API error {response.status}: {error_text[:100]}"
                    if settings.debug_mode:
                        print(f"❌ [GROQ] API error {response.status}: {error_text[:100]}")
                    return False
                    
        except asyncio.TimeoutError:
            self.last_error = "Connection timeout"
            if settings.debug_mode:
                print("❌ [GROQ] Connection timeout")
            return False
        except Exception as e:
            self.last_error = f"Connection test failed: {str(e)}"
            if settings.debug_mode:
                print(f"❌ [GROQ] Connection test failed: {e}")
            return False
    
    def _build_enhanced_prompt(self, query: str, personality_context: str, 
                              memory_context: str, is_current_info: bool) -> list:
        """Build enhanced prompt for current information queries"""
        messages = []
        
        # System message with ENHANCED current date context
        if is_current_info:
            now = datetime.now()
            current_date = now.strftime("%A, %B %d, %Y")
            current_time = now.strftime("%I:%M %p")
            current_year = now.year
            
            enhanced_system = f"""You are Pascal, a helpful AI assistant with access to current information.

🎯 CRITICAL - CURRENT DATE & TIME INFORMATION:
Today is: {current_date}
Current time: {current_time}  
Current year: {current_year}

IMPORTANT INSTRUCTIONS FOR CURRENT INFO QUERIES:
- For "what day is today" questions: Answer with today's day name ({now.strftime("%A")})
- For "what date is today" questions: Answer with today's full date ({current_date})
- For "what time is it" questions: Answer with current time ({current_time})
- For "current president" questions: Use your knowledge of current office holders
- For news/weather queries: Provide helpful information based on your knowledge

Always be specific and direct when providing current information.

{personality_context[:500] if personality_context else ''}"""
        else:
            enhanced_system = f"""You are Pascal, a helpful AI assistant.

{personality_context[:500] if personality_context else ''}"""
        
        messages.append({"role": "system", "content": enhanced_system})
        
        # Add memory context if available
        if memory_context:
            messages.append({"role": "system", "content": f"Recent context: {memory_context[:300]}"})
        
        # User query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    async def generate_response(self, query: str, personality_context: str, 
                               memory_context: str) -> str:
        """Generate response from Groq with enhanced current info handling"""
        if not self.available:
            return "Online services are not available right now. Please check your Groq API key configuration."
        
        # Detect if this is a current info query
        is_current_info = self._detect_current_info(query)
        
        if settings.debug_mode:
            current_info_flag = " [CURRENT INFO]" if is_current_info else ""
            print(f"[GROQ] 🎯 Processing query{current_info_flag}: {query[:50]}...")
        
        try:
            start_time = time.time()
            
            messages = self._build_enhanced_prompt(query, personality_context, memory_context, is_current_info)
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": settings.max_response_tokens,
                "temperature": 0.1 if is_current_info else settings.temperature,  # Lower temp for current info
                "stream": False
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=25)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'choices' in data and data['choices']:
                        content = data['choices'][0]['message']['content']
                        
                        # Update stats
                        self.request_count += 1
                        self.success_count += 1
                        response_time = time.time() - start_time
                        self.total_time += response_time
                        self.response_times.append(response_time)
                        
                        # Keep only last 20 measurements
                        if len(self.response_times) > 20:
                            self.response_times = self.response_times[-20:]
                        
                        if settings.debug_mode:
                            print(f"[GROQ] ✅ Response generated in {response_time:.2f}s")
                        
                        return content.strip()
                    else:
                        self.failure_count += 1
                        return "I received an invalid response from the online service."
                
                elif response.status == 429:
                    self.failure_count += 1
                    return "I'm being rate limited. Please try again in a moment."
                elif response.status in [401, 403]:
                    self.failure_count += 1
                    return "There's an issue with the Groq API configuration. Please check the API key."
                else:
                    error_text = await response.text()
                    self.failure_count += 1
                    if settings.debug_mode:
                        print(f"[GROQ] ❌ API error {response.status}: {error_text[:100]}")
                    return f"Online service error. Please try again."
        
        except asyncio.TimeoutError:
            self.failure_count += 1
            if settings.debug_mode:
                print("[GROQ] ❌ Request timed out")
            return "The request timed out. Please try again."
        except Exception as e:
            self.failure_count += 1
            if settings.debug_mode:
                print(f"[GROQ] ❌ Error: {e}")
            return "I'm having trouble connecting to online services right now."
    
    async def generate_response_stream(self, query: str, personality_context: str, 
                                     memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response from Groq with enhanced current info handling"""
        if not self.available:
            yield "Online services are not available right now. Please check your Groq API key configuration."
            return
        
        # Detect if this is a current info query
        is_current_info = self._detect_current_info(query)
        
        if settings.debug_mode:
            current_info_flag = " [CURRENT INFO]" if is_current_info else ""
            print(f"[GROQ] 🌊 Streaming query{current_info_flag}: {query[:50]}...")
        
        try:
            start_time = time.time()
            
            messages = self._build_enhanced_prompt(query, personality_context, memory_context, is_current_info)
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": settings.max_response_tokens,
                "temperature": 0.1 if is_current_info else settings.temperature,  # Lower temp for current info
                "stream": True
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    response_received = False
                    
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                line_str = line_str[6:]
                                if line_str == '[DONE]':
                                    break
                                try:
                                    data = json.loads(line_str)
                                    if 'choices' in data and data['choices']:
                                        delta = data['choices'][0].get('delta', {})
                                        if 'content' in delta and delta['content']:
                                            text_chunk = delta['content']
                                            if text_chunk:
                                                yield text_chunk
                                                response_received = True
                                except json.JSONDecodeError:
                                    continue
                    
                    if response_received:
                        # Update stats
                        self.request_count += 1
                        self.success_count += 1
                        response_time = time.time() - start_time
                        self.total_time += response_time
                        self.response_times.append(response_time)
                        
                        if len(self.response_times) > 20:
                            self.response_times = self.response_times[-20:]
                            
                        if settings.debug_mode:
                            print(f"[GROQ] ✅ Streaming completed in {response_time:.2f}s")
                    else:
                        self.failure_count += 1
                        yield "I didn't receive a proper response from the online service."
                        
                elif response.status == 429:
                    self.failure_count += 1
                    yield "I'm being rate limited. Please try again in a moment."
                elif response.status in [401, 403]:
                    self.failure_count += 1
                    yield "There's an issue with the Groq API configuration."
                else:
                    self.failure_count += 1
                    yield "Online service error. Please try again."
                    
        except asyncio.TimeoutError:
            self.failure_count += 1
            if settings.debug_mode:
                print("[GROQ] ❌ Streaming timed out")
            yield "The request timed out. Please try again."
        except Exception as e:
            self.failure_count += 1
            if settings.debug_mode:
                print(f"[GROQ] ❌ Streaming error: {e}")
            yield "I'm having trouble connecting to online services right now."
    
    def _detect_current_info(self, query: str) -> bool:
        """Enhanced current info detection to match router logic"""
        query_lower = query.lower().strip()
        
        # Enhanced patterns matching the router
        current_patterns = [
            # Date/time queries
            'what day is today', 'what date is today', 'what time is it',
            'current date', 'current time', 'todays date', "today's date",
            'what is today', 'tell me the date', 'what day is it',
            'what is the date', 'what is the time', 'date today',
            'time now', 'current day', 'what day', 'what date',
            
            # Current status queries
            'current president', 'current prime minister', 'current pm',
            'who is the current', 'current leader', 'current government',
            'who is president now', 'current us president',
            
            # News and events
            'latest news', 'recent news', 'news today', 'breaking news',
            'current events', "what's happening", 'in the news',
            'news now', 'today news', 'current news',
            
            # Weather
            'weather today', 'current weather', 'weather now',
            'what is the weather', 'todays weather',
            
            # Other current info indicators
            'right now', 'at the moment', 'currently'
        ]
        
        for pattern in current_patterns:
            if pattern in query_lower:
                if settings.debug_mode:
                    print(f"[GROQ] 🎯 Current info pattern detected: '{pattern}'")
                return True
        
        # Single word triggers with context
        single_triggers = ['today', 'now', 'current', 'latest', 'recent']
        words = query_lower.split()
        
        for word in words:
            if word in single_triggers:
                # Avoid false positives for educational queries
                if any(avoid in query_lower for avoid in ['explain', 'definition', 'what is', 'how does', 'why', 'example']):
                    continue
                if settings.debug_mode:
                    print(f"[GROQ] 🎯 Current info trigger detected: '{word}'")
                return True
        
        return False
    
    def get_provider_stats(self) -> dict:
        """Get provider statistics for compatibility"""
        avg_time = self.total_time / max(self.request_count, 1)
        
        return {
            'aiohttp_available': AIOHTTP_AVAILABLE,
            'initialization_successful': self.initialization_successful,
            'last_error': self.last_error,
            'available_providers': ['groq'] if self.available else [],
            'preferred_provider': 'groq' if self.available else None,
            'providers': {
                'groq': {
                    'available': self.available,
                    'success_count': self.success_count,
                    'failure_count': self.failure_count,
                    'avg_response_time': avg_time,
                    'api_key_configured': bool(self.api_key),
                    'current_model': self.model,
                    'supports_current_info': True,
                    'timeout': 30.0,
                    'enhanced_current_info': True
                }
            }
        }
    
    def is_available(self) -> bool:
        """Check if Groq provider is available"""
        return self.initialization_successful and self.available
    
    def get_performance_stats(self) -> dict:
        """Get detailed performance statistics"""
        avg_time = self.total_time / max(self.request_count, 1)
        success_rate = (self.success_count / max(self.request_count, 1)) * 100
        
        recent_avg = 0
        if self.response_times:
            recent_avg = sum(self.response_times[-5:]) / len(self.response_times[-5:])
        
        return {
            'provider': 'groq',
            'model': self.model,
            'available': self.available,
            'initialization_successful': self.initialization_successful,
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'failed_requests': self.failure_count,
            'success_rate_percent': success_rate,
            'avg_response_time': avg_time,
            'recent_avg_time': recent_avg,
            'api_key_configured': bool(self.api_key),
            'last_error': self.last_error,
            'supports_streaming': True,
            'supports_current_info': True,
            'enhanced_current_info_detection': True
        }
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
        self.available = False
        if settings.debug_mode:
            print("[GROQ] 🔌 Connection closed")
