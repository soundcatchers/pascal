"""
Pascal AI Assistant - Online LLM with Google Search Integration
Provides real current information using Google Custom Search API
"""

import asyncio
import json
import time
import re
import os
from typing import Optional, AsyncGenerator, Dict, Any, List
from datetime import datetime, timezone

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from config.settings import settings

class OnlineLLM:
    """Online LLM with Google Search for current information"""
    
    def __init__(self):
        self.session = None
        self.available = False
        self.last_error = None
        self.initialization_successful = False
        
        # Groq configuration
        self.api_key = settings.groq_api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"
        
        # Google Search configuration
        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY', '').strip()
        self.google_search_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '').strip()
        self.google_search_available = bool(self.google_api_key and self.google_search_id)
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_time = 0.0
        self.response_times = []
        self.search_count = 0
        self.search_success_count = 0
        
        # Temporal indicators for current info detection
        self.strong_temporal_indicators = [
            'today', 'now', 'currently', 'right now', 'at the moment',
            'latest', 'recent', 'breaking', 'current', 'live', 'real-time',
            'as of', 'in 2025', 'this year'
        ]
    
    async def initialize(self) -> bool:
        """Initialize Groq client and Google Search"""
        if not AIOHTTP_AVAILABLE:
            self.last_error = "aiohttp not installed - install with: pip install aiohttp"
            if settings.debug_mode:
                print("❌ [GROQ] aiohttp not available")
            return False
        
        if not self.api_key or not self._validate_api_key(self.api_key):
            self.last_error = "Invalid or missing Groq API key"
            if settings.debug_mode:
                print("❌ [GROQ] Invalid/missing API key")
            return False
        
        try:
            # Create session
            timeout = aiohttp.ClientTimeout(total=30, connect=5, sock_read=25)
            connector = aiohttp.TCPConnector(
                limit=5,
                limit_per_host=3,
                force_close=False,
                enable_cleanup_closed=True,
                use_dns_cache=True,
                keepalive_timeout=300
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout, 
                connector=connector,
                headers={'Content-Type': 'application/json'}
            )
            
            # Test Groq connection
            if await self._test_connection_fast():
                self.available = True
                self.initialization_successful = True
                
                if settings.debug_mode:
                    print("✅ [GROQ] API initialized")
                    if self.google_search_available:
                        print("✅ [GOOGLE] Search configured and ready")
                    else:
                        print("⚠️  [GOOGLE] Search not configured (add GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID to .env)")
                
                return True
            else:
                return False
                
        except Exception as e:
            self.last_error = f"Initialization failed: {str(e)}"
            if settings.debug_mode:
                print(f"❌ [GROQ] Initialization error: {e}")
            return False
    
    def _validate_api_key(self, key: str) -> bool:
        """Validate API key format"""
        if not key or not isinstance(key, str):
            return False
        
        key = key.strip()
        
        # Check for placeholder values
        invalid_values = [
            '', 'your_groq_api_key_here', 'your_grok_api_key_here',
            'gsk_your_groq_api_key_here', 'gsk-your_groq_api_key_here'
        ]
        
        if key.lower() in [v.lower() for v in invalid_values]:
            return False
        
        if key.startswith('gsk_') or key.startswith('gsk-'):
            return len(key) > 20
        
        return False
    
    async def _test_connection_fast(self) -> bool:
        """Quick connection test"""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "temperature": 0.1
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    response_text = await response.text()
                    data = self._safe_json_parse(response_text)
                    return bool(data and 'choices' in data and data['choices'])
                elif response.status == 429:
                    return True  # Rate limited but functional
                else:
                    return False
                    
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ [GROQ] Connection test error: {e}")
            return False
    
    def _safe_json_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Safe JSON parsing"""
        if not text or not text.strip():
            return None
        
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON
            try:
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                if start_idx >= 0 and end_idx >= 0 and end_idx > start_idx:
                    return json.loads(text[start_idx:end_idx + 1])
            except:
                pass
            return None
    
    def detect_needs_search(self, query: str) -> bool:
        """Detect if query needs Google search"""
        query_lower = query.lower().strip()
        
        # Check for temporal indicators
        has_temporal = any(indicator in query_lower for indicator in self.strong_temporal_indicators)
        
        # Check for question words + current/latest/recent
        current_question_patterns = [
            r'\bwhat\s+is\s+(?:the\s+)?current\b',
            r'\bwho\s+is\s+(?:the\s+)?current\b',
            r'\bwhat\'?s\s+(?:the\s+)?(?:current|latest)\b',
            r'\b(?:current|latest|recent)\s+\w+\s+(?:record|price|rate|value)\b',
        ]
        
        for pattern in current_question_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # If has temporal indicators and looks like a factual question
        if has_temporal and '?' in query:
            return True
        
        return False
    
    async def google_search(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """Search Google and return top results"""
        if not self.google_search_available:
            if settings.debug_mode:
                print("⚠️  [GOOGLE] Search not available - API keys not configured")
            return []
        
        try:
            self.search_count += 1
            
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_search_id,
                'q': query,
                'num': num_results
            }
            
            if settings.debug_mode:
                print(f"[GOOGLE] 🔍 Searching: {query}")
            
            async with self.session.get(
                search_url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    for item in data.get('items', [])[:num_results]:
                        results.append({
                            'title': item.get('title', ''),
                            'link': item.get('link', ''),
                            'snippet': item.get('snippet', ''),
                            'source': item.get('displayLink', '')
                        })
                    
                    self.search_success_count += 1
                    
                    if settings.debug_mode:
                        print(f"[GOOGLE] ✅ Found {len(results)} results")
                    
                    return results
                    
                elif response.status == 429:
                    if settings.debug_mode:
                        print("⚠️  [GOOGLE] Rate limited")
                    return []
                elif response.status in [401, 403]:
                    if settings.debug_mode:
                        print("❌ [GOOGLE] Invalid API key or Search Engine ID")
                    return []
                else:
                    if settings.debug_mode:
                        print(f"❌ [GOOGLE] Search error: {response.status}")
                    return []
                    
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ [GOOGLE] Search exception: {e}")
            return []
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM prompt"""
        if not results:
            return ""
        
        formatted = "\n\nREAL-TIME SEARCH RESULTS FROM GOOGLE:\n"
        formatted += "=" * 60 + "\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"\n[Result {i}] {result['title']}\n"
            formatted += f"Source: {result['source']}\n"
            formatted += f"Link: {result['link']}\n"
            formatted += f"Info: {result['snippet']}\n"
        
        formatted += "=" * 60 + "\n"
        return formatted
    
    async def generate_response_stream(self, query: str, personality_context: str, 
                                     memory_context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response with Google Search integration"""
        if not self.available:
            yield "Online services are not available."
            return
        
        # Check if we need to search
        needs_search = self.detect_needs_search(query)
        
        try:
            start_time = time.time()
            
            # Get datetime info
            now = datetime.now()
            datetime_info = {
                'current_date': now.strftime("%A, %B %d, %Y"),
                'current_time': now.strftime("%I:%M %p"),
                'current_day': now.strftime("%A"),
                'current_year': now.year,
            }
            
            # Search Google if needed
            search_results = []
            if needs_search and self.google_search_available:
                yield "🔍 Searching Google... "
                search_results = await self.google_search(query, num_results=3)
            
            # Build enhanced prompt
            messages = []
            
            # System message with current info
            system_content = f"""You are Pascal, a helpful AI assistant with access to real-time information.

CURRENT DATE & TIME:
Today is: {datetime_info['current_date']}
Current time: {datetime_info['current_time']}
Current year: {datetime_info['current_year']}

{personality_context[:200] if personality_context else ''}"""

            # Add search results if available
            if search_results:
                system_content += self._format_search_results(search_results)
                system_content += "\n\nIMPORTANT: Use the search results above to answer the user's question with current, accurate information. Cite the sources naturally in your response."
            
            messages.append({"role": "system", "content": system_content})
            
            # Add memory context
            if memory_context:
                messages.append({"role": "system", "content": f"Context: {memory_context[-200:]}"})
            
            # User query
            messages.append({"role": "user", "content": query})
            
            # Call Groq API
            headers = {'Authorization': f'Bearer {self.api_key}'}
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.1 if search_results else 0.3,
                "stream": True
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=25)
            ) as response:
                if response.status == 200:
                    response_received = False
                    
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str and line_str.startswith('data: '):
                                json_str = line_str[6:].strip()
                                
                                if json_str == '[DONE]':
                                    break
                                
                                data = self._safe_json_parse(json_str)
                                if data and 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta and delta['content']:
                                        yield delta['content']
                                        response_received = True
                    
                    if response_received:
                        self.request_count += 1
                        self.success_count += 1
                        response_time = time.time() - start_time
                        self.total_time += response_time
                        self.response_times.append(response_time)
                        
                        if len(self.response_times) > 20:
                            self.response_times = self.response_times[-20:]
                        
                        if settings.debug_mode:
                            search_note = " (with Google search)" if search_results else ""
                            print(f"[GROQ] ✅ Response in {response_time:.2f}s{search_note}")
                    else:
                        self.failure_count += 1
                        yield "\n\nI didn't receive a proper response."
                        
                elif response.status == 429:
                    self.failure_count += 1
                    yield "\n\nI'm being rate limited. Please try again in a moment."
                else:
                    self.failure_count += 1
                    yield "\n\nOnline service error. Please try again."
                    
        except asyncio.TimeoutError:
            self.failure_count += 1
            yield "\n\nThe request timed out. Please try again."
        except Exception as e:
            self.failure_count += 1
            if settings.debug_mode:
                print(f"[GROQ] ❌ Error: {e}")
            yield "\n\nI'm having trouble with online services right now."
    
    async def generate_response(self, query: str, personality_context: str, 
                               memory_context: str) -> str:
        """Generate non-streaming response with Google Search"""
        parts = []
        async for chunk in self.generate_response_stream(query, personality_context, memory_context):
            parts.append(chunk)
        return ''.join(parts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = self.total_time / max(self.request_count, 1)
        success_rate = (self.success_count / max(self.request_count, 1)) * 100
        search_success_rate = (self.search_success_count / max(self.search_count, 1)) * 100 if self.search_count > 0 else 0
        
        recent_avg = 0
        if self.response_times:
            recent_avg = sum(self.response_times[-5:]) / len(self.response_times[-5:])
        
        return {
            'provider': 'groq',
            'model': self.model,
            'available': self.available,
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'failed_requests': self.failure_count,
            'success_rate_percent': success_rate,
            'avg_response_time': avg_time,
            'recent_avg_time': recent_avg,
            'google_search': {
                'configured': self.google_search_available,
                'total_searches': self.search_count,
                'successful_searches': self.search_success_count,
                'search_success_rate': search_success_rate
            },
            'enhancements': [
                '✅ Google Custom Search API integrated',
                '✅ Real-time web search for current info',
                '✅ Automatic search triggering',
                '✅ Search results included in LLM context',
                '✅ Source attribution in responses'
            ]
        }
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
        self.available = False
        
        if settings.debug_mode:
            if self.request_count > 0:
                avg_time = self.total_time / self.request_count
                success_rate = (self.success_count / self.request_count) * 100
                print(f"[GROQ] 📊 Session: {self.request_count} requests, {avg_time:.2f}s avg, {success_rate:.1f}% success")
                if self.search_count > 0:
                    print(f"[GOOGLE] 📊 {self.search_count} searches, {self.search_success_count} successful")
            print("[GROQ] 🔌 Connection closed")
