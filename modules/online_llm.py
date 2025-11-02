"""
Pascal AI Assistant - Online LLM with Brave Search API
FREE tier search with better accuracy than Google Custom Search
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
    """Online LLM with improved sports search handling"""
    
    def __init__(self):
        self.session = None
        self.available = False
        self.last_error = None
        self.initialization_successful = False
        
        # Groq configuration
        self.api_key = settings.groq_api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"
        
        # Brave Search configuration
        self.brave_api_key = os.getenv('BRAVE_API_KEY', '').strip()
        self.brave_search_available = bool(self.brave_api_key and self.brave_api_key != 'your_brave_api_key_here')
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_time = 0.0
        self.response_times = []
        self.search_count = 0
        self.search_success_count = 0
        self.news_search_count = 0
        self.news_search_success_count = 0
        
        # Temporal indicators
        self.strong_temporal_indicators = [
            'today', 'now', 'currently', 'right now', 'at the moment',
            'latest', 'recent', 'recently', 'breaking', 'current', 'live', 'real-time',
            'as of', 'in 2025', 'this year', 'this week', 'this month',
            'yesterday', 'last night', 'last week', 'last month',
            'who won', 'what happened', 'won', 'happened'
        ]
        
        # NEWS-SPECIFIC indicators (NON-SPORTS)
        self.news_indicators = [
            # Event queries
            'what happened', 'what\'s happening', 'happening in',
            'latest news', 'breaking news', 'recent news', 'news about',
            
            # Political/government
            'government', 'president', 'prime minister', 'election',
            'parliament', 'senate', 'congress', 'political',
            
            # Conflict/war
            'war', 'conflict', 'gaza', 'ukraine', 'military',
            'attack', 'fighting', 'crisis',
            
            # General events (non-sports)
            'last week', 'this week', 'yesterday', 'recently',
            'just happened', 'announced', 'reported',
            
            # Update queries
            'update on', 'latest with', 'latest on',
            'current situation', 'status of'
        ]
    
    async def initialize(self) -> bool:
        """Initialize Groq client and Brave Search"""
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
                    if self.brave_search_available:
                        print("✅ [BRAVE] Intelligent search routing (general + news)")
                    else:
                        print("⚠️  [BRAVE] Search not configured")
                
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
            try:
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                if start_idx >= 0 and end_idx >= 0 and end_idx > start_idx:
                    return json.loads(text[start_idx:end_idx + 1])
            except:
                pass
            return None
    
    def _optimize_search_query(self, query: str) -> str:
        """FIXED: Optimize search query for better results, especially sports"""
        query_lower = query.lower().strip()
        
        # CRITICAL: Skip hardcoded optimizations for enhanced follow-up queries!
        # Enhanced queries already have full context from conversation history
        if 'follow-up:' in query_lower or 'previous q:' in query_lower:
            # Only do minimal cleanup - don't destroy the enhanced context!
            if settings.debug_mode:
                print(f"[BRAVE] Enhanced follow-up query - skipping hardcoded optimizations")
            # Just clean up extra spaces and return
            optimized = ' '.join(query.split())
            # Limit to reasonable length for Brave API
            if len(optimized) > 400:
                optimized = optimized[:400]
            return optimized
        
        # SPORTS-SPECIFIC optimizations (only for non-enhanced queries)
        sports_patterns = {
            # F1/Formula 1
            r'who won (?:the )?last f1 race': 'F1 race winner latest 2025',
            r'last f1 race': 'F1 race results latest 2025',
            r'f1 race.*where': 'F1 latest race winner location 2025',
            r'next f1 race': 'F1 race schedule next 2025',
            r'where.*next f1': 'F1 next race location 2025',
            
            # BTCC
            r'who won (?:the )?last btcc': 'BTCC race winner latest 2025',
            r'btcc.*last race': 'BTCC latest race results 2025',
            
            # Generic sports
            r'who won (?:the )?last (\w+) race': r'\1 race winner latest results 2025',
            r'who won (?:the )?last (\w+) game': r'\1 game winner latest results',
            r'who won (?:the )?last (\w+) match': r'\1 match winner latest results',
            
            # Results queries
            r'latest (\w+) results': r'\1 results latest 2025',
            r'recent (\w+) results': r'\1 results recent 2025',
        }
        
        # Check sports patterns first
        for pattern, replacement in sports_patterns.items():
            if re.search(pattern, query_lower):
                optimized = re.sub(pattern, replacement, query_lower)
                if settings.debug_mode:
                    print(f"[BRAVE] Query optimized for sports: '{query}' -> '{optimized}'")
                return optimized
        
        # POLITICS/NEWS optimizations
        politics_patterns = {
            r'who is (?:the )?current (?:english |uk |british )?(?:pm|prime minister)': 'UK prime minister current 2025',
            r'current (?:english |uk |british )?(?:pm|prime minister)': 'UK prime minister current 2025',
            r'who is (?:the )?(?:english |uk |british )?(?:pm|prime minister)': 'UK prime minister current 2025',
        }
        
        for pattern, replacement in politics_patterns.items():
            if re.search(pattern, query_lower):
                if settings.debug_mode:
                    print(f"[BRAVE] Query optimized for politics: '{query}' -> '{replacement}'")
                return replacement
        
        # GENERAL optimizations - make queries shorter and more targeted
        # Remove filler words
        filler_words = [
            'can you tell me', 'please tell me', 'i want to know',
            'what is the', 'who is the', 'where is the',
            'do you know', 'could you', 'would you',
            'tell me about', 'information about'
        ]
        
        optimized = query_lower
        for filler in filler_words:
            optimized = optimized.replace(filler, '')
        
        # Simplify temporal phrases
        temporal_replacements = {
            'what happened recently': 'latest news',
            'what\'s happening': 'current news',
            'right now': 'current',
            'at the moment': 'current',
            'as of today': '2025',
        }
        
        for old, new in temporal_replacements.items():
            optimized = optimized.replace(old, new)
        
        # Clean up extra spaces
        optimized = ' '.join(optimized.split())
        
        # Limit query length (Brave works better with shorter queries)
        words = optimized.split()
        if len(words) > 8:
            # Keep most important words (nouns, verbs, years, names)
            important_words = []
            for word in words:
                if (word.isdigit() or  # years
                    word in ['2025', '2024', 'latest', 'current', 'winner', 'results'] or  # key terms
                    word[0].isupper() or  # proper nouns
                    len(word) > 4):  # longer words tend to be more specific
                    important_words.append(word)
            
            if len(important_words) > 0:
                optimized = ' '.join(important_words[:8])
        
        if settings.debug_mode and optimized != query_lower:
            print(f"[BRAVE] Query optimized: '{query}' -> '{optimized}'")
        
        return optimized if optimized != query_lower else query
    
    def detect_needs_search(self, query: str) -> bool:
        """Detect if query needs Brave search"""
        query_lower = query.lower().strip()
        
        # High-confidence search patterns (includes sports which will use GENERAL search)
        high_confidence_search_patterns = [
            # Sports results
            r'\bwho\s+won\s+(?:the\s+)?(?:last|latest|recent|yesterday\'?s?|today\'?s?)',
            r'\bwho\s+won\s+(?:the\s+)?(?:\w+\s+)?(?:race|game|match|championship|election)',
            r'\b(?:last|latest|next)\s+(?:f1|formula|btcc|race|game|match)',
            r'\b(?:f1|formula\s*(?:1|one)|btcc|nascar|indycar)\s+(?:race|results?|winner)',
            
            # Event queries
            r'\bwhat\s+happened\s+(?:in|with|to|at)\s+\w+\s+(?:recently|lately|today|yesterday|this\s+week)',
            r'\bwhat\s+happened\s+(?:recently|lately|today|yesterday)',
            r'\bwhat\'?s\s+happening\s+(?:in|with|today)',
            
            # Current patterns
            r'\bwhat\s+is\s+(?:the\s+)?current\s+\w+',
            r'\bwhat\'?s\s+(?:the\s+)?current\s+\w+',
            r'\bcurrent\s+(?:world|land|speed|temperature|stock|price|exchange|population)\s+record',
            
            # Temporal phrases
            r'\bas\s+of\s+(?:today|now|this\s+year|\d{4})',
            r'\b(?:latest|breaking|recent|today\'?s?)\s+(?:news|headlines|events)',
            r'\bcurrent\s+events',
            
            # Political
            r'\b(?:who\s+is|who\'?s)\s+(?:the\s+)?current\s+(?:\w+\s+)?(?:president|prime\s+minister|pm|leader)',
        ]
        
        # Check high-confidence patterns first
        for pattern in high_confidence_search_patterns:
            if re.search(pattern, query_lower):
                if settings.debug_mode:
                    print(f"[BRAVE] High-confidence search trigger")
                return True
        
        # Check for temporal indicators
        has_temporal = any(indicator in query_lower for indicator in self.strong_temporal_indicators)
        
        if has_temporal and '?' in query:
            if settings.debug_mode:
                print(f"[BRAVE] Temporal + question mark search trigger")
            return True
        
        return False
    
    def detect_needs_news_search(self, query: str) -> bool:
        """Detect if query needs NEWS search (not sports)"""
        query_lower = query.lower().strip()
        
        # EXCLUSION: Sports queries should NOT use news search
        sports_exclusion_patterns = [
            r'\b(?:f1|formula|btcc|nascar|indycar|race|game|match)\b',
            r'\bwho\s+won\s+(?:the\s+)?(?:last|latest)',
            r'\b(?:sports?|scores?|results?)\b',
        ]
        
        # Check if it's a sports query - if so, DON'T use news search
        for pattern in sports_exclusion_patterns:
            if re.search(pattern, query_lower):
                if settings.debug_mode:
                    print(f"[BRAVE] Sports query detected - using GENERAL search instead of news")
                return False
        
        # Check for news indicators
        news_keyword_count = sum(1 for indicator in self.news_indicators if indicator in query_lower)
        
        if news_keyword_count >= 2:
            if settings.debug_mode:
                print(f"[BRAVE NEWS] Multiple news indicators ({news_keyword_count})")
            return True
        
        return False
    
    async def brave_search(self, query: str, num_results: int = 5, search_type: str = 'general') -> List[Dict[str, Any]]:
        """
        Brave Search API integration with query optimization
        
        Args:
            query: Search query
            num_results: Number of results to return (default 5)
            search_type: 'general' or 'news'
        """
        if not self.brave_search_available:
            if settings.debug_mode:
                print("⚠️  [BRAVE] Search not available")
            return []
        
        try:
            if search_type == 'news':
                self.news_search_count += 1
            else:
                self.search_count += 1
            
            # CRITICAL: Optimize the query for better results
            optimized_query = self._optimize_search_query(query)
            
            # Use dedicated news endpoint for news searches
            if search_type == 'news':
                search_url = "https://api.search.brave.com/res/v1/news/search"
            else:
                search_url = "https://api.search.brave.com/res/v1/web/search"
            
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip',
                'X-Subscription-Token': self.brave_api_key
            }
            params = {
                'q': optimized_query,
                'count': min(num_results, 20)  # Brave max is 20
            }
            
            # Add news-specific parameters for Brave
            if search_type == 'news':
                params['freshness'] = 'pd'  # Past day for news
            
            if settings.debug_mode:
                search_label = "NEWS" if search_type == 'news' else "WEB"
                print(f"[BRAVE {search_label}] 🔍 Searching: {optimized_query}")
            
            async with self.session.get(
                search_url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    # Brave API returns different structures for web vs news
                    if search_type == 'news':
                        # News endpoint returns results directly
                        news_results = data.get('results', [])
                        for item in news_results[:num_results]:
                            results.append({
                                'title': item.get('title', ''),
                                'link': item.get('url', ''),
                                'snippet': item.get('description', ''),
                                'source': item.get('meta_url', {}).get('hostname', ''),
                                'search_type': search_type
                            })
                    else:
                        # Web endpoint returns results in 'web' key
                        web_results = data.get('web', {}).get('results', [])
                        for item in web_results[:num_results]:
                            results.append({
                                'title': item.get('title', ''),
                                'link': item.get('url', ''),
                                'snippet': item.get('description', ''),
                                'source': item.get('meta_url', {}).get('hostname', ''),
                                'search_type': search_type
                            })
                    
                    if search_type == 'news':
                        self.news_search_success_count += 1
                    else:
                        self.search_success_count += 1
                    
                    if settings.debug_mode:
                        search_label = "NEWS" if search_type == 'news' else "WEB"
                        print(f"[BRAVE {search_label}] ✅ Found {len(results)} results")
                        if len(results) == 0:
                            print(f"[BRAVE {search_label}] ⚠️  No results for: {optimized_query}")
                    
                    return results
                    
                elif response.status == 429:
                    if settings.debug_mode:
                        print("⚠️  [BRAVE] Rate limited")
                    return []
                elif response.status in [401, 403]:
                    if settings.debug_mode:
                        print("❌ [BRAVE] Invalid API key")
                    return []
                else:
                    if settings.debug_mode:
                        error_text = await response.text()
                        print(f"❌ [BRAVE] Search error: {response.status} - {error_text[:100]}")
                    return []
                    
        except Exception as e:
            if settings.debug_mode:
                print(f"❌ [BRAVE] Search exception: {e}")
            return []
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM prompt"""
        if not results:
            return ""
        
        search_type = results[0].get('search_type', 'general')
        label = "REAL-TIME NEWS SEARCH RESULTS" if search_type == 'news' else "REAL-TIME SEARCH RESULTS FROM BRAVE"
        
        formatted = f"\n\n{label}:\n"
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
        """Generate streaming response with improved search"""
        if not self.available:
            yield "Online services are not available."
            return
        
        # Determine search type
        needs_search = self.detect_needs_search(query)
        needs_news_search = self.detect_needs_news_search(query)
        
        if needs_news_search:
            search_type = 'news'
            needs_search = True
        else:
            search_type = 'general'
        
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
            
            # Search Brave/News if needed
            search_results = []
            if needs_search and self.brave_search_available:
                if search_type == 'news':
                    yield "📰 Searching latest news... "
                else:
                    yield "🔍 Searching Brave... "
                
                # FIXED: Use more results for better coverage
                search_results = await self.brave_search(query, num_results=5, search_type=search_type)
                
                # If no results, try a fallback search with simplified query
                if not search_results and search_type == 'general':
                    if settings.debug_mode:
                        print("[BRAVE] No results, trying fallback search...")
                    
                    # Extract key terms for fallback
                    fallback_query = self._extract_key_terms(query)
                    if fallback_query and fallback_query != query:
                        search_results = await self.brave_search(fallback_query, num_results=5, search_type='general')
            
            # Build enhanced prompt
            messages = []
            
            # System message with current info and conversation context
            system_content = f"""You are Pascal, a helpful AI assistant with access to real-time information.

CURRENT DATE & TIME:
Today is: {datetime_info['current_date']}
Current time: {datetime_info['current_time']}
Current year: {datetime_info['current_year']}

{personality_context[:200] if personality_context else ''}"""

            # Add conversation context early for follow-up resolution (max 2048 chars)
            if memory_context:
                context_safe = memory_context[:2048] if len(memory_context) > 2048 else memory_context
                system_content += f"\n\nRECENT CONVERSATION:\n{context_safe}"

            # Add search results if available
            if search_results:
                system_content += self._format_search_results(search_results)
                if search_type == 'news':
                    system_content += "\n\n🚨 CRITICAL INSTRUCTION - YOU MUST FOLLOW THIS:\n"
                    system_content += "1. Answer ONLY using the NEWS search results above\n"
                    system_content += "2. DO NOT use your training data or knowledge cutoff\n"
                    system_content += "3. If the search results contain the answer, provide it with source citations\n"
                    system_content += "4. If the search results don't contain the answer, say 'I couldn't find current information about this in the search results'\n"
                    system_content += "5. NEVER say 'I don't have information as of my knowledge cutoff' when search results are provided"
                else:
                    system_content += "\n\n🚨 CRITICAL INSTRUCTION - YOU MUST FOLLOW THIS:\n"
                    system_content += "1. Answer ONLY using the web search results above\n"
                    system_content += "2. DO NOT use your training data or knowledge cutoff information\n"
                    system_content += "3. The search results contain CURRENT information from {datetime_info['current_year']}\n"
                    system_content += "4. If the search results contain the answer, state it clearly with source citations\n"
                    system_content += "5. If the search results don't contain the answer, say 'The search results don't contain this specific information'\n"
                    system_content += "6. NEVER mention your knowledge cutoff or training data when search results are provided"
            elif needs_search:
                # No results found but search was needed
                system_content += "\n\nNote: Search was attempted but no results were found. Provide the best answer possible with available knowledge, and suggest checking current sources for the latest information."
            
            messages.append({"role": "system", "content": system_content})
            
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
                            if search_results:
                                search_label = "news" if search_type == 'news' else "web"
                                print(f"[GROQ] ✅ Response in {response_time:.2f}s (with {search_label} search)")
                            else:
                                print(f"[GROQ] ✅ Response in {response_time:.2f}s")
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
    
    def _extract_key_terms(self, query: str) -> str:
        """Extract key terms for fallback search"""
        query_lower = query.lower()
        
        # Key terms to keep
        key_terms = []
        
        # Extract sports terms
        sports = ['f1', 'formula', 'btcc', 'nascar', 'race', 'game', 'match']
        for sport in sports:
            if sport in query_lower:
                key_terms.append(sport)
        
        # Extract temporal terms
        temporal = ['last', 'latest', 'recent', 'next', '2025', '2024']
        for term in temporal:
            if term in query_lower:
                key_terms.append(term)
        
        # Extract action terms
        actions = ['won', 'winner', 'results', 'champion']
        for action in actions:
            if action in query_lower:
                key_terms.append(action)
        
        # If we have some terms, return them
        if key_terms:
            return ' '.join(key_terms)
        
        # Otherwise return first 3-4 meaningful words
        words = query.split()
        meaningful = [w for w in words if len(w) > 3 and w.lower() not in ['what', 'where', 'when', 'who', 'which', 'that', 'this', 'there']]
        return ' '.join(meaningful[:4]) if meaningful else query
    
    async def generate_response(self, query: str, personality_context: str, 
                               memory_context: str) -> str:
        """Generate non-streaming response"""
        parts = []
        async for chunk in self.generate_response_stream(query, personality_context, memory_context):
            parts.append(chunk)
        return ''.join(parts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = self.total_time / max(self.request_count, 1)
        success_rate = (self.success_count / max(self.request_count, 1)) * 100
        
        search_success_rate = (self.search_success_count / max(self.search_count, 1)) * 100 if self.search_count > 0 else 0
        news_search_success_rate = (self.news_search_success_count / max(self.news_search_count, 1)) * 100 if self.news_search_count > 0 else 0
        
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
            'brave_search': {
                'configured': self.brave_search_available,
                'total_web_searches': self.search_count,
                'total_news_searches': self.news_search_count,
                'successful_web_searches': self.search_success_count,
                'successful_news_searches': self.news_search_success_count,
                'web_search_success_rate': search_success_rate,
                'news_search_success_rate': news_search_success_rate
            },
            'enhancements': [
                '✅ Brave Search API integrated (FREE tier)',
                '✅ IMPROVED: Query optimization for sports results',
                '✅ FIXED: Better search terms extraction',
                '✅ FIXED: F1/BTCC queries now work properly',
                '✅ Fallback search for no results',
                '✅ 5 results per search for better coverage',
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
                if self.search_count > 0 or self.news_search_count > 0:
                    print(f"[BRAVE] 📊 Web: {self.search_count} searches ({self.search_success_count} successful)")
                    print(f"[BRAVE NEWS] 📊 News: {self.news_search_count} searches ({self.news_search_success_count} successful)")
            print("[GROQ] 🔌 Connection closed")
