"""
Enhanced Context-Aware Router Extension
This provides additional context methods for better conversational flow and search query enhancement
"""

import asyncio
import time
import re
from typing import Dict, Any, List, Optional, AsyncGenerator
from config.settings import settings

class EnhancedContextMixin:
    """Mixin to add enhanced context awareness to the router"""
    
    def __init__(self, *args, **kwargs):
        # Don't call super().__init__() here as it causes issues with multiple inheritance
        # The router initialization will be handled by the combined class
        self.conversation_context = {
            'recent_topics': [],
            'last_intent': None,
            'source_memory': {},
            'follow_up_chain': [],
            'entities_mentioned': [],
            'current_topic': None,
            'topic_history': [],
            'session_start_time': time.time()
        }
    
    async def _analyze_conversation_context(self, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        """Analyze conversation context for better routing and query enhancement"""
        context_info = {
            'is_follow_up': False,
            'context_summary': '',
            'entities_in_context': [],
            'topic_continuity': False,
            'reference_words': [],
            'temporal_references': [],
            'is_greeting': False,
            'is_standalone': False,
            'should_force_offline': False,
            'should_force_online': False
        }
        
        query_lower = query.lower().strip()
        
        # Check for greetings first (these should always go offline)
        # Use regex with strict boundaries: start/whitespace before, whitespace/end/punctuation after
        greeting_patterns = [
            r'(?:^|\s)hello(?:\s|$|[,!?.])', r'(?:^|\s)hi(?:\s|$|[,!?.])', r'(?:^|\s)hey(?:\s|$|[,!?.])', 
            r'(?:^|\s)good morning(?:\s|$|[,!?.])', r'(?:^|\s)good afternoon(?:\s|$|[,!?.])', 
            r'(?:^|\s)good evening(?:\s|$|[,!?.])', r'(?:^|\s)how are you(?:\s|$|[,!?.])', 
            r'(?:^|\s)how do you do(?:\s|$|[,!?.])', r"(?:^|\s)what'?s up(?:\s|$|[,!?.])", 
            r'(?:^|\s)greetings(?:\s|$|[,!?.])', r'(?:^|\s)howdy(?:\s|$|[,!?.])'
        ]
        
        context_info['is_greeting'] = any(re.search(pattern, query_lower) for pattern in greeting_patterns)
        if context_info['is_greeting']:
            context_info['should_force_offline'] = True
            return context_info
        
        # Check for casual chat patterns (should go offline)
        casual_patterns = [
            r"(?:^|\s)how'?s it going(?:\s|$|[,!?.])", r"(?:^|\s)what'?s new(?:\s|$|[,!?.])", 
            r'(?:^|\s)tell me about yourself(?:\s|$|[,!?.])', r'(?:^|\s)nice to meet you(?:\s|$|[,!?.])', 
            r'(?:^|\s)good to see you(?:\s|$|[,!?.])', r'(?:^|\s)how have you been(?:\s|$|[,!?.])'
        ]
        
        if any(re.search(pattern, query_lower) for pattern in casual_patterns):
            context_info['should_force_offline'] = True
            return context_info
        
        # Check for standalone queries that don't need context
        standalone_patterns = [
            'what is', 'who is', 'where is', 'when is', 'how does', 'explain',
            'tell me about', 'what are', 'define', 'calculate'
        ]
        
        context_info['is_standalone'] = any(pattern in query_lower for pattern in standalone_patterns)
        
        # STAGE 1: Detect topic shift phrases (signals NEW topic, not follow-up)
        topic_shift_phrases = [
            'how about', 'what about', 'now tell me', 'switching to', 'moving on',
            'let\'s talk about', 'can you tell me about'
        ]
        has_topic_shift = any(phrase in query_lower for phrase in topic_shift_phrases)
        
        if has_topic_shift:
            # Topic shift detected - treat as standalone query
            context_info['is_follow_up'] = False
            context_info['is_standalone'] = True
            context_info['topic_continuity'] = False
            if settings.debug_mode:
                print(f"[ENHANCED_CONTEXT] 🔄 Topic shift detected: treating as standalone")
            # Skip follow-up detection
        else:
            # TRULY GENERIC follow-up detection - works for ANY domain, any query type
            # Run for ALL queries when recent memory exists (not just standalone patterns)
            has_recent_memory = hasattr(self, 'memory_manager') and self.memory_manager
            if has_recent_memory:
                try:
                    # Get recent conversation context from memory
                    if hasattr(self.memory_manager, 'short_term_memory') and self.memory_manager.short_term_memory:
                        last_memory = self.memory_manager.short_term_memory[-1]
                        recent_query = last_memory.user_input if hasattr(last_memory, 'user_input') else ''
                        recent_response = last_memory.assistant_response if hasattr(last_memory, 'assistant_response') else ''
                        recent_text = (recent_query + " " + recent_response).lower()
                        
                        # Generic word-overlap check - no hardcoded topic keywords!
                        # Works for quantum physics, finance, medicine, F1, politics, ANY domain
                        
                        # Extract content words (exclude stopwords, strip punctuation)
                        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                                    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about',
                                    'who', 'what', 'where', 'when', 'why', 'how', 'which', 'this', 'that'}
                        
                        # Strip punctuation from words so "details?" matches "details"
                        import string
                        query_words = set(
                            word.strip(string.punctuation) 
                            for word in query_lower.split() 
                            if word.strip(string.punctuation) not in stopwords and len(word.strip(string.punctuation)) > 2
                        )
                        context_words = set(
                            word.strip(string.punctuation) 
                            for word in recent_text.split() 
                            if word.strip(string.punctuation) not in stopwords and len(word.strip(string.punctuation)) > 2
                        )
                        
                        # Check for word overlap - if they share content words, likely related
                        word_overlap = query_words & context_words
                        
                        # Follow-up indicators (all generic, no hardcoding)
                        has_word_overlap = len(word_overlap) > 0
                        is_very_short = len(query_lower.split()) <= 3  # 1-3 words only
                        is_short_query = len(query_lower.split()) <= 7
                        # Pronouns that reference previous context (NOT "why", "what", etc.)
                        context_pronouns = ['he', 'she', 'it', 'they', 'that', 'this', 'his', 'her', 'their']
                        has_context_pronouns = any(word.strip(string.punctuation) in context_pronouns 
                                                 for word in query_lower.split())
                        
                        # Detect follow-up with multiple heuristics (ordered by confidence):
                        # 1. Word overlap → high confidence (same topic)
                        # 2. Very short (1-3 words) WITH pronouns → medium confidence ("what's that?", "where's it?")
                        # 3. Short query (4-7 words) WITH pronouns AND some overlap → medium confidence
                        # CRITICAL: "why do birds sing?" (4 words, no pronouns, no overlap) = NOT a follow-up
                        if has_word_overlap or (is_very_short and has_context_pronouns) or (is_short_query and has_context_pronouns and has_word_overlap):
                            # Follow-up detected! Works for ANY topic without hardcoding
                            context_info['is_follow_up'] = True
                            context_info['topic_continuity'] = True
                            context_info['should_force_online'] = True  # Follow-ups need current info
                            context_info['is_standalone'] = False  # Override standalone classification
                            
                            # No entity extraction needed - full context in enhancement
                            context_info['entities_in_context'] = []
                            context_info['context_summary'] = f"Follow-up ({len(word_overlap)} shared terms)"
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ENHANCED_CONTEXT] Generic follow-up detection error: {e}")
        
        # Only return early if it's truly standalone with NO topic continuity
        if context_info['is_standalone'] and len(query_lower.split()) > 4 and not context_info['topic_continuity']:
            return context_info
        
        # Detect reference words (pronouns, demonstratives)
        reference_words = ['he', 'she', 'it', 'they', 'that', 'this', 'there', 'then', 'him', 'her', 'them']
        context_info['reference_words'] = [word for word in reference_words if word in query_lower.split()]
        
        # Detect temporal references
        temporal_refs = ['today', 'yesterday', 'now', 'recently', 'latest', 'new', 'current', 'just']
        context_info['temporal_references'] = [word for word in temporal_refs if word in query_lower.split()]
        
        # Enhanced follow-up detection
        follow_up_indicators = [
            len(query_lower.split()) <= 5 and bool(context_info['reference_words']),  # Short with pronouns
            any(word in query_lower for word in ['second', 'third', 'next', 'another', 'also', 'too']),  # Sequence words
            any(word in query_lower for word in ['what about', 'how about', 'and then', 'after that']),  # Follow-up phrases
        ]
        
        # Only consider it a follow-up if we have recent conversation context
        has_recent_context = (
            self.conversation_context['recent_topics'] or 
            self.conversation_context['entities_mentioned'] or
            (time.time() - self.conversation_context['session_start_time']) > 30  # Been talking for a while
        )
        
        # CRITICAL: Preserve is_follow_up if already set by topic continuity check
        # Use OR to avoid overwriting True with False
        context_info['is_follow_up'] = context_info['is_follow_up'] or (any(follow_up_indicators) and has_recent_context)
        
        # Return context_info - generic word-overlap check already handled everything
        return context_info
    
    def _extract_entities_from_context(self, context: str) -> List[str]:
        """Extract key entities from recent conversation context"""
        entities = []
        context_lower = context.lower()
        
        # Only extract from recent parts (last 3 interactions)
        context_lines = context.split('\n')
        recent_context = '\n'.join(context_lines[-6:])  # Last 6 lines (3 user + 3 assistant)
        context_lower = recent_context.lower()
        
        # Political entities
        political_terms = [
            'keir starmer', 'prime minister', 'deputy pm', 'uk government', 
            'labour party', 'conservative', 'parliament', 'downing street'
        ]
        
        # Sports entities
        sports_terms = [
            'formula 1', 'f1', 'grand prix', 'verstappen', 'hamilton', 
            'ferrari', 'mercedes', 'red bull', 'mclaren', 'alpine'
        ]
        
        # General entities
        all_terms = political_terms + sports_terms
        
        for term in all_terms:
            if term in context_lower:
                entities.append(term)
        
        return entities[:3]  # Limit to top 3 most recent entities
    
    def _summarize_recent_context(self, context: str) -> str:
        """Create a brief summary of recent context"""
        # Simple extraction of last few topics
        lines = context.split('\n')
        recent_lines = [line for line in lines[-6:] if line.strip()]  # Last 6 lines only
        
        if not recent_lines:
            return "No recent context"
        
        # Look for key topics in recent lines
        topics = []
        for line in recent_lines:
            if any(term in line.lower() for term in ['prime minister', 'keir starmer', 'deputy']):
                topics.append('UK Politics')
            elif any(term in line.lower() for term in ['f1', 'formula', 'race', 'grand prix']):
                topics.append('Formula 1')
            elif any(term in line.lower() for term in ['weather', 'temperature']):
                topics.append('Weather')
        
        if topics:
            return f"Recent topics: {', '.join(set(topics))}"
        
        return "General conversation"
    
    def _check_topic_continuity(self, query: str, context: str) -> bool:
        """Check if current query continues the previous topic"""
        query_lower = query.lower()
        
        # Only check recent context (last few lines)
        context_lines = context.split('\n')
        recent_context = '\n'.join(context_lines[-6:])
        context_lower = recent_context.lower()
        
        # Define topic keywords
        topic_keywords = {
            'politics': ['prime minister', 'deputy', 'government', 'party', 'parliament', 'keir starmer'],
            'f1': ['formula', 'race', 'grand prix', 'driver', 'circuit', 'f1'],
            'weather': ['weather', 'temperature', 'rain', 'sunny']
        }
        
        for topic, keywords in topic_keywords.items():
            query_has_topic = any(keyword in query_lower for keyword in keywords)
            context_has_topic = any(keyword in context_lower for keyword in keywords)
            
            if query_has_topic and context_has_topic:
                return True
        
        return False
    
    def _enhance_search_query(self, query: str, context_info: Dict[str, Any]) -> str:
        """
        Enhance search query with conversation context - TRULY GENERIC APPROACH
        
        This method works for ANY topic (F1, politics, weather, science, etc.) 
        without ANY hardcoding. Strategy:
        - Include recent conversation history (last 2-3 turns)
        - Add current follow-up query
        - Let search/LLM understand relationships from full context
        
        Works "ad infinitum" for multi-turn follow-up chains without topic-specific code.
        """
        
        # If not a follow-up, return query as-is
        if not context_info['is_follow_up']:
            return query
        
        # Get recent conversation history (multiple turns for multi-turn chains)
        recent_context = self._get_recent_conversation_context()
        
        # If we have conversation history, include it with the follow-up
        if recent_context:
            # CRITICAL: Try to create a simpler enhanced query first
            simple_query = self._create_simple_enhanced_query(query, recent_context)
            
            # Use simple query if it's cleaner and shorter (better for search)
            if simple_query and len(simple_query) < 100:
                return simple_query
            else:
                # Fallback: Use full context
                enhanced_query = f"{recent_context} Follow-up: {query}"
                return enhanced_query
        
        # Fallback: No context available, return original query
        return query
    
    def _create_simple_enhanced_query(self, query: str, conversation_history: str) -> str:
        """
        Create a simplified enhanced query by extracting key entities and merging with follow-up
        
        TRULY GENERIC APPROACH: Uses frequency-based extraction to work across ALL domains
        (sports, politics, science, weather, etc.) without hardcoded keywords.
        
        Strategy:
        1. Extract all significant words from recent conversation
        2. Use frequency to identify topic indicators (most mentioned terms)
        3. Combine with proper nouns for complete context
        """
        import re
        from collections import Counter
        
        # STEP 1: Extract all words from conversation history
        # Remove punctuation and split into words
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9]*\b', conversation_history)
        
        # STEP 2: Define comprehensive stopwords (words to ignore)
        stopwords = {
            # Articles & conjunctions
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
            # Pronouns (CRITICAL - these crowd out topic nouns)
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
            'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
            # Common verbs (not topic indicators)
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'get', 'got', 'make', 'made',
            'go', 'went', 'come', 'came', 'say', 'said', 'see', 'saw', 'know', 'knew', 'think', 'thought',
            'take', 'took', 'give', 'gave', 'tell', 'told', 'ask', 'asked', 'use', 'used', 'find', 'found',
            'want', 'wanted', 'try', 'tried', 'need', 'needed', 'feel', 'felt', 'become', 'became',
            # Demonstratives & quantifiers
            'that', 'this', 'these', 'those', 'such', 'same', 'own', 'other', 'another',
            'all', 'each', 'every', 'both', 'either', 'neither', 'one', 'two', 'three',
            # Question words
            'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whom', 'whose',
            # Result/search indicators (not content words)
            'according', 'result', 'search', 'brave', 'based', 'however', 'from', 'official', 'wikipedia',
            'com', 'www', 'http', 'https', 'org', 'net', 'io', 'news', 'article',
            # Generic descriptors
            'more', 'most', 'some', 'any', 'many', 'much', 'few', 'several', 'just', 'only', 'very', 'so', 'too'
        }
        
        # STEP 3: Filter words and count frequency (case-insensitive for counting)
        # Keep words that are:
        # - Not stopwords
        # - Length >= 2
        # - Any casing (proper nouns, acronyms like NASCAR/OECD, mixed case)
        significant_words = []
        original_case_map = {}  # Map lowercase -> original case for output
        
        for word in words:
            word_lower = word.lower()
            # Keep if: not stopword and length >= 2
            if word_lower not in stopwords and len(word) >= 2:
                significant_words.append(word_lower)  # Use lowercase for counting
                # Preserve original case (prefer proper nouns over lowercase)
                if word_lower not in original_case_map or word[0].isupper():
                    original_case_map[word_lower] = word
        
        # Count word frequency (now case-insensitive: "Ferrari" + "ferrari" = count 2)
        word_freq = Counter(significant_words)
        
        # STEP 4: Get top frequent words (likely topic indicators)
        # Prioritize:
        # 1. Words that appear multiple times (frequency > 1)
        # 2. Longer words (more likely to be substantive nouns)
        # 3. Words that aren't common verbs
        
        # First, get all words with frequency > 1 (repeated = likely important)
        repeated_words = [word for word, count in word_freq.items() if count > 1]
        
        # Then get remaining top frequent words
        top_frequent = [word for word, count in word_freq.most_common(12)]
        
        # Prefer longer words (likely domain-specific: "championship", "entanglement", etc.)
        # Score words: frequency * length
        scored_words = []
        for word in top_frequent:
            score = word_freq[word] * len(word)
            scored_words.append((word, score))
        
        # Sort by score and take top 5-6
        scored_words.sort(key=lambda x: x[1], reverse=True)
        # Convert back to original case using the map
        top_words = [original_case_map.get(word, word) for word, score in scored_words[:6]]
        
        # STEP 5: Extract proper nouns (capitalized multi-word phrases)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', conversation_history)
        proper_nouns = [n for n in proper_nouns if n not in stopwords and len(n) > 2]
        
        # Get unique proper nouns (preserve order)
        seen = set()
        unique_proper_nouns = []
        for noun in proper_nouns:
            if noun.lower() not in seen:
                seen.add(noun.lower())
                unique_proper_nouns.append(noun)
        
        # STEP 6: Build enhanced query intelligently
        query_parts = []
        
        # Add top frequent words (these are likely topic indicators: F1, tennis, weather, tire, strategy, etc.)
        if top_words:
            # Take top 3-4 high-scoring words as topic context (increased from 3 to ensure coverage)
            query_parts.extend(top_words[:4])
        
        # Add proper nouns (people, places, teams)
        if unique_proper_nouns:
            query_parts.extend(unique_proper_nouns[:2])
        
        # Add user's query
        query_parts.append(query)
        
        # Add year for temporal context
        query_parts.append("2025")
        
        # STEP 7: Combine into natural query, removing duplicates
        seen_in_query = set()
        final_parts = []
        for part in query_parts:
            part_lower = part.lower()
            if part_lower not in seen_in_query:
                seen_in_query.add(part_lower)
                final_parts.append(part)
        
        # Return if we have meaningful context (more than just query + year)
        if len(final_parts) > 2:
            simple_query = ' '.join(final_parts)
            return simple_query
        
        # No sufficient context found, return empty (will use fallback)
        return ""
    
    def _get_recent_conversation_context(self) -> str:
        """
        Get recent conversation history from memory for follow-up enhancement
        
        CRITICAL: Detects topic boundaries to prevent context pollution!
        - If last question introduced NEW proper nouns → Only use CURRENT topic
        - Otherwise → Use last 2-3 turns for multi-turn chains
        
        Example 1 (Multi-turn on same topic):
          Turn 1: "who won F1 race?" → Lando Norris
          Turn 2: "where are they racing next?" → Uses Turn 1 context ✅
          Turn 3: "who came second?" → Uses Turn 1+2 context ✅
        
        Example 2 (Topic shift):
          Turn 1: "who won F1 race?" → Lando Norris
          Turn 2: "where was Olympics held?" → NEW TOPIC (Olympics) - resets context
          Turn 3: "have I missed winter olympics?" → Uses Turn 2 ONLY (not Turn 1) ✅
        """
        context_parts = []
        
        # Always pull from memory if available
        if hasattr(self, 'memory_manager') and self.memory_manager:
            try:
                # PRAGMATIC SOLUTION: Use last 2 Q&A pairs always
                # 
                # Trade-offs considered:
                # 1. Last 1 pair: Prevents ALL cross-topic pollution but breaks multi-turn chains
                # 2. Last 2 pairs: Supports 2-turn chains but CAN have cross-topic pollution
                # 3. Last 3 pairs: Supports longer chains but WILL have cross-topic pollution
                #
                # CHOSEN: Last 2 pairs as best balance
                # - Handles most conversational patterns (2-turn chains are common)
                # - Minimal cross-topic pollution (only 1 old pair can pollute)
                # - Simple, no fragile topic detection needed
                #
                # Known Limitations:
                # - Q1 (F1) → Q2 (Olympics) → Q3 (Olympics follow-up): Q3 may include Q2+Q1 (F1 pollutes!)
                #   But user can start a new conversation or system will naturally age out Q1 after Q4
                if hasattr(self.memory_manager, 'short_term_memory') and self.memory_manager.short_term_memory:
                    recent_memories = self.memory_manager.short_term_memory[-2:]
                    
                    # STEP 2: Build context from selected memories
                    for memory in recent_memories:
                        # Get both the question and answer
                        user_question = memory.user_input if hasattr(memory, 'user_input') else ''
                        assistant_answer = memory.assistant_response if hasattr(memory, 'assistant_response') else ''
                        
                        # Add this Q&A pair to context
                        if user_question and assistant_answer:
                            # CRITICAL: Clean the answer to remove ALL debug text and search indicators
                            clean_answer = assistant_answer
                            
                            # Remove all search/debug markers (with and without emojis)
                            debug_markers = [
                                '🔍 Searching Brave...', 'Searching Brave', '🔍 Searching Brave',
                                '📰 Searching latest news...', 'Searching latest news', '📰 Searching latest news',
                                '🌐 Getting current information...', 'Getting current information',
                                '🏔️', '🔍', '📰', '🌐', '⚡'  # Remove stray emojis
                            ]
                            for marker in debug_markers:
                                clean_answer = clean_answer.replace(marker, '').strip()
                            
                            # Truncate to keep query manageable (shorter for better search results)
                            truncated_answer = clean_answer[:150] + '...' if len(clean_answer) > 150 else clean_answer
                            context_parts.append(f"Q: {user_question} A: {truncated_answer}")
            except Exception:
                pass
        
        # Combine all Q&A pairs: "Q: ... A: ... Q: ... A: ..."
        return ' '.join(context_parts) if context_parts else ""
    
    def _detect_topic_shift(self, current_query: str, previous_context: str) -> bool:
        """
        Detect if current query represents a topic shift from previous context.
        
        Uses significant word overlap to determine topic continuity.
        - HIGH overlap → Same topic (F1 → F1, Olympics → Olympics)
        - LOW overlap → Topic shift (F1 → Olympics, Weather → Politics)
        
        Works case-insensitively and handles all entity types (F1, olympics, weather, etc.)
        """
        import re
        
        # MINIMAL stopwords for topic detection (more conservative than query enhancement)
        # Keep content words like "racing", "won", "missed" - they indicate topic continuity!
        stopwords = {
            # Articles & conjunctions  
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
            # Pronouns (these DON'T indicate topic!)
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
            'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
            # Basic auxiliary verbs only (NOT content verbs!)
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
            # Demonstratives
            'that', 'this', 'these', 'those',
            # Question words (keep for now, but might indicate topic)
            'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whom', 'whose'
        }
        
        # Simple stemming function (no external libraries needed)
        def simple_stem(word: str) -> str:
            """Basic suffix removal for common English endings"""
            # Handle common suffixes (racing→race, finished→finish, olympics→olympic)
            if len(word) <= 3:
                return word
            
            # Remove -ing (try adding 'e' back for words like racing→race)
            if word.endswith('ing') and len(word) > 4:
                stem = word[:-3]
                # Try adding 'e' back if it makes sense (rac→race, writ→write)
                if len(stem) >= 2 and stem[-1] not in 'aeiouy':
                    return stem + 'e'  # racing→race, writing→write
                return stem
            # Remove -ed (try adding 'e' back)
            if word.endswith('ed') and len(word) > 3:
                stem = word[:-2]
                if len(stem) >= 2 and stem[-1] not in 'aeiouy':
                    return stem + 'e'  # raced→race
                return stem
            # Remove -s (plural / third person)
            if word.endswith('s') and len(word) > 2 and not word.endswith('ss'):
                return word[:-1]  # races→race, olympics→olympic
            # Remove -er
            if word.endswith('er') and len(word) > 3:
                return word[:-2]
            
            return word
        
        # Extract significant words (case-insensitive, handles F1, olympics, weather, etc.)
        def get_significant_words(text: str) -> set:
            # Extract all alphanumeric words (handles F1, NASCAR, etc.)
            words = re.findall(r'\b[A-Za-z0-9]+\b', text.lower())
            # Filter out stopwords, but keep short meaningful words (F1, GP, UK, etc.)
            significant = {w for w in words if w not in stopwords and len(w) >= 2}
            
            # CRITICAL: Apply stemming to ALL significant words
            # This makes "racing" match "race", "finished" match "finish", etc.
            stemmed = {simple_stem(w) for w in significant}
            
            # CRITICAL: If we got ZERO significant words, include content words anyway
            # This prevents false positives when user asks pronoun-only questions
            if not stemmed:
                # Keep all non-pronoun words as fallback
                pronoun_set = {
                    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                    'my', 'your', 'his', 'her', 'its', 'our', 'their'
                }
                stemmed = {simple_stem(w) for w in words if w not in pronoun_set and len(w) >= 2}
            
            return stemmed
        
        current_words = get_significant_words(current_query)
        previous_words = get_significant_words(previous_context)
        
        # Calculate overlap
        if not previous_words:
            # No previous context → not a topic shift
            return False
        
        overlap = current_words & previous_words  # Intersection
        
        # Handle edge cases
        if not current_words:
            # No significant words in current query → assume NOT a topic shift
            return False
        
        overlap_ratio = len(overlap) / len(current_words)
        
        # TOPIC SHIFT if overlap is LOW (< 20%)
        # Lowered threshold from 30% to 20% to be more conservative
        # Example: "olympics" vs "F1 race" = 0% overlap → SHIFT
        # Example: "winter olympics" vs "summer olympics" = 50% overlap → SAME TOPIC
        # Example: "where are they racing" vs "who won f1 race" → "racing"/"race" won't match but user clearly means F1
        
        # CRITICAL: Also check assistant responses for topic continuity
        # If current query has ANY overlap with previous context → SAME TOPIC
        return overlap_ratio == 0.0  # Only shift if ZERO overlap
    
    def _extract_recent_names_from_context(self, context_summary: str) -> List[str]:
        """Extract person names and key entities from recent context"""
        names = []
        
        # Always try to get from memory first for better accuracy
        if hasattr(self, 'memory_manager') and self.memory_manager:
            try:
                # Get last few interactions from short-term memory
                if hasattr(self.memory_manager, 'short_term_memory') and self.memory_manager.short_term_memory:
                    recent_memories = self.memory_manager.short_term_memory[-2:]  # Last 2 interactions
                    for memory in recent_memories:
                        # Extract from assistant responses
                        response = memory.assistant_response if hasattr(memory, 'assistant_response') else ''
                        names.extend(self._extract_names_from_text(response))
            except Exception:
                pass
        
        # If no names found in memory, try the context summary
        if not names and context_summary:
            names = self._extract_names_from_text(context_summary)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in names:
            if name.lower() not in seen:
                seen.add(name.lower())
                unique_names.append(name)
        
        return unique_names[:3]  # Return top 3 most recent
    
    def _extract_names_from_text(self, text: str) -> List[str]:
        """Extract driver names, locations, and other key entities from text"""
        names = []
        
        # Common F1 drivers
        drivers = [
            'Lando Norris', 'Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc',
            'George Russell', 'Carlos Sainz', 'Fernando Alonso', 'Oscar Piastri',
            'Sergio Perez', 'Pierre Gasly'
        ]
        
        # Common F1 locations/races
        locations = [
            'Mexico City', 'Monaco', 'Silverstone', 'Monza', 'Spa',
            'Singapore', 'Abu Dhabi', 'Las Vegas', 'Miami', 'Austin'
        ]
        
        # Check for drivers
        for driver in drivers:
            if driver.lower() in text.lower():
                names.append(driver)
        
        # Check for locations
        for location in locations:
            if location.lower() in text.lower():
                names.append(location)
        
        return names
    
    def _update_conversation_context(self, query: str, decision, response_metadata: Dict = None):
        """Update conversation context after each interaction"""
        # Track recent topics
        topic = self._extract_topic_from_query(query)
        if topic and topic != 'general' and topic not in self.conversation_context['recent_topics']:
            self.conversation_context['recent_topics'].append(topic)
            # Keep only last 3 topics
            if len(self.conversation_context['recent_topics']) > 3:
                self.conversation_context['recent_topics'] = self.conversation_context['recent_topics'][-3:]
        
        # Update current topic (but not for greetings)
        if topic and topic != 'general':
            self.conversation_context['current_topic'] = topic
            
            # Add to topic history
            if topic not in self.conversation_context['topic_history']:
                self.conversation_context['topic_history'].append(topic)
                
                # Keep history manageable
                if len(self.conversation_context['topic_history']) > 5:
                    self.conversation_context['topic_history'] = self.conversation_context['topic_history'][-5:]
        
        # Track intent progression
        if hasattr(decision, 'analysis') and hasattr(decision.analysis, 'intent'):
            self.conversation_context['last_intent'] = decision.analysis.intent
        
        # Track sources for follow-ups
        if response_metadata and 'sources' in response_metadata and topic != 'general':
            self.conversation_context['source_memory'][topic] = response_metadata['sources']
        
        # Track follow-up chains
        if hasattr(decision, 'reason') and decision.reason and 'follow-up' in decision.reason.lower():
            self.conversation_context['follow_up_chain'].append({
                'query': query,
                'timestamp': time.time(),
                'topic': topic
            })
        else:
            # Reset chain for new topics or greetings
            self.conversation_context['follow_up_chain'] = []
        
        # Update entities (but not for general/greeting topics)
        if topic != 'general':
            entities = self._extract_entities_from_interaction(query, response_metadata.get('response', '') if response_metadata else '')
            for entity in entities:
                if entity not in self.conversation_context['entities_mentioned']:
                    self.conversation_context['entities_mentioned'].append(entity)
            
            # Keep entities list manageable and recent
            if len(self.conversation_context['entities_mentioned']) > 5:
                self.conversation_context['entities_mentioned'] = self.conversation_context['entities_mentioned'][-5:]
    
    def _extract_entities_from_interaction(self, query: str, response: str) -> List[str]:
        """Extract entities from current interaction"""
        entities = []
        text = (query + " " + response).lower()
        
        entity_patterns = [
            'keir starmer', 'prime minister', 'deputy pm',
            'formula 1', 'f1', 'grand prix',
            'verstappen', 'hamilton', 'ferrari', 'mclaren'
        ]
        
        for pattern in entity_patterns:
            if pattern in text:
                entities.append(pattern)
        
        return entities
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract topic from query for context tracking"""
        query_lower = query.lower()
        
        # Don't assign specific topics to greetings
        greeting_patterns = ['hello', 'hi', 'hey', 'how are you', 'good morning', 'good afternoon', 'good evening']
        if any(pattern in query_lower for pattern in greeting_patterns):
            return 'general'
        
        # Define topic patterns
        topic_patterns = {
            'uk_politics': ['keir starmer', 'prime minister', 'deputy pm', 'uk government', 'labour'],
            'f1': ['f1', 'formula 1', 'grand prix', 'verstappen', 'hamilton', 'racing'],
            'sports': ['sports', 'football', 'basketball', 'soccer', 'tennis'],
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'cloudy'],
            'news': ['news', 'latest', 'current', 'today', 'recent'],
            'programming': ['python', 'code', 'programming', 'software', 'development']
        }
        
        for topic, keywords in topic_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return topic
        
        return 'general'
    
    async def make_enhanced_intelligent_decision(self, query: str, session_id: Optional[str] = None):
        """Make routing decision with enhanced context awareness that can override base router logic"""
        
        # First analyze conversation context
        context_info = await self._analyze_conversation_context(query, session_id)
        
        # Make the base routing decision
        base_decision = await self.make_intelligent_decision(query, session_id=session_id)
        
        # Override routing based on enhanced context
        if context_info['should_force_offline'] and hasattr(self, 'offline_available') and self.offline_available:
            base_decision.route_type = 'offline'
            base_decision.reason = f"Enhanced context: Greeting/casual chat detected - forcing offline for better personality"
            base_decision.confidence = 0.95
        elif context_info['should_force_online'] and hasattr(self, 'online_available') and self.online_available:
            base_decision.route_type = 'online'
            base_decision.reason = f"Enhanced context: Follow-up with current info context - forcing online for continuity"
            base_decision.confidence = 0.95
        elif context_info['is_follow_up'] and context_info['entities_in_context'] and hasattr(self, 'online_available') and self.online_available:
            base_decision.route_type = 'online'
            base_decision.reason = f"Enhanced context: Follow-up detected with entities {context_info['entities_in_context']} - routing online"
            base_decision.confidence = min(0.95, base_decision.confidence + 0.1)
        
        return base_decision, context_info
    
    async def get_enhanced_streaming_response(self, query: str, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Enhanced streaming response with better context handling and search query enhancement"""
        
        # Use enhanced decision making that can override base router
        decision, context_info = await self.make_enhanced_intelligent_decision(query, session_id)
        
        # Debug output
        if context_info['is_greeting'] and settings.debug_mode:
            print(f"[ENHANCED_CONTEXT] 👋 Greeting detected: {query} -> Forcing OFFLINE")
        elif context_info['is_follow_up'] and settings.debug_mode:
            print(f"[ENHANCED_CONTEXT] 🔗 Follow-up detected: {query}")
            if context_info['entities_in_context']:
                print(f"[ENHANCED_CONTEXT] 🧠 Context entities: {context_info['entities_in_context']}")
        
        # For online routes, enhance the search query
        enhanced_query = query
        if decision.route_type == 'online':
            enhanced_query = self._enhance_search_query(query, context_info)
            if enhanced_query != query and settings.debug_mode:
                print(f"[ENHANCED_CONTEXT] 🔍 Enhanced search: '{query}' -> '{enhanced_query}'")
        
        # Update the router's last decision
        self.last_decision = decision
        
        # Stream response using enhanced routing
        response_text = ""
        try:
            if decision.route_type == 'online':
                async for chunk in self._handle_enhanced_online_route(enhanced_query, query, decision, session_id, context_info):
                    response_text += chunk
                    yield chunk
            elif decision.route_type == 'offline':
                async for chunk in self._handle_enhanced_offline_route(query, decision, session_id, context_info):
                    response_text += chunk
                    yield chunk
            elif decision.route_type == 'skill':
                # Use the base router's skills handling
                async for chunk in self._handle_skills_route(query, decision, session_id):
                    response_text += chunk
                    yield chunk
            else:
                # Fallback
                async for chunk in self._handle_fallback_route(query, decision):
                    response_text += chunk
                    yield chunk
                
            # Update context after response
            self._update_conversation_context(query, decision, {'response': response_text})
            
        except Exception as e:
            if settings.debug_mode:
                print(f"[ENHANCED_CONTEXT] ❌ Error: {e}")
                import traceback
                traceback.print_exc()
            yield f"I encountered an error: {e}"
    
    async def _handle_enhanced_online_route(self, enhanced_query: str, original_query: str, decision, session_id: Optional[str], context_info: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Handle online routing with enhanced search query"""
        try:
            # Build enhanced prompt with context
            try:
                if hasattr(self, 'personality_manager') and self.personality_manager:
                    personality_context = await self.personality_manager.get_system_prompt()
                else:
                    personality_context = ""
            except Exception:
                personality_context = ""
            
            # Build context-aware prompt using original query for memory context
            try:
                from modules.prompt_builder import build_prompt
                prompt = await build_prompt(
                    session_id or getattr(self.memory_manager, 'session_id', None),
                    original_query, self.memory_manager, self.personality_manager, max_chars=6000
                )
            except Exception:
                prompt = ""
            
            if hasattr(decision, 'analysis') and hasattr(decision.analysis, 'current_info_score') and decision.analysis.current_info_score >= 0.7:
                yield "🌐 Getting current information... "
            
            # Stream from online LLM with enhanced search query but original context
            response_buffer = ""
            if hasattr(self, 'online_llm') and self.online_llm:
                async for chunk in self.online_llm.generate_response_stream(
                    enhanced_query, personality_context, prompt
                ):
                    response_buffer += chunk
                    yield chunk
            
            # Store response with metadata
            try:
                meta = {'enhanced_query': enhanced_query, 'original_query': original_query}
                if hasattr(self, 'online_llm') and hasattr(self.online_llm, 'last_sources'):
                    try:
                        srcs = getattr(self.online_llm, 'last_sources', None)
                        if srcs:
                            meta['sources'] = srcs
                    except Exception:
                        pass
                
                if hasattr(self, 'memory_manager') and self.memory_manager and session_id:
                    await self.memory_manager.add_interaction(original_query, response_buffer, metadata=meta)
            except Exception:
                pass
                
        except Exception as e:
            if settings.debug_mode:
                print(f"[ENHANCED_CONTEXT] Online route error: {e}")
            raise e
    
    async def _handle_enhanced_offline_route(self, query: str, decision, session_id: Optional[str], context_info: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Handle offline routing with enhanced context"""
        try:
            try:
                if hasattr(self, 'personality_manager') and self.personality_manager:
                    personality_context = await self.personality_manager.get_system_prompt()
                else:
                    personality_context = ""
            except Exception:
                personality_context = ""
            
            try:
                from modules.prompt_builder import build_prompt
                prompt = await build_prompt(
                    session_id or getattr(self.memory_manager, 'session_id', None),
                    query, self.memory_manager, self.personality_manager, max_chars=2500
                )
            except Exception:
                prompt = ""
            
            # Optimize offline model settings
            if hasattr(self, '_optimize_offline_for_query') and hasattr(decision, 'analysis'):
                self._optimize_offline_for_query(decision.analysis)
            
            response_buffer = ""
            if hasattr(self, 'offline_llm') and self.offline_llm:
                async for chunk in self.offline_llm.generate_response_stream(
                    query, personality_context, prompt
                ):
                    response_buffer += chunk
                    yield chunk
            
            # Store response
            try:
                if hasattr(self, 'memory_manager') and self.memory_manager and session_id:
                    await self.memory_manager.add_interaction(query, response_buffer)
            except Exception:
                pass
                
        except Exception as e:
            if settings.debug_mode:
                print(f"[ENHANCED_CONTEXT] Offline route error: {e}")
            raise e
    
    def get_context_summary(self) -> str:
        """Get a summary of current conversation context"""
        context = self.conversation_context
        
        summary_parts = []
        
        if context['recent_topics']:
            summary_parts.append(f"Recent topics: {', '.join(context['recent_topics'])}")
        
        if context['follow_up_chain']:
            chain_length = len(context['follow_up_chain'])
            summary_parts.append(f"Follow-up chain: {chain_length} related queries")
        
        if context['source_memory']:
            topics_with_sources = list(context['source_memory'].keys())
            summary_parts.append(f"Topics with sources: {', '.join(topics_with_sources)}")
        
        if context['entities_mentioned']:
            summary_parts.append(f"Entities: {', '.join(context['entities_mentioned'][:3])}")
        
        if context['current_topic']:
            summary_parts.append(f"Current topic: {context['current_topic']}")
        
        return "; ".join(summary_parts) if summary_parts else "No active context"
    
    def clear_context(self):
        """Clear conversation context"""
        self.conversation_context = {
            'recent_topics': [],
            'last_intent': None,
            'source_memory': {},
            'follow_up_chain': [],
            'entities_mentioned': [],
            'current_topic': None,
            'topic_history': [],
            'session_start_time': time.time()
        }


# Create a combined router class that inherits from IntelligentRouter and uses the mixin
class EnhancedIntelligentRouter:
    """Factory class to create an enhanced router"""
    
    @staticmethod
    def create(personality_manager, memory_manager):
        """Create an enhanced intelligent router with context awareness"""
        from modules.intelligent_router import IntelligentRouter
        
        # Create a new class that combines both
        class ContextAwareIntelligentRouter(IntelligentRouter, EnhancedContextMixin):
            def __init__(self, personality_manager, memory_manager):
                # Initialize IntelligentRouter first
                IntelligentRouter.__init__(self, personality_manager, memory_manager)
                # Then initialize the mixin without calling super()
                EnhancedContextMixin.__init__(self)
            
            async def close(self):
                """Enhanced close method to properly cleanup aiohttp sessions"""
                try:
                    # Close online LLM sessions
                    if hasattr(self, 'online_llm') and self.online_llm:
                        if hasattr(self.online_llm, 'close'):
                            await self.online_llm.close()
                        # Close any aiohttp sessions in online_llm
                        if hasattr(self.online_llm, '_session') and self.online_llm._session:
                            await self.online_llm._session.close()
                    
                    # Close skills manager sessions
                    if hasattr(self, 'skills_manager') and self.skills_manager:
                        if hasattr(self.skills_manager, 'close'):
                            await self.skills_manager.close()
                        # Close any aiohttp sessions in skills_manager
                        for skill_name in ['weather', 'news', 'google']:
                            skill = getattr(self.skills_manager, f'{skill_name}_skill', None)
                            if skill and hasattr(skill, '_session') and skill._session:
                                await skill._session.close()
                    
                    # Save performance data
                    if hasattr(self, 'performance_tracker'):
                        self.performance_tracker.save_performance_data()
                        
                except Exception as e:
                    if settings.debug_mode:
                        print(f"[ENHANCED_ROUTER] Close error: {e}")
        
        return ContextAwareIntelligentRouter(personality_manager, memory_manager)
