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
        
        # If it's clearly standalone and long enough, it's not a follow-up
        if context_info['is_standalone'] and len(query_lower.split()) > 4:
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
        
        context_info['is_follow_up'] = any(follow_up_indicators) and has_recent_context
        
        # Get recent conversation context from memory only if it seems like a follow-up
        if context_info['is_follow_up']:
            try:
                if session_id and hasattr(self, 'memory_manager') and self.memory_manager:
                    recent_context = await self.memory_manager.get_context(include_long_term=False)
                    if recent_context:
                        # Extract entities and topics from recent context
                        context_info['entities_in_context'] = self._extract_entities_from_context(recent_context)
                        context_info['context_summary'] = self._summarize_recent_context(recent_context)
                        
                        # Check topic continuity
                        if self.conversation_context['current_topic']:
                            context_info['topic_continuity'] = self._check_topic_continuity(query, recent_context)
                        
                        # If follow-up has current info context, force online
                        if context_info['entities_in_context'] and any(
                            entity in ['f1', 'formula 1', 'grand prix', 'prime minister', 'news'] 
                            for entity in context_info['entities_in_context']
                        ):
                            context_info['should_force_online'] = True
            except Exception as e:
                if settings.debug_mode:
                    print(f"[ENHANCED_CONTEXT] Context analysis error: {e}")
        
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
        """Enhance search query with conversation context"""
        
        # For standalone F1 queries, improve the search terms
        query_lower = query.lower()
        if any(term in query_lower for term in ['f1', 'formula', 'grand prix']):
            # Make F1 searches more current and broader
            if 'last' in query_lower or 'latest' in query_lower:
                # Remove specific year references and make more general
                enhanced = query.replace('2025', '').replace('2024', '')
                enhanced = enhanced.replace('last f1 race', 'latest F1 race results')
                enhanced = enhanced.replace('latest f1 race', 'latest F1 race results')
                return enhanced.strip()
        
        if not context_info['is_follow_up'] or not context_info['entities_in_context']:
            return query
        
        enhanced_query = query
        entities = context_info.get('entities_in_context', [])
        
        # If query has pronouns and we have entities, substitute context
        if context_info['reference_words'] and entities:
            query_lower = query.lower()
            
            # Handle specific cases
            if 'he' in query_lower and any('keir starmer' in entity.lower() for entity in entities):
                enhanced_query = query.replace('he', 'Keir Starmer').replace('He', 'Keir Starmer')
            
            elif 'it' in query_lower and any('formula' in entity.lower() or 'f1' in entity.lower() for entity in entities):
                enhanced_query = query.replace('it', 'Formula 1 race').replace('It', 'Formula 1 race')
            
            # Handle F1 follow-ups specifically
            elif any(word in query_lower for word in ['second', 'third', 'who came', 'who finished']) and any('f1' in entity.lower() or 'formula' in entity.lower() or 'grand prix' in entity.lower() for entity in entities):
                # Add F1 context to the query
                enhanced_query = f"F1 Formula 1 latest race results {enhanced_query} podium positions"
            
            # Add contextual terms for better search
            elif 'deputy' in query_lower and any('keir starmer' in entity.lower() or 'prime minister' in entity.lower() for entity in entities):
                enhanced_query = f"UK Deputy Prime Minister {enhanced_query}"
            
            elif any(word in query_lower for word in ['new', 'today', 'recent']) and entities:
                # Add the most relevant entity as context
                main_entity = entities[0]
                enhanced_query = f"{main_entity} {enhanced_query}"
        
        return enhanced_query
    
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
