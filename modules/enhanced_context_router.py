"""
Enhanced Context-Aware Router Extension
This provides additional context methods for better conversational flow and search query enhancement
"""

import asyncio
import time
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
            'topic_history': []
        }
    
    async def _analyze_conversation_context(self, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        """Analyze conversation context for better routing and query enhancement"""
        context_info = {
            'is_follow_up': False,
            'context_summary': '',
            'entities_in_context': [],
            'topic_continuity': False,
            'reference_words': [],
            'temporal_references': []
        }
        
        query_lower = query.lower().strip()
        
        # Detect reference words (pronouns, demonstratives)
        reference_words = ['he', 'she', 'it', 'they', 'that', 'this', 'there', 'then', 'him', 'her', 'them']
        context_info['reference_words'] = [word for word in reference_words if word in query_lower.split()]
        
        # Detect temporal references
        temporal_refs = ['today', 'yesterday', 'now', 'recently', 'latest', 'new', 'current', 'just']
        context_info['temporal_references'] = [word for word in temporal_refs if word in query_lower.split()]
        
        # Check if this looks like a follow-up
        follow_up_indicators = [
            len(query_lower.split()) <= 6,  # Short queries
            bool(context_info['reference_words']),  # Has pronouns
            bool(context_info['temporal_references']),  # Has temporal refs
            any(word in query_lower for word in ['also', 'too', 'and', 'what about', 'how about'])
        ]
        
        context_info['is_follow_up'] = any(follow_up_indicators)
        
        # Get recent conversation context from memory
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
        except Exception as e:
            if settings.debug_mode:
                print(f"[ENHANCED_CONTEXT] Context analysis error: {e}")
        
        return context_info
    
    def _extract_entities_from_context(self, context: str) -> List[str]:
        """Extract key entities from recent conversation context"""
        entities = []
        context_lower = context.lower()
        
        # Political entities
        political_terms = [
            'keir starmer', 'prime minister', 'deputy pm', 'uk government', 
            'labour party', 'conservative', 'parliament', 'downing street'
        ]
        
        # Sports entities
        sports_terms = [
            'formula 1', 'f1', 'grand prix', 'verstappen', 'hamilton', 
            'ferrari', 'mercedes', 'red bull'
        ]
        
        # General entities
        all_terms = political_terms + sports_terms
        
        for term in all_terms:
            if term in context_lower:
                entities.append(term)
        
        return entities[:5]  # Limit to top 5 entities
    
    def _summarize_recent_context(self, context: str) -> str:
        """Create a brief summary of recent context"""
        # Simple extraction of last few topics
        lines = context.split('\n')
        recent_lines = [line for line in lines[-10:] if line.strip()]
        
        if not recent_lines:
            return "No recent context"
        
        # Look for key topics in recent lines
        topics = []
        for line in recent_lines:
            if 'prime minister' in line.lower():
                topics.append('UK Politics')
            elif any(term in line.lower() for term in ['f1', 'formula', 'race']):
                topics.append('Formula 1')
            elif any(term in line.lower() for term in ['weather', 'temperature']):
                topics.append('Weather')
        
        if topics:
            return f"Recent topics: {', '.join(set(topics))}"
        
        return "General conversation"
    
    def _check_topic_continuity(self, query: str, context: str) -> bool:
        """Check if current query continues the previous topic"""
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Define topic keywords
        topic_keywords = {
            'politics': ['prime minister', 'deputy', 'government', 'party', 'parliament'],
            'f1': ['formula', 'race', 'grand prix', 'driver', 'circuit'],
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
        
        if not context_info['is_follow_up']:
            return query
        
        enhanced_query = query
        entities = context_info.get('entities_in_context', [])
        
        # If query has pronouns and we have entities, substitute context
        if context_info['reference_words'] and entities:
            query_lower = query.lower()
            
            # Handle specific cases
            if 'he' in query_lower and any('keir starmer' in entity.lower() for entity in entities):
                enhanced_query = query.replace('he', 'Keir Starmer').replace('He', 'Keir Starmer')
            
            elif 'it' in query_lower and any('formula' in entity.lower() for entity in entities):
                enhanced_query = query.replace('it', 'Formula 1 race').replace('It', 'Formula 1 race')
            
            # Add contextual terms for better search
            if 'deputy' in query_lower and any('keir starmer' in entity.lower() for entity in entities):
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
        if topic and topic not in self.conversation_context['recent_topics']:
            self.conversation_context['recent_topics'].append(topic)
            # Keep only last 5 topics
            if len(self.conversation_context['recent_topics']) > 5:
                self.conversation_context['recent_topics'] = self.conversation_context['recent_topics'][-5:]
        
        # Update current topic
        if topic:
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
        if response_metadata and 'sources' in response_metadata:
            self.conversation_context['source_memory'][topic] = response_metadata['sources']
        
        # Track follow-up chains
        if hasattr(decision, 'reason') and decision.reason and 'follow-up' in decision.reason.lower():
            self.conversation_context['follow_up_chain'].append({
                'query': query,
                'timestamp': time.time(),
                'topic': topic
            })
        else:
            # Reset chain for new topics
            self.conversation_context['follow_up_chain'] = []
        
        # Update entities
        entities = self._extract_entities_from_interaction(query, response_metadata.get('response', '') if response_metadata else '')
        for entity in entities:
            if entity not in self.conversation_context['entities_mentioned']:
                self.conversation_context['entities_mentioned'].append(entity)
        
        # Keep entities list manageable
        if len(self.conversation_context['entities_mentioned']) > 10:
            self.conversation_context['entities_mentioned'] = self.conversation_context['entities_mentioned'][-10:]
    
    def _extract_entities_from_interaction(self, query: str, response: str) -> List[str]:
        """Extract entities from current interaction"""
        entities = []
        text = (query + " " + response).lower()
        
        entity_patterns = [
            'keir starmer', 'prime minister', 'deputy pm',
            'formula 1', 'f1', 'grand prix',
            'verstappen', 'hamilton', 'ferrari'
        ]
        
        for pattern in entity_patterns:
            if pattern in text:
                entities.append(pattern)
        
        return entities
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract topic from query for context tracking"""
        query_lower = query.lower()
        
        # Define topic patterns
        topic_patterns = {
            'uk_politics': ['keir starmer', 'prime minister', 'deputy pm', 'uk government', 'labour'],
            'f1': ['f1', 'formula 1', 'grand prix', 'verstappen', 'hamilton', 'racing'],
            'sports': ['sports', 'football', 'basketball', 'soccer', 'tennis'],
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'cloudy'],
            'news': ['news', 'latest', 'current', 'today', 'recent'],
            'programming': ['python', 'code', 'programming', 'software', 'development'],
            'general': []
        }
        
        for topic, keywords in topic_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return topic
        
        return 'general'
    
    def _enhance_follow_up_detection(self, query: str, analysis) -> bool:
        """Enhanced follow-up detection using conversation context"""
        # Original detection
        try:
            if hasattr(self, '_looks_like_followup'):
                basic_follow_up = self._looks_like_followup(query, analysis)
            else:
                basic_follow_up = False
        except:
            basic_follow_up = False
        
        if basic_follow_up:
            return True
        
        # Enhanced detection using context
        query_lower = query.lower()
        
        # Check for temporal references
        temporal_refs = ['then', 'next', 'after', 'before', 'later', 'earlier']
        if any(ref in query_lower for ref in temporal_refs):
            return True
        
        # Check for ranking/position references
        position_refs = ['second', 'third', 'fourth', 'next', 'another', 'other']
        if any(ref in query_lower for ref in position_refs) and self.conversation_context['recent_topics']:
            return True
        
        # Check if query relates to recent topics
        recent_topics = self.conversation_context['recent_topics']
        if recent_topics:
            current_topic = self._extract_topic_from_query(query)
            if current_topic in recent_topics[-2:]:  # Last 2 topics
                return True
        
        return False
    
    async def get_enhanced_streaming_response(self, query: str, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Enhanced streaming response with better context handling and search query enhancement"""
        
        # Analyze conversation context
        context_info = await self._analyze_conversation_context(query, session_id)
        
        # Make routing decision with enhanced context
        decision = await self.make_intelligent_decision(query, session_id=session_id)
        
        # Check if this is a follow-up with enhanced detection
        is_follow_up = self._enhance_follow_up_detection(query, decision.analysis)
        
        # Enhance decision with context
        if (context_info['is_follow_up'] and 
            context_info['entities_in_context'] and 
            hasattr(self, 'online_available') and self.online_available):
            
            if decision.route_type != 'online':
                decision.route_type = 'online'
                decision.reason = f"Follow-up with context detected: {context_info['context_summary']} - routing online for continuity"
                decision.confidence = min(0.95, decision.confidence + 0.1)
        
        if is_follow_up and settings.debug_mode:
            print(f"[ENHANCED_CONTEXT] 🔗 Follow-up detected: {query}")
            if context_info['entities_in_context']:
                print(f"[ENHANCED_CONTEXT] 🧠 Context entities: {context_info['entities_in_context']}")
        
        # For online routes with follow-ups, enhance the search query
        enhanced_query = query
        if decision.route_type == 'online' and context_info['is_follow_up']:
            enhanced_query = self._enhance_search_query(query, context_info)
            if enhanced_query != query and settings.debug_mode:
                print(f"[ENHANCED_CONTEXT] 🔍 Enhanced search: '{query}' -> '{enhanced_query}'")
        
        # Stream response using enhanced routing
        response_text = ""
        try:
            if decision.route_type == 'online' and enhanced_query != query:
                # Handle enhanced online route
                async for chunk in self._handle_enhanced_online_route(enhanced_query, query, decision, session_id, context_info):
                    response_text += chunk
                    yield chunk
            else:
                # Use standard routing
                async for chunk in self.get_streaming_response(query, session_id=session_id):
                    response_text += chunk
                    yield chunk
                
            # Update context after response
            self._update_conversation_context(query, decision, {'response': response_text})
            
        except Exception as e:
            if settings.debug_mode:
                print(f"[ENHANCED_CONTEXT] ❌ Error: {e}")
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
            
            if hasattr(decision, 'analysis') and decision.analysis.current_info_score >= 0.7:
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
        
        return ContextAwareIntelligentRouter(personality_manager, memory_manager)
