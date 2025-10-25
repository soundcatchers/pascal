"""
Enhanced Context-Aware Router Extension
This provides additional context methods for better conversational flow
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, AsyncGenerator

class EnhancedContextMixin:
    """Mixin to add enhanced context awareness to the router"""
    
    def __init__(self):
        super().__init__()
        self.conversation_context = {
            'recent_topics': [],
            'last_intent': None,
            'source_memory': {},
            'follow_up_chain': []
        }
    
    def _update_conversation_context(self, query: str, decision, response_metadata: Dict = None):
        """Update conversation context after each interaction"""
        # Track recent topics
        topic = self._extract_topic_from_query(query)
        if topic and topic not in self.conversation_context['recent_topics']:
            self.conversation_context['recent_topics'].append(topic)
            # Keep only last 5 topics
            if len(self.conversation_context['recent_topics']) > 5:
                self.conversation_context['recent_topics'] = self.conversation_context['recent_topics'][-5:]
        
        # Track intent progression
        self.conversation_context['last_intent'] = decision.analysis.intent
        
        # Track sources for follow-ups
        if response_metadata and 'sources' in response_metadata:
            self.conversation_context['source_memory'][topic] = response_metadata['sources']
        
        # Track follow-up chains
        if decision.reason and 'follow-up' in decision.reason.lower():
            self.conversation_context['follow_up_chain'].append({
                'query': query,
                'timestamp': time.time(),
                'topic': topic
            })
        else:
            # Reset chain for new topics
            self.conversation_context['follow_up_chain'] = []
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract topic from query for context tracking"""
        query_lower = query.lower()
        
        # Define topic patterns
        topic_patterns = {
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
        basic_follow_up = self._looks_like_followup(query, analysis)
        
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
        """Enhanced streaming response with better context handling"""
        
        # Make routing decision with enhanced context
        decision = await self.make_intelligent_decision(query, session_id=session_id)
        
        # Check if this is a follow-up with enhanced detection
        is_follow_up = self._enhance_follow_up_detection(query, decision.analysis)
        
        if is_follow_up and settings.debug_mode:
            print(f"[ENHANCED_CONTEXT] 🔗 Follow-up detected: {query}")
        
        # Update routing decision if enhanced follow-up is detected
        if is_follow_up and not decision.reason.startswith("Short follow-up"):
            # Get last sources from memory
            try:
                last_sources = await self.memory_manager.get_last_assistant_sources()
                if last_sources:
                    # If we have sources and this is a follow-up, prefer online for continuity
                    if self.online_available and decision.route_type != 'online':
                        decision.route_type = 'online'
                        decision.reason = f"Enhanced follow-up detected with prior sources - routing to online for continuity"
                        decision.confidence = min(0.95, decision.confidence + 0.1)
            except:
                pass
        
        # Stream response using the original method
        response_text = ""
        try:
            async for chunk in self.get_streaming_response(query, session_id=session_id):
                response_text += chunk
                yield chunk
                
            # Update context after response
            self._update_conversation_context(query, decision, {'response': response_text})
            
        except Exception as e:
            yield f"I encountered an error: {e}"
    
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
        
        return "; ".join(summary_parts) if summary_parts else "No active context"
