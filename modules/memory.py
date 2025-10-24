"""
Pascal AI Assistant - Memory Management (extended)

Handles short-term and long-term memory for conversations.
Added: optional metadata / sources for assistant responses and helper to get last assistant sources.

This is a full-file replacement — save as modules/memory.py
"""

import json
import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta

from config.settings import settings

class MemoryInteraction:
    """Represents a single interaction in memory"""

    def __init__(self, user_input: str, assistant_response: str, timestamp: float = None, metadata: Dict[str, Any] = None):
        self.user_input = user_input
        self.assistant_response = assistant_response
        self.timestamp = timestamp or time.time()
        self.datetime = datetime.fromtimestamp(self.timestamp)
        # metadata may contain 'sources': [{'url':..., 'title':...}, ...] and other info
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'user_input': self.user_input,
            'assistant_response': self.assistant_response,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryInteraction':
        """Create from dictionary"""
        return cls(
            user_input=data.get('user_input', ''),
            assistant_response=data.get('assistant_response', ''),
            timestamp=data.get('timestamp'),
            metadata=data.get('metadata', {})
        )


class MemoryManager:
    """Manages Pascal's memory system"""

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.short_term_memory: List[MemoryInteraction] = []
        self.long_term_memory: List[MemoryInteraction] = []
        self.user_preferences: Dict[str, Any] = {}
        self.learned_facts: Dict[str, Any] = {}

        # Memory limits
        self.short_term_limit = getattr(settings, 'short_term_memory_limit', 50)
        self.long_term_enabled = getattr(settings, 'long_term_memory_enabled', False)

        # Auto-save timer
        self.last_save_time = time.time()
        self.save_interval = getattr(settings, 'memory_save_interval', 300)

        # Memory file path
        self.memory_file = settings.get_memory_path(session_id)

    async def load_session(self, session_id: str = None) -> bool:
        """Load memory from file"""
        try:
            if session_id:
                self.session_id = session_id
                self.memory_file = settings.get_memory_path(session_id)

            if not self.memory_file.exists():
                if settings.debug_mode:
                    print(f"No existing memory file for session: {self.session_id}")
                return True  # Not an error, just a new session

            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load interactions (backwards-compatible keys)
            if 'short_term_memory' in data:
                self.short_term_memory = [
                    MemoryInteraction.from_dict(item)
                    for item in data['short_term_memory']
                ]

            if 'long_term_memory' in data and self.long_term_enabled:
                self.long_term_memory = [
                    MemoryInteraction.from_dict(item)
                    for item in data['long_term_memory']
                ]

            # Load user data
            self.user_preferences = data.get('user_preferences', {})
            self.learned_facts = data.get('learned_facts', {})

            # Clean old memories if needed
            await self._cleanup_old_memories()

            if settings.debug_mode:
                print(f"Loaded memory session: {self.session_id}")
                print(f"Short-term memories: {len(self.short_term_memory)}")
                print(f"Long-term memories: {len(self.long_term_memory)}")

            return True

        except Exception as e:
            if settings.debug_mode:
                print(f"Failed to load memory session: {e}")
            return False

    async def save_session(self) -> bool:
        """Save memory to file"""
        try:
            # Prepare data for saving
            data = {
                'session_id': self.session_id,
                'last_updated': time.time(),
                'short_term_memory': [
                    interaction.to_dict()
                    for interaction in self.short_term_memory
                ],
                'user_preferences': self.user_preferences,
                'learned_facts': self.learned_facts
            }

            # Add long-term memory if enabled
            if self.long_term_enabled:
                data['long_term_memory'] = [
                    interaction.to_dict()
                    for interaction in self.long_term_memory
                ]

            # Ensure directory exists
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.last_save_time = time.time()

            if settings.debug_mode:
                print(f"Saved memory session: {self.session_id}")

            return True

        except Exception as e:
            if settings.debug_mode:
                print(f"Failed to save memory session: {e}")
            return False

    async def add_interaction(self, user_input: str, assistant_response: str, metadata: Dict[str, Any] = None):
        """
        Add a new interaction to memory.
        metadata optional dict; commonly contains 'sources': [{'url':..., 'title':...}, ...]
        """
        interaction = MemoryInteraction(user_input, assistant_response, metadata=metadata)

        # Add to short-term memory
        self.short_term_memory.append(interaction)

        # Manage short-term memory size
        if len(self.short_term_memory) > self.short_term_limit:
            # Move oldest to long-term if enabled
            if self.long_term_enabled:
                oldest = self.short_term_memory.pop(0)
                self.long_term_memory.append(oldest)
            else:
                # Just remove oldest
                self.short_term_memory.pop(0)

        # Extract potential learning opportunities
        await self._extract_learning(user_input, assistant_response)

        # Auto-save if needed
        if time.time() - self.last_save_time > self.save_interval:
            await self.save_session()

    async def _extract_learning(self, user_input: str, assistant_response: str):
        """Extract learning opportunities from interactions"""
        user_lower = user_input.lower()

        # Extract user preferences
        preference_indicators = {
            'i like': 'likes',
            'i love': 'loves',
            'i hate': 'dislikes',
            'i prefer': 'prefers',
            'my favorite': 'favorite',
            'i usually': 'habits',
            'i always': 'habits',
            'i never': 'habits'
        }

        for indicator, category in preference_indicators.items():
            if indicator in user_lower:
                # Extract the preference
                parts = user_lower.split(indicator, 1)
                if len(parts) > 1:
                    preference = parts[1].strip()
                    if category not in self.user_preferences:
                        self.user_preferences[category] = []
                    if preference not in self.user_preferences[category]:
                        self.user_preferences[category].append(preference)

        # Extract facts about the user
        fact_indicators = [
            'my name is', 'i am', "i'm", 'i work', 'i live', 'my job'
        ]

        for indicator in fact_indicators:
            if indicator in user_lower:
                parts = user_lower.split(indicator, 1)
                if len(parts) > 1:
                    fact = parts[1].strip()
                    timestamp = time.time()
                    self.learned_facts[indicator.replace(' ', '_')] = {
                        'value': fact,
                        'timestamp': timestamp,
                        'confidence': 0.8
                    }

    async def _cleanup_old_memories(self):
        """Clean up old memories based on age and relevance"""
        if not self.long_term_enabled:
            return

        # Remove memories older than 30 days
        cutoff_time = time.time() - (30 * 24 * 60 * 60)  # 30 days

        self.long_term_memory = [
            memory for memory in self.long_term_memory
            if memory.timestamp > cutoff_time
        ]

        # Limit long-term memory size (keep most recent 1000)
        if len(self.long_term_memory) > 1000:
            self.long_term_memory = self.long_term_memory[-1000:]

    async def get_context(self, include_long_term: bool = True) -> str:
        """Get memory context for LLM prompts"""
        context_parts = []

        # Add recent short-term memory
        if self.short_term_memory:
            context_parts.append("Recent conversation:")
            recent_memories = self.short_term_memory[-10:]  # Last 10 interactions

            for memory in recent_memories:
                context_parts.append(f"User: {memory.user_input}")
                # If assistant response had metadata.sources, include small citation markers
                sources = memory.metadata.get('sources') if isinstance(memory.metadata, dict) else None
                if sources:
                    # Create inline source markers to help follow-ups
                    src_texts = []
                    for i, s in enumerate(sources, 1):
                        title = s.get('title') if isinstance(s, dict) else None
                        url = s.get('url') if isinstance(s, dict) else str(s)
                        src_texts.append(f"[{i}] {title or url}")
                    context_parts.append(f"Assistant: {memory.assistant_response} (sources: {', '.join(src_texts)})")
                else:
                    context_parts.append(f"Assistant: {memory.assistant_response}")

        # Add relevant long-term context
        if include_long_term and self.long_term_enabled and self.long_term_memory:
            # For now, just add a summary of long-term memory count
            context_parts.append(f"\nPrevious conversations: {len(self.long_term_memory)} interactions")

        # Add user preferences
        if self.user_preferences:
            context_parts.append("\nUser preferences:")
            for category, items in self.user_preferences.items():
                if items:
                    context_parts.append(f"- {category.title()}: {', '.join(items[:3])}")  # Limit to 3 items

        # Add learned facts
        if self.learned_facts:
            context_parts.append("\nKnown facts about user:")
            for fact_type, fact_data in self.learned_facts.items():
                if isinstance(fact_data, dict) and 'value' in fact_data:
                    fact_type_readable = fact_type.replace('_', ' ').title()
                    context_parts.append(f"- {fact_type_readable}: {fact_data['value']}")

        return "\n".join(context_parts)

    async def get_last_assistant_sources(self) -> List[Dict[str, Any]]:
        """
        Return the sources from the most recent assistant response in short-term memory,
        or empty list if none.
        """
        if not self.short_term_memory:
            return []
        last = self.short_term_memory[-1]
        metadata = last.metadata or {}
        sources = metadata.get('sources', [])
        # Normalize to list of dicts
        if not isinstance(sources, list):
            return []
        return sources

    async def search_memory(self, query: str, limit: int = 5) -> List[MemoryInteraction]:
        """Search through memory for relevant interactions"""
        query_lower = query.lower()
        relevant_memories = []

        # Search through all memories
        all_memories = self.short_term_memory + (self.long_term_memory if self.long_term_enabled else [])

        for memory in all_memories:
            # Simple keyword matching
            if (query_lower in memory.user_input.lower() or
                query_lower in memory.assistant_response.lower()):
                relevant_memories.append(memory)

        # Sort by recency and return limited results
        relevant_memories.sort(key=lambda x: x.timestamp, reverse=True)
        return relevant_memories[:limit]

    async def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.short_term_memory:
            return "No conversation history yet."

        total_interactions = len(self.short_term_memory) + len(self.long_term_memory)
        recent_topics = []

        # Extract topics from recent interactions
        for memory in self.short_term_memory[-5:]:
            # Simple topic extraction based on key words
            words = memory.user_input.lower().split()
            important_words = [word for word in words if len(word) > 4 and word.isalpha()]
            recent_topics.extend(important_words[:2])  # Take first 2 important words

        # Remove duplicates while preserving order
        unique_topics = []
        for topic in recent_topics:
            if topic not in unique_topics:
                unique_topics.append(topic)

        summary = f"Conversation summary: {total_interactions} total interactions"
        if unique_topics:
            summary += f". Recent topics: {', '.join(unique_topics[:5])}"

        return summary

    async def clear_session(self):
        """Clear current session memory"""
        self.short_term_memory.clear()
        # Keep long-term memory and user data
        await self.save_session()

        if settings.debug_mode:
            print("Cleared short-term memory")

    async def clear_all_memory(self):
        """Clear all memory including long-term"""
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        self.user_preferences.clear()
        self.learned_facts.clear()
        await self.save_session()

        if settings.debug_mode:
            print("Cleared all memory")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'session_id': self.session_id,
            'short_term_count': len(self.short_term_memory),
            'long_term_count': len(self.long_term_memory) if self.long_term_enabled else 0,
            'user_preferences_count': len(self.user_preferences),
            'learned_facts_count': len(self.learned_facts),
            'memory_file_exists': self.memory_file.exists(),
            'last_save_time': self.last_save_time,
            'total_interactions': len(self.short_term_memory) + (len(self.long_term_memory) if self.long_term_enabled else 0)
        }

    async def export_memory(self, export_path: str = None) -> str:
        """Export memory to a file"""
        if not export_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"pascal_memory_export_{timestamp}.json"

        export_data = {
            'export_timestamp': time.time(),
            'session_id': self.session_id,
            'memory_stats': self.get_memory_stats(),
            'short_term_memory': [memory.to_dict() for memory in self.short_term_memory],
            'long_term_memory': [memory.to_dict() for memory in self.long_term_memory] if self.long_term_enabled else [],
            'user_preferences': self.user_preferences,
            'learned_facts': self.learned_facts
        }

        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return export_path

    async def import_memory(self, import_path: str) -> bool:
        """Import memory from a file"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            # Import short-term memory
            if 'short_term_memory' in import_data:
                self.short_term_memory = [
                    MemoryInteraction.from_dict(item)
                    for item in import_data['short_term_memory']
                ]

            # Import long-term memory
            if 'long_term_memory' in import_data and self.long_term_enabled:
                self.long_term_memory = [
                    MemoryInteraction.from_dict(item)
                    for item in import_data['long_term_memory']
                ]

            # Import user data
            if 'user_preferences' in import_data:
                self.user_preferences.update(import_data['user_preferences'])

            if 'learned_facts' in import_data:
                self.learned_facts.update(import_data['learned_facts'])

            # Save the imported data
            await self.save_session()

            if settings.debug_mode:
                print(f"Imported memory from: {import_path}")

            return True

        except Exception as e:
            if settings.debug_mode:
                print(f"Failed to import memory: {e}")
            return False
