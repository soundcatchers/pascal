"""
Pascal AI Assistant - Memory Management
Handles short-term and long-term memory for conversations
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
    
    def __init__(self, user_input: str, assistant_response: str, timestamp: float = None):
        self.user_input = user_input
        self.assistant_response = assistant_response
        self.timestamp = timestamp or time.time()
        self.datetime = datetime.fromtimestamp(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'user_input': self.user_input,
            'assistant_response': self.assistant_response,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryInteraction':
        """Create from dictionary"""
        return cls(
            user_input=data['user_input'],
            assistant_response=data['assistant_response'],
            timestamp=data['timestamp']
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
        self.short_term_limit = settings.short_term_memory_limit
        self.long_term_enabled = settings.long_term_memory_enabled
        
        # Auto-save timer
        self.last_save_time = time.time()
        self.save_interval = settings.memory_save_interval
        
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
            
            # Load interactions
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
    
    async def add_interaction(self, user_input: str, assistant_response: str):
        """Add a new interaction to memory"""
        interaction = MemoryInteraction(user_input, assistant_response)
        
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
        """Extract learning opportunities from interactions
