"""
Pascal AI Assistant - Personality Management
Handles personality loading, switching, and context generation
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

from config.settings import settings

class PersonalityManager:
    """Manages Pascal's personality system"""
    
    def __init__(self):
        self.current_personality = None
        self.personality_data = {}
        self.available_personalities = []
        self.personality_cache = {}
        
        # Initialize by scanning for available personalities
        asyncio.create_task(self._scan_personalities())
    
    async def _scan_personalities(self):
        """Scan for available personality files"""
        personalities_dir = settings.config_dir / "personalities"
        
        if not personalities_dir.exists():
            if settings.debug_mode:
                print("Personalities directory not found")
            return
        
        self.available_personalities = []
        
        for file_path in personalities_dir.glob("*.json"):
            personality_name = file_path.stem
            self.available_personalities.append(personality_name)
            
            if settings.debug_mode:
                print(f"Found personality: {personality_name}")
    
    async def load_personality(self, personality_name: str) -> bool:
        """Load a personality configuration"""
        try:
            # Check if already cached
            if personality_name in self.personality_cache:
                self.personality_data = self.personality_cache[personality_name]
                self.current_personality = personality_name
                return True
            
            # Load from file
            personality_path = settings.get_personality_path(personality_name)
            
            if not personality_path.exists():
                raise FileNotFoundError(f"Personality file not found: {personality_path}")
            
            with open(personality_path, 'r', encoding='utf-8') as f:
                personality_data = json.load(f)
            
            # Validate personality data
            if not self._validate_personality(personality_data):
                raise ValueError(f"Invalid personality configuration: {personality_name}")
            
            # Cache and set as current
            self.personality_cache[personality_name] = personality_data
            self.personality_data = personality_data
            self.current_personality = personality_name
            
            if settings.debug_mode:
                print(f"Loaded personality: {personality_name}")
            
            return True
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Failed to load personality {personality_name}: {e}")
            return False
    
    def _validate_personality(self, data: Dict[str, Any]) -> bool:
        """Validate personality configuration structure"""
        required_fields = ['name', 'description', 'traits', 'speaking_style', 'system_prompt']
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate traits
        if not isinstance(data['traits'], dict):
            return False
        
        # Validate speaking style
        if not isinstance(data['speaking_style'], dict):
            return False
        
        return True
    
    async def get_system_prompt(self) -> str:
        """Generate system prompt based on current personality"""
        if not self.personality_data:
            return "You are Pascal, a helpful AI assistant."
        
        base_prompt = self.personality_data.get('system_prompt', '')
        
        # Add personality traits context
        traits = self.personality_data.get('traits', {})
        if traits:
            traits_text = self._format_traits(traits)
            base_prompt += f"\n\nPersonality traits: {traits_text}"
        
        # Add speaking style context
        speaking_style = self.personality_data.get('speaking_style', {})
        if speaking_style:
            style_text = self._format_speaking_style(speaking_style)
            base_prompt += f"\n\nSpeaking style: {style_text}"
        
        # Add conversation style phrases
        conv_style = self.personality_data.get('conversation_style', {})
        if conv_style:
            base_prompt += f"\n\nUse these conversation patterns when appropriate:"
            for key, phrase in conv_style.items():
                base_prompt += f"\n- {key.title()}: {phrase}"
        
        return base_prompt
    
    def _format_traits(self, traits: Dict[str, float]) -> str:
        """Format personality traits for prompt"""
        trait_descriptions = []
        
        for trait, value in traits.items():
            if value >= 0.8:
                intensity = "very high"
            elif value >= 0.6:
                intensity = "high"
            elif value >= 0.4:
                intensity = "moderate"
            elif value >= 0.2:
                intensity = "low"
            else:
                intensity = "very low"
            
            trait_descriptions.append(f"{trait} ({intensity}: {value})")
        
        return ", ".join(trait_descriptions)
    
    def _format_speaking_style(self, style: Dict[str, Any]) -> str:
        """Format speaking style for prompt"""
        style_parts = []
        
        for key, value in style.items():
            if isinstance(value, bool):
                if value:
                    style_parts.append(f"{key}: enabled")
            else:
                style_parts.append(f"{key}: {value}")
        
        return ", ".join(style_parts)
    
    async def get_greeting(self) -> str:
        """Get personality-appropriate greeting"""
        if not self.personality_data:
            return "Hello! I'm Pascal. How can I help you today?"
        
        conv_style = self.personality_data.get('conversation_style', {})
        return conv_style.get('greeting', "Hello! I'm Pascal. How can I help you today?")
    
    async def get_thinking_phrase(self) -> str:
        """Get personality-appropriate thinking phrase"""
        if not self.personality_data:
            return "Let me think about that..."
        
        conv_style = self.personality_data.get('conversation_style', {})
        return conv_style.get('thinking', "Let me think about that...")
    
    async def get_clarification_phrase(self) -> str:
        """Get personality-appropriate clarification phrase"""
        if not self.personality_data:
            return "Could you help me understand what you mean by"
        
        conv_style = self.personality_data.get('conversation_style', {})
        return conv_style.get('clarification', "Could you help me understand what you mean by")
    
    async def get_completion_phrase(self) -> str:
        """Get personality-appropriate completion phrase"""
        if not self.personality_data:
            return "I hope that helps! Is there anything else you'd like to know?"
        
        conv_style = self.personality_data.get('conversation_style', {})
        return conv_style.get('completion', "I hope that helps! Is there anything else you'd like to know?")
    
    async def get_error_phrase(self) -> str:
        """Get personality-appropriate error phrase"""
        if not self.personality_data:
            return "I'm having trouble with that. Let me try a different approach."
        
        conv_style = self.personality_data.get('conversation_style', {})
        return conv_style.get('error', "I'm having trouble with that. Let me try a different approach.")
    
    async def adjust_response_style(self, response: str) -> str:
        """Adjust response to match personality style"""
        if not self.personality_data:
            return response
        
        # Get style preferences
        speaking_style = self.personality_data.get('speaking_style', {})
        traits = self.personality_data.get('traits', {})
        
        # Apply style adjustments (basic implementation)
        # In a more advanced version, this could use NLP to modify tone, formality, etc.
        
        # Adjust formality
        formality = traits.get('formality', 0.5)
        if formality < 0.3:
            # Make more casual
            response = response.replace("I would", "I'd")
            response = response.replace("You are", "You're")
            response = response.replace("cannot", "can't")
        elif formality > 0.7:
            # Make more formal
            response = response.replace("can't", "cannot")
            response = response.replace("won't", "will not")
            response = response.replace("I'd", "I would")
        
        # Adjust verbosity
        verbosity = speaking_style.get('verbosity', 'moderate')
        if verbosity == 'concise' and len(response) > 200:
            # Could implement response shortening logic here
            pass
        
        return response
    
    def get_personality_info(self) -> Dict[str, Any]:
        """Get current personality information"""
        if not self.personality_data:
            return {}
        
        return {
            'name': self.personality_data.get('name', 'Unknown'),
            'description': self.personality_data.get('description', ''),
            'current': self.current_personality,
            'traits': self.personality_data.get('traits', {}),
            'knowledge_focus': self.personality_data.get('knowledge_focus', [])
        }
    
    def list_available_personalities(self) -> List[str]:
        """Get list of available personalities"""
        return self.available_personalities.copy()
    
    async def create_personality(self, name: str, config: Dict[str, Any]) -> bool:
        """Create a new personality configuration"""
        try:
            # Validate configuration
            if not self._validate_personality(config):
                return False
            
            # Save to file
            personality_path = settings.get_personality_path(name)
            with open(personality_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Update available personalities
            if name not in self.available_personalities:
                self.available_personalities.append(name)
            
            if settings.debug_mode:
                print(f"Created personality: {name}")
            
            return True
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Failed to create personality {name}: {e}")
            return False
    
    async def delete_personality(self, name: str) -> bool:
        """Delete a personality configuration"""
        try:
            # Don't delete if it's the current personality
            if name == self.current_personality:
                return False
            
            # Remove file
            personality_path = settings.get_personality_path(name)
            if personality_path.exists():
                personality_path.unlink()
            
            # Remove from cache and available list
            if name in self.personality_cache:
                del self.personality_cache[name]
            
            if name in self.available_personalities:
                self.available_personalities.remove(name)
            
            if settings.debug_mode:
                print(f"Deleted personality: {name}")
            
            return True
            
        except Exception as e:
            if settings.debug_mode:
                print(f"Failed to delete personality {name}: {e}")
            return False
