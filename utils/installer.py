"""
Pascal AI Assistant - Installation Utilities
Handles initial setup and configuration creation
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

def create_default_personality() -> Dict[str, Any]:
    """Create default personality configuration"""
    return {
        "name": "Pascal",
        "description": "A helpful, intelligent AI assistant with a friendly personality",
        "traits": {
            "helpfulness": 0.9,
            "curiosity": 0.8,
            "formality": 0.3,
            "humor": 0.6,
            "patience": 0.9
        },
        "speaking_style": {
            "tone": "friendly and approachable",
            "complexity": "adaptive to user level",
            "verbosity": "concise but thorough",
            "examples": True
        },
        "knowledge_focus": [
            "programming and technology",
            "problem-solving",
            "learning and education",
            "creative projects"
        ],
        "conversation_style": {
            "greeting": "Hello! I'm Pascal. How can I help you today?",
            "thinking": "Let me think about that...",
            "clarification": "Could you help me understand what you mean by",
            "completion": "I hope that helps! Is there anything else you'd like to know?",
            "error": "I'm having trouble with that. Let me try a different approach."
        },
        "system_prompt": "You are Pascal, a helpful AI assistant. You are knowledgeable, friendly, and always eager to help. You explain things clearly and ask for clarification when needed. You maintain a consistent personality across all interactions."
    }

def create_assistant_personality() -> Dict[str, Any]:
    """Create assistant personality configuration"""
    return {
        "name": "Pascal Assistant",
        "description": "A more formal, professional version of Pascal for business use",
        "traits": {
            "helpfulness": 0.95,
            "curiosity": 0.7,
            "formality": 0.8,
            "humor": 0.3,
            "patience": 0.95
        },
        "speaking_style": {
            "tone": "professional and courteous",
            "complexity": "technical when appropriate",
            "verbosity": "detailed and comprehensive",
            "examples": True
        },
        "knowledge_focus": [
            "business and productivity",
            "technical documentation",
            "project management",
            "professional communication"
        ],
        "conversation_style": {
            "greeting": "Good day. I'm Pascal, your AI assistant. How may I assist you?",
            "thinking": "Processing your request...",
            "clarification": "To provide the most accurate assistance, could you specify",
            "completion": "I trust this information is helpful. Please let me know if you require further assistance.",
            "error": "I apologize for the difficulty. Let me attempt an alternative approach."
        },
        "system_prompt": "You are Pascal, a professional AI assistant. You are knowledgeable, efficient, and maintain a courteous, business-appropriate demeanor. You provide detailed, accurate information and maintain professionalism in all interactions."
    }

def create_env_template():
    """Create .env.example template file"""
    env_content = """# Pascal AI Assistant Environment Variables
# Copy this file to .env and add your actual API keys

# OpenAI API (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google API (for Gemini models)
GOOGLE_API_KEY=your_google_api_key_here

# Debug settings
DEBUG=false
LOG_LEVEL=INFO

# Performance settings
MAX_CONCURRENT_REQUESTS=3
CACHE_EXPIRY=3600
"""
    return env_content

def create_skills_config() -> Dict[str, Any]:
    """Create initial skills configuration"""
    return {
        "enabled_skills": [],
        "skill_settings": {
            "maps": {
                "default_location": "London, UK",
                "api_key": ""
            },
            "weather": {
                "default_location": "London, UK",
                "api_key": "",
                "units": "metric"
            },
            "music": {
                "default_service": "local",
                "spotify_client_id": "",
                "spotify_client_secret": ""
            },
            "messaging": {
                "default_service": "sms",
                "twilio_account_sid": "",
                "twilio_auth_token": ""
            }
        }
    }

def setup_initial_config():
    """Set up initial configuration files"""
    print("🔧 Creating initial configuration files...")
    
    # Get base directory (pascal/)
    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / "config"
    personalities_dir = config_dir / "personalities"
    
    # Create directories if they don't exist
    personalities_dir.mkdir(parents=True, exist_ok=True)
    
    # Create personality files
    personalities = {
        "default.json": create_default_personality(),
        "assistant.json": create_assistant_personality()
    }
    
    for filename, config in personalities.items():
        filepath = personalities_dir / filename
        if not filepath.exists():
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ Created {filename}")
        else:
            print(f"⏭️  {filename} already exists, skipping")
    
    # Create skills config
    skills_config_path = config_dir / "skills_config.json"
    if not skills_config_path.exists():
        with open(skills_config_path, 'w', encoding='utf-8') as f:
            json.dump(create_skills_config(), f, indent=2)
        print("✅ Created skills_config.json")
    else:
        print("⏭️  skills_config.json already exists, skipping")
    
    # Create .env.example
    env_example_path = base_dir / ".env.example"
    if not env_example_path.exists():
        with open(env_example_path, 'w', encoding='utf-8') as f:
            f.write(create_env_template())
        print("✅ Created .env.example")
    else:
        print("⏭️  .env.example already exists, skipping")
    
    # Create cache directory with .gitkeep
    cache_dir = base_dir / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gitkeep_path = cache_dir / ".gitkeep"
    if not gitkeep_path.exists():
        gitkeep_path.touch()
    
    print("✅ Initial configuration complete!")

if __name__ == "__main__":
    setup_initial_config()
