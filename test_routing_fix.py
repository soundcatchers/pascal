#!/usr/bin/env python3
"""
Test script to verify routing fix for current information queries
"""

import asyncio
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

async def test_routing():
    """Test the routing logic for current info queries"""
    print("🧪 Testing Pascal Routing Fix")
    print("=" * 50)
    
    from modules.router import LightningRouter
    from modules.personality import PersonalityManager
    from modules.memory import MemoryManager
    
    # Create router
    personality_manager = PersonalityManager()
    memory_manager = MemoryManager()
    router = LightningRouter(personality_manager, memory_manager)
    
    # Test queries
    test_queries = [
        ("Hello, how are you?", "Should route OFFLINE (simple greeting)"),
        ("What is 2+2?", "Should route OFFLINE (simple math)"),
        ("What day is today?", "Should route ONLINE (current date)"),
        ("What's the current date?", "Should route ONLINE (current date)"),
        ("What's happening in the news today?", "Should route ONLINE (current news)"),
        ("Who is the current president?", "Should route ONLINE (current info)"),
        ("Explain what AI is", "Should route OFFLINE (general knowledge)"),
    ]
    
    print("\nRouting Decision Tests:")
    print("-" * 50)
    
    for query, expected in test_queries:
        # Check if needs current info
        needs_current = router._needs_current_information(query)
        
        # Get routing decision
        decision = router._decide_route(query)
        route = "OFFLINE" if decision.use_offline else "ONLINE"
        
        print(f"\nQuery: '{query}'")
        print(f"  Needs current info: {needs_current}")
