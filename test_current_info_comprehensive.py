#!/usr/bin/env python3
"""
FIXED: Comprehensive test script for current information routing
Tests all aspects of the fixed system with Groq + Gemini only
"""

import asyncio
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode for comprehensive testing
os.environ['DEBUG'] = 'true'

async def test_comprehensive_current_info():
    """FIXED: Comprehensive test of current info system"""
    print("🔧 FIXED: Pascal Current Information System Test (Groq + Gemini)")
    print("=" * 70)
    
    try:
        # Test 1: Import all modules
        print("\n📦 Testing Module Imports:")
        print("-" * 40)
        
        from config.settings import settings
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        
        print("✅ All modules imported successfully")
        
        # Test 2: Settings validation
        print("\n⚙️ Testing Settings Configuration:")
        print("-" * 40)
        
        print(f"Debug mode: {settings.debug_mode}")
        print(f"Pascal version: {settings.version}")
        
        # Check API keys - Groq + Gemini only
        groq_configured = settings.groq_api_key and settings.validate_groq_api_key(settings.groq_api_key)
        gemini_configured = settings.gemini_api_key and settings.validate_gemini_api_key(settings.gemini_api_key)
        
        print(f"Groq API: {'✅ Configured' if groq_configured else '❌ Not configured'}")
        print(f"Gemini API: {'✅ Configured' if gemini_configured else '❌ Not configured'}")
        
        if settings.groq_api_key:
            if settings.groq_api_key.startswith('gsk_'):
                print("  ✅ Groq key format: Correct (gsk_)")
            elif settings.groq_api_key.startswith('gsk-'):
                print("  ⚠️ Groq key format: Deprecated (gsk-) but works")
            else:
                print("  ❌ Groq key format: Invalid")
        
        print(f"Online available: {settings.is_online_available()}")
        print(f"Current info priority: {settings.force_online_current_info}")
        
        # Test 3: Current info detection patterns
        print("\n🎯 Testing Current Info Detection Patterns:")
        print("-" * 40)
        
        # Create router for testing
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        router = LightningRouter(personality_manager, memory_manager)
        
        # Comprehensive test queries - FIXED with all variations
        test_queries = [
            # These SHOULD be detected as current info
            ("What day is today?", True, "Primary date question"),
            ("What is today's date?", True, "Date with apostrophe"),
            ("What is todays date?", True, "Date without apostrophe - FIXED"),
            ("What is the date today?", True, "Date with different word order - FIXED"),
            ("Tell me todays date", True, "Tell me variation - FIXED"),
            ("Tell me today's date", True, "Tell me with apostrophe - FIXED"),
            ("Give me today's date", True, "Give me variation - FIXED"),
            ("What's the current date?", True, "Current date query"),
            ("What time is it?", True, "Time query"),
            ("Who is the current president?", True, "Current status query"),
            ("Who is current president?", True, "Current status without 'the' - FIXED"),
            ("What's happening in the news today?", True, "Current news query"),
            ("Current weather", True, "Current weather"),
            ("Today's weather", True, "Today's weather"),
            
            # These should NOT be detected as current info  
            ("Hello, how are you?", False, "Simple greeting"),
            ("What is 2+2?", False, "Simple math"),
            ("Explain what AI is", False, "General knowledge"),
            ("Write a Python function", False, "Programming task"),
            ("What is the capital of France?", False, "Static knowledge"),
        ]
        
        print("Current Info Detection Tests:")
        correct_detections = 0
        total_tests = len(test_queries)
        
        for query, expected_current_info, description in test_queries:
            detected = router._needs_current_information(query)
            status = "✅" if detected == expected_current_info else "❌"
            
            print(f"  {status} '{
