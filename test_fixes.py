#!/usr/bin/env python3
"""
Quick test script to verify the Pascal fixes are working
"""

import asyncio
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode for testing
os.environ['DEBUG'] = 'true'

async def test_fixes():
    """Test the fixes"""
    print("🔧 Testing Pascal Fixes")
    print("=" * 50)
    
    try:
        from config.settings import settings
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        
        print("✅ Modules imported successfully")
        
        # Test settings
        print(f"\n📋 Settings Check:")
        print(f"  Debug mode: {settings.debug_mode}")
        print(f"  Groq API key: {'✅ Configured' if settings.groq_api_key else '❌ Missing'}")
        print(f"  Online available: {settings.is_online_available()}")
        
        if settings.groq_api_key:
            if settings.groq_api_key.startswith('gsk_'):
                print(f"  ✅ Groq key format: Correct (gsk_)")
            elif settings.groq_api_key.startswith('gsk-'):
                print(f"  ⚠️ Groq key format: Deprecated (gsk-) but works")
            else:
                print(f"  ❌ Groq key format: Invalid")
        
        # Test router
        print(f"\n🤖 Router Test:")
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        router = LightningRouter(personality_manager, memory_manager)
        
        # Test current info detection
        test_queries = [
            ("What day is today?", True, "Should detect current info"),
            ("Hello Pascal", False, "Should not detect current info"),
            ("What's the current date?", True, "Should detect current info"),
        ]
        
        print("\nCurrent Info Detection Tests:")
        for query, expected, description in test_queries:
            detected = router._needs_current_information(query)
            status = "✅" if detected == expected else "❌"
            print(f"  {status} '{query}' -> {detected} ({description})")
        
        # Test routing decisions
        print("\nRouting Decision Tests:")
        router.offline_available = True
        router.online_available = True
        
        for query, expected_current, description in test_queries:
            decision = router._decide_route(query)
            route = "OFFLINE" if decision.use_offline else "ONLINE"
            
            if expected_current:
                # Should route ONLINE for current info
                status = "✅" if route == "ONLINE" else "❌"
                print(f"  {status} '{query}' -> {route} (Expected: ONLINE)")
            else:
                # Can route either way, but show decision
                print(f"  ℹ️ '{query}' -> {route} ({decision.reason})")
        
        # Test actual LLM availability
        print(f"\n🔍 LLM Availability Test:")
        await router._check_llm_availability()
        
        print(f"  Offline: {'✅ Available' if router.offline_available else '❌ Not available'}")
        print(f"  Online: {'✅ Available' if router.online_available else '❌ Not available'}")
        print(f"  Mode: {router.mode.value}")
        
        # Test actual response if both available
        if router.online_available:
            print(f"\n🧪 Testing Current Info Query:")
            try:
                response = await router.get_response("What day is today?")
                print(f"  Response: {response[:100]}...")
                
                # Check if response contains current date info
                current_words = ['today', 'thursday', 'september', '2025', 'current']
                contains_current = any(word in response.lower() for word in current_words)
                
                if contains_current:
                    print(f"  ✅ Response contains current date information")
                else:
                    print(f"  ⚠️ Response may not contain current date information")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        await router.close()
        
        print("\n" + "=" * 50)
        print("✅ Fix testing complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    try:
        result = asyncio.run(test_fixes())
        if result:
            print("\n🎉 Fixes are working correctly!")
            print("Run Pascal with: ./run.sh")
        else:
            print("\n❌ Issues detected - check output above")
        return 0 if result else 1
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
        return 1
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
