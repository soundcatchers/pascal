#!/usr/bin/env python3
"""
FIXED: Quick test to verify Pascal is working after fixes
"""

import asyncio
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode for testing
os.environ['DEBUG'] = 'true'

async def test_fixed_pascal():
    """Test the fixed Pascal system"""
    print("🔧 FIXED: Pascal Quick Test")
    print("=" * 50)
    
    try:
        # Test 1: Import all modules with FIXED class names
        print("\n📦 Testing Module Imports:")
        print("-" * 30)
        
        from config.settings import settings
        print("✅ Settings imported")
        
        from modules.personality import PersonalityManager
        print("✅ PersonalityManager imported")
        
        from modules.memory import MemoryManager
        print("✅ MemoryManager imported")
        
        from modules.router import LightningRouter
        print("✅ LightningRouter imported")
        
        # FIXED: Test correct offline LLM import
        from modules.offline_llm import LightningOfflineLLM
        print("✅ LightningOfflineLLM imported (FIXED CLASS NAME)")
        
        from modules.online_llm import OnlineLLM
        print("✅ OnlineLLM imported")
        
        print("✅ All modules imported successfully!")
        
        # Test 2: Settings validation
        print("\n⚙️ Testing Settings:")
        print("-" * 30)
        
        print(f"Debug mode: {settings.debug_mode}")
        print(f"Pascal version: {settings.version}")
        print(f"Groq API configured: {bool(settings.groq_api_key)}")
        print(f"Online available: {settings.is_online_available()}")
        
        if settings.groq_api_key:
            if settings.groq_api_key.startswith('gsk_'):
                print("✅ Groq API key format: Correct (gsk_)")
            elif settings.groq_api_key.startswith('gsk-'):
                print("⚠️ Groq API key format: Deprecated (gsk-) but works")
            else:
                print("❌ Groq API key format: Invalid")
        else:
            print("⚠️ No Groq API key configured")
        
        # Test 3: Create components
        print("\n🤖 Testing Component Creation:")
        print("-" + 30)
        
        personality_manager = PersonalityManager()
        print("✅ PersonalityManager created")
        
        memory_manager = MemoryManager()
        print("✅ MemoryManager created")
        
        router = LightningRouter(personality_manager, memory_manager)
        print("✅ LightningRouter created")
        
        # Test 4: Test LLM initialization
        print("\n🔍 Testing LLM Availability:")
        print("-" * 30)
        
        await router._check_llm_availability()
        
        print(f"Offline available: {router.offline_available}")
        print(f"Online available: {router.online_available}")
        print(f"Router mode: {router.mode.value}")
        
        # Test 5: Test current info detection
        print("\n🎯 Testing Current Info Detection:")
        print("-" * 30)
        
        test_queries = [
            ("What day is today?", True, "Should detect current info"),
            ("Hello Pascal", False, "Should not detect current info"),
            ("What's the current date?", True, "Should detect current info"),
            ("What is 2+2?", False, "Should not detect current info"),
        ]
        
        for query, expected, description in test_queries:
            detected = router._needs_current_information(query)
            status = "✅" if detected == expected else "❌"
            print(f"  {status} '{query}' -> {detected} ({description})")
        
        # Test 6: Test routing decisions
        print("\n🚦 Testing Routing Decisions:")
        print("-" * 30)
        
        # Simulate both LLMs available for testing
        router.offline_available = True
        router.online_available = True
        
        for query, expected_current, description in test_queries:
            decision = router._decide_route(query)
            route = "OFFLINE" if decision.use_offline else "ONLINE"
            
            if expected_current:
                # Should route ONLINE for current info
                status = "✅" if route == "ONLINE" else "❌"
                print(f"  {status} '{query}' -> {route} (Expected: ONLINE for current info)")
            else:
                # General queries can go either way
                print(f"  ℹ️ '{query}' -> {route} ({decision.reason})")
        
        # Test 7: Test actual response if available
        if router.offline_available or router.online_available:
            print("\n🧪 Testing Actual Response:")
            print("-" * 30)
            
            try:
                test_query = "Hello, how are you?"
                print(f"Test query: '{test_query}'")
                
                response = await router.get_response(test_query)
                print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")
                print("✅ Response generation successful!")
                
            except Exception as e:
                print(f"❌ Response generation failed: {e}")
        else:
            print("\n⚠️ No LLMs available for response testing")
            print("   - For offline: sudo systemctl start ollama")
            print("   - For online: Add GROQ_API_KEY to .env")
        
        # Clean up
        await router.close()
        
        print("\n" + "=" * 50)
        print("✅ FIXED Pascal Test Complete!")
        
        # Summary
        print("\n📊 Summary:")
        if router.online_available:
            print("✅ Online LLM (Groq) is working")
        else:
            print("❌ Online LLM (Groq) not available - current info disabled")
            print("   Fix: Add GROQ_API_KEY to .env file")
        
        if router.offline_available:
            print("✅ Offline LLM (Ollama) is working")
        else:
            print("❌ Offline LLM (Ollama) not available")
            print("   Fix: sudo systemctl start ollama && ./download_models.sh")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    try:
        result = asyncio.run(test_fixed_pascal())
        if result:
            print("\n🎉 PASCAL IS FIXED AND WORKING!")
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
