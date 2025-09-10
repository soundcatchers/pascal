#!/usr/bin/env python3
"""
Test script to verify routing fix for current information queries
"""

import asyncio
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode for testing
os.environ['DEBUG'] = 'true'

async def test_routing():
    """Test the routing logic for current info queries"""
    print("🧪 Testing Pascal Routing Fix")
    print("=" * 50)
    
    try:
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        from config.settings import settings
        
        print("✅ Modules imported successfully")
        
        # Enable debug mode
        settings.debug_mode = True
        
        # Create router components
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        router = LightningRouter(personality_manager, memory_manager)
        
        print("\n📋 Testing Routing Logic:")
        print("-" * 50)
        
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
        
        for query, expected in test_queries:
            # Check if needs current info
            needs_current = router._needs_current_information(query)
            
            # Get routing decision (simulate both LLMs available)
            router.offline_available = True
            router.online_available = True
            decision = router._decide_route(query)
            route = "OFFLINE" if decision.use_offline else "ONLINE"
            
            print(f"\nQuery: '{query}'")
            print(f"  Needs current info: {needs_current}")
            print(f"  Decision: {route} - {decision.reason}")
            print(f"  Expected: {expected}")
            
            # Check if routing is correct
            if "ONLINE" in expected and route == "ONLINE":
                print("  ✅ Correct routing!")
            elif "OFFLINE" in expected and route == "OFFLINE":
                print("  ✅ Correct routing!")
            else:
                print("  ❌ INCORRECT ROUTING!")
        
        print("\n" + "=" * 50)
        
        # Now test actual LLM availability
        print("\n🔍 Checking Actual LLM Availability:")
        print("-" * 50)
        
        await router._check_llm_availability()
        
        print(f"\nOffline Available: {router.offline_available}")
        print(f"Online Available: {router.online_available}")
        print(f"Router Mode: {router.mode.value}")
        
        # Check API configuration
        print("\n🔑 API Configuration:")
        print("-" * 50)
        
        groq_configured = settings.groq_api_key and settings.groq_api_key not in ['', 'your_groq_api_key_here', 'gsk-your_groq_api_key_here']
        gemini_configured = settings.gemini_api_key and settings.gemini_api_key not in ['', 'your_gemini_api_key_here']
        openai_configured = settings.openai_api_key and settings.openai_api_key not in ['', 'sk-your_openai_api_key_here']
        
        print(f"Groq API: {'✅ Configured' if groq_configured else '❌ Not configured'}")
        print(f"Gemini API: {'✅ Configured' if gemini_configured else '❌ Not configured'}")
        print(f"OpenAI API: {'✅ Configured' if openai_configured else '❌ Not configured'}")
        
        # Test actual responses if both are available
        if router.offline_available and router.online_available:
            print("\n" + "=" * 50)
            print("\n🧪 Testing Actual Response Generation:")
            print("-" * 50)
            
            test_queries_actual = [
                "What day is today?",
                "Hello, how are you?"
            ]
            
            for test_query in test_queries_actual:
                print(f"\nTest Query: '{test_query}'")
                decision = router._decide_route(test_query)
                print(f"Routing Decision: {'OFFLINE' if decision.use_offline else 'ONLINE'}")
                print("Getting response...")
                
                try:
                    response = await router.get_response(test_query)
                    print(f"Response preview: {response[:150]}...")
                    
                    # Check response quality
                    if "day is today" in test_query.lower():
                        if any(phrase in response.lower() for phrase in ["don't have real-time", "knowledge cutoff", "cannot provide current", "don't have access"]):
                            print("⚠️ WARNING: Response suggests offline model was used for date query!")
                            print("The routing may need adjustment.")
                        else:
                            print("✅ Response appears to be from online source (should have current info)")
                    else:
                        print("✅ Response generated successfully")
                        
                except Exception as e:
                    print(f"❌ Error getting response: {e}")
        else:
            print("\n⚠️ Cannot test actual responses - one or both LLMs unavailable")
            if not router.online_available:
                print("   Online LLM not available - check API keys in .env")
            if not router.offline_available:
                print("   Offline LLM not available - check Ollama is running")
        
        # Cleanup
        await router.close()
        
        print("\n" + "=" * 50)
        print("✅ Test Complete!")
        
        # Summary
        print("\n📊 Summary:")
        if router.online_available:
            print("✅ Online API is configured and available")
        else:
            print("❌ Online API is NOT available - current info queries won't work properly")
            print("   Fix: Add GROQ_API_KEY to your .env file")
        
        if router.offline_available:
            print("✅ Offline Ollama is available")
        else:
            print("⚠️ Offline Ollama is NOT available")
            print("   Fix: sudo systemctl start ollama")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you're in the pascal directory and virtual environment is activated")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    print("Starting routing test...\n")
    try:
        asyncio.run(test_routing())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
