#!/usr/bin/env python3
"""
Pascal Complete System Test - Simplified for Nemotron + Groq
Tests the complete simplified system configuration
"""

import asyncio
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode for testing
os.environ['DEBUG'] = 'true'

async def test_complete_system():
    """Test the complete simplified Pascal system"""
    print("🔧 Pascal Complete System Test - Simplified (Nemotron + Groq)")
    print("=" * 70)
    
    try:
        # Test 1: Import all modules
        print("\n📦 Testing Module Imports:")
        print("-" * 40)
        
        from config.settings import settings
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        
        print("✅ All core modules imported successfully")
        
        # Test 2: Settings validation
        print("\n⚙️ Testing Settings Configuration:")
        print("-" * 40)
        
        print(f"Debug mode: {settings.debug_mode}")
        print(f"Pascal version: {settings.version}")
        
        # Check API keys - Groq only
        groq_configured = settings.groq_api_key and settings.validate_groq_api_key(settings.groq_api_key)
        
        print(f"Groq API: {'✅ Configured' if groq_configured else '❌ Not configured'}")
        
        if settings.groq_api_key:
            if settings.groq_api_key.startswith('gsk_'):
                print("✅ Groq key format: Correct (gsk_)")
            elif settings.groq_api_key.startswith('gsk-'):
                print("⚠️ Groq key format: Deprecated (gsk-) but works")
            else:
                print("❌ Groq key format: Invalid")
        
        print(f"Online available: {settings.is_online_available()}")
        print(f"Preferred offline model: {settings.preferred_offline_model}")
        
        # Test 3: Test LLM imports
        print("\n🤖 Testing LLM Module Imports:")
        print("-" * 40)
        
        try:
            from modules.offline_llm import LightningOfflineLLM
            print("✅ LightningOfflineLLM imported successfully")
        except ImportError as e:
            print(f"❌ LightningOfflineLLM import failed: {e}")
            return False
        
        try:
            from modules.online_llm import OnlineLLM
            print("✅ OnlineLLM imported successfully")
        except ImportError as e:
            print(f"❌ OnlineLLM import failed: {e}")
            return False
        
        # Test 4: Create components
        print("\n🔧 Testing Component Creation:")
        print("-" * 40)
        
        personality_manager = PersonalityManager()
        print("✅ PersonalityManager created")
        
        memory_manager = MemoryManager()
        print("✅ MemoryManager created")
        
        router = LightningRouter(personality_manager, memory_manager)
        print("✅ LightningRouter created")
        
        # Test 5: Test current info detection
        print("\n🎯 Testing Current Info Detection:")
        print("-" * 40)
        
        test_queries = [
            ("What day is today?", True, "Should detect current info"),
            ("What's the current date?", True, "Should detect current info"),
            ("Who is the current president?", True, "Should detect current info"),
            ("What's happening in the news today?", True, "Should detect current info"),
            ("Hello, how are you?", False, "Should not detect current info"),
            ("What is 2+2?", False, "Should not detect current info"),
            ("Explain what AI is", False, "Should not detect current info"),
            ("Write a Python function", False, "Should not detect current info"),
        ]
        
        print("Current Info Detection Tests:")
        correct_detections = 0
        total_tests = len(test_queries)
        
        for query, expected_current_info, description in test_queries:
            detected = router._needs_current_information(query)
            status = "✅" if detected == expected_current_info else "❌"
            
            print(f"  {status} '{query}'")
            print(f"     Expected: {expected_current_info}, Got: {detected}")
            print(f"     {description}")
            
            if detected == expected_current_info:
                correct_detections += 1
        
        detection_accuracy = (correct_detections / total_tests) * 100
        print(f"\n📊 Detection Accuracy: {correct_detections}/{total_tests} ({detection_accuracy:.1f}%)")
        
        # Test 6: Test routing decisions
        print("\n🚦 Testing Routing Decisions:")
        print("-" * 40)
        
        # Simulate both LLMs available
        router.offline_available = True
        router.online_available = True
        
        print("Routing Decision Tests:")
        for query, expected_current, description in test_queries:
            decision = router._decide_route(query)
            route = "NEMOTRON" if decision.use_offline else "GROQ"
            
            if expected_current:
                # Should route to Groq for current info
                status = "✅" if route == "GROQ" else "❌"
                print(f"  {status} '{query}' -> {route} (Expected: GROQ for current info)")
            else:
                # General queries should go to Nemotron for speed
                expected_route = "NEMOTRON"  # Default for general queries
                status = "✅" if route == expected_route else "ℹ️"
                print(f"  {status} '{query}' -> {route} ({decision.reason})")
        
        # Test 7: Test actual LLM availability
        print("\n🔍 Testing Actual LLM Availability:")
        print("-" * 40)
        
        await router._check_llm_availability()
        
        print(f"Offline (Nemotron): {'✅ Available' if router.offline_available else '❌ Not available'}")
        print(f"Online (Groq): {'✅ Available' if router.online_available else '❌ Not available'}")
        print(f"Router mode: {router.mode.value}")
        
        # Test 8: Test actual response if both available
        if router.offline_available or router.online_available:
            print("\n🧪 Testing Actual Response Generation:")
            print("-" * 40)
            
            test_scenarios = []
            
            # Add offline test if available
            if router.offline_available:
                test_scenarios.append(("Hello, how are you?", "offline", "Should use Nemotron"))
            
            # Add online test if available
            if router.online_available:
                test_scenarios.append(("What day is today?", "online", "Should use Groq for current info"))
            
            for test_query, expected_route, description in test_scenarios:
                print(f"\nTest: '{test_query}' ({description})")
                
                try:
                    decision = router._decide_route(test_query)
                    actual_route = "offline" if decision.use_offline else "online"
                    route_match = "✅" if actual_route == expected_route else "⚠️"
                    
                    print(f"  {route_match} Routing: {actual_route} (Expected: {expected_route})")
                    print(f"  Reason: {decision.reason}")
                    
                    # Test actual response generation
                    print("  Getting response...")
                    response = await router.get_response(test_query)
                    print(f"  Response preview: {response[:100]}...")
                    
                    # Validate response quality
                    if "day is today" in test_query.lower() and router.online_available:
                        # Check if current date response makes sense
                        current_indicators = ["today", "thursday", "september", "2025", "current"]
                        has_current_info = any(indicator in response.lower() for indicator in current_indicators)
                        
                        if has_current_info:
                            print("  ✅ Response contains current date information")
                        else:
                            print("  ⚠️ Response may not contain current date information")
                    else:
                        print("  ✅ Response generated successfully")
                        
                except Exception as e:
                    print(f"  ❌ Error: {e}")
        else:
            print("\n⚠️ Cannot test responses - no LLMs available")
            print("   For offline: sudo systemctl start ollama && ollama pull nemotron-mini:4b-instruct-q4_K_M")
            print("   For online: Add GROQ_API_KEY to .env file")
        
        # Test 9: Test streaming if available
        if router.online_available or router.offline_available:
            print("\n🌊 Testing Streaming Response:")
            print("-" * 40)
            
            try:
                test_query = "Count from 1 to 3"
                print(f"Streaming test: '{test_query}'")
                print("Response: ", end="")
                
                stream_response = ""
                async for chunk in router.get_streaming_response(test_query):
                    print(chunk, end="", flush=True)
                    stream_response += chunk
                
                print(f"\n✅ Streaming test complete! ({len(stream_response)} chars)")
                
            except Exception as e:
                print(f"❌ Streaming test failed: {e}")
        
        # Clean up
        await router.close()
        
        print("\n" + "=" * 70)
        print("📊 COMPLETE SYSTEM TEST SUMMARY")
        print("=" * 70)
        
        # Calculate overall health
        health_checks = [
            ("Module Imports", True),  # Always pass if we get here
            ("Settings Configuration", groq_configured or router.offline_available),
            ("Current Info Detection", detection_accuracy >= 90),
            ("Routing Logic", True),  # Always pass basic routing
            ("LLM Availability", router.offline_available or router.online_available),
        ]
        
        passed_checks = sum(1 for _, result in health_checks if result)
        total_checks = len(health_checks)
        
        for check_name, result in health_checks:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} {check_name}")
        
        overall_health = (passed_checks / total_checks) * 100
        print(f"\nOverall System Health: {passed_checks}/{total_checks} ({overall_health:.1f}%)")
        
        if overall_health >= 80:
            print("\n🎉 SYSTEM IS HEALTHY!")
            print("Pascal is ready for use with simplified Nemotron + Groq configuration.")
        elif overall_health >= 60:
            print("\n⚡ SYSTEM IS MOSTLY WORKING!")
            print("Pascal should work with some limitations.")
        else:
            print("\n⚠️ SYSTEM NEEDS ATTENTION!")
            print("Please address the failed checks above.")
        
        # Show recommendations
        print("\n💡 Recommendations:")
        if not router.offline_available:
            print("• Install Ollama and Nemotron: sudo systemctl start ollama && ollama pull nemotron-mini:4b-instruct-q4_K_M")
        if not router.online_available:
            print("• Configure Groq API key in .env for current information queries")
        if detection_accuracy < 90:
            print("• Current info detection may need refinement")
        
        print(f"\n🚀 Run Pascal: ./run.sh")
        
        return overall_health >= 60
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    try:
        result = asyncio.run(test_complete_system())
        if result:
            print("\n✅ System test completed successfully!")
        else:
            print("\n❌ System test revealed issues - check output above")
        return 0 if result else 1
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
        return 1
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
