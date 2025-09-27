#!/usr/bin/env python3
"""
Test Script: Current Info Detection Fix
Verify that current information queries are properly routed to online
"""

import asyncio
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode for testing
os.environ['DEBUG'] = 'true'

async def test_current_info_detection():
    """Test the fixed current info detection and routing"""
    print("🔧 Testing FIXED Current Info Detection")
    print("=" * 50)
    
    try:
        # Import the fixed modules
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        from modules.skills_manager import EnhancedSkillsManager
        
        print("✅ FIXED modules imported successfully")
        
        # Create components
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        router = LightningRouter(personality_manager, memory_manager)
        skills_manager = EnhancedSkillsManager()
        
        print("✅ Components created")
        
        # Initialize everything
        await router._check_llm_availability()
        await skills_manager.initialize()
        
        print(f"System Status:")
        print(f"  Offline Available: {router.offline_available}")
        print(f"  Online Available: {router.online_available}")
        print(f"  Skills Available: {router.skills_available}")
        
        # Test queries - these should now work correctly
        test_queries = [
            # Current info queries (should go to ONLINE)
            {
                "query": "What day is today?",
                "expected_route": "online",
                "expected_current_info": True,
                "description": "Should detect current info and route to Groq"
            },
            {
                "query": "What's today's date?",
                "expected_route": "online", 
                "expected_current_info": True,
                "description": "Should detect current info and route to Groq"
            },
            {
                "query": "What is the current date?",
                "expected_route": "online",
                "expected_current_info": True,
                "description": "Should detect current info and route to Groq"
            },
            {
                "query": "Who is the current president?",
                "expected_route": "online",
                "expected_current_info": True,
                "description": "Should detect current info and route to Groq"
            },
            {
                "query": "What's happening in the news today?",
                "expected_route": "online",
                "expected_current_info": True,
                "description": "Should detect current info and route to Groq"
            },
            {
                "query": "What's the weather like?",
                "expected_route": "online",
                "expected_current_info": True,
                "description": "Weather should be treated as current info"
            },
            
            # Simple instant queries (should go to SKILLS)
            {
                "query": "What time is it?",
                "expected_route": "skill",
                "expected_current_info": False,
                "description": "Simple time query should go to skills"
            },
            {
                "query": "Time?",
                "expected_route": "skill",
                "expected_current_info": False,
                "description": "Simple time query should go to skills"
            },
            {
                "query": "15 + 23",
                "expected_route": "skill",
                "expected_current_info": False,
                "description": "Calculator should go to skills"
            },
            
            # General queries (should go to OFFLINE)
            {
                "query": "Hello, how are you?",
                "expected_route": "offline",
                "expected_current_info": False,
                "description": "General greeting should go to offline"
            },
            {
                "query": "Explain what Python is",
                "expected_route": "offline",
                "expected_current_info": False,
                "description": "General explanation should go to offline"
            }
        ]
        
        print(f"\n🧪 Testing Fixed Current Info Detection:")
        print("-" * 50)
        
        correct_detections = 0
        correct_routes = 0
        total_tests = len(test_queries)
        
        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            expected_route = test_case["expected_route"]
            expected_current_info = test_case["expected_current_info"]
            description = test_case["description"]
            
            print(f"\nTest {i}/{total_tests}: '{query}'")
            print(f"  Expected: {expected_route} (current_info: {expected_current_info})")
            print(f"  Description: {description}")
            
            # Test current info detection
            is_current_info = router._detect_current_info_enhanced(query)
            current_info_correct = (is_current_info == expected_current_info)
            
            if current_info_correct:
                print(f"  ✅ Current info detection: {is_current_info} (CORRECT)")
                correct_detections += 1
            else:
                print(f"  ❌ Current info detection: {is_current_info} (WRONG, expected {expected_current_info})")
            
            # Test skills detection (for non-current-info queries)
            if not expected_current_info and expected_route == "skill":
                skill_detected = router._detect_instant_skill(query)
                if skill_detected:
                    print(f"  ✅ Skill detection: {skill_detected}")
                else:
                    print(f"  ❌ Skill detection: None (expected skill)")
            
            # Test routing decision
            decision = router._decide_route_enhanced(query)
            actual_route = decision.route_type
            route_correct = (actual_route == expected_route or 
                           (expected_route == "online" and actual_route == "offline" and not router.online_available) or
                           (expected_route == "skill" and actual_route == "offline" and not router.skills_available))
            
            if route_correct:
                print(f"  ✅ Routing: {actual_route} (CORRECT)")
                print(f"    Reason: {decision.reason}")
                print(f"    Current info flag: {decision.is_current_info}")
                correct_routes += 1
            else:
                print(f"  ❌ Routing: {actual_route} (WRONG, expected {expected_route})")
                print(f"    Reason: {decision.reason}")
                print(f"    Current info flag: {decision.is_current_info}")
        
        # Calculate accuracy
        detection_accuracy = (correct_detections / total_tests) * 100
        routing_accuracy = (correct_routes / total_tests) * 100
        
        print(f"\n📊 FIXED Current Info Detection Results:")
        print("=" * 50)
        print(f"Current Info Detection: {correct_detections}/{total_tests} ({detection_accuracy:.1f}%)")
        print(f"Routing Accuracy: {correct_routes}/{total_tests} ({routing_accuracy:.1f}%)")
        
        # Test actual responses if systems are available
        if router.online_available or router.offline_available:
            print(f"\n🧪 Testing Actual Responses:")
            print("-" * 30)
            
            # Test a current info query
            if router.online_available:
                test_query = "What day is today?"
                print(f"\nTesting current info query: '{test_query}'")
                print(f"Expected: Should route to online (Groq) and provide current date")
                
                try:
                    decision = router._decide_route_enhanced(test_query)
                    print(f"Decision: {decision.route_type} - {decision.reason}")
                    print(f"Current info detected: {decision.is_current_info}")
                    
                    if decision.use_online and decision.is_current_info:
                        print("✅ CORRECT: Current info query routed to online!")
                        
                        # Get actual response
                        print("Getting response...")
                        response = await router.get_response(test_query)
                        print(f"Response preview: {response[:150]}...")
                        
                        # Check if response contains current date info
                        import datetime
                        now = datetime.datetime.now()
                        current_indicators = [
                            now.strftime("%A").lower(),  # Day name
                            now.strftime("%B").lower(),  # Month name
                            str(now.year),               # Year
                            "today",
                            "current"
                        ]
                        
                        response_lower = response.lower()
                        has_current_info = any(indicator in response_lower for indicator in current_indicators)
                        
                        if has_current_info:
                            print("✅ EXCELLENT: Response contains current date information!")
                        else:
                            print("⚠️ Response may not contain current date information")
                    else:
                        print("❌ PROBLEM: Current info query not routed to online properly")
                        
                except Exception as e:
                    print(f"❌ Error testing response: {e}")
            
            # Test a simple time query
            test_query = "What time is it?"
            print(f"\nTesting simple time query: '{test_query}'")
            print(f"Expected: Should route to skills for instant response")
            
            try:
                decision = router._decide_route_enhanced(test_query)
                print(f"Decision: {decision.route_type} - {decision.reason}")
                print(f"Current info detected: {decision.is_current_info}")
                
                if decision.use_skill and not decision.is_current_info:
                    print("✅ CORRECT: Simple time query routed to skills!")
                elif decision.use_offline or decision.use_online:
                    print("⚠️ Acceptable: Routed to LLM (skills may not be available)")
                else:
                    print("❌ PROBLEM: Routing not working properly")
                    
            except Exception as e:
                print(f"❌ Error testing routing: {e}")
        
        # Cleanup
        await router.close()
        await skills_manager.close()
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")
        print("=" * 30)
        
        if detection_accuracy >= 90 and routing_accuracy >= 80:
            print("🎉 EXCELLENT: Current info detection fix is working!")
            print("✅ Current information queries should now route to online properly")
            print("✅ Simple instant queries route to skills")
            print("✅ General queries route to offline")
            success = True
        elif detection_accuracy >= 70 and routing_accuracy >= 70:
            print("✅ GOOD: Current info detection is mostly working")
            print("⚠️ Some edge cases may need refinement")
            success = True
        else:
            print("❌ ISSUES: Current info detection needs more work")
            print(f"Detection accuracy: {detection_accuracy:.1f}% (need >90%)")
            print(f"Routing accuracy: {routing_accuracy:.1f}% (need >80%)")
            success = False
        
        print(f"\n💡 Next Steps:")
        if success:
            print("1. Test Pascal: ./run.sh")
            print("2. Try current info queries: 'What day is today?'")
            print("3. Verify they route to online (should see 🌐 indicator)")
            print("4. Try simple queries: 'What time is it?' (should be instant)")
        else:
            print("1. Review router.py patterns for missed cases")
            print("2. Check skills_manager.py pattern exclusions")
            print("3. Test with debug mode to see routing decisions")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    try:
        result = asyncio.run(test_current_info_detection())
        if result:
            print("\n✅ Current info detection fix test PASSED!")
        else:
            print("\n❌ Current info detection fix test FAILED!")
        return 0 if result else 1
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
        return 1
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
