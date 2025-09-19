#!/usr/bin/env python3
"""
Pascal Enhanced Performance Test Suite
Comprehensive testing of the rewritten offline LLM and routing system
"""

import asyncio
import sys
import time
import statistics
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode for testing
import os
os.environ['DEBUG'] = 'true'

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

class PerformanceTest:
    """Enhanced performance testing class"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def log_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Log test result with details"""
        self.results[test_name] = {
            'success': success,
            'timestamp': time.time(),
            'details': details
        }
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name}")
        
        if details.get('time'):
            print(f"     Time: {details['time']:.2f}s")
        if details.get('response_preview'):
            print(f"     Response: {details['response_preview']}")
        if not success and details.get('error'):
            print(f"     Error: {details['error']}")

async def test_direct_ollama_enhanced():
    """Enhanced direct Ollama testing with response validation"""
    print("🔧 Enhanced Direct Ollama Test")
    print("=" * 50)
    
    if not AIOHTTP_AVAILABLE:
        print("❌ aiohttp not available - install with: pip install aiohttp")
        return False
    
    test = PerformanceTest()
    
    try:
        # Test connection
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get('http://localhost:11434/api/tags') as response:
                if response.status != 200:
                    test.log_result("Ollama Connection", False, {"error": f"HTTP {response.status}"})
                    return False
                
                data = await response.json()
                models = [model['name'] for model in data.get('models', [])]
                
                if not models:
                    test.log_result("Model Availability", False, {"error": "No models found"})
                    return False
                
                # Choose best model
                preferred_models = [
                    'nemotron-mini:4b-instruct-q4_K_M',
                    'nemotron-fast',
                    'qwen2.5:3b',
                    'phi3:mini'
                ]
                
                model_to_use = None
                for preferred in preferred_models:
                    for model in models:
                        if preferred == model or preferred in model:
                            model_to_use = model
                            break
                    if model_to_use:
                        break
                
                if not model_to_use:
                    model_to_use = models[0]
                
                test.log_result("Model Selection", True, {"model": model_to_use, "total_models": len(models)})
                
                # Enhanced test queries with expected response validation
                test_queries = [
                    {
                        "query": "Hello, how are you?",
                        "expected_type": "greeting",
                        "max_time": 3.0,
                        "validate": lambda r: any(word in r.lower() for word in ['hello', 'hi', 'good', 'fine', 'well', 'help'])
                    },
                    {
                        "query": "What is 15 + 27?",
                        "expected_type": "calculation", 
                        "max_time": 2.0,
                        "validate": lambda r: '42' in r or 'forty' in r.lower()
                    },
                    {
                        "query": "Explain Python in one sentence.",
                        "expected_type": "explanation",
                        "max_time": 4.0,
                        "validate": lambda r: 'python' in r.lower() and len(r.split()) >= 5
                    },
                    {
                        "query": "Say hello briefly.",
                        "expected_type": "simple_request",
                        "max_time": 2.0,
                        "validate": lambda r: any(word in r.lower() for word in ['hello', 'hi', 'hey'])
                    },
                    {
                        "query": "Count from 1 to 3.",
                        "expected_type": "counting",
                        "max_time": 3.0,
                        "validate": lambda r: all(num in r for num in ['1', '2', '3'])
                    }
                ]
                
                response_times = []
                quality_scores = []
                
                print(f"\nTesting model: {model_to_use}")
                print("-" * 40)
                
                for i, test_case in enumerate(test_queries, 1):
                    query = test_case["query"]
                    print(f"Test {i}/{len(test_queries)}: '{query}'")
                    
                    start_time = time.time()
                    
                    # Optimized payload for speed
                    payload = {
                        "model": model_to_use,
                        "prompt": f"User: {query}\nAssistant: ",
                        "stream": False,
                        "options": {
                            "num_predict": 50,
                            "num_ctx": 256,
                            "temperature": 0.3,
                            "top_p": 0.8,
                            "top_k": 20,
                            "repeat_penalty": 1.05
                        }
                    }
                    
                    try:
                        async with session.post(
                            'http://localhost:11434/api/generate',
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=test_case["max_time"] + 2)
                        ) as resp:
                            elapsed = time.time() - start_time
                            
                            if resp.status == 200:
                                result = await resp.json()
                                response_text = result.get('response', '').strip()
                                
                                # Calculate quality score
                                quality_score = 0
                                
                                # Time score (40%)
                                if elapsed <= test_case["max_time"]:
                                    time_score = max(0, 40 - (elapsed / test_case["max_time"]) * 20)
                                    quality_score += time_score
                                
                                # Content validation score (40%)
                                if response_text and test_case["validate"](response_text):
                                    quality_score += 40
                                elif response_text:
                                    quality_score += 20  # Partial credit for any response
                                
                                # Length appropriateness score (20%)
                                if 5 <= len(response_text.split()) <= 50:
                                    quality_score += 20
                                elif response_text:
                                    quality_score += 10
                                
                                response_times.append(elapsed)
                                quality_scores.append(quality_score)
                                
                                # Performance evaluation
                                eval_count = result.get('eval_count', 0)
                                eval_duration = result.get('eval_duration', 1)
                                tokens_per_sec = eval_count / max(eval_duration / 1e9, 0.001) if eval_count > 0 else 0
                                
                                # Determine status
                                if quality_score >= 80:
                                    status = "✅ EXCELLENT"
                                elif quality_score >= 60:
                                    status = "✅ GOOD"
                                elif quality_score >= 40:
                                    status = "⚠️ FAIR"
                                else:
                                    status = "❌ POOR"
                                
                                print(f"  {status} {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s, Q:{quality_score:.0f}%)")
                                print(f"     Response: '{response_text[:80]}{'...' if len(response_text) > 80 else ''}'")
                                
                            else:
                                print(f"  ❌ HTTP {resp.status} - {elapsed:.2f}s")
                                response_times.append(elapsed)
                                quality_scores.append(0)
                                
                    except asyncio.TimeoutError:
                        elapsed = time.time() - start_time
                        print(f"  ❌ TIMEOUT - {elapsed:.2f}s")
                        response_times.append(elapsed)
                        quality_scores.append(0)
                    except Exception as e:
                        elapsed = time.time() - start_time
                        print(f"  ❌ ERROR - {elapsed:.2f}s: {str(e)[:50]}")
                        response_times.append(elapsed)
                        quality_scores.append(0)
                
                # Calculate overall performance metrics
                if response_times:
                    avg_time = statistics.mean(response_times)
                    min_time = min(response_times)
                    max_time = max(response_times)
                    avg_quality = statistics.mean(quality_scores)
                    under_3s = sum(1 for t in response_times if t < 3)
                    under_2s = sum(1 for t in response_times if t < 2)
                    good_quality = sum(1 for q in quality_scores if q >= 60)
                    
                    print(f"\n📊 Enhanced Ollama Results:")
                    print(f"Average time: {avg_time:.2f}s")
                    print(f"Time range: {min_time:.2f}s - {max_time:.2f}s")
                    print(f"Average quality: {avg_quality:.1f}%")
                    print(f"Under 3s: {under_3s}/{len(test_queries)}")
                    print(f"Under 2s: {under_2s}/{len(test_queries)}")
                    print(f"Good quality: {good_quality}/{len(test_queries)}")
                    
                    # Overall grade
                    if avg_time < 2 and avg_quality >= 70:
                        grade = "A+ (Excellent)"
                    elif avg_time < 3 and avg_quality >= 60:
                        grade = "A (Very Good)"
                    elif avg_time < 4 and avg_quality >= 50:
                        grade = "B (Good)"
                    elif avg_time < 6 and avg_quality >= 40:
                        grade = "C (Fair)"
                    else:
                        grade = "D (Needs Improvement)"
                    
                    print(f"Overall Grade: {grade}")
                    
                    test.log_result("Overall Performance", avg_time < 4 and avg_quality >= 50, {
                        "avg_time": avg_time,
                        "avg_quality": avg_quality,
                        "grade": grade
                    })
                    
                    return avg_time < 4 and avg_quality >= 50
                else:
                    test.log_result("Response Generation", False, {"error": "No successful responses"})
                    return False
                    
    except Exception as e:
        print(f"❌ Direct Ollama test failed: {e}")
        test.log_result("Test Execution", False, {"error": str(e)})
        return False

async def test_pascal_offline_llm():
    """Test Pascal's rewritten offline LLM"""
    print("\n🚀 Pascal Offline LLM Test")
    print("=" * 50)
    
    test = PerformanceTest()
    
    try:
        # Import the rewritten LLM
        from modules.offline_llm import LightningOfflineLLM
        
        test.log_result("Module Import", True, {"module": "LightningOfflineLLM"})
        
        # Create and initialize
        llm = LightningOfflineLLM()
        
        print("Initializing LightningOfflineLLM...")
        success = await llm.initialize()
        
        if not success:
            test.log_result("LLM Initialization", False, {"error": llm.last_error})
            return False
        
        test.log_result("LLM Initialization", True, {
            "model": llm.current_model,
            "profile": llm.current_profile
        })
        
        # Test different performance profiles
        profiles = ['speed', 'balanced', 'quality']
        profile_results = {}
        
        for profile in profiles:
            print(f"\n🧪 Testing {profile.upper()} profile:")
            print("-" * 30)
            
            llm.set_performance_profile(profile)
            
            # Profile-specific test queries
            if profile == 'speed':
                test_queries = [
                    "Hello",
                    "Hi there",
                    "How are you?",
                    "What's 2+2?",
                    "Say hello"
                ]
            elif profile == 'balanced':
                test_queries = [
                    "Hello, how are you today?",
                    "Explain Python briefly",
                    "What is 15 + 27?",
                    "Tell me about AI",
                    "Write a short greeting"
                ]
            else:  # quality
                test_queries = [
                    "Explain the concept of machine learning",
                    "Write a Python function to sort a list",
                    "What are the benefits of renewable energy?",
                    "Describe how HTTP works",
                    "Explain object-oriented programming"
                ]
            
            times = []
            quality_scores = []
            
            for i, query in enumerate(test_queries, 1):
                print(f"  Test {i}/5: '{query[:40]}...'")
                
                start_time = time.time()
                
                try:
                    response = await llm.generate_response(query, "", "")
                    elapsed = time.time() - start_time
                    
                    # Quality validation
                    quality_score = 0
                    
                    if response and not response.startswith("Model error"):
                        # Check for nonsense responses
                        if llm._validate_response(response, query):
                            quality_score = 80
                            
                            # Additional quality checks
                            if len(response.split()) >= 3:
                                quality_score += 10
                            if query.lower() in response.lower() or any(word in response.lower() for word in query.lower().split()[:2]):
                                quality_score += 10
                        else:
                            quality_score = 20  # Nonsense response detected
                    
                    times.append(elapsed)
                    quality_scores.append(quality_score)
                    
                    if quality_score >= 70:
                        status = "✅"
                    elif quality_score >= 50:
                        status = "⚠️"
                    else:
                        status = "❌"
                    
                    print(f"    {status} {elapsed:.2f}s (Q:{quality_score}%) - {response[:60]}...")
                    
                except Exception as e:
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    quality_scores.append(0)
                    print(f"    ❌ {elapsed:.2f}s - Error: {str(e)[:50]}")
            
            # Profile summary
            if times:
                avg_time = statistics.mean(times)
                avg_quality = statistics.mean(quality_scores)
                under_target = sum(1 for t in times if t < llm.profiles[profile]['timeout'])
                good_quality = sum(1 for q in quality_scores if q >= 70)
                
                profile_results[profile] = {
                    'avg_time': avg_time,
                    'avg_quality': avg_quality,
                    'under_target': under_target,
                    'good_quality': good_quality
                }
                
                print(f"    📊 {profile}: {avg_time:.2f}s avg, {avg_quality:.0f}% quality, {good_quality}/5 good responses")
        
        # Test streaming
        print(f"\n🌊 Testing Streaming:")
        print("-" * 30)
        
        streaming_query = "Count from 1 to 5"
        print(f"Query: '{streaming_query}'")
        print("Response: ", end="", flush=True)
        
        start_time = time.time()
        first_chunk_time = None
        response_text = ""
        chunk_count = 0
        
        try:
            async for chunk in llm.generate_response_stream(streaming_query, "", ""):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                
                print(chunk, end="", flush=True)
                response_text += chunk
                chunk_count += 1
            
            total_time = time.time() - start_time
            print(f"\n    ✅ Streaming: {total_time:.2f}s total, {first_chunk_time:.2f}s first chunk, {chunk_count} chunks")
            
            test.log_result("Streaming Test", True, {
                "total_time": total_time,
                "first_chunk_time": first_chunk_time,
                "chunk_count": chunk_count
            })
            
        except Exception as e:
            print(f"\n    ❌ Streaming failed: {e}")
            test.log_result("Streaming Test", False, {"error": str(e)})
        
        # Overall assessment
        print(f"\n📊 Pascal Offline LLM Results:")
        best_profile = min(profile_results.keys(), key=lambda p: profile_results[p]['avg_time'])
        best_time = profile_results[best_profile]['avg_time']
        best_quality = profile_results[best_profile]['avg_quality']
        
        print(f"Best profile: {best_profile} ({best_time:.2f}s avg, {best_quality:.0f}% quality)")
        
        # Check for improvements over original test
        improvement_detected = best_time < 3.0 and best_quality >= 70
        
        if improvement_detected:
            print("✅ MAJOR IMPROVEMENT detected over original implementation!")
        
        await llm.close()
        
        test.log_result("Pascal Offline LLM", improvement_detected, {
            "best_profile": best_profile,
            "best_time": best_time,
            "best_quality": best_quality
        })
        
        return improvement_detected
        
    except Exception as e:
        print(f"❌ Pascal LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        test.log_result("Pascal Offline LLM", False, {"error": str(e)})
        return False

async def test_routing_accuracy():
    """Test the enhanced routing system"""
    print("\n🚦 Enhanced Routing Test")
    print("=" * 50)
    
    test = PerformanceTest()
    
    try:
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        
        # Create components
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        router = LightningRouter(personality_manager, memory_manager)
        
        test.log_result("Router Creation", True, {"router": "LightningRouter"})
        
        # Test routing decisions
        routing_tests = [
            # Current info queries (should go online if available)
            ("What day is today?", "online", True, "Current date query"),
            ("What time is it?", "skill", False, "Time query (instant skill)"),
            ("Who is the current president?", "online", True, "Current political info"),
            ("Latest news headlines", "online", True, "News query"),
            ("Current weather in London", "online", True, "Weather query"),
            
            # General queries (should prefer offline)
            ("Hello, how are you?", "offline", False, "Greeting"),
            ("Explain Python programming", "offline", False, "Educational content"),
            ("What is 2+2?", "skill", False, "Math (instant skill)"),
            ("Write a Python function", "offline", False, "Programming request"),
            ("Tell me about machine learning", "offline", False, "General explanation"),
            
            # Borderline cases
            ("Explain current events", "online", True, "Explanation + current"),
            ("What is the definition of AI?", "offline", False, "Definition query"),
            ("How does Python work today?", "offline", False, "Technical + temporal but not current info"),
        ]
        
        print("Testing routing decisions:")
        print("-" * 40)
        
        correct_routes = 0
        total_tests = len(routing_tests)
        
        for query, expected_route, expected_current, description in routing_tests:
            # Test current info detection
            detected_current = router._detect_current_info_enhanced(query)
            
            # Test routing decision
            decision = router._decide_route_enhanced(query)
            actual_route = decision.route_type
            
            # Evaluate correctness
            current_correct = detected_current == expected_current
            route_reasonable = (
                actual_route == expected_route or
                (expected_route == "online" and actual_route == "offline" and not router.online_available) or
                (expected_route == "skill" and actual_route == "offline" and not router.skills_available)
            )
            
            if current_correct and route_reasonable:
                correct_routes += 1
                status = "✅"
            elif route_reasonable:
                status = "⚠️"
            else:
                status = "❌"
            
            print(f"  {status} '{query}'")
            print(f"     Expected: {expected_route} (current: {expected_current})")
            print(f"     Actual: {actual_route} (current: {detected_current}) - {decision.reason}")
        
        accuracy = (correct_routes / total_tests) * 100
        print(f"\n📊 Routing Accuracy: {correct_routes}/{total_tests} ({accuracy:.1f}%)")
        
        routing_success = accuracy >= 80
        
        test.log_result("Routing Accuracy", routing_success, {
            "accuracy": accuracy,
            "correct_routes": correct_routes,
            "total_tests": total_tests
        })
        
        await router.close()
        return routing_success
        
    except Exception as e:
        print(f"❌ Routing test failed: {e}")
        test.log_result("Routing Test", False, {"error": str(e)})
        return False

async def test_end_to_end_performance():
    """Test end-to-end system performance"""
    print("\n🎯 End-to-End Performance Test")
    print("=" * 50)
    
    test = PerformanceTest()
    
    try:
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        
        # Initialize full system
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        await personality_manager.load_personality("default")
        await memory_manager.load_session()
        
        router = LightningRouter(personality_manager, memory_manager)
        await router._check_llm_availability()
        
        test.log_result("System Initialization", True, {
            "offline_available": router.offline_available,
            "online_available": router.online_available,
            "skills_available": router.skills_available
        })
        
        # End-to-end test scenarios
        scenarios = [
            {
                "query": "Hello Pascal, how are you?",
                "expected_route": "offline",
                "target_time": 2.0,
                "description": "Basic greeting"
            },
            {
                "query": "What's 15 + 27?",
                "expected_route": "skill",
                "target_time": 0.5,
                "description": "Simple calculation"
            },
            {
                "query": "What day is today?",
                "expected_route": "online" if router.online_available else "offline",
                "target_time": 4.0 if router.online_available else 2.0,
                "description": "Current date query"
            },
            {
                "query": "Explain what Python is briefly",
                "expected_route": "offline",
                "target_time": 3.0,
                "description": "Educational content"
            }
        ]
        
        print("Running end-to-end scenarios:")
        print("-" * 40)
        
        successful_scenarios = 0
        
        for i, scenario in enumerate(scenarios, 1):
            query = scenario["query"]
            target_time = scenario["target_time"]
            
            print(f"\nScenario {i}: {scenario['description']}")
            print(f"Query: '{query}'")
            
            start_time = time.time()
            
            try:
                response = await router.get_response(query)
                elapsed = time.time() - start_time
                
                # Validate response
                valid_response = (
                    response and
                    len(response.strip()) > 5 and
                    not response.startswith("I'm sorry") and
                    not response.startswith("Error") and
                    elapsed <= target_time * 1.5  # Allow 50% tolerance
                )
                
                if valid_response:
                    successful_scenarios += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"{status} {elapsed:.2f}s (target: {target_time:.1f}s)")
                print(f"Response: {response[:100]}...")
                
                if router.last_decision:
                    decision = router.last_decision
                    print(f"Route: {decision.route_type} - {decision.reason}")
                
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ {elapsed:.2f}s - Error: {str(e)[:50]}")
        
        success_rate = (successful_scenarios / len(scenarios)) * 100
        print(f"\n📊 End-to-End Success Rate: {successful_scenarios}/{len(scenarios)} ({success_rate:.1f}%)")
        
        end_to_end_success = success_rate >= 75
        
        test.log_result("End-to-End Performance", end_to_end_success, {
            "success_rate": success_rate,
            "successful_scenarios": successful_scenarios,
            "total_scenarios": len(scenarios)
        })
        
        await router.close()
        return end_to_end_success
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        test.log_result("End-to-End Performance", False, {"error": str(e)})
        return False

async def main():
    """Main test runner with comprehensive reporting"""
    print("⚡ Pascal Enhanced Performance Test Suite")
    print("=" * 60)
    print("Testing rewritten offline LLM and enhanced routing system")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Direct Ollama Enhanced", test_direct_ollama_enhanced),
        ("Pascal Offline LLM", test_pascal_offline_llm),
        ("Routing Accuracy", test_routing_accuracy),
        ("End-to-End Performance", test_end_to_end_performance)
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            results[test_name] = False
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("📊 ENHANCED PERFORMANCE TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\nOverall: {passed}/{total} tests passed ({success_rate:.1f}%)")
    print(f"Total test time: {total_time:.1f}s")
    
    # Performance evaluation
    if success_rate >= 90:
        grade = "A+ (Excellent)"
        message = "🎉 OUTSTANDING PERFORMANCE! All systems optimized."
    elif success_rate >= 75:
        grade = "A (Very Good)"
        message = "✅ GOOD PERFORMANCE! Minor issues detected."
    elif success_rate >= 60:
        grade = "B (Good)"
        message = "⚠️ ACCEPTABLE PERFORMANCE with some issues."
    elif success_rate >= 40:
        grade = "C (Fair)"
        message = "⚠️ PERFORMANCE ISSUES detected - needs attention."
    else:
        grade = "D (Poor)"
        message = "❌ CRITICAL ISSUES - major problems detected."
    
    print(f"\nPerformance Grade: {grade}")
    print(message)
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if not results.get("Direct Ollama Enhanced", False):
        print("• Check Ollama service and model availability")
        print("• Run: sudo systemctl start ollama && ollama pull nemotron-mini:4b-instruct-q4_K_M")
    
    if not results.get("Pascal Offline LLM", False):
        print("• The rewritten LLM module may need debugging")
        print("• Check import paths and dependencies")
    
    if not results.get("Routing Accuracy", False):
        print("• Routing logic may need fine-tuning")
        print("• Check current info detection patterns")
    
    if not results.get("End-to-End Performance", False):
        print("• Full system integration needs work")
        print("• Check component interactions")
    
    if success_rate >= 75:
        print("• Run Pascal: ./run.sh")
        print("• Test with various query types")
        print("• Monitor performance with 'status' command")
    
    return success_rate >= 60

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Performance test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
