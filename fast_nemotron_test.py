#!/usr/bin/env python3
"""
Pascal Speed Optimization Test Suite
Comprehensive speed testing for Pascal's optimized components
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

async def test_direct_ollama_speed():
    """Test direct Ollama API speed without Pascal overhead"""
    print("🔧 Direct Ollama Speed Test")
    print("=" * 40)
    
    if not AIOHTTP_AVAILABLE:
        print("❌ aiohttp not available - install with: pip install aiohttp")
        return False
    
    try:
        # Get available models
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get('http://localhost:11434/api/tags') as response:
                if response.status != 200:
                    print("❌ Ollama not responding")
                    return False
                
                data = await response.json()
                models = [model['name'] for model in data.get('models', [])]
                
                if not models:
                    print("❌ No Ollama models found")
                    return False
                
                # Choose best model
                model_to_use = None
                for preferred in ['nemotron-fast:latest', 'nemotron-fast', 'nemotron-mini:4b-instruct-q4_K_M']:
                    for model in models:
                        if preferred == model or preferred in model:
                            model_to_use = model
                            break
                    if model_to_use:
                        break
                
                if not model_to_use:
                    model_to_use = models[0]
                
                print(f"✅ Testing direct Ollama API...")
                print(f"Using model: {model_to_use}")
                
                # Test queries
                test_queries = ["Hi", "Hello", "Good morning", "How are you?", "Thanks"]
                response_times = []
                
                for i, query in enumerate(test_queries, 1):
                    print(f"Test {i}/5: '{query}'")
                    
                    start_time = time.time()
                    
                    payload = {
                        "model": model_to_use,
                        "prompt": query,
                        "stream": False,
                        "options": {
                            "num_predict": 50,
                            "num_ctx": 256,
                            "temperature": 0.3,
                            "top_p": 0.8,
                            "top_k": 20
                        }
                    }
                    
                    try:
                        async with session.post(
                            'http://localhost:11434/api/generate',
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=8)
                        ) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                elapsed = time.time() - start_time
                                response_times.append(elapsed)
                                
                                response_text = result.get('response', '').strip()
                                eval_count = result.get('eval_count', 0)
                                eval_duration = result.get('eval_duration', 1)
                                tokens_per_sec = eval_count / max(eval_duration / 1e9, 0.001) if eval_count > 0 else 0
                                
                                print(f"  ✅ {elapsed:.2f}s - {tokens_per_sec:.1f} tok/s")
                                print(f"     Response: '{response_text[:50]}{'...' if len(response_text) > 50 else ''}'")
                            else:
                                elapsed = time.time() - start_time
                                response_times.append(elapsed)
                                print(f"  ❌ {elapsed:.2f}s - HTTP {resp.status}")
                    except asyncio.TimeoutError:
                        elapsed = time.time() - start_time
                        response_times.append(elapsed)
                        print(f"  ❌ {elapsed:.2f}s - TIMEOUT")
                    except Exception as e:
                        elapsed = time.time() - start_time
                        response_times.append(elapsed)
                        print(f"  ❌ {elapsed:.2f}s - ERROR: {e}")
                
                if response_times:
                    avg_time = statistics.mean(response_times)
                    min_time = min(response_times)
                    max_time = max(response_times)
                    under_3s = sum(1 for t in response_times if t < 3)
                    under_2s = sum(1 for t in response_times if t < 2)
                    
                    print(f"\n📊 Direct Ollama Results:")
                    print(f"Average time: {avg_time:.2f}s")
                    print(f"Min time: {min_time:.2f}s")
                    print(f"Max time: {max_time:.2f}s")
                    print(f"Under 3s: {under_3s}/5")
                    print(f"Under 2s: {under_2s}/5")
                    
                    if avg_time < 3:
                        print("✅ GOOD - Direct Ollama meets speed target")
                        return True
                    else:
                        print("⚠️ SLOW - Direct Ollama needs optimization")
                        return False
                else:
                    print("❌ No successful responses")
                    return False
                    
    except Exception as e:
        print(f"❌ Direct Ollama test failed: {e}")
        return False

async def test_pascal_optimized_speed():
    """Test Pascal's optimized offline LLM performance"""
    print("🚀 Pascal Optimized Speed Test")
    print("=" * 40)
    
    try:
        # Import Pascal's optimized LLM
        from modules.offline_llm import LightningOfflineLLM
        from config.settings import settings
        
        print("✅ Modules imported successfully")
        
        # Create and initialize
        llm = LightningOfflineLLM()
        
        if not await llm.initialize():
            print("❌ LLM initialization failed")
            print(f"Error: {llm.last_error}")
            return False
        
        print("✅ LightningOfflineLLM initialized")
        
        # Test different profiles
        profiles = ['speed', 'balanced']
        results = {}
        
        for profile in profiles:
            print(f"\n🧪 Testing {profile} profile:")
            llm.set_performance_profile(profile)
            
            test_queries = ["Hi", "Hello", "What's 2+2?", "Good morning", "Thanks"]
            profile_times = []
            
            for i, query in enumerate(test_queries, 1):
                print(f"  Test {i}/5: '{query}'")
                
                start_time = time.time()
                try:
                    response = await llm.generate_response(query, "", "")
                    elapsed = time.time() - start_time
                    profile_times.append(elapsed)
                    
                    if response and not response.startswith("Model error"):
                        print(f"    ✅ {elapsed:.2f}s - {response[:40]}...")
                    else:
                        print(f"    ❌ {elapsed:.2f}s - Error")
                except Exception as e:
                    elapsed = time.time() - start_time
                    profile_times.append(elapsed)
                    print(f"    ❌ {elapsed:.2f}s - Exception: {str(e)[:40]}...")
            
            if profile_times:
                avg_time = statistics.mean(profile_times)
                under_3s = sum(1 for t in profile_times if t < 3)
                results[profile] = {
                    'avg_time': avg_time,
                    'times': profile_times,
                    'under_3s': under_3s
                }
                
                print(f"    📊 {profile}: {avg_time:.2f}s avg, {under_3s}/5 under 3s")
        
        # Overall assessment
        if results:
            best_profile = min(results.keys(), key=lambda p: results[p]['avg_time'])
            best_time = results[best_profile]['avg_time']
            
            print(f"\n📊 Pascal Optimized Results:")
            print(f"Best profile: {best_profile} ({best_time:.2f}s avg)")
            
            if best_time < 3:
                print("✅ GOOD - Pascal optimization meets speed target")
                await llm.close()
                return True
            else:
                print("⚠️ SLOW - Pascal optimization needs work")
                await llm.close()
                return False
        else:
            print("❌ No results collected")
            await llm.close()
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_router_overhead():
    """Test router routing overhead"""
    print("🚦 Router Overhead Test")
    print("=" * 40)
    
    try:
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        from config.settings import settings
        
        print("✅ Router modules imported")
        
        # Create router components
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        router = LightningRouter(personality_manager, memory_manager)
        
        print("✅ Router components created")
        
        # Test routing decisions (no actual LLM calls)
        test_queries = [
            ("Hello", False),
            ("What day is today?", True),
            ("How are you?", False),
            ("Current weather", True),
            ("What's 2+2?", False)
        ]
        
        decision_times = []
        
        for query, expected_current in test_queries:
            start_time = time.time()
            
            # Test routing decision speed
            decision = router._decide_route_fast(query)
            current_detected = router._detect_current_info_fast(query)
            
            elapsed = time.time() - start_time
            decision_times.append(elapsed)
            
            route_type = "GROQ" if decision.use_online else "NEMOTRON"
            current_match = "✅" if current_detected == expected_current else "❌"
            
            print(f"  '{query}' -> {route_type} ({elapsed*1000:.1f}ms) {current_match}")
        
        if decision_times:
            avg_decision_time = statistics.mean(decision_times)
            max_decision_time = max(decision_times)
            
            print(f"\n📊 Router Overhead Results:")
            print(f"Average decision time: {avg_decision_time*1000:.2f}ms")
            print(f"Max decision time: {max_decision_time*1000:.2f}ms")
            
            if avg_decision_time < 0.01:  # Under 10ms
                print("✅ EXCELLENT - Router overhead is minimal")
                await router.close()
                return True
            elif avg_decision_time < 0.05:  # Under 50ms
                print("✅ GOOD - Router overhead is acceptable")
                await router.close()
                return True
            else:
                print("⚠️ SLOW - Router overhead is high")
                await router.close()
                return False
        else:
            print("❌ No decision times collected")
            await router.close()
            return False
            
    except Exception as e:
        print(f"❌ Router test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_streaming_vs_regular():
    """Test streaming vs regular response performance"""
    print("🌊 Streaming vs Regular Benchmark")
    print("=" * 40)
    
    try:
        from modules.offline_llm import LightningOfflineLLM
        
        llm = LightningOfflineLLM()
        if not await llm.initialize():
            print("❌ LLM initialization failed")
            return False
        
        llm.set_performance_profile('speed')
        test_query = "Count from 1 to 3"
        
        # Test regular response
        print("Testing regular response...")
        start_time = time.time()
        regular_response = await llm.generate_response(test_query, "", "")
        regular_time = time.time() - start_time
        
        print(f"  Regular: {regular_time:.2f}s")
        print(f"  Response: {regular_response[:50]}...")
        
        # Test streaming response
        print("Testing streaming response...")
        start_time = time.time()
        first_chunk_time = None
        streaming_response = ""
        chunk_count = 0
        
        async for chunk in llm.generate_response_stream(test_query, "", ""):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
            streaming_response += chunk
            chunk_count += 1
        
        streaming_total_time = time.time() - start_time
        
        print(f"  Streaming total: {streaming_total_time:.2f}s")
        print(f"  First chunk: {first_chunk_time:.2f}s")
        print(f"  Chunks: {chunk_count}")
        print(f"  Response: {streaming_response[:50]}...")
        
        # Analysis
        print(f"\n📊 Streaming vs Regular Results:")
        
        if first_chunk_time and first_chunk_time < regular_time:
            improvement = ((regular_time - first_chunk_time) / regular_time) * 100
            print(f"✅ Streaming first chunk {improvement:.1f}% faster")
            print(f"✅ User experience improvement: {first_chunk_time:.2f}s vs {regular_time:.2f}s")
            
            await llm.close()
            return True
        else:
            print("⚠️ Streaming doesn't show significant improvement")
            await llm.close()
            return False
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    print("⚡ Pascal Speed Optimization Test Suite")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Direct Ollama", test_direct_ollama_speed),
        ("Pascal Optimized", test_pascal_optimized_speed),
        ("Router Efficiency", test_router_overhead),
        ("Streaming", test_streaming_vs_regular)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nTEST {len(results)+1}: {test_name}")
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n" + "=" * 50)
    print("📊 SPEED TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\nOverall: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("🎉 EXCELLENT PERFORMANCE!")
        print("Pascal speed optimizations are working well.")
    elif success_rate >= 50:
        print("✅ GOOD PERFORMANCE!")
        print("Most optimizations working, minor issues detected.")
    elif success_rate >= 25:
        print("⚠️ SPEED ISSUES DETECTED")
        print("Multiple optimizations failed - check configuration.")
    else:
        print("❌ CRITICAL SPEED ISSUES")
        print("Major optimization problems detected.")
    
    # Recommendations
    print(f"\n💡 Next steps:")
    if not results.get("Direct Ollama", False):
        print("• Check Ollama service: sudo systemctl status ollama")
        print("• Verify model availability: ollama list")
        print("• Run optimization script: ./ollama_optimization.sh")
    
    if not results.get("Pascal Optimized", False):
        print("• Check Pascal configuration: python test_complete_system.py")
        print("• Verify imports: python -c 'from modules.offline_llm import LightningOfflineLLM'")
    
    if not results.get("Router Efficiency", False):
        print("• Check router configuration and imports")
    
    if not results.get("Streaming", False):
        print("• Streaming may need optimization for your setup")
    
    return success_rate >= 50

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Speed test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
