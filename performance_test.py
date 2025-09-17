#!/usr/bin/env python3
"""
Pascal Performance Testing Script
Comprehensive testing of optimized offline LLM performance
"""

import asyncio
import sys
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode
import os
os.environ['DEBUG'] = 'true'

async def test_offline_llm_performance():
    """Test offline LLM performance comprehensively"""
    print("🔧 Pascal Offline LLM Performance Test")
    print("=" * 50)
    
    try:
        # Import the optimized module
        from modules.offline_llm import LightningOfflineLLM
        from config.settings import settings
        
        print("✅ Modules imported successfully")
        
        # Create and initialize LLM
        print("\n🚀 Initializing Optimized Offline LLM...")
        llm = LightningOfflineLLM()
        
        success = await llm.initialize()
        if not success:
            print("❌ LLM initialization failed")
            print(f"Error: {llm.last_error}")
            return False
        
        print("✅ LLM initialized successfully")
        status = llm.get_status()
        print(f"Model: {status['current_model']}")
        print(f"Profile: {status['performance_profile']} ({status['profile_description']})")
        
        # Test each performance profile
        profiles = ['speed', 'balanced', 'quality']
        test_results = {}
        
        for profile in profiles:
            print(f"\n🧪 Testing Profile: {profile.upper()}")
            print("-" * 30)
            
            llm.set_performance_profile(profile)
            profile_results = await test_profile_performance(llm, profile)
            test_results[profile] = profile_results
            
            # Show profile results
            avg_time = statistics.mean(profile_results['response_times'])
            print(f"Average time: {avg_time:.2f}s")
            print(f"Min time: {min(profile_results['response_times']):.2f}s")
            print(f"Max time: {max(profile_results['response_times']):.2f}s")
            print(f"Success rate: {profile_results['success_rate']:.1f}%")
            print(f"Under 4s: {profile_results['under_4s']}/{profile_results['total_tests']}")
            
            # Grade the performance
            grade = calculate_profile_grade(profile_results, profile)
            print(f"Performance Grade: {grade}")
        
        # Test streaming performance
        print(f"\n🌊 Testing Streaming Performance...")
        print("-" * 30)
        
        streaming_results = await test_streaming_performance(llm)
        
        # Overall performance report
        print(f"\n📊 PERFORMANCE REPORT")
        print("=" * 50)
        
        # Show profile comparison
        print("Profile Comparison:")
        for profile, results in test_results.items():
            avg_time = statistics.mean(results['response_times'])
            under_4s_percent = (results['under_4s'] / results['total_tests']) * 100
            print(f"  {profile.ljust(8)}: {avg_time:.2f}s avg, {under_4s_percent:.0f}% under 4s")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        best_profile = recommend_best_profile(test_results)
        print(f"• Best profile for your Pi 5: {best_profile}")
        
        overall_grade = calculate_overall_grade(test_results)
        print(f"• Overall performance grade: {overall_grade}")
        
        if streaming_results['works']:
            print(f"• Streaming: Works ({streaming_results['first_chunk_time']:.2f}s to first chunk)")
        else:
            print(f"• Streaming: Issues detected")
        
        # Optimization suggestions
        print(f"\n🔧 Optimization Suggestions:")
        suggestions = generate_optimization_suggestions(test_results, status)
        for suggestion in suggestions:
            print(f"• {suggestion}")
        
        # Final cleanup
        await llm.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_profile_performance(llm, profile: str) -> Dict[str, Any]:
    """Test performance for a specific profile"""
    test_queries = [
        "Hello, how are you?",
        "What is 15 + 27?", 
        "Explain Python in one sentence.",
        "Write a short greeting.",
        "What's 2+2?",
        "Tell me about AI briefly.",
        "Count from 1 to 3.",
        "Say hello in Spanish."
    ]
    
    response_times = []
    success_count = 0
    total_tests = len(test_queries)
    
    for i, query in enumerate(test_queries, 1):
        print(f"  Test {i}/{total_tests}: '{query[:30]}...'")
        
        start_time = time.time()
        try:
            response = await llm.generate_response(query, "", "")
            elapsed = time.time() - start_time
            
            if response and not response.startswith("Model error"):
                success_count += 1
                response_times.append(elapsed)
                print(f"    ✅ {elapsed:.2f}s - {response[:50]}...")
            else:
                print(f"    ❌ {elapsed:.2f}s - Error: {response[:50]}...")
                response_times.append(elapsed)  # Include failed responses
        except Exception as e:
            elapsed = time.time() - start_time
            response_times.append(elapsed)
            print(f"    ❌ {elapsed:.2f}s - Exception: {str(e)[:50]}...")
    
    under_4s = sum(1 for t in response_times if t < 4)
    under_2s = sum(1 for t in response_times if t < 2)
    
    return {
        'response_times': response_times,
        'success_count': success_count,
        'total_tests': total_tests,
        'success_rate': (success_count / total_tests) * 100,
        'under_4s': under_4s,
        'under_2s': under_2s,
        'profile': profile
    }

async def test_streaming_performance(llm) -> Dict[str, Any]:
    """Test streaming response performance"""
    query = "Count slowly from 1 to 5"
    
    try:
        start_time = time.time()
        first_chunk_time = None
        total_chunks = 0
        full_response = ""
        
        print(f"Testing streaming with: '{query}'")
        print("Response: ", end="", flush=True)
        
        async for chunk in llm.generate_response_stream(query, "", ""):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
            
            print(chunk, end="", flush=True)
            full_response += chunk
            total_chunks += 1
        
        total_time = time.time() - start_time
        print(f"\n")
        
        return {
            'works': True,
            'first_chunk_time': first_chunk_time or 0,
            'total_time': total_time,
            'total_chunks': total_chunks,
            'response_length': len(full_response)
        }
        
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        return {
            'works': False,
            'error': str(e)
        }

def calculate_profile_grade(results: Dict[str, Any], profile: str) -> str:
    """Calculate performance grade for a profile"""
    if not results['response_times']:
        return "F (No data)"
    
    avg_time = statistics.mean(results['response_times'])
    success_rate = results['success_rate']
    under_4s_percent = (results['under_4s'] / results['total_tests']) * 100
    
    # Different standards for different profiles
    if profile == 'speed':
        if avg_time < 1.5 and success_rate >= 90 and under_4s_percent >= 95:
            return "A+ (Excellent)"
        elif avg_time < 2.5 and success_rate >= 80 and under_4s_percent >= 90:
            return "A (Very Good)"
        elif avg_time < 4 and success_rate >= 70:
            return "B (Good)"
        else:
            return "C (Needs optimization)"
    
    elif profile == 'balanced':
        if avg_time < 3 and success_rate >= 90 and under_4s_percent >= 85:
            return "A+ (Excellent)"
        elif avg_time < 4 and success_rate >= 85 and under_4s_percent >= 75:
            return "A (Very Good)"
        elif avg_time < 6 and success_rate >= 75:
            return "B (Good)"
        else:
            return "C (Needs optimization)"
    
    elif profile == 'quality':
        if avg_time < 5 and success_rate >= 95:
            return "A+ (Excellent)"
        elif avg_time < 8 and success_rate >= 90:
            return "A (Very Good)"
        elif avg_time < 10 and success_rate >= 80:
            return "B (Good)"
        else:
            return "C (Needs optimization)"
    
    return "D (Poor)"

def recommend_best_profile(test_results: Dict[str, Dict[str, Any]]) -> str:
    """Recommend the best profile based on test results"""
    profile_scores = {}
    
    for profile, results in test_results.items():
        if not results['response_times']:
            profile_scores[profile] = 0
            continue
        
        avg_time = statistics.mean(results['response_times'])
        success_rate = results['success_rate']
        under_4s_percent = (results['under_4s'] / results['total_tests']) * 100
        
        # Scoring algorithm (higher is better)
        score = 0
        
        # Success rate is most important
        score += success_rate * 0.4
        
        # Time performance
        if avg_time < 2:
            score += 40
        elif avg_time < 4:
            score += 30
        elif avg_time < 6:
            score += 20
        elif avg_time < 8:
            score += 10
        
        # Consistency (under 4s percentage)
        score += under_4s_percent * 0.3
        
        profile_scores[profile] = score
    
    # Find best profile
    best_profile = max(profile_scores, key=profile_scores.get)
    return best_profile

def calculate_overall_grade(test_results: Dict[str, Dict[str, Any]]) -> str:
    """Calculate overall performance grade"""
    all_times = []
    total_success = 0
    total_tests = 0
    
    for results in test_results.values():
        all_times.extend(results['response_times'])
        total_success += results['success_count']
        total_tests += results['total_tests']
    
    if not all_times:
        return "F (No data)"
    
    avg_time = statistics.mean(all_times)
    success_rate = (total_success / total_tests) * 100
    under_4s_percent = (sum(1 for t in all_times if t < 4) / len(all_times)) * 100
    
    if avg_time < 3 and success_rate >= 90 and under_4s_percent >= 80:
        return "A+ (Excellent Pi 5 performance)"
    elif avg_time < 4 and success_rate >= 85 and under_4s_percent >= 70:
        return "A (Very good Pi 5 performance)"
    elif avg_time < 6 and success_rate >= 75 and under_4s_percent >= 60:
        return "B (Good Pi 5 performance)"
    elif avg_time < 8 and success_rate >= 65:
        return "C (Fair Pi 5 performance)"
    else:
        return "D (Poor - needs optimization)"

def generate_optimization_suggestions(test_results: Dict[str, Dict[str, Any]], status: Dict[str, Any]) -> List[str]:
    """Generate optimization suggestions based on test results"""
    suggestions = []
    
    # Analyze all response times
    all_times = []
    for results in test_results.values():
        all_times.extend(results['response_times'])
    
    if all_times:
        avg_time = statistics.mean(all_times)
        max_time = max(all_times)
        
        if avg_time > 6:
            suggestions.append("Average response time is slow - run optimization script")
        
        if max_time > 15:
            suggestions.append("Some responses are very slow - check system resources")
        
        slow_responses = sum(1 for t in all_times if t > 8)
        if slow_responses > len(all_times) * 0.3:
            suggestions.append("Many slow responses - consider using 'speed' profile")
    
    # Check current configuration
    if status.get('current_model') != 'nemotron-fast':
        suggestions.append("Use optimized 'nemotron-fast' model for better performance")
    
    if status.get('current_settings', {}).get('num_ctx', 0) > 512:
        suggestions.append("Reduce context window (num_ctx) to 512 for faster responses")
    
    # Check success rates
    for profile, results in test_results.items():
        if results['success_rate'] < 80:
            suggestions.append(f"{profile} profile has low success rate - check Ollama logs")
    
    # System-level suggestions
    suggestions.append("Ensure Pi 5 has adequate cooling for sustained performance")
    suggestions.append("Consider overclocking Pi 5 CPU if thermals allow")
    suggestions.append("Use fast NVMe storage for best model loading times")
    
    return suggestions[:5]  # Limit to top 5 suggestions

async def quick_performance_test():
    """Quick 30-second performance test"""
    print("⚡ Quick Performance Test (30 seconds)")
    print("=" * 40)
    
    try:
        from modules.offline_llm import LightningOfflineLLM
        
        llm = LightningOfflineLLM()
        if not await llm.initialize():
            print("❌ LLM initialization failed")
            return False
        
        # Test balanced profile with 5 queries
        llm.set_performance_profile('balanced')
        
        queries = [
            "Hello",
            "What's 2+2?",
            "Say hi",
            "Count to 3",
            "Python is?"
        ]
        
        times = []
        for i, query in enumerate(queries, 1):
            print(f"Test {i}/5: {query}")
            start = time.time()
            response = await llm.generate_response(query, "", "")
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  {elapsed:.2f}s - {response[:40]}...")
        
        avg_time = statistics.mean(times)
        print(f"\nQuick Test Results:")
        print(f"Average time: {avg_time:.2f}s")
        print(f"Target: <4s for good Pi 5 performance")
        
        if avg_time < 2:
            print("🎉 Excellent performance!")
        elif avg_time < 4:
            print("✅ Good performance!")
        elif avg_time < 6:
            print("⚠️ Fair performance - consider optimization")
        else:
            print("❌ Poor performance - run full optimization")
        
        await llm.close()
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main function with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pascal Performance Testing')
    parser.add_argument('--quick', action='store_true', help='Run quick 30-second test')
    parser.add_argument('--full', action='store_true', help='Run full comprehensive test')
    args = parser.parse_args()
    
    if args.quick:
        result = asyncio.run(quick_performance_test())
    elif args.full or not (args.quick or args.full):
        result = asyncio.run(test_offline_llm_performance())
    
    if result:
        print("\n✅ Performance test completed successfully!")
        print("\n💡 Next steps:")
        print("• If performance is poor, run: ./ollama_optimization.sh")
        print("• Test Pascal: ./run.sh")
        print("• Monitor performance in Pascal with 'status' command")
    else:
        print("\n❌ Performance test failed - check output above")
    
    return 0 if result else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹️ Performance test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
