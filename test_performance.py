#!/usr/bin/env python3
"""
Pascal AI Assistant - Performance Testing Script
Tests offline LLM performance across different profiles on Pi 5
"""

import asyncio
import time
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.offline_llm import OptimizedOfflineLLM
from config.settings import settings

class PerformanceTester:
    """Performance testing for Pascal's offline LLM"""
    
    def __init__(self):
        self.llm = None
        self.test_queries = {
            'simple': [
                "Hello, how are you?",
                "What is 2+2?",
                "Thanks for your help!",
                "What's the weather like?",
                "Can you help me?"
            ],
            'medium': [
                "Explain the concept of machine learning in simple terms.",
                "What are the benefits and drawbacks of renewable energy?",
                "How do I create a simple Python function to calculate fibonacci numbers?",
                "What's the difference between a list and a tuple in Python?",
                "Explain the water cycle to a 10-year-old."
            ],
            'complex': [
                "Compare and contrast different sorting algorithms, including their time complexity and best use cases.",
                "Analyze the potential impacts of artificial intelligence on the job market over the next decade.",
                "Write a detailed explanation of how neural networks work, including backpropagation.",
                "Discuss the ethical implications of gene editing technology and its potential future applications.",
                "Create a comprehensive plan for implementing a microservices architecture in a web application."
            ]
        }
    
    async def initialize(self):
        """Initialize the offline LLM"""
        print("🔧 Initializing Pascal's offline LLM...")
        self.llm = OptimizedOfflineLLM()
        success = await self.llm.initialize()
        
        if success:
            print("✅ LLM initialized successfully")
            return True
        else:
            print("❌ LLM initialization failed")
            return False
    
    async def test_profile_performance(self, profile: str, queries: list):
        """Test performance for a specific profile"""
        print(f"\n🧪 Testing {profile.upper()} profile:")
        print("=" * 50)
        
        self.llm.set_performance_profile(profile)
        
        total_time = 0
        successful_responses = 0
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query[:50]}{'...' if len(query) > 50 else ''}")
            
            try:
                start_time = time.time()
                response = await self.llm.generate_response(query, "", "")
                end_time = time.time()
                
                response_time = end_time - start_time
                total_time += response_time
                successful_responses += 1
                
                print(f"⏱️  Response time: {response_time:.2f}s")
                print(f"📝 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
                print(f"✅ Success")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        if successful_responses > 0:
            avg_time = total_time / successful_responses
            print(f"\n📊 {profile.upper()} Profile Summary:")
            print(f"   • Successful responses: {successful_responses}/{len(queries)}")
            print(f"   • Average response time: {avg_time:.2f}s")
            print(f"   • Total time: {total_time:.2f}s")
            
            # Performance rating
            if avg_time <= 2:
                rating = "🚀 Excellent"
            elif avg_time <= 4:
                rating = "⚡ Good"
            elif avg_time <= 6:
                rating = "✅ Acceptable"
            else:
                rating = "⚠️ Slow"
            
            print(f"   • Performance rating: {rating}")
            
            return avg_time, successful_responses
        else:
            print(f"❌ All queries failed for {profile} profile")
            return None, 0
    
    async def test_model_switching(self):
        """Test switching between available models"""
        print("\n🔄 Testing Model Switching:")
        print("=" * 50)
        
        available_models = self.llm.list_available_models()
        
        if len(available_models) <= 1:
            print("Only one model available, skipping switching test")
            return
        
        test_query = "Hello, can you tell me your name?"
        
        for model_info in available_models[:3]:  # Test up to 3 models
            model_name = model_info['name']
            print(f"\n🤖 Testing model: {model_name}")
            
            try:
                start_time = time.time()
                success = await self.llm.switch_model(model_name)
                switch_time = time.time() - start_time
                
                if success:
                    print(f"✅ Model switch successful ({switch_time:.2f}s)")
                    
                    # Test response
                    response_start = time.time()
                    response = await self.llm.generate_response(test_query, "", "")
                    response_time = time.time() - response_start
                    
                    print(f"⏱️  Response time: {response_time:.2f}s")
                    print(f"📝 Response: {response[:80]}...")
                else:
                    print(f"❌ Model switch failed")
                    
            except Exception as e:
                print(f"❌ Error testing model {model_name}: {e}")
    
    async def stress_test(self, duration_seconds: int = 60):
        """Perform stress test for specified duration"""
        print(f"\n💪 Stress Test ({duration_seconds}s):")
        print("=" * 50)
        
        self.llm.set_performance_profile('speed')  # Use speed profile for stress test
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        request_count = 0
        successful_responses = 0
        total_response_time = 0
        
        test_queries = [
            "Count from 1 to 5",
            "What is AI?",
            "Hello!",
            "Explain Python",
            "Name three colors"
        ]
        
        while time.time() < end_time:
            query = test_queries[request_count % len(test_queries)]
            request_count += 1
            
            try:
                query_start = time.time()
                response = await self.llm.generate_response(query, "", "")
                query_end = time.time()
                
                response_time = query_end - query_start
                total_response_time += response_time
                successful_responses += 1
                
                print(f"Request {request_count}: {response_time:.2f}s ✅")
                
            except Exception as e:
                print(f"Request {request_count}: Error - {e} ❌")
            
            # Brief pause to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        actual_duration = time.time() - start_time
        
        print(f"\n📊 Stress Test Results:")
        print(f"   • Duration: {actual_duration:.2f}s")
        print(f"   • Total requests: {request_count}")
        print(f"   • Successful responses: {successful_responses}")
        print(f"   • Success rate: {(successful_responses/request_count)*100:.1f}%")
        
        if successful_responses > 0:
            avg_response_time = total_response_time / successful_responses
            requests_per_second = successful_responses / actual_duration
            
            print(f"   • Average response time: {avg_response_time:.2f}s")
            print(f"   • Requests per second: {requests_per_second:.2f}")
    
    async def system_info_test(self):
        """Display system information and model details"""
        print("\n🖥️  System Information:")
        print("=" * 50)
        
        # Hardware info
        hw_info = settings.get_hardware_info()
        print(f"Hardware: {hw_info}")
        
        # Model info
        if self.llm:
            model_stats = self.llm.get_performance_stats()
            print(f"\nModel Statistics:")
            for key, value in model_stats.items():
                print(f"   • {key}: {value}")
            
            # Available models
            available_models = self.llm.list_available_models()
            print(f"\nAvailable Models ({len(available_models)}):")
            for model in available_models:
                status = "🟢 LOADED" if model['loaded'] else "⚪ Available"
                print(f"   • {model['name']} - {model['ram_usage']} RAM - Speed: {model['speed_rating']} - Quality: {model['quality_rating']} {status}")
    
    async def run_comprehensive_test(self):
        """Run comprehensive performance test"""
        print("🤖 Pascal AI Assistant - Performance Test Suite")
        print("=" * 60)
        
        if not await self.initialize():
            return False
        
        # System information
        await self.system_info_test()
        
        # Test all profiles
        profile_results = {}
        for profile in ['speed', 'balanced', 'quality']:
            complexity = {
                'speed': 'simple',
                'balanced': 'medium', 
                'quality': 'complex'
            }[profile]
            
            queries = self.test_queries[complexity][:3]  # Test 3 queries per profile
            avg_time, success_count = await self.test_profile_performance(profile, queries)
            profile_results[profile] = {'avg_time': avg_time, 'success_count': success_count}
        
        # Model switching test
        await self.test_model_switching()
        
        # Stress test (30 seconds)
        await self.stress_test(30)
        
        # Final summary
        print("\n🎯 Final Performance Summary:")
        print("=" * 50)
        
        for profile, results in profile_results.items():
            if results['avg_time']:
                print(f"{profile.upper()} Profile: {results['avg_time']:.2f}s avg, {results['success_count']} successes")
        
        print("\n✅ Performance testing complete!")
        
        # Close LLM
        if self.llm:
            await self.llm.close()
        
        return True

async def main():
    """Main test function"""
    tester = PerformanceTester()
    
    # Check if we should run specific test
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if not await tester.initialize():
            return
        
        if test_type == 'quick':
            # Quick test - one query per profile
            for profile in ['speed', 'balanced', 'quality']:
                await tester.test_profile_performance(profile, [tester.test_queries['simple'][0]])
        
        elif test_type == 'stress':
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            await tester.stress_test(duration)
        
        elif test_type == 'models':
            await tester.test_model_switching()
        
        elif test_type == 'info':
            await tester.system_info_test()
        
        else:
            print(f"Unknown test type: {test_type}")
            print("Available tests: quick, stress [duration], models, info")
        
        if tester.llm:
            await tester.llm.close()
    
    else:
        # Run comprehensive test
        await tester.run_comprehensive_test()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
