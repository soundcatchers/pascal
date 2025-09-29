#!/usr/bin/env python3
"""
Pascal Intelligent Routing Test Suite - FIXED
Tests the routing logic with comprehensive test cases
"""

import asyncio
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.intelligent_router import IntelligentRouter
from core.config_manager import ConfigManager
from core.personality_manager import PersonalityManager
from core.memory_manager import MemoryManager


@dataclass
class TestCase:
    """Test case for routing"""
    query: str
    expected_route: str  # 'offline', 'online', 'skill'
    category: str
    description: str = ""


class RoutingTester:
    """Tests intelligent routing system"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.personality_manager = PersonalityManager(self.config_manager)
        self.memory_manager = MemoryManager(self.config_manager)
        self.router = None
        self.results = []
        
    async def initialize(self):
        """Initialize the router"""
        print("\n🚀 Initializing FIXED Enhanced Skills Manager...")
        self.router = IntelligentRouter(
            self.config_manager,
            self.personality_manager,
            self.memory_manager
        )
        await self.router.initialize()
        print(f"✅ Router initialized - Systems: offline={self.router.offline_available}, online={self.router.online_available}, skills={self.router.skills_available}")
        
    def get_test_cases(self) -> Dict[str, List[TestCase]]:
        """Get comprehensive test cases organized by category"""
        return {
            "OFFLINE_SIMPLE": [
                TestCase("What is 2+2?", "offline", "OFFLINE_SIMPLE", "Basic math"),
                TestCase("Explain what a variable is in programming", "offline", "OFFLINE_SIMPLE", "Programming concept"),
                TestCase("What are the primary colors?", "offline", "OFFLINE_SIMPLE", "Basic knowledge"),
            ],
            "OFFLINE_REASONING": [
                TestCase("Compare bubble sort and quicksort algorithms", "offline", "OFFLINE_REASONING", "Technical comparison"),
                TestCase("Explain the benefits of object-oriented programming", "offline", "OFFLINE_REASONING", "Conceptual explanation"),
                TestCase("How does recursion work?", "offline", "OFFLINE_REASONING", "Technical deep dive"),
            ],
            "COMPLEX": [
                TestCase("Compare Python and JavaScript for web development", "offline", "COMPLEX", "Multi-faceted comparison"),
                TestCase("Explain database normalization with examples", "offline", "COMPLEX", "Technical with examples"),
                TestCase("What are the tradeoffs between SQL and NoSQL?", "offline", "COMPLEX", "Architectural decision"),
            ],
            "CURRENT_EVENTS": [
                TestCase("What's happening in the news today?", "online", "CURRENT_EVENTS", "Today's news"),
                TestCase("Latest technology announcements this week", "online", "CURRENT_EVENTS", "Recent tech news"),
                TestCase("Current weather in London", "skill", "CURRENT_EVENTS", "Weather query"),
            ],
            "SKILL_BASED": [
                TestCase("What's the weather like?", "skill", "SKILL_BASED", "Weather without location"),
                TestCase("Show me today's news headlines", "skill", "SKILL_BASED", "News headlines"),
                TestCase("Weather forecast for Paris", "skill", "SKILL_BASED", "Weather with location"),
            ],
            "TIME_SENSITIVE": [
                TestCase("What time is it?", "offline", "TIME_SENSITIVE", "Current time - can be local"),
                TestCase("What's the current Bitcoin price?", "online", "TIME_SENSITIVE", "Real-time pricing"),
                TestCase("Who won the latest election?", "online", "TIME_SENSITIVE", "Recent event result"),
            ],
            "VERIFICATION": [
                TestCase("Is Python 3.13 released yet?", "online", "VERIFICATION", "Software release status"),
                TestCase("Verify if Raspberry Pi 5 has 16GB RAM option", "online", "VERIFICATION", "Hardware spec verification"),
                TestCase("Check if TypeScript supports decorators", "offline", "VERIFICATION", "Known feature check"),
            ],
        }
    
    async def test_single_query(self, test: TestCase, verbose: bool = False) -> Dict[str, Any]:
        """Test a single query and return results"""
        start_time = time.time()
        
        try:
            # Route the query
            result = await self.router.route_query(test.query, use_history=[])
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract the route from result
            # The route_query returns a dict with 'response', 'route', 'reasoning' etc.
            actual_route = result.get('route', 'unknown')
            reasoning = result.get('reasoning', 'No reasoning provided')
            response_preview = result.get('response', '')[:100] if result.get('response') else 'No response'
            
            # Determine if test passed
            passed = actual_route == test.expected_route
            
            test_result = {
                'query': test.query,
                'expected_route': test.expected_route,
                'actual_route': actual_route,
                'passed': passed,
                'response_time': response_time,
                'reasoning': reasoning,
                'response_preview': response_preview,
                'category': test.category
            }
            
            # Print result
            status = "✅" if passed else "❌"
            print(f"{status} {test.query}")
            if verbose or not passed:
                print(f"   Expected: {test.expected_route}, Got: {actual_route}")
                print(f"   Reasoning: {reasoning}")
                print(f"   Response time: {response_time:.2f}s")
                if not passed:
                    print(f"   Response preview: {response_preview}")
            
            return test_result
            
        except Exception as e:
            print(f"❌ {test.query}")
            print(f"   Error: {str(e)}")
            return {
                'query': test.query,
                'expected_route': test.expected_route,
                'actual_route': 'error',
                'passed': False,
                'response_time': time.time() - start_time,
                'reasoning': f'Error: {str(e)}',
                'response_preview': '',
                'category': test.category
            }
    
    async def run_category_tests(self, category: str, tests: List[TestCase], verbose: bool = False):
        """Run all tests in a category"""
        print(f"\n📊 Testing Category: {category}")
        print("-" * 40)
        
        category_results = []
        for test in tests:
            result = await self.test_single_query(test, verbose)
            category_results.append(result)
            self.results.append(result)
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.5)
        
        # Category summary
        passed = sum(1 for r in category_results if r['passed'])
        total = len(category_results)
        avg_time = sum(r['response_time'] for r in category_results) / total if total > 0 else 0
        
        print(f"\n📈 {category} Results: {passed}/{total} passed (avg: {avg_time:.2f}s)")
        
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("🎯 TEST SUITE SUMMARY")
        print("=" * 60)
        
        # Overall stats
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        # Route distribution
        print("\n📊 Route Distribution:")
        route_counts = {}
        for result in self.results:
            route = result['actual_route']
            route_counts[route] = route_counts.get(route, 0) + 1
        
        for route, count in sorted(route_counts.items()):
            print(f"   {route}: {count} queries")
        
        # Performance stats
        avg_response_time = sum(r['response_time'] for r in self.results) / total_tests
        print(f"\n⚡ Average Response Time: {avg_response_time:.2f}s")
        
        # Failed tests detail
        if failed_tests > 0:
            print(f"\n❌ Failed Tests Detail:")
            for result in self.results:
                if not result['passed']:
                    print(f"\n   Query: {result['query']}")
                    print(f"   Expected: {result['expected_route']}, Got: {result['actual_route']}")
                    print(f"   Reasoning: {result['reasoning']}")
        
        # Final verdict
        print("\n" + "=" * 60)
        if failed_tests == 0:
            print("✅ Tests completed successfully!")
        else:
            print(f"⚠️  {failed_tests} test(s) need attention")
        print("=" * 60)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.router:
            await self.router.cleanup()


async def run_comprehensive_test(verbose: bool = False, specific_category: str = None):
    """Run comprehensive routing tests"""
    print("\n🧪 Pascal Intelligent Routing Test Suite")
    print("=" * 60)
    
    tester = RoutingTester()
    
    try:
        # Initialize
        await tester.initialize()
        
        # Get test cases
        test_categories = tester.get_test_cases()
        
        # Run tests
        if specific_category:
            if specific_category in test_categories:
                await tester.run_category_tests(
                    specific_category,
                    test_categories[specific_category],
                    verbose
                )
            else:
                print(f"❌ Unknown category: {specific_category}")
                print(f"Available: {', '.join(test_categories.keys())}")
                return
        else:
            # Run all categories
            for category, tests in test_categories.items():
                await tester.run_category_tests(category, tests, verbose)
        
        # Print summary
        tester.print_summary()
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await tester.cleanup()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Pascal intelligent routing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--category', '-c', type=str, help='Test specific category only')
    parser.add_argument('--full', '-f', action='store_true', help='Run full test suite')
    
    args = parser.parse_args()
    
    if args.full or args.category or args.verbose:
        print("Running comprehensive test suite...")
        asyncio.run(run_comprehensive_test(
            verbose=args.verbose,
            specific_category=args.category
        ))
    else:
        print("Usage: python test_routing_intelligence.py [--full] [--verbose] [--category CATEGORY]")
        print("\nAvailable categories:")
        print("  - OFFLINE_SIMPLE")
        print("  - OFFLINE_REASONING") 
        print("  - COMPLEX")
        print("  - CURRENT_EVENTS")
        print("  - SKILL_BASED")
        print("  - TIME_SENSITIVE")
        print("  - VERIFICATION")


if __name__ == "__main__":
    main()
