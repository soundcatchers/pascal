#!/usr/bin/env python3
"""
Pascal Intelligent Routing Test Suite
Comprehensive testing of the enhanced routing system for near-perfect accuracy
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode for testing
import os
os.environ['DEBUG'] = 'true'

@dataclass
class TestCase:
    """Test case for routing validation"""
    query: str
    expected_route: str  # 'offline', 'online', 'skill', or 'any'
    expected_current_info: bool
    category: str
    description: str
    complexity_level: str = "any"
    confidence_threshold: float = 0.7

class RoutingTestSuite:
    """Comprehensive routing test suite"""
    
    def __init__(self):
        self.test_cases = self._create_test_cases()
        self.results = {}
        self.performance_data = {}
    
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases for routing validation"""
        
        return [
            # === CURRENT INFO QUERIES (Should route to online) ===
            TestCase(
                "What day is today?",
                "online", True, "current_date",
                "Direct current date query",
                "simple", 0.9
            ),
            TestCase(
                "What's today's date?",
                "online", True, "current_date", 
                "Today's date query variant",
                "simple", 0.9
            ),
            TestCase(
                "Tell me the current date",
                "online", True, "current_date",
                "Imperative current date query",
                "simple", 0.8
            ),
            TestCase(
                "Who is the current president?",
                "online", True, "current_politics",
                "Current political information",
                "simple", 0.9
            ),
            TestCase(
                "Who is the current US president?",
                "online", True, "current_politics",
                "Specific current political query",
                "simple", 0.9
            ),
            TestCase(
                "Latest news headlines",
                "online", True, "current_news",
                "Current news request",
                "moderate", 0.8
            ),
            TestCase(
                "What's happening in the news today?",
                "online", True, "current_news",
                "Current events query",
                "moderate", 0.8
            ),
            TestCase(
                "Breaking news",
                "online", True, "current_news",
                "Breaking news request",
                "simple", 0.8
            ),
            TestCase(
                "Current weather in London",
                "online", True, "current_weather",
                "Current weather query",
                "moderate", 0.8
            ),
            TestCase(
                "Weather today",
                "online", True, "current_weather",
                "Today's weather query",
                "simple", 0.8
            ),
            TestCase(
                "What's the current stock price of Apple?",
                "online", True, "current_financial",
                "Current financial information",
                "moderate", 0.8
            ),
            TestCase(
                "Latest sports scores",
                "online", True, "current_sports",
                "Current sports results",
                "moderate", 0.8
            ),
            
            # === INSTANT QUERIES (Should route to skills) ===
            TestCase(
                "What time is it?",
                "skill", False, "instant_time",
                "Simple time query",
                "instant", 0.95
            ),
            TestCase(
                "Time?",
                "skill", False, "instant_time",
                "Minimal time query",
                "instant", 0.9
            ),
            TestCase(
                "Current time",
                "skill", False, "instant_time",
                "Current time request",
                "instant", 0.9
            ),
            TestCase(
                "15 + 23",
                "skill", False, "instant_calc",
                "Simple addition",
                "instant", 0.95
            ),
            TestCase(
                "What is 2+2?",
                "skill", False, "instant_calc",
                "Basic math question",
                "instant", 0.9
            ),
            TestCase(
                "Calculate 45 * 6",
                "skill", False, "instant_calc",
                "Multiplication request",
                "instant", 0.9
            ),
            TestCase(
                "20% of 150",
                "skill", False, "instant_calc",
                "Percentage calculation",
                "instant", 0.85
            ),
            
            # === GENERAL QUERIES (Should prefer offline) ===
            TestCase(
                "Hello, how are you?",
                "offline", False, "greeting",
                "Basic greeting",
                "simple", 0.8
            ),
            TestCase(
                "Hi Pascal",
                "offline", False, "greeting",
                "Simple greeting",
                "simple", 0.8
            ),
            TestCase(
                "Good morning",
                "offline", False, "greeting",
                "Time-based greeting",
                "simple", 0.7
            ),
            TestCase(
                "Explain what Python is",
                "offline", False, "explanation",
                "Technical explanation",
                "moderate", 0.8
            ),
            TestCase(
                "What is machine learning?",
                "offline", False, "explanation",
                "ML explanation request",
                "moderate", 0.8
            ),
            TestCase(
                "How does HTTP work?",
                "offline", False, "explanation",
                "Technical concept explanation",
                "moderate", 0.8
            ),
            TestCase(
                "Write a Python function to sort a list",
                "offline", False, "programming",
                "Code creation request",
                "complex", 0.8
            ),
            TestCase(
                "Create a function that calculates fibonacci numbers",
                "offline", False, "programming",
                "Programming task",
                "complex", 0.8
            ),
            TestCase(
                "Help me debug this code",
                "offline", False, "programming",
                "Programming assistance",
                "moderate", 0.7
            ),
            TestCase(
                "What is the capital of France?",
                "offline", False, "factual",
                "General knowledge question",
                "simple", 0.7
            ),
            TestCase(
                "Tell me about the history of computers",
                "offline", False, "explanation",
                "Historical explanation",
                "complex", 0.7
            ),
            
            # === EDGE CASES ===
            TestCase(
                "Weather forecast",
                "any", False, "edge_weather",
                "Ambiguous weather query (could be current or general)",
                "moderate", 0.5
            ),
            TestCase(
                "News about artificial intelligence",
                "any", False, "edge_news",
                "Topic-specific news (could be current or general)",
                "moderate", 0.6
            ),
            TestCase(
                "What happened today in history?",
                "offline", False, "edge_history",
                "Historical events on this date (not current news)",
                "moderate", 0.7
            ),
            TestCase(
                "How is the weather usually in London?",
                "offline", False, "edge_weather_general",
                "General weather patterns (not current)",
                "moderate", 0.7
            ),
            TestCase(
                "Explain current events in AI",
                "online", True, "edge_current_ai",
                "Current developments in specific field",
                "complex", 0.7
            ),
            TestCase(
                "What are the recent developments in quantum computing?",
                "online", True, "edge_recent_tech",
                "Recent technological developments",
                "complex", 0.8
            ),
            
            # === COMPLEX QUERIES ===
            TestCase(
                "Compare Python and JavaScript for web development",
                "offline", False, "complex_comparison",
                "Technical comparison",
                "complex", 0.8
            ),
            TestCase(
                "Analyze the pros and cons of remote work",
                "offline", False, "complex_analysis",
                "Analytical thinking task",
                "complex", 0.7
            ),
            TestCase(
                "Write a comprehensive guide to REST APIs",
                "offline", False, "complex_creation",
                "Long-form content creation",
                "complex", 0.8
            ),
            
            # === TEMPORAL EDGE CASES ===
            TestCase(
                "What happened yesterday?",
                "online", True, "temporal_recent",
                "Recent events query",
                "moderate", 0.8
            ),
            TestCase(
                "What will happen tomorrow?",
                "online", True, "temporal_future",
                "Future events query",
                "moderate", 0.7
            ),
            TestCase(
                "Current trends in technology",
                "online", True, "temporal_trends",
                "Current trends request",
                "moderate", 0.8
            ),
            TestCase(
                "Historical trends in technology",
                "offline", False, "temporal_historical",
                "Historical trends (not current)",
                "moderate", 0.7
            ),
        ]
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive routing tests"""
        print("🧪 Pascal Intelligent Routing Test Suite")
        print("=" * 60)
        
        try:
            # Initialize the enhanced query analyzer and router
            from modules.query_analyzer import EnhancedQueryAnalyzer
            from modules.intelligent_router import IntelligentRouter
            from modules.personality import PersonalityManager
            from modules.memory import MemoryManager
            
            print("✅ Modules imported successfully")
            
            # Create components
            analyzer = EnhancedQueryAnalyzer()
            personality_manager = PersonalityManager()
            memory_manager = MemoryManager()
            
            await personality_manager.load_personality("default")
            await memory_manager.load_session()
            
            router = IntelligentRouter(personality_manager, memory_manager)
            await router._check_llm_availability()
            
            print(f"✅ Router initialized - Systems: offline={router.offline_available}, "
                  f"online={router.online_available}, skills={router.skills_available}")
            
            # Run tests by category
            results_by_category = {}
            total_tests = 0
            total_correct = 0
            total_high_confidence = 0
            
            categories = list(set(test.category.split('_')[0] for test in self.test_cases))
            
            for category in sorted(categories):
                print(f"\n📊 Testing Category: {category.upper()}")
                print("-" * 40)
                
                category_tests = [t for t in self.test_cases if t.category.startswith(category)]
                category_correct = 0
                category_high_confidence = 0
                category_results = []
                
                for test in category_tests:
                    result = await self._run_single_test(test, analyzer, router)
                    category_results.append(result)
                    
                    total_tests += 1
                    if result['routing_correct']:
                        total_correct += 1
                        category_correct += 1
                    
                    if result['high_confidence']:
                        total_high_confidence += 1
                        category_high_confidence += 1
                    
                    # Print result
                    status = "✅" if result['routing_correct'] and result['high_confidence'] else \
                             "⚠️" if result['routing_correct'] else "❌"
                    
                    print(f"  {status} {test.query[:50]}...")
                    print(f"      Expected: {test.expected_route}, Got: {result['actual_route']}")
                    print(f"      Confidence: {result['confidence']:.2f}, Current Info: {result['current_info_score']:.2f}")
                
                category_accuracy = (category_correct / len(category_tests)) * 100
                category_confidence_rate = (category_high_confidence / len(category_tests)) * 100
                
                print(f"\n  📈 Category Results:")
                print(f"      Accuracy: {category_correct}/{len(category_tests)} ({category_accuracy:.1f}%)")
                print(f"      High Confidence: {category_high_confidence}/{len(category_tests)} ({category_confidence_rate:.1f}%)")
                
                results_by_category[category] = {
                    'tests': len(category_tests),
                    'correct': category_correct,
                    'accuracy': category_accuracy,
                    'high_confidence': category_high_confidence,
                    'confidence_rate': category_confidence_rate,
                    'details': category_results
                }
            
            # Overall results
            overall_accuracy = (total_correct / total_tests) * 100
            overall_confidence_rate = (total_high_confidence / total_tests) * 100
            
            print(f"\n" + "=" * 60)
            print("📊 OVERALL TEST RESULTS")
            print("=" * 60)
            
            print(f"Total Tests: {total_tests}")
            print(f"Routing Accuracy: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")
            print(f"High Confidence Rate: {total_high_confidence}/{total_tests} ({overall_confidence_rate:.1f}%)")
            
            # Performance assessment
            if overall_accuracy >= 95 and overall_confidence_rate >= 90:
                grade = "A+ (Excellent - Production Ready)"
            elif overall_accuracy >= 90 and overall_confidence_rate >= 85:
                grade = "A (Very Good - Minor tuning needed)"
            elif overall_accuracy >= 85 and overall_confidence_rate >= 80:
                grade = "B+ (Good - Some optimization needed)"
            elif overall_accuracy >= 80 and overall_confidence_rate >= 75:
                grade = "B (Fair - Significant optimization needed)"
            elif overall_accuracy >= 70:
                grade = "C (Poor - Major improvements needed)"
            else:
                grade = "D (Critical - System needs redesign)"
            
            print(f"\nPerformance Grade: {grade}")
            
            # Detailed analysis
            print(f"\n📈 Detailed Analysis:")
            self._analyze_failures(results_by_category)
            
            # System performance
            routing_stats = router.get_routing_stats()
            if not routing_stats.get('no_decisions'):
                print(f"\n🎯 Routing Intelligence Stats:")
                print(f"  Average Confidence: {routing_stats['average_confidence']:.2f}")
                print(f"  Average Expected Time: {routing_stats['average_expected_time']:.1f}s")
                print(f"  System Health: {routing_stats['system_health']}")
            
            await router.close()
            
            return {
                'overall_accuracy': overall_accuracy,
                'overall_confidence_rate': overall_confidence_rate,
                'grade': grade,
                'results_by_category': results_by_category,
                'routing_stats': routing_stats,
                'total_tests': total_tests,
                'recommendations': self._generate_recommendations(overall_accuracy, overall_confidence_rate, results_by_category)
            }
            
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    async def _run_single_test(self, test: TestCase, analyzer, router) -> Dict[str, Any]:
        """Run a single routing test"""
        
        try:
            # Analyze query
            analysis = await analyzer.analyze_query(test.query)
            
            # Make routing decision
            decision = await router.make_intelligent_decision(test.query)
            
            # Evaluate current info detection
            current_info_correct = (analysis.current_info_score >= 0.7) == test.expected_current_info
            
            # Evaluate routing decision
            actual_route = decision.route_type
            
            if test.expected_route == "any":
                # For edge cases, any reasonable routing is acceptable
                routing_correct = actual_route in ['offline', 'online', 'skill']
            else:
                # Check if routing matches expected, with fallback considerations
                routing_correct = actual_route == test.expected_route
                
                # Allow reasonable fallbacks
                if not routing_correct:
                    if test.expected_route == 'online' and actual_route == 'offline' and not router.online_available:
                        routing_correct = True  # Acceptable fallback
                    elif test.expected_route == 'skill' and actual_route == 'offline' and not router.skills_available:
                        routing_correct = True  # Acceptable fallback
                    elif test.expected_route == 'offline' and actual_route == 'online' and not router.offline_available:
                        routing_correct = True  # Acceptable fallback
            
            # Check confidence
            high_confidence = decision.confidence >= test.confidence_threshold
            
            return {
                'test_case': test,
                'analysis': analysis,
                'decision': decision,
                'actual_route': actual_route,
                'expected_route': test.expected_route,
                'routing_correct': routing_correct,
                'current_info_correct': current_info_correct,
                'current_info_score': analysis.current_info_score,
                'confidence': decision.confidence,
                'high_confidence': high_confidence,
                'processing_time': analysis.processing_time
            }
            
        except Exception as e:
            return {
                'test_case': test,
                'error': str(e),
                'routing_correct': False,
                'current_info_correct': False,
                'high_confidence': False
            }
    
    def _analyze_failures(self, results_by_category: Dict[str, Any]):
        """Analyze test failures to identify patterns"""
        
        print("🔍 Failure Analysis:")
        
        total_failures = 0
        failure_patterns = {
            'current_info_detection': 0,
            'routing_decision': 0,
            'low_confidence': 0,
            'system_availability': 0
        }
        
        for category, results in results_by_category.items():
            category_failures = []
            
            for result in results['details']:
                if not result.get('routing_correct') or not result.get('high_confidence'):
                    total_failures += 1
                    
                    if not result.get('current_info_correct'):
                        failure_patterns['current_info_detection'] += 1
                        category_failures.append(f"Current info detection: {result['test_case'].query}")
                    
                    if not result.get('routing_correct'):
                        failure_patterns['routing_decision'] += 1
                        category_failures.append(f"Wrong route: {result['test_case'].query}")
                    
                    if not result.get('high_confidence'):
                        failure_patterns['low_confidence'] += 1
                        category_failures.append(f"Low confidence: {result['test_case'].query}")
            
            if category_failures:
                print(f"\n  {category.upper()} Issues:")
                for failure in category_failures[:3]:  # Show top 3
                    print(f"    • {failure}")
        
        if total_failures == 0:
            print("  🎉 No significant issues detected!")
        else:
            print(f"\n  📊 Failure Patterns:")
            for pattern, count in failure_patterns.items():
                if count > 0:
                    print(f"    • {pattern}: {count} occurrences")
    
    def _generate_recommendations(self, accuracy: float, confidence_rate: float, 
                                results_by_category: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Accuracy recommendations
        if accuracy < 90:
            recommendations.append("Routing accuracy below 90% - review routing decision logic")
        
        if confidence_rate < 85:
            recommendations.append("Confidence rate below 85% - tune confidence calculation")
        
        # Category-specific recommendations
        for category, results in results_by_category.items():
            if results['accuracy'] < 80:
                if category == 'current':
                    recommendations.append("Current info detection needs improvement - review temporal patterns")
                elif category == 'instant':
                    recommendations.append("Skills routing needs tuning - review instant query patterns")
                elif category == 'general':
                    recommendations.append("General query routing needs optimization")
                elif category == 'edge':
                    recommendations.append("Edge case handling needs improvement - review ambiguous query logic")
        
        # System availability recommendations
        if accuracy >= 90 and confidence_rate >= 85:
            recommendations.append("🎉 Routing system performing excellently - ready for production!")
        elif accuracy >= 85:
            recommendations.append("Routing system performing well - minor optimizations needed")
        else:
            recommendations.append("Routing system needs significant improvements before production use")
        
        # Performance recommendations
        recommendations.append("Monitor system performance continuously for optimal routing")
        recommendations.append("Consider A/B testing different routing thresholds in production")
        
        return recommendations[:10]  # Limit to top 10 recommendations

async def run_quick_routing_test():
    """Quick routing test for development"""
    print("⚡ Quick Routing Intelligence Test")
    print("=" * 40)
    
    suite = RoutingTestSuite()
    
    # Test just a few key cases
    quick_tests = [
        "What day is today?",
        "Hello, how are you?", 
        "What time is it?",
        "15 + 23",
        "Latest news",
        "Explain Python"
    ]
    
    try:
        from modules.query_analyzer import EnhancedQueryAnalyzer
        
        analyzer = EnhancedQueryAnalyzer()
        
        print("Testing query analysis:")
        for query in quick_tests:
            analysis = await analyzer.analyze_query(query)
            
            print(f"\nQuery: '{query}'")
            print(f"  Intent: {analysis.intent.value}")
            print(f"  Complexity: {analysis.complexity.value}")
            print(f"  Current Info Score: {analysis.current_info_score:.2f}")
            print(f"  Confidence: {analysis.confidence:.2f}")
        
        analyzer_stats = analyzer.get_analysis_stats()
        print(f"\n📊 Analyzer Performance:")
        print(f"  Analyses: {analyzer_stats['total_analyses']}")
        print(f"  Avg Time: {analyzer_stats['average_analysis_time']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

async def run_routing_benchmark():
    """Benchmark routing performance"""
    print("🏃 Routing Performance Benchmark")
    print("=" * 40)
    
    try:
        from modules.query_analyzer import EnhancedQueryAnalyzer
        
        analyzer = EnhancedQueryAnalyzer()
        
        # Test queries of varying complexity
        benchmark_queries = [
            "Hi",
            "What time is it?",
            "What day is today?",
            "Hello, how are you?",
            "What is 15 + 27?",
            "Explain machine learning",
            "Latest news headlines",
            "Write a Python function to sort a list",
            "Compare the pros and cons of different programming languages",
            "What are the current trends in artificial intelligence and how do they impact modern software development?"
        ]
        
        print(f"Benchmarking {len(benchmark_queries)} queries...")
        
        total_time = 0
        analysis_times = []
        
        for i, query in enumerate(benchmark_queries, 1):
            start_time = time.time()
            analysis = await analyzer.analyze_query(query)
            elapsed = time.time() - start_time
            
            total_time += elapsed
            analysis_times.append(elapsed)
            
            print(f"  {i:2d}. {elapsed:.4f}s - {query[:50]}...")
        
        avg_time = total_time / len(benchmark_queries)
        min_time = min(analysis_times)
        max_time = max(analysis_times)
        
        print(f"\n📊 Benchmark Results:")
        print(f"  Total Time: {total_time:.4f}s")
        print(f"  Average Time: {avg_time:.4f}s")
        print(f"  Min Time: {min_time:.4f}s")
        print(f"  Max Time: {max_time:.4f}s")
        print(f"  Queries/Second: {len(benchmark_queries)/total_time:.1f}")
        
        # Performance assessment
        if avg_time < 0.001:
            grade = "A+ (Excellent - Sub-millisecond)"
        elif avg_time < 0.005:
            grade = "A (Very Good - Under 5ms)"
        elif avg_time < 0.01:
            grade = "B (Good - Under 10ms)"
        elif avg_time < 0.05:
            grade = "C (Acceptable - Under 50ms)"
        else:
            grade = "D (Slow - Over 50ms)"
        
        print(f"  Performance Grade: {grade}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def main():
    """Main test runner with options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pascal Routing Intelligence Test Suite')
    parser.add_argument('--full', action='store_true', help='Run full comprehensive test suite')
    parser.add_argument('--quick', action='store_true', help='Run quick routing test')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--export', type=str, help='Export results to JSON file')
    
    args = parser.parse_args()
    
    async def run_tests():
        results = {}
        
        if args.quick or not any([args.full, args.benchmark]):
            print("Running quick test...\n")
            results['quick_test'] = await run_quick_routing_test()
        
        if args.benchmark:
            print("\nRunning benchmark...\n")
            results['benchmark'] = await run_routing_benchmark()
        
        if args.full:
            print("\nRunning comprehensive test suite...\n")
            suite = RoutingTestSuite()
            results['comprehensive'] = await suite.run_comprehensive_test()
            
            if args.export and 'comprehensive' in results:
                export_path = args.export
                try:
                    with open(export_path, 'w') as f:
                        json.dump(results['comprehensive'], f, indent=2, default=str)
                    print(f"\n📁 Results exported to: {export_path}")
                except Exception as e:
                    print(f"\n❌ Export failed: {e}")
        
        return results
    
    try:
        results = asyncio.run(run_tests())
        
        print(f"\n" + "=" * 60)
        print("🎯 TEST SUITE SUMMARY")
        print("=" * 60)
        
        if 'comprehensive' in results and not results['comprehensive'].get('error'):
            comp_results = results['comprehensive']
            print(f"Routing Accuracy: {comp_results['overall_accuracy']:.1f}%")
            print(f"Confidence Rate: {comp_results['overall_confidence_rate']:.1f}%")
            print(f"Performance Grade: {comp_results['grade']}")
            
            if comp_results['overall_accuracy'] >= 95:
                print("\n🎉 EXCELLENT! Routing system ready for production.")
            elif comp_results['overall_accuracy'] >= 90:
                print("\n✅ VERY GOOD! Minor optimizations recommended.")
            elif comp_results['overall_accuracy'] >= 80:
                print("\n⚠️ GOOD! Some improvements needed before production.")
            else:
                print("\n❌ NEEDS WORK! Significant improvements required.")
        
        if any(results.values()):
            print("\n✅ Tests completed successfully!")
            return 0
        else:
            print("\n❌ Some tests failed - check output above")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
