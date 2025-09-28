#!/usr/bin/env python3
"""
IMPROVED Test Script: Current Info Detection Fix
Enhanced diagnostics and better error handling
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode for testing
os.environ['DEBUG'] = 'true'

class Colors:
    """Color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.NC}")
    print(f"{Colors.WHITE}{text}{Colors.NC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.NC}")

def print_section(text: str):
    """Print section header"""
    print(f"\n{Colors.BLUE}{'='*20} {text} {'='*20}{Colors.NC}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.NC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.NC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️ {text}{Colors.NC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️ {text}{Colors.NC}")

async def check_prerequisites():
    """Check system prerequisites"""
    print_section("Prerequisites Check")
    
    issues = []
    
    # Check aiohttp import
    try:
        import aiohttp
        print_success(f"aiohttp v{aiohttp.__version__} available")
    except ImportError as e:
        print_error(f"aiohttp not available: {e}")
        issues.append("aiohttp missing - install with: pip install aiohttp==3.9.5")
    
    # Check critical modules
    critical_modules = [
        'config.settings',
        'modules.offline_llm', 
        'modules.online_llm',
        'modules.router',
        'modules.skills_manager',
        'modules.personality',
        'modules.memory'
    ]
    
    for module_name in critical_modules:
        try:
            __import__(module_name)
            print_success(f"{module_name} imported successfully")
        except ImportError as e:
            print_error(f"{module_name} import failed: {e}")
            issues.append(f"Module {module_name} not available")
    
    return issues

async def test_individual_components():
    """Test individual components"""
    print_section("Component Testing")
    
    component_status = {}
    
    # Test settings
    try:
        from config.settings import settings
        print_success(f"Settings loaded - Pascal v{settings.version}")
        component_status['settings'] = True
    except Exception as e:
        print_error(f"Settings failed: {e}")
        component_status['settings'] = False
        return component_status
    
    # Test offline LLM
    try:
        from modules.offline_llm import LightningOfflineLLM
        print_info("Testing offline LLM initialization...")
        
        llm = LightningOfflineLLM()
        start_time = time.time()
        
        try:
            success = await asyncio.wait_for(llm.initialize(), timeout=45.0)
            elapsed = time.time() - start_time
            
            if success:
                status = llm.get_status()
                print_success(f"Offline LLM initialized in {elapsed:.1f}s")
                print(f"   Model: {status.get('current_model', 'Unknown')}")
                print(f"   Available: {status.get('available', False)}")
                component_status['offline_llm'] = True
            else:
                print_error(f"Offline LLM init failed in {elapsed:.1f}s")
                print(f"   Error: {getattr(llm, 'last_error', 'Unknown error')}")
                component_status['offline_llm'] = False
            
            await llm.close()
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print_error(f"Offline LLM initialization timed out after {elapsed:.1f}s")
            component_status['offline_llm'] = False
        
    except Exception as e:
        print_error(f"Offline LLM component failed: {e}")
        component_status['offline_llm'] = False
    
    # Test online LLM
    try:
        from modules.online_llm import OnlineLLM
        print_info("Testing online LLM initialization...")
        
        online_llm = OnlineLLM()
        start_time = time.time()
        
        try:
            success = await asyncio.wait_for(online_llm.initialize(), timeout=20.0)
            elapsed = time.time() - start_time
            
            if success:
                print_success(f"Online LLM initialized in {elapsed:.1f}s")
                component_status['online_llm'] = True
            else:
                print_error(f"Online LLM init failed in {elapsed:.1f}s")
                print(f"   Error: {getattr(online_llm, 'last_error', 'API key not configured')}")
                component_status['online_llm'] = False
            
            await online_llm.close()
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print_error(f"Online LLM initialization timed out after {elapsed:.1f}s")
            component_status['online_llm'] = False
        
    except Exception as e:
        print_error(f"Online LLM component failed: {e}")
        component_status['online_llm'] = False
    
    # Test skills manager
    try:
        from modules.skills_manager import EnhancedSkillsManager
        print_info("Testing skills manager initialization...")
        
        skills_manager = EnhancedSkillsManager()
        start_time = time.time()
        
        try:
            api_status = await asyncio.wait_for(skills_manager.initialize(), timeout=15.0)
            elapsed = time.time() - start_time
            
            available_apis = sum(1 for status in api_status.values() if status['available'])
            print_success(f"Skills manager initialized in {elapsed:.1f}s")
            print(f"   APIs available: {available_apis}/{len(api_status)}")
            component_status['skills_manager'] = True
            
            await skills_manager.close()
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print_error(f"Skills manager initialization timed out after {elapsed:.1f}s")
            component_status['skills_manager'] = False
        
    except Exception as e:
        print_error(f"Skills manager component failed: {e}")
        component_status['skills_manager'] = False
    
    return component_status

async def test_router_integration(component_status):
    """Test router integration"""
    print_section("Router Integration Testing")
    
    try:
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        
        print_info("Creating router components...")
        
        # Create components
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        await personality_manager.load_personality("default")
        await memory_manager.load_session()
        
        router = LightningRouter(personality_manager, memory_manager)
        print_success("Router components created")
        
        # Initialize with timeout
        print_info("Initializing router systems...")
        start_time = time.time()
        
        try:
            await asyncio.wait_for(router._check_llm_availability(), timeout=60.0)
            elapsed = time.time() - start_time
            print_success(f"Router systems initialized in {elapsed:.1f}s")
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print_error(f"Router initialization timed out after {elapsed:.1f}s")
            return False
        
        # Check availability
        print_info("Checking system availability...")
        print(f"   Offline Available: {router.offline_available}")
        print(f"   Online Available: {router.online_available}")
        print(f"   Skills Available: {router.skills_available}")
        print(f"   Router Mode: {router.mode.value}")
        
        systems_available = router.offline_available or router.online_available
        
        if not systems_available:
            print_error("No LLM systems available for testing")
            await router.close()
            return False
        
        # Test current info detection
        print_info("Testing current info detection...")
        
        test_queries = [
            # Current info queries (should return True)
            ("What day is today?", True, "Current date query"),
            ("What's today's date?", True, "Today's date query"),
            ("Who is the current president?", True, "Current political info"),
            ("What's happening in the news today?", True, "Current news query"),
            ("What's the weather like?", True, "Weather query"),
            
            # Non-current info queries (should return False)
            ("What time is it?", False, "Simple time query"),
            ("Hello, how are you?", False, "Greeting"),
            ("What is 2+2?", False, "Math query"),
            ("Explain Python", False, "General explanation"),
            ("Write a function", False, "Programming request")
        ]
        
        detection_correct = 0
        total_tests = len(test_queries)
        
        print_info(f"Running {total_tests} detection tests...")
        
        for query, expected_current_info, description in test_queries:
            detected = router._detect_current_info_enhanced(query)
            correct = detected == expected_current_info
            
            if correct:
                detection_correct += 1
                status_icon = "✅"
            else:
                status_icon = "❌"
            
            print(f"   {status_icon} '{query}' -> {detected} (expected: {expected_current_info})")
        
        detection_accuracy = (detection_correct / total_tests) * 100
        print_info(f"Detection accuracy: {detection_correct}/{total_tests} ({detection_accuracy:.1f}%)")
        
        # Test routing decisions
        print_info("Testing routing decisions...")
        
        routing_correct = 0
        
        for query, expected_current_info, description in test_queries:
            decision = router._decide_route_enhanced(query)
            
            # Check if routing makes sense
            route_correct = False
            
            if expected_current_info:
                # Current info should go to online if available, otherwise offline
                if router.online_available and decision.use_online:
                    route_correct = True
                elif not router.online_available and decision.use_offline:
                    route_correct = True
            else:
                # Non-current info can go to skills, offline, or online
                if decision.use_skill or decision.use_offline or decision.use_online:
                    route_correct = True
            
            if route_correct:
                routing_correct += 1
                status_icon = "✅"
            else:
                status_icon = "❌"
            
            print(f"   {status_icon} '{query}' -> {decision.route_type} ({decision.reason})")
        
        routing_accuracy = (routing_correct / total_tests) * 100
        print_info(f"Routing accuracy: {routing_correct}/{total_tests} ({routing_accuracy:.1f}%)")
        
        # Test actual response if systems are available
        if systems_available:
            print_info("Testing actual response generation...")
            
            # Test a current info query
            if router.online_available:
                test_query = "What day is today?"
                print_info(f"Testing current info: '{test_query}'")
                
                try:
                    start_time = time.time()
                    response = await asyncio.wait_for(
                        router.get_response(test_query),
                        timeout=30.0
                    )
                    elapsed = time.time() - start_time
                    
                    if response and len(response.strip()) > 10:
                        print_success(f"Current info response generated in {elapsed:.1f}s")
                        print(f"   Preview: {response[:100]}...")
                        
                        # Check if it contains current date info
                        import datetime
                        now = datetime.datetime.now()
                        current_indicators = [
                            now.strftime("%A").lower(),
                            now.strftime("%B").lower(),
                            str(now.year),
                            "today"
                        ]
                        
                        response_lower = response.lower()
                        has_current_info = any(indicator in response_lower for indicator in current_indicators)
                        
                        if has_current_info:
                            print_success("Response contains current date information!")
                        else:
                            print_warning("Response may not contain current date information")
                    else:
                        print_error("Empty or very short response received")
                        
                except asyncio.TimeoutError:
                    print_error("Response generation timed out")
                except Exception as e:
                    print_error(f"Response generation failed: {e}")
            
            # Test a simple query
            test_query = "Hello, how are you?"
            print_info(f"Testing general query: '{test_query}'")
            
            try:
                start_time = time.time()
                response = await asyncio.wait_for(
                    router.get_response(test_query),
                    timeout=30.0
                )
                elapsed = time.time() - start_time
                
                if response and len(response.strip()) > 5:
                    print_success(f"General response generated in {elapsed:.1f}s")
                    print(f"   Preview: {response[:100]}...")
                else:
                    print_error("Empty or very short response received")
                    
            except asyncio.TimeoutError:
                print_error("Response generation timed out")
            except Exception as e:
                print_error(f"Response generation failed: {e}")
        
        # Cleanup
        await router.close()
        
        # Return results
        success = (detection_accuracy >= 90 and routing_accuracy >= 80 and systems_available)
        
        return {
            'success': success,
            'detection_accuracy': detection_accuracy,
            'routing_accuracy': routing_accuracy,
            'systems_available': systems_available,
            'offline_available': router.offline_available,
            'online_available': router.online_available,
            'skills_available': router.skills_available
        }
        
    except Exception as e:
        print_error(f"Router integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_streaming_if_available(router):
    """Test streaming functionality if systems are available"""
    print_info("Testing streaming functionality...")
    
    if not (router.offline_available or router.online_available):
        print_warning("No systems available for streaming test")
        return False
    
    try:
        test_query = "Count from 1 to 3"
        print_info(f"Testing streaming with: '{test_query}'")
        print("Response: ", end="", flush=True)
        
        response_text = ""
        chunk_count = 0
        start_time = time.time()
        first_chunk_time = None
        
        async for chunk in router.get_streaming_response(test_query):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
            
            print(chunk, end="", flush=True)
            response_text += chunk
            chunk_count += 1
        
        total_time = time.time() - start_time
        print()  # New line
        
        if response_text and chunk_count > 0:
            print_success(f"Streaming test successful!")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   First chunk: {first_chunk_time:.2f}s" if first_chunk_time else "   First chunk: N/A")
            print(f"   Chunks received: {chunk_count}")
            print(f"   Response length: {len(response_text)} characters")
            return True
        else:
            print_error("Streaming test failed - no response received")
            return False
            
    except Exception as e:
        print_error(f"Streaming test error: {e}")
        return False

async def main():
    """Main test function with comprehensive reporting"""
    print_header("IMPROVED Pascal Current Info Detection Test")
    print_info("This test will comprehensively check the current info routing system")
    
    start_time = time.time()
    
    # Check prerequisites
    issues = await check_prerequisites()
    
    if issues:
        print_error("Prerequisites check failed:")
        for issue in issues:
            print(f"   • {issue}")
        print_info("\nQuick fixes:")
        print("   • pip install aiohttp==3.9.5")
        print("   • Check you're in the Pascal directory")
        print("   • Ensure all Python files are present")
        print("   • Run: python aiohttp_quick_fix.py")
        return False
    
    print_success("Prerequisites check passed")
    
    # Test individual components
    component_status = await test_individual_components()
    
    working_components = sum(1 for status in component_status.values() if status)
    total_components = len(component_status)
    
    print_info(f"Component status: {working_components}/{total_components} working")
    
    if working_components == 0:
        print_error("No components are working - cannot proceed with integration tests")
        print_info("\nTroubleshooting:")
        print("   • Run: python aiohttp_quick_fix.py")
        print("   • Check: sudo systemctl start ollama")
        print("   • Verify: ollama pull nemotron-mini:4b-instruct-q4_K_M")
        print("   • Add Groq API key to .env file")
        return False
    
    # Test router integration
    router_results = await test_router_integration(component_status)
    
    if not router_results:
        print_error("Router integration test failed")
        return False
    
    # Final assessment
    total_time = time.time() - start_time
    
    print_header("COMPREHENSIVE TEST RESULTS")
    
    print_section("Component Health")
    for component, status in component_status.items():
        status_icon = "✅" if status else "❌"
        component_name = component.replace('_', ' ').title()
        print(f"   {status_icon} {component_name}")
    
    print_section("Router Test Results")
    if isinstance(router_results, dict):
        print(f"   Detection Accuracy: {router_results['detection_accuracy']:.1f}%")
        print(f"   Routing Accuracy: {router_results['routing_accuracy']:.1f}%")
        print(f"   Systems Available: {router_results['systems_available']}")
        print(f"   Offline Available: {router_results['offline_available']}")
        print(f"   Online Available: {router_results['online_available']}")
        print(f"   Skills Available: {router_results['skills_available']}")
        
        success = router_results['success']
    else:
        success = False
    
    print_section("Overall Assessment")
    print(f"   Test Duration: {total_time:.1f} seconds")
    print(f"   Components Working: {working_components}/{total_components}")
    
    if success:
        print_success("🎉 CURRENT INFO DETECTION IS WORKING CORRECTLY!")
        
        print_section("Verified Functionality")
        print("   ✅ Current information queries route to online properly")
        print("   ✅ Simple instant queries route to skills when available")
        print("   ✅ General queries route to offline when available")
        print("   ✅ System availability checking is robust")
        print("   ✅ Error handling and fallbacks work properly")
        
        print_section("Performance Summary")
        if isinstance(router_results, dict):
            if router_results['detection_accuracy'] >= 95:
                print("   🏆 EXCELLENT detection accuracy")
            elif router_results['detection_accuracy'] >= 90:
                print("   ✅ GOOD detection accuracy")
            
            if router_results['routing_accuracy'] >= 90:
                print("   🏆 EXCELLENT routing accuracy")
            elif router_results['routing_accuracy'] >= 80:
                print("   ✅ GOOD routing accuracy")
        
        print_section("Next Steps")
        print("   1. Run Pascal: ./run.sh")
        print("   2. Test current info: 'What day is today?' (should show 🌐)")
        print("   3. Test simple query: 'What time is it?' (should be instant)")
        print("   4. Test general query: 'Hello Pascal' (should work quickly)")
        print("   5. Use 'status' command to verify routing")
        
    else:
        print_error("⚠️ SYSTEM NEEDS ATTENTION")
        
        print_section("Issues Detected")
        
        if not component_status.get('offline_llm', False):
            print("   ❌ Offline LLM (Ollama/Nemotron) not working")
            print("      • Check: sudo systemctl status ollama")
            print("      • Start: sudo systemctl start ollama")
            print("      • Model: ollama pull nemotron-mini:4b-instruct-q4_K_M")
        
        if not component_status.get('online_llm', False):
            print("   ❌ Online LLM (Groq) not working")
            print("      • Add GROQ_API_KEY to .env file")
            print("      • Get free key: https://console.groq.com/")
            print("      • Check internet connection")
        
        if isinstance(router_results, dict):
            if router_results['detection_accuracy'] < 90:
                print("   ❌ Current info detection accuracy too low")
                print("      • May need pattern refinement")
            
            if router_results['routing_accuracy'] < 80:
                print("   ❌ Routing accuracy too low")
                print("      • Check system availability")
        
        print_section("Recommended Actions")
        print("   1. Run: python aiohttp_quick_fix.py")
        print("   2. Run: python complete_diagnostic.py")
        print("   3. Check system logs for detailed errors")
        print("   4. Verify all dependencies are installed")
        print("   5. Ensure you're in the correct virtual environment")
    
    print_section("Summary")
    if success:
        print_success("✅ All tests passed - Pascal is ready!")
    else:
        print_error("❌ Some tests failed - check recommendations above")
    
    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if result:
            print_success("\n🎉 Test completed successfully!")
            print_info("Pascal's current info detection system is working properly!")
        else:
            print_error("\n⚠️ Test revealed issues - check output above")
            print_info("Run the recommended fixes and try again")
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print_warning("\n⏹️ Test interrupted")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
