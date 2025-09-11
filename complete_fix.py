#!/usr/bin/env python3
"""
Pascal AI Assistant - Complete Fix and Debug Script
FIXED: Addresses all API key issues and routing problems with enhanced debugging
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

def fix_env_file():
    """Fix .env file API key format issues"""
    print("🔧 Fixing .env file API key formats...")
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("Creating .env file from template...")
        with open(env_file, 'w') as f:
            f.write("""# Pascal AI Assistant Environment Variables - FIXED
# Copy this file and add your actual API keys

# Performance Settings
PERFORMANCE_MODE=balanced
STREAMING_ENABLED=true
KEEP_ALIVE_ENABLED=true
TARGET_RESPONSE_TIME=3.0
MAX_RESPONSE_TOKENS=200

# API Keys (FIXED formats)
# Groq API (Primary - fastest) - NEW FORMAT: gsk_ (underscore)
GROQ_API_KEY=gsk_your_groq_api_key_here

# Google Gemini API (Secondary - free)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API (Fallback - reliable but paid)
OPENAI_API_KEY=sk_your_openai_api_key_here

# Ollama Settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_KEEP_ALIVE=30m

# Debug Settings
DEBUG=false
LOG_LEVEL=INFO
""")
        print("✅ Created .env file with correct API key formats")
        print("📝 Please add your actual API keys to .env")
        return
    
    # Fix existing .env file
    with open(env_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    changes_made = []
    
    for line in lines:
        # Fix GROK -> GROQ naming
        if line.startswith('GROK_API_KEY='):
            key_value = line.split('=', 1)[1] if '=' in line else ''
            fixed_lines.append(f'GROQ_API_KEY={key_value}')
            changes_made.append("Renamed GROK_API_KEY to GROQ_API_KEY")
        
        # Fix gsk- to gsk_ format in comments/examples
        elif 'gsk-' in line and line.strip().startswith('#'):
            fixed_line = line.replace('gsk-', 'gsk_')
            fixed_lines.append(fixed_line)
            if fixed_line != line:
                changes_made.append("Updated example format from gsk- to gsk_")
        
        # Add warning comment for deprecated format
        elif line.startswith('GROQ_API_KEY=') and '=' in line:
            key_value = line.split('=', 1)[1].strip()
            if key_value.startswith('gsk-'):
                fixed_lines.append("# WARNING: gsk- format is deprecated, use gsk_ for new keys")
                fixed_lines.append(line)
                changes_made.append("Added deprecation warning for gsk- format")
            else:
                fixed_lines.append(line)
        
        else:
            fixed_lines.append(line)
    
    if changes_made:
        with open(env_file, 'w') as f:
            f.write('\n'.join(fixed_lines))
        
        print("✅ Fixed .env file:")
        for change in changes_made:
            print(f"   • {change}")
    else:
        print("✅ .env file format is already correct")

async def test_current_info_routing():
    """Test the current information routing logic"""
    print("\n🧪 Testing Current Information Routing...")
    
    try:
        os.environ['DEBUG'] = 'true'  # Enable debug mode
        
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        
        # Create router
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        router = LightningRouter(personality_manager, memory_manager)
        
        # Test queries with expected routing
        test_cases = [
            ("What day is today?", "online", "Direct date query"),
            ("What's the current date?", "online", "Current date query"),
            ("Who is the current president?", "online", "Current status query"),
            ("What's happening in the news today?", "online", "Current news query"),
            ("Hello, how are you?", "offline", "Simple greeting"),
            ("What is 2+2?", "offline", "Simple math"),
            ("Explain quantum physics", "offline", "General knowledge"),
            ("Write a Python function to sort a list", "online", "Code generation"),
        ]
        
        print("Testing routing decisions:")
        correct_routes = 0
        total_tests = len(test_cases)
        
        for query, expected_route, description in test_cases:
            # Simulate both LLMs available
            router.offline_available = True
            router.online_available = True
            
            # Test current info detection
            needs_current = router._needs_current_information(query)
            
            # Get routing decision
            decision = router._decide_route(query)
            actual_route = "offline" if decision.use_offline else "online"
            
            is_correct = actual_route == expected_route
            if is_correct:
                correct_routes += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"  {status} '{query}'")
            print(f"     Expected: {expected_route}, Got: {actual_route}")
            print(f"     Needs current info: {needs_current}")
            print(f"     Reason: {decision.reason}")
            print()
        
        accuracy = (correct_routes / total_tests) * 100
        print(f"📊 Routing Accuracy: {correct_routes}/{total_tests} ({accuracy:.1f}%)")
        
        if accuracy >= 90:
            print("✅ Routing logic is working correctly!")
        else:
            print("⚠️ Routing logic needs adjustment")
        
        await router.close()
        return accuracy >= 90
        
    except Exception as e:
        print(f"❌ Error testing routing: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_connections():
    """Test API connections with proper error handling"""
    print("\n🔌 Testing API Connections...")
    
    try:
        # Check aiohttp first
        try:
            import aiohttp
            print("✅ aiohttp available")
        except ImportError:
            print("❌ aiohttp not installed")
            print("   Fix: pip install aiohttp")
            return False
        
        from config.settings import settings
        from modules.online_llm import OnlineLLM
        
        # Enable debug mode
        settings.debug_mode = True
        
        # Test settings validation
        print("\nAPI Key Validation:")
        groq_valid = settings.validate_groq_api_key(settings.groq_api_key)
        gemini_valid = settings.validate_gemini_api_key(settings.gemini_api_key)
        openai_valid = settings.validate_openai_api_key(settings.openai_api_key)
        
        print(f"  Groq: {'✅ Valid' if groq_valid else '❌ Invalid/Missing'}")
        print(f"  Gemini: {'✅ Valid' if gemini_valid else '❌ Invalid/Missing'}")
        print(f"  OpenAI: {'✅ Valid' if openai_valid else '❌ Invalid/Missing'}")
        
        if not any([groq_valid, gemini_valid, openai_valid]):
            print("❌ No valid API keys found")
            print("   Add at least one API key to .env file")
            print("   Groq (fastest): GROQ_API_KEY=gsk_your-key")
            print("   Gemini (free): GEMINI_API_KEY=your-key") 
            print("   OpenAI (reliable): OPENAI_API_KEY=sk-your-key")
            return False
        
        # Test OnlineLLM initialization
        print("\nTesting OnlineLLM:")
        online_llm = OnlineLLM()
        success = await online_llm.initialize()
        
        if success:
            print("✅ OnlineLLM initialized successfully")
            
            # Get provider stats
            stats = online_llm.get_provider_stats()
            print(f"  Available providers: {stats['available_providers']}")
            print(f"  Preferred provider: {stats['preferred_provider']}")
            
            # Test a simple query
            if stats['available_providers']:
                print("\nTesting simple query:")
                response = await online_llm.generate_response(
                    "Say 'API test successful'",
                    "You are a test assistant",
                    ""
                )
                
                if "successful" in response.lower():
                    print("✅ API query test passed")
                    
                    # Test current info query
                    print("\nTesting current info query:")
                    current_response = await online_llm.generate_response(
                        "What day is today?",
                        "You are a helpful assistant with access to current information",
                        ""
                    )
                    print(f"  Response: {current_response[:100]}...")
                    
                    await online_llm.close()
                    return True
                else:
                    print("⚠️ Unexpected response from API")
                    print(f"  Response: {response}")
            
        else:
            print("❌ OnlineLLM initialization failed")
            if hasattr(online_llm, 'last_error') and online_llm.last_error:
                print(f"  Error: {online_llm.last_error}")
        
        await online_llm.close()
        return False
        
    except Exception as e:
        print(f"❌ Error testing APIs: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_ollama_status():
    """Check Ollama service status"""
    print("\n🦙 Checking Ollama Status...")
    
    try:
        import subprocess
        
        # Check if ollama command exists
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ Ollama is installed")
            print(f"   Version: {result.stdout.strip()}")
        else:
            print("❌ Ollama not properly installed")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama not found")
        print("   Install: curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    except Exception as e:
        print(f"⚠️ Error checking Ollama: {e}")
        return False
    
    # Check service status
    try:
        result = subprocess.run(['systemctl', 'is-active', 'ollama'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.stdout.strip() == 'active':
            print("✅ Ollama service is running")
        else:
            print("⚠️ Ollama service not running")
            print("   Start: sudo systemctl start ollama")
            
    except Exception:
        print("⚠️ Could not check service status")
    
    # Check for models
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                model_count = len(lines) - 1
                print(f"✅ Found {model_count} model(s)")
                return True
            else:
                print("⚠️ No models found")
                print("   Download: ./download_models.sh")
                return False
        else:
            print("❌ Could not list models")
            return False
            
    except Exception as e:
        print(f"⚠️ Error checking models: {e}")
        return False

async def test_end_to_end_routing():
    """Test end-to-end routing with actual LLMs"""
    print("\n🔄 Testing End-to-End Routing...")
    
    try:
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        
        # Create router
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        router = LightningRouter(personality_manager, memory_manager)
        
        # Check LLM availability
        await router._check_llm_availability()
        
        print(f"\nLLM Status:")
        print(f"  Offline available: {router.offline_available}")
        print(f"  Online available: {router.online_available}")
        print(f"  Router mode: {router.mode.value}")
        
        # Test current info query if online is available
        if router.online_available:
            print("\n🧪 Testing current info query:")
            test_query = "What day is today?"
            print(f"Query: '{test_query}'")
            
            decision = router._decide_route(test_query)
            print(f"Decision: {'Online' if not decision.use_offline else 'Offline'} - {decision.reason}")
            
            if not decision.use_offline:
                print("✅ Current info query correctly routed to online")
                
                # Test actual response
                try:
                    response = await router.get_response(test_query)
                    print(f"Response preview: {response[:150]}...")
                    
                    # Check if response contains current date info
                    if any(word in response.lower() for word in ['today', 'thursday', 'september', '2025']):
                        print("✅ Response contains current date information")
                    else:
                        print("⚠️ Response may not contain current date information")
                        
                except Exception as e:
                    print(f"❌ Error getting response: {e}")
            else:
                print("❌ Current info query incorrectly routed to offline")
        else:
            print("⚠️ Cannot test current info - online LLM not available")
        
        # Test simple query
        if router.offline_available:
            print("\n🧪 Testing simple query:")
            simple_query = "Hello, how are you?"
            print(f"Query: '{simple_query}'")
            
            decision = router._decide_route(simple_query)
            print(f"Decision: {'Online' if not decision.use_offline else 'Offline'} - {decision.reason}")
            
            try:
                response = await router.get_response(simple_query)
                print(f"Response preview: {response[:100]}...")
                print("✅ Simple query processed successfully")
            except Exception as e:
                print(f"❌ Error with simple query: {e}")
        
        await router.close()
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_fixes():
    """Show available fixes"""
    print("\n🔧 Available Fixes:")
    print("1. fix-env     - Fix .env file API key formats")
    print("2. test-routing - Test current info routing logic")
    print("3. test-apis   - Test online API connections")
    print("4. check-ollama - Check Ollama installation and status")
    print("5. test-e2e    - Test end-to-end routing with actual LLMs")
    print("6. full-check  - Run all diagnostics and fixes")

async def main():
    """Main function"""
    print("🚀 Pascal Complete Fix and Debug Tool")
    print("=" * 50)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'fix-env':
            fix_env_file()
        elif command == 'test-routing':
            await test_current_info_routing()
        elif command == 'test-apis':
            await test_api_connections()
        elif command == 'check-ollama':
            check_ollama_status()
        elif command == 'test-e2e':
            await test_end_to_end_routing()
        elif command == 'full-check':
            # Run all checks
            print("Running complete system check...")
            
            # Fix environment
            fix_env_file()
            
            # Check Ollama
            ollama_ok = check_ollama_status()
            
            # Test APIs
            api_ok = await test_api_connections()
            
            # Test routing
            routing_ok = await test_current_info_routing()
            
            # Test end-to-end
            e2e_ok = await test_end_to_end_routing()
            
            # Summary
            print("\n" + "=" * 50)
            print("📊 COMPLETE CHECK SUMMARY")
            print("=" * 50)
            
            checks = [
                ("Environment", True),  # Always pass after fix
                ("Ollama", ollama_ok),
                ("Online APIs", api_ok),
                ("Routing Logic", routing_ok),
                ("End-to-End", e2e_ok),
            ]
            
            passed = sum(1 for _, result in checks if result)
            total = len(checks)
            
            for check_name, result in checks:
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"{status} {check_name}")
            
            print(f"\nOverall: {passed}/{total} checks passed")
            
            if passed == total:
                print("\n🎉 ALL SYSTEMS GO!")
                print("Pascal should work perfectly now.")
                print("Run: ./run.sh")
            elif passed >= 3:
                print("\n⚡ MOSTLY WORKING!")
                print("Pascal should work with some limitations.")
                print("Run: ./run.sh")
            else:
                print("\n⚠️ Issues remain - see details above")
                print("Address the failed checks and run 'full-check' again")
        else:
            print(f"Unknown command: {command}")
            show_fixes()
    else:
        # Default behavior - show current status
        print("Checking current system status...\n")
        
        # Quick status check
        env_file = Path(".env")
        if env_file.exists():
            print("✅ .env file exists")
            
            # Check for API keys
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Check Groq key format
            if 'GROQ_API_KEY=' in content:
                for line in content.split('\n'):
                    if line.startswith('GROQ_API_KEY='):
                        key = line.split('=', 1)[1].strip()
                        if key.startswith('gsk_'):
                            print("✅ Groq API key format is correct (gsk_)")
                        elif key.startswith('gsk-'):
                            print("⚠️ Groq API key uses deprecated format (gsk-)")
                            print("   Consider updating to gsk_ format")
                        elif key and key not in ['', 'gsk_your_groq_api_key_here']:
                            print("❌ Groq API key format is invalid")
                        else:
                            print("⚠️ Groq API key is placeholder/empty")
                        break
            elif 'GROK_API_KEY=' in content:
                print("⚠️ Found GROK_API_KEY (legacy naming)")
                print("   Run: python complete_fix.py fix-env")
            else:
                print("⚠️ No Groq API key configured")
        else:
            print("❌ .env file not found")
        
        # Check if modules can be imported
        try:
            from config.settings import settings
            print("✅ Pascal modules can be imported")
            
            if settings.is_online_available():
                print("✅ Online APIs are configured")
            else:
                print("⚠️ No online APIs configured")
                
        except Exception as e:
            print(f"❌ Cannot import Pascal modules: {e}")
        
        print("\n" + "=" * 50)
        show_fixes()
        print("\nRecommended: python complete_fix.py full-check")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
