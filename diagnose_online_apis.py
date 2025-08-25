#!/usr/bin/env python3
"""
Pascal AI Assistant - Online LLM Diagnostic Script
Complete diagnostic tool for online API connectivity issues
"""

import sys
import os
import asyncio
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

async def diagnose_online_apis():
    """Complete diagnostic for online API connectivity"""
    print("🌐 Pascal AI - Complete Online LLM Diagnostics")
    print("=" * 60)
    
    # Step 1: Check aiohttp availability
    try:
        import aiohttp
        print("✅ aiohttp module: Available")
        aiohttp_ok = True
    except ImportError:
        print("❌ aiohttp module: Not installed")
        print("\nTo fix:")
        print("1. Activate virtual environment: source venv/bin/activate")
        print("2. Install aiohttp: pip install aiohttp")
        print("3. Or reinstall all: pip install -r requirements.txt")
        return False
    
    # Step 2: Import settings
    try:
        from config.settings import settings
        print("✅ Settings module: Loaded")
    except ImportError as e:
        print(f"❌ Settings module: Failed to load - {e}")
        return False
    
    # Step 3: Check API key configuration
    print("\n🔍 Checking API Key Configuration:")
    
    api_keys = {
        'Grok (xAI)': getattr(settings, 'grok_api_key', None),
        'OpenAI': getattr(settings, 'openai_api_key', None),
        'Anthropic': getattr(settings, 'anthropic_api_key', None)
    }
    
    configured_apis = []
    for name, key in api_keys.items():
        provider_name = name.lower().split()[0]  # Extract provider name
        invalid_keys = [None, '', 'your_api_key_here', f'your_{provider_name}_api_key_here']
        
        if key and key not in invalid_keys:
            print(f"✅ {name}: Configured (key length: {len(str(key))})")
            configured_apis.append(name)
        else:
            print(f"❌ {name}: Not configured")
    
    if not configured_apis:
        print("\n❌ No API keys properly configured!")
        print("\nTo configure:")
        print("1. Copy template: cp .env.example .env")
        print("2. Edit .env file: nano .env")
        print("3. Add real API keys:")
        print("   GROK_API_KEY=xai-your-actual-key-here")
        print("   OPENAI_API_KEY=sk-your-actual-key-here")  
        print("   ANTHROPIC_API_KEY=sk-ant-your-actual-key-here")
        print("\nNote: You only need one valid API key for online functionality")
        return False
    
    # Step 4: Test online LLM initialization
    print(f"\n🔄 Testing Online LLM with {len(configured_apis)} API(s)...")
    
    try:
        from modules.online_llm import OnlineLLM
        online_llm = OnlineLLM()
        
        # Initialize with detailed error reporting
        success = await online_llm.initialize()
        
        if success:
            print("✅ Online LLM initialized successfully")
            
            # Get detailed provider statistics
            stats = online_llm.get_provider_stats()
            
            print(f"\n📊 System Status:")
            print(f"  • aiohttp available: {stats['aiohttp_available']}")
            print(f"  • Initialization successful: {stats['initialization_successful']}")
            print(f"  • Available providers: {stats['available_providers']}")
            print(f"  • Preferred provider: {stats['preferred_provider']}")
            
            if stats.get('last_error'):
                print(f"  • Last error: {stats['last_error']}")
            
            print(f"\n📋 Provider Details:")
            for provider_name, provider_stats in stats['providers'].items():
                available = "✅ Available" if provider_stats['available'] else "❌ Not Available"
                configured = "🔑 Configured" if provider_stats['api_key_configured'] else "🚫 No Key"
                success_count = provider_stats['success_count']
                failure_count = provider_stats['failure_count']
                
                print(f"  • {provider_name.title()}: {available} ({configured})")
                print(f"    Success: {success_count}, Failures: {failure_count}")
            
            # Step 5: Test actual API functionality
            print(f"\n🧪 Testing Live API Response:")
            try:
                test_query = "Respond with exactly: 'API test successful'"
                response = await online_llm.generate_response(
                    test_query,
                    "You are a helpful assistant.", 
                    ""
                )
                
                if "API test successful" in response:
                    print("✅ API response test: SUCCESS")
                    print(f"Full response: {response}")
                elif response and len(response) > 10 and not response.startswith("I'm having trouble"):
                    print("⚠️ API working but gave unexpected response:")
                    print(f"Response: {response[:200]}...")
                    print("This usually means the API is working correctly.")
                else:
                    print("❌ API response test failed:")
                    print(f"Response: {response}")
                    
                    # Additional debugging
                    print("\nDebugging info:")
                    if hasattr(online_llm, 'last_error') and online_llm.last_error:
                        print(f"Last error: {online_llm.last_error}")
            
            except Exception as e:
                print(f"❌ API response test failed with exception: {e}")
                print("\nThis indicates a connectivity or API key issue.")
            
            await online_llm.close()
            return True
        
        else:
            print("❌ Online LLM initialization failed")
            
            # Detailed failure analysis
            stats = online_llm.get_provider_stats()
            
            print(f"\n🔍 Failure Analysis:")
            if stats.get('last_error'):
                print(f"Primary error: {stats['last_error']}")
            
            print(f"\nProvider analysis:")
            for provider_name, provider_stats in stats['providers'].items():
                configured = provider_stats['api_key_configured']
                available = provider_stats['available']
                print(f"  • {provider_name.title()}:")
                print(f"    Key configured: {configured}")
                print(f"    Connection test passed: {available}")
            
            print(f"\n🔧 Troubleshooting:")
            print("1. Verify internet connection")
            print("2. Check API keys are valid and have quota/credits")
            print("3. Ensure no extra spaces/quotes in .env file")
            print("4. Test API keys directly with curl/postman")
            print("5. Check firewall/proxy settings")
            
            await online_llm.close()
            return False
    
    except Exception as e:
        print(f"❌ Critical error testing OnlineLLM: {e}")
        print(f"\nError type: {type(e).__name__}")
        print(f"This suggests a code or import issue.")
        
        # Show traceback for debugging
        if settings.debug_mode:
            import traceback
            traceback.print_exc()
        
        return False

def check_environment():
    """Comprehensive environment check"""
    print("\n🔍 Environment Analysis:")
    
    # Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"⚠️ Python: {python_version.major}.{python_version.minor}.{python_version.micro} (3.8+ recommended)")
    
    # Virtual environment
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if in_venv:
        print("✅ Virtual environment: Active")
    else:
        print("⚠️ Virtual environment: Not active")
        print("   Activate with: source venv/bin/activate")
    
    # Current directory
    current_dir = Path.cwd()
    if current_dir.name == 'pascal':
        print("✅ Directory: In pascal project")
    else:
        print(f"⚠️ Directory: {current_dir} (should be in pascal/)")
    
    # Critical files
    critical_files = {
        'main.py': 'Main entry point',
        'requirements.txt': 'Dependencies',
        'modules/online_llm.py': 'Online LLM module',
        'modules/router.py': 'Router module',
        '.env': 'Environment config'
    }
    
    print(f"\n📁 File Status:")
    for file, description in critical_files.items():
        if Path(file).exists():
            print(f"✅ {file}: Found ({description})")
        else:
            print(f"❌ {file}: Missing ({description})")
    
    # Dependencies check
    print(f"\n📦 Key Dependencies:")
    dependencies = ['aiohttp', 'openai', 'anthropic', 'requests']
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}: Available")
        except ImportError:
            print(f"❌ {dep}: Not installed")

def main():
    """Main diagnostic function"""
    try:
        # Environment check first
        check_environment()
        
        # Online API diagnostics
        result = asyncio.run(diagnose_online_apis())
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        if result:
            print("✅ Online LLM diagnostics PASSED!")
            print("\nReady to use online services!")
            print("\nNext steps:")
            print("1. Run Pascal: ./run.sh")
            print("2. Test offline: 'what is 2+2?'")
            print("3. Test online: 'what is today's date?'")
        else:
            print("❌ Online LLM diagnostics FAILED!")
            print("\nPlease fix the issues above.")
            print("\nYou can still use Pascal offline-only if Ollama is working.")
            print("Run Pascal with: ./run.sh")
        
        print("\n" + "=" * 60)
        return 0 if result else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Diagnostics interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected diagnostic error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
