#!/usr/bin/env python3
"""
Pascal AI Assistant - Online LLM Diagnostic Script
Diagnoses online API connectivity and configuration issues
"""

import sys
import os
import asyncio
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

async def diagnose_online_apis():
    """Diagnose online API connectivity"""
    print("🌐 Pascal AI - Online LLM Diagnostics")
    print("=" * 50)
    
    # Check if aiohttp is available
    try:
        import aiohttp
        print("✅ aiohttp module: Available")
    except ImportError:
        print("❌ aiohttp module: Not installed")
        print("\nTo fix this issue:")
        print("1. Activate virtual environment: source venv/bin/activate")
        print("2. Install missing dependencies: pip install aiohttp")
        print("3. Or reinstall all requirements: pip install -r requirements.txt")
        return False
    
    # Import settings after confirming aiohttp is available
    try:
        from config.settings import settings
        print("✅ Settings module: Loaded successfully")
    except ImportError as e:
        print(f"❌ Settings module: Import failed - {e}")
        return False
    
    # Check environment variables
    print("\n🔍 Checking API Key Configuration:")
    
    api_keys = {
        'Grok (xAI)': settings.grok_api_key,
        'OpenAI': settings.openai_api_key,
        'Anthropic': settings.anthropic_api_key
    }
    
    configured_apis = []
    for name, key in api_keys.items():
        if key and key not in ['', 'your_api_key_here', f'your_{name.lower().split()[0]}_api_key_here', None]:
            print(f"✅ {name}: Configured")
            configured_apis.append(name)
        else:
            print(f"❌ {name}: Not configured")
    
    if not configured_apis:
        print("\n❌ No API keys configured!")
        print("\nTo configure API keys:")
        print("1. Copy .env.example to .env: cp .env.example .env")
        print("2. Edit .env and add your API keys:")
        print("   GROK_API_KEY=your_actual_grok_key")
        print("   OPENAI_API_KEY=your_actual_openai_key")
        print("   ANTHROPIC_API_KEY=your_actual_anthropic_key")
        print("\nNote: You only need one API key to enable online functionality")
        return False
    
    # Test online LLM initialization
    print(f"\n🔄 Testing Online LLM with {len(configured_apis)} configured API(s)...")
    
    try:
        from modules.online_llm import OnlineLLM
        online_llm = OnlineLLM()
        success = await online_llm.initialize()
        
        if success:
            print("✅ Online LLM initialized successfully")
            
            # Get detailed stats
            stats = online_llm.get_provider_stats()
            
            print(f"\n📊 Provider Status:")
            print(f"  • aiohttp available: {stats['aiohttp_available']}")
            print(f"  • Initialization successful: {stats['initialization_successful']}")
            print(f"  • Available providers: {stats['available_providers']}")
            print(f"  • Preferred provider: {stats['preferred_provider']}")
            
            if stats.get('last_error'):
                print(f"  • Last error: {stats['last_error']}")
            
            print(f"\n📋 Individual Provider Status:")
            for provider_name, provider_stats in stats['providers'].items():
                status = "✅ Available" if provider_stats['available'] else "❌ Not Available"
                key_status = "🔑 Configured" if provider_stats['api_key_configured'] else "🚫 No Key"
                success_count = provider_stats['success_count']
                failure_count = provider_stats['failure_count']
                print(f"  • {provider_name.title()}: {status} ({key_status}) - Success: {success_count}, Failures: {failure_count}")
            
            # Test actual API call
            print(f"\n🧪 Testing API Response:")
            try:
                response = await online_llm.generate_response(
                    "Say 'Online API test successful!' in exactly those words",
                    "You are a helpful assistant.", 
                    ""
                )
                
                if "Online API test successful!" in response:
                    print("✅ API response test: SUCCESS")
                    print(f"Response: {response}")
                elif response and not response.startswith("I'm sorry"):
                    print("⚠️ API responded but with unexpected content:")
                    print(f"Response: {response[:200]}...")
                else:
                    print("❌ API response test failed")
                    print(f"Response: {response}")
            
            except Exception as e:
                print(f"❌ API response test failed: {e}")
            
            await online_llm.close()
            return True
        
        else:
            print("❌ Online LLM initialization failed")
            
            # Get error details
            stats = online_llm.get_provider_stats()
            if stats.get('last_error'):
                print(f"Last error: {stats['last_error']}")
            
            print(f"\n📋 Provider Details:")
            for provider_name, provider_stats in stats['providers'].items():
                key_configured = provider_stats['api_key_configured']
                print(f"  • {provider_name.title()}: Key configured = {key_configured}")
            
            print("\n🔧 Troubleshooting Steps:")
            print("1. Check your internet connection")
            print("2. Verify API keys are valid and have credits/quota")
            print("3. Check API key format (no extra spaces/quotes in .env)")
            print("4. Try testing individual providers")
            print("5. Check firewall/proxy settings")
            
            await online_llm.close()
            return False
    
    except Exception as e:
        print(f"❌ Failed to import or test OnlineLLM: {e}")
        print("\n🔧 This might indicate:")
        print("1. Missing dependencies (run: pip install -r requirements.txt)")
        print("2. Corrupted module files")
        print("3. Python environment issues")
        import traceback
        traceback.print_exc()
        return False

def check_environment():
    """Check Python environment and dependencies"""
    print("\n🔍 Environment Check:")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment: Active")
    else:
        print("⚠️ Virtual environment: Not active (recommended to use venv)")
    
    # Check current directory
    current_dir = Path.cwd()
    if current_dir.name == 'pascal':
        print("✅ Directory: In pascal project directory")
    else:
        print(f"⚠️ Directory: {current_dir} (should be in pascal/)")
    
    # Check key files
    key_files = ['main.py', 'requirements.txt', 'modules/online_llm.py', '.env']
    for file in key_files:
        if Path(file).exists():
            print(f"✅ {file}: Found")
        else:
            print(f"❌ {file}: Missing")

def main():
    """Main diagnostic function"""
    try:
        # Basic environment check first
        check_environment()
        
        # Then test online APIs
        result = asyncio.run(diagnose_online_apis())
        
        print("\n" + "=" * 50)
        if result:
            print("✅ Online LLM diagnostics PASSED!")
            print("Pascal should now work with online services.")
            print("\nNext steps:")
            print("1. Run Pascal: ./run.sh")
            print("2. Test online query: 'what is today's date?'")
            print("3. Test offline query: 'what is 2+2?'")
        else:
            print("❌ Online LLM diagnostics FAILED!")
            print("Fix the issues above, then run this diagnostic again.")
            print("\nYou can still use Pascal in offline-only mode if Ollama is working.")
        
        return 0 if result else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Diagnostics interrupted")
        return 1
    except Exception as e:
        print(f"\n💥 Diagnostic error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
