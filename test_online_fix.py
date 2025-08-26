#!/usr/bin/env python3
"""
Quick test script for fixed online LLM functionality
"""

import asyncio
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

async def test_online_fix():
    """Test the fixed online LLM"""
    print("🔧 Testing Fixed Online LLM Integration")
    print("=" * 50)
    
    try:
        # Import fixed modules
        from modules.online_llm import OnlineLLM
        from config.settings import settings
        
        # Check API keys
        print("📋 API Key Status:")
        grok_key = getattr(settings, 'grok_api_key', None)
        openai_key = getattr(settings, 'openai_api_key', None)
        anthropic_key = getattr(settings, 'anthropic_api_key', None)
        
        invalid_keys = [None, '', 'your_api_key_here', 'your_grok_api_key_here', 'your_openai_api_key_here', 'your_anthropic_api_key_here', 'your_actual_grok_api_key_here']
        
        grok_configured = grok_key and grok_key not in invalid_keys
        openai_configured = openai_key and openai_key not in invalid_keys
        anthropic_configured = anthropic_key and anthropic_key not in invalid_keys
        
        print(f"  Grok: {'✅ Configured' if grok_configured else '❌ Not configured'}")
        print(f"  OpenAI: {'✅ Configured' if openai_configured else '❌ Not configured'}")  
        print(f"  Anthropic: {'✅ Configured' if anthropic_configured else '❌ Not configured'}")
        
        if not any([grok_configured, openai_configured, anthropic_configured]):
            print("\n❌ No API keys configured!")
            print("Edit your .env file and add your actual API keys.")
            print("At minimum, you need ONE working API key.")
            return False
        
        # Test online LLM initialization
        print(f"\n🔄 Initializing Online LLM...")
        online_llm = OnlineLLM()
        
        success = await online_llm.initialize()
        
        if success:
            print("✅ Online LLM initialized successfully!")
            
            # Get provider stats
            stats = online_llm.get_provider_stats()
            print(f"\n📊 Provider Status:")
            print(f"  Available providers: {stats['available_providers']}")
            print(f"  Primary provider: {stats['preferred_provider']}")
            
            for provider_name, provider_stats in stats['providers'].items():
                status = "✅ Available" if provider_stats['available'] else "❌ Not Available"
                configured = "🔑 Key OK" if provider_stats['api_key_configured'] else "🚫 No Key"
                print(f"    {provider_name.title()}: {status} ({configured})")
            
            # Test actual API call
            print(f"\n🧪 Testing API Response...")
            try:
                response = await online_llm.generate_response(
                    "Say 'Online test successful' and nothing else",
                    "You are a helpful assistant.",
                    ""
                )
                
                if "Online test successful" in response:
                    print("✅ API response test: SUCCESS")
                    print(f"Response: {response}")
                elif response and len(response) > 5:
                    print("⚠️ API working but gave different response:")
                    print(f"Response: {response[:200]}")
                else:
                    print("❌ API response test failed:")
                    print(f"Response: {response}")
                    
            except Exception as e:
                print(f"❌ API test error: {e}")
            
            await online_llm.close()
            
        else:
            print("❌ Online LLM initialization failed")
            
            # Get detailed error info
            stats = online_llm.get_provider_stats()
            if stats.get('last_error'):
                print(f"Error: {stats['last_error']}")
            
            print("🔧 Troubleshooting:")
            print("1. Check internet connection")  
            print("2. Verify API keys are correct (not placeholder values)")
            print("3. Check API key has credits/quota")
            print("4. Try with a different provider")
            
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    try:
        result = asyncio.run(test_online_fix())
        
        print("\n" + "=" * 50)
        if result:
            print("🎉 ONLINE LLM FIX SUCCESSFUL!")
            print("\nNext steps:")
            print("1. Run Pascal: ./run.sh")
            print("2. Test with a query that needs online info: 'what day is it today?'")
        else:
            print("❌ ONLINE LLM STILL HAS ISSUES")
            print("\nPlease fix the API key configuration in .env file")
        print("=" * 50)
        
        return 0 if result else 1
        
    except Exception as e:
        print(f"❌ Test script error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
