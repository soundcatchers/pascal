#!/usr/bin/env python3
"""
Enhanced debug test script for Groq API integration
"""

import asyncio
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode
os.environ['DEBUG'] = 'true'

async def test_groq_debug():
    """Test Groq with detailed debugging"""
    print("🔍 Enhanced Groq API Debug Test")
    print("=" * 50)
    
    try:
        # First check if aiohttp is installed
        try:
            import aiohttp
            print("✅ aiohttp is installed")
            print(f"   Version: {aiohttp.__version__}")
        except ImportError:
            print("❌ aiohttp is NOT installed!")
            print("   Install with: pip install aiohttp")
            return False
        
        # Import modules
        from config.settings import settings
        from modules.online_llm import OnlineLLM, APIProvider
        
        # Enable debug mode
        settings.debug_mode = True
        
        # Check settings
        print("\n📋 Settings Check:")
        groq_key = getattr(settings, 'groq_api_key', None)
        print(f"  groq_api_key exists: {groq_key is not None}")
        if groq_key:
            print(f"  groq_api_key starts with: {groq_key[:15]}...")
            print(f"  groq_api_key length: {len(groq_key)}")
        
        # Test direct Groq API call
        print("\n🔧 Testing Direct Groq API Call:")
        
        session = aiohttp.ClientSession()
        try:
            # Test with the Groq API directly
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {groq_key}'
            }
            
            # List of models to try
            models_to_test = [
                'llama3-8b-8192',
                'llama3-70b-8192', 
                'mixtral-8x7b-32768',
                'gemma-7b-it'
            ]
            
            working_model = None
            
            for model in models_to_test:
                print(f"\n  Testing model: {model}")
                
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "Say 'test successful' and nothing else"}
                    ],
                    "max_tokens": 10,
                    "temperature": 0.1,
                    "stream": False
                }
                
                try:
                    async with session.post(
                        'https://api.groq.com/openai/v1/chat/completions',
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        print(f"    Response status: {response.status}")
                        
                        if response.status == 200:
                            data = await response.json()
                            if 'choices' in data and data['choices']:
                                content = data['choices'][0]['message']['content']
                                print(f"    ✅ Model works! Response: {content}")
                                working_model = model
                                break
                            else:
                                print(f"    ⚠️ Unexpected response format")
                        elif response.status == 401:
                            print(f"    ❌ Authentication failed - check API key")
                            error_data = await response.text()
                            print(f"    Error: {error_data[:200]}")
                        elif response.status == 429:
                            print(f"    ❌ Rate limited or quota exceeded")
                        else:
                            error_data = await response.text()
                            print(f"    ❌ Error: {error_data[:200]}")
                            
                except asyncio.TimeoutError:
                    print(f"    ❌ Request timed out")
                except Exception as e:
                    print(f"    ❌ Exception: {str(e)[:100]}")
            
            if working_model:
                print(f"\n✅ Found working Groq model: {working_model}")
            else:
                print("\n❌ No Groq models worked with this API key")
                
        finally:
            await session.close()
        
        # Now test the OnlineLLM class
        print("\n🔄 Testing OnlineLLM Class:")
        online_llm = OnlineLLM()
        
        # Check initialization state
        print(f"  api_configs initialized: {hasattr(online_llm, 'api_configs')}")
        print(f"  Number of configs: {len(online_llm.api_configs) if hasattr(online_llm, 'api_configs') else 0}")
        
        # Initialize
        success = await online_llm.initialize()
        
        print(f"\n  Initialization result: {success}")
        print(f"  Last error: {online_llm.last_error}")
        
        if success:
            # Get stats
            stats = online_llm.get_provider_stats()
            print(f"\n📊 Provider Statistics:")
            print(f"  Available providers: {stats['available_providers']}")
            print(f"  Preferred provider: {stats['preferred_provider']}")
            
            # Test actual generation if Groq is available
            if 'groq' in stats['available_providers']:
                print(f"\n🧪 Testing Groq Response Generation:")
                try:
                    response = await online_llm.generate_response(
                        "What is 2+2? Answer with just the number.",
                        "You are a helpful assistant.",
                        ""
                    )
                    print(f"  Response: {response}")
                    
                    if response and not response.startswith("I'm having trouble"):
                        print("  ✅ Groq response generation successful!")
                        return True
                    else:
                        print("  ❌ Groq failed to generate proper response")
                        
                except Exception as e:
                    print(f"  ❌ Generation error: {e}")
        
        await online_llm.close()
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return False

def main():
    """Main test function"""
    try:
        result = asyncio.run(test_groq_debug())
        
        print("\n" + "=" * 50)
        if result:
            print("🎉 GROQ API WORKING!")
            print("\nYour Groq integration is functional.")
            print("Run Pascal with: ./run.sh")
        else:
            print("❌ GROQ API ISSUES REMAIN")
            print("\nPossible issues:")
            print("1. API key might not have access to standard models")
            print("2. Account might be rate limited")
            print("3. Network connectivity issues")
            print("\nTry:")
            print("1. Check your Groq dashboard: https://console.groq.com/")
            print("2. Verify your API key and available models")
            print("3. Check if you have remaining credits")
        
        return 0 if result else 1
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
