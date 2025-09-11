#!/usr/bin/env python3
"""
FIXED: Enhanced debug test script for Groq API integration
Updated for new gsk_ API key format (gsk- is deprecated)
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
    """Test Groq with detailed debugging using current API key format"""
    print("🔍 FIXED: Groq API Debug Test (New gsk_ Format)")
    print("=" * 50)
    
    try:
        # Check aiohttp availability
        try:
            import aiohttp
            print("✅ aiohttp is installed")
            print(f"   Version: {aiohttp.__version__}")
        except ImportError:
            print("❌ aiohttp is NOT installed!")
            print("   Install with: pip install aiohttp")
            return False
        
        # Import Pascal modules
        from config.settings import settings
        from modules.online_llm import OnlineLLM, APIProvider
        
        # Enable debug mode
        settings.debug_mode = True
        
        # Check settings - FIXED: Check for proper gsk_ format
        print("\n📋 Settings Check:")
        groq_key = getattr(settings, 'groq_api_key', None)
        print(f"  groq_api_key exists: {groq_key is not None}")
        
        # Check environment variables directly
        env_groq = os.getenv("GROQ_API_KEY")
        env_grok = os.getenv("GROK_API_KEY")  # Legacy support
        
        print(f"  GROQ_API_KEY in env: {env_groq is not None}")
        print(f"  GROK_API_KEY in env (legacy): {env_grok is not None}")
        
        if groq_key:
            print(f"  groq_api_key starts with: {groq_key[:15]}...")
            print(f"  groq_api_key length: {len(groq_key)}")
            
            # FIXED: Check for new gsk_ format specifically
            invalid_values = ['', 'your_groq_api_key_here', 'gsk_your_groq_api_key_here', 'your_grok_api_key_here']
            if groq_key in invalid_values:
                print("  ❌ API key appears to be placeholder/invalid")
                return False
            elif groq_key.startswith('gsk_'):
                print("  ✅ API key format is correct (gsk_)")
            elif groq_key.startswith('gsk-'):
                print("  ⚠️ API key uses deprecated format (gsk-)")
                print("  Consider updating to new gsk_ format")
            else:
                print("  ❌ API key format is invalid")
                print(f"  Expected: gsk_ or gsk-, got: {groq_key[:10]}...")
                return False
        else:
            print("  ❌ No Groq API key found")
            print("\n🔧 API Key Setup:")
            print("  1. Get API key from: https://console.groq.com/")
            print("  2. Add to .env file: GROQ_API_KEY=gsk_your-actual-key")
            print("  3. Make sure .env file is in the pascal directory")
            print("  4. NEW KEYS start with gsk_ (underscore)")
            return False
        
        # Test direct Groq API call with current models
        print("\n🔧 Testing Direct Groq API Call:")
        
        session = aiohttp.ClientSession()
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {groq_key}'
            }
            
            # Current Groq models (updated for 2024/2025)
            models_to_test = [
                'llama-3.1-8b-instant',       # Primary: Fast and reliable
                'llama-3.1-70b-versatile',    # High quality
                'llama-3.2-11b-text-preview', # Balanced option
                'llama-3.2-90b-text-preview', # Highest quality
                'gemma2-9b-it',               # Google model on Groq
                'mixtral-8x7b-32768'          # Alternative model
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
                                print(f"    Response: {data}")
                        elif response.status == 401:
                            print(f"    ❌ Authentication failed - check API key")
                            error_data = await response.text()
                            print(f"    Error: {error_data[:200]}")
                            return False  # Bad auth, no point testing other models
                        elif response.status == 429:
                            print(f"    ⚠️ Rate limited or quota exceeded")
                            error_data = await response.text()
                            print(f"    Error: {error_data[:200]}")
                        elif response.status == 400:
                            print(f"    ❌ Bad request - model may not exist")
                            error_data = await response.text()
                            print(f"    Error: {error_data[:200]}")
                        else:
                            error_data = await response.text()
                            print(f"    ❌ Error {response.status}: {error_data[:200]}")
                            
                except asyncio.TimeoutError:
                    print(f"    ❌ Request timed out")
                except Exception as e:
                    print(f"    ❌ Exception: {str(e)[:100]}")
            
            if working_model:
                print(f"\n✅ Found working Groq model: {working_model}")
                
                # Test current info query
                print(f"\n🧪 Testing Current Info Query:")
                current_info_payload = {
                    "model": working_model,
                    "messages": [
                        {"role": "user", "content": "What day is today? Respond with the current day."}
                    ],
                    "max_tokens": 50,
                    "temperature": 0.1,
                    "stream": False
                }
                
                try:
                    async with session.post(
                        'https://api.groq.com/openai/v1/chat/completions',
                        headers=headers,
                        json=current_info_payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'choices' in data and data['choices']:
                                content = data['choices'][0]['message']['content']
                                print(f"    ✅ Current info response: {content}")
                            else:
                                print(f"    ⚠️ Unexpected response format")
                        else:
                            print(f"    ❌ Current info test failed: {response.status}")
                            
                except Exception as e:
                    print(f"    ❌ Current info test error: {str(e)}")
                
                # Test streaming
                print(f"\n🔄 Testing streaming with {working_model}:")
                
                stream_payload = {
                    "model": working_model,
                    "messages": [
                        {"role": "user", "content": "Count from 1 to 5"}
                    ],
                    "max_tokens": 20,
                    "temperature": 0.1,
                    "stream": True
                }
                
                try:
                    async with session.post(
                        'https://api.groq.com/openai/v1/chat/completions',
                        headers=headers,
                        json=stream_payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            print("    ✅ Streaming response:")
                            chunks_received = 0
                            async for line in response.content:
                                if line:
                                    line_str = line.decode('utf-8').strip()
                                    if line_str.startswith('data: '):
                                        line_str = line_str[6:]
                                        if line_str == '[DONE]':
                                            break
                                        try:
                                            import json
                                            data = json.loads(line_str)
                                            if 'choices' in data and data['choices']:
                                                delta = data['choices'][0].get('delta', {})
                                                if 'content' in delta:
                                                    print(f"    Chunk: {delta['content']}", end='')
                                                    chunks_received += 1
                                        except json.JSONDecodeError:
                                            continue
                            print(f"\n    ✅ Received {chunks_received} chunks")
                        else:
                            print(f"    ❌ Streaming failed: {response.status}")
                            
                except Exception as e:
                    print(f"    ❌ Streaming error: {str(e)}")
            else:
                print("\n❌ No Groq models worked with this API key")
                print("\n🔧 Troubleshooting:")
                print("  1. Verify API key at: https://console.groq.com/")
                print("  2. Check if account has quota/credits")
                print("  3. Try regenerating the API key")
                print("  4. Ensure key format is gsk_ (not gsk-)")
                return False
                
        finally:
            await session.close()
        
        # Test OnlineLLM class
        print("\n🔄 Testing OnlineLLM Class:")
        online_llm = OnlineLLM()
        
        # Check initialization
        success = await online_llm.initialize()
        
        print(f"\n  Initialization result: {success}")
        print(f"  Last error: {online_llm.last_error}")
        
        if success:
            # Get stats
            stats = online_llm.get_provider_stats()
            print(f"\n📊 Provider Statistics:")
            print(f"  Available providers: {stats['available_providers']}")
            print(f"  Preferred provider: {stats['preferred_provider']}")
            
            # Show detailed Groq stats
            if 'providers' in stats and 'groq' in stats['providers']:
                groq_stats = stats['providers']['groq']
                print(f"\n🚀 Groq Statistics:")
                print(f"  Available: {groq_stats.get('available', False)}")
                print(f"  API key configured: {groq_stats.get('api_key_configured', False)}")
                print(f"  Current model: {groq_stats.get('current_model', 'Unknown')}")
                print(f"  Success count: {groq_stats.get('success_count', 0)}")
                print(f"  Failure count: {groq_stats.get('failure_count', 0)}")
            
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
                        
                        # Test current info routing
                        print(f"\n🧪 Testing Current Info Routing:")
                        current_info_response = await online_llm.generate_response(
                            "What day is today?",
                            "You are a helpful assistant.",
                            ""
                        )
                        print(f"  Current info response: {current_info_response}")
                        
                        # Test streaming
                        print(f"\n🌊 Testing Groq Streaming:")
                        print("  Streaming response: ", end='')
                        stream_response = ""
                        async for chunk in online_llm.generate_response_stream(
                            "Count from 1 to 3",
                            "You are a helpful assistant.",
                            ""
                        ):
                            print(chunk, end='')
                            stream_response += chunk
                        print(f"\n  ✅ Streaming test complete! Total: {len(stream_response)} chars")
                        
                        await online_llm.close()
                        return True
                    else:
                        print("  ❌ Groq failed to generate proper response")
                        print(f"  Response was: {response}")
                        
                except Exception as e:
                    print(f"  ❌ Generation error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("  ❌ Groq not in available providers")
                print(f"  Available: {stats['available_providers']}")
        
        await online_llm.close()
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return False

async def test_env_file():
    """Test .env file configuration for new API key format"""
    print("\n📁 Testing .env File Configuration:")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("  ❌ .env file not found")
        print("  Create .env file with your Groq API key:")
        print("  GROQ_API_KEY=gsk_your-actual-key-here")
        return False
    
    print("  ✅ .env file exists")
    
    # Read and check content
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Check for GROQ_API_KEY (primary)
        has_groq = 'GROQ_API_KEY=' in content
        has_grok = 'GROK_API_KEY=' in content  # Legacy
        
        if has_groq:
            print("  ✅ GROQ_API_KEY found in .env")
            
            # Extract the key
            for line in content.split('\n'):
                if line.startswith('GROQ_API_KEY='):
                    key_value = line.split('=', 1)[1].strip()
                    if key_value in ['', 'your_groq_api_key_here', 'gsk_your_groq_api_key_here']:
                        print("  ❌ GROQ_API_KEY is placeholder/empty")
                        return False
                    elif key_value.startswith('gsk_'):
                        print("  ✅ GROQ_API_KEY format is correct (gsk_)")
                        return True
                    elif key_value.startswith('gsk-'):
                        print("  ⚠️ GROQ_API_KEY uses deprecated format (gsk-)")
                        print("  Consider updating to new gsk_ format")
                        return True  # Still works
                    else:
                        print("  ❌ GROQ_API_KEY format is invalid")
                        print(f"  Key starts with: {key_value[:10]}...")
                        return False
        elif has_grok:
            print("  ⚠️ Found GROK_API_KEY (legacy) - should rename to GROQ_API_KEY")
            
            # Extract the key
            for line in content.split('\n'):
                if line.startswith('GROK_API_KEY='):
                    key_value = line.split('=', 1)[1].strip()
                    if key_value in ['', 'your_grok_api_key_here']:
                        print("  ❌ GROK_API_KEY is placeholder/empty")
                        return False
                    else:
                        print("  ✅ GROK_API_KEY has value (will work as fallback)")
                        print("  📝 Rename GROK_API_KEY to GROQ_API_KEY in .env")
                        return True
        else:
            print("  ❌ Neither GROQ_API_KEY nor GROK_API_KEY found in .env")
            return False
            
    except Exception as e:
        print(f"  ❌ Error reading .env file: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    try:
        print("🚀 Pascal Groq Integration Test Suite (FIXED for gsk_)")
        print("=" * 60)
        
        # Test environment
        env_ok = asyncio.run(test_env_file())
        
        if not env_ok:
            print("\n❌ Environment configuration issues found")
            print("\nTo fix:")
            print("1. Create or edit .env file: nano .env")
            print("2. Add your Groq API key: GROQ_API_KEY=gsk_your-actual-key")
            print("3. Get API key from: https://console.groq.com/")
            print("4. IMPORTANT: New keys start with gsk_ (underscore)")
            print("5. If you have GROK_API_KEY, rename it to GROQ_API_KEY")
            return 1
        
        # Run main test
        result = asyncio.run(test_groq_debug())
        
        print("\n" + "=" * 60)
        if result:
            print("🎉 GROQ API WORKING PERFECTLY!")
            print("\nYour Groq integration is fully functional with new gsk_ format.")
            print("Pascal will use Groq as the primary online provider.")
            print("Current information queries will automatically route to Groq.")
            print("\nRun Pascal with: ./run.sh")
        else:
            print("❌ GROQ API ISSUES DETECTED")
            print("\nPossible issues:")
            print("1. API key might be invalid or expired")
            print("2. Account might be rate limited or out of credits")
            print("3. API key format issue (must start with gsk_)")
            print("4. Network connectivity issues")
            print("\nTroubleshooting steps:")
            print("1. Check your Groq console: https://console.groq.com/")
            print("2. Verify your API key is active and has credits")
            print("3. Try regenerating your API key (new format: gsk_)")
            print("4. Check your internet connection")
            print("5. Make sure you're using GROQ_API_KEY not GROK_API_KEY")
            print("6. Ensure key starts with gsk_ (underscore, not dash)")
        
        return 0 if result else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
