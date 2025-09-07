#!/usr/bin/env python3
"""
Enhanced debug test script for Groq API integration with current models
Updated for 2024 Groq API models and endpoints
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
    """Test Groq with detailed debugging using current models"""
    print("🔍 Enhanced Groq API Debug Test (2024 Models)")
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
            
            # Check for common invalid values
            invalid_values = ['', 'your_groq_api_key_here', 'gsk-your_groq_api_key_here']
            if groq_key in invalid_values:
                print("  ❌ API key appears to be placeholder/invalid")
                return False
            elif not groq_key.startswith('gsk-'):
                print("  ⚠️ API key doesn't start with 'gsk-' (expected Groq format)")
            else:
                print("  ✅ API key format looks correct")
        
        # Test direct Groq API call with current models
        print("\n🔧 Testing Direct Groq API Call with Current Models:")
        
        session = aiohttp.ClientSession()
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {groq_key}'
            }
            
            # Current Groq models as of 2024 (in priority order)
            models_to_test = [
                'llama-3.1-8b-instant',       # Fast and reliable
                'llama-3.1-70b-versatile',    # High quality
                'llama-3.2-11b-text-preview', # Balanced option
                'llama-3.2-90b-text-preview', # Highest quality
                'gemma2-9b-it',               # Google model on Groq
                'mixtral-8x7b-32768'          # Fallback if available
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
                            return False  # No point testing other models with bad auth
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
                
                # Test streaming with working model
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
                return False
                
        finally:
            await session.close()
        
        # Now test the OnlineLLM class
        print("\n🔄 Testing OnlineLLM Class:")
        online_llm = OnlineLLM()
        
        # Check initialization state
        print(f"  api_configs initialized: {hasattr(online_llm, 'api_configs')}")
        if hasattr(online_llm, 'api_configs'):
            print(f"  Number of configs: {len(online_llm.api_configs)}")
            
            # Check if Groq config exists
            if APIProvider.GROQ in online_llm.api_configs:
                groq_config = online_llm.api_configs[APIProvider.GROQ]
                print(f"  Groq config exists: {groq_config}")
                print(f"  Groq models: {groq_config.get('models', [])}")
                print(f"  Groq default model: {groq_config.get('default_model', 'None')}")
        
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
                        
                        # Test streaming
                        print(f"\n🌊 Testing Groq Streaming:")
                        print("  Streaming response: ", end='')
                        async for chunk in online_llm.generate_response_stream(
                            "Count from 1 to 3",
                            "You are a helpful assistant.",
                            ""
                        ):
                            print(chunk, end='')
                        print("\n  ✅ Streaming test complete!")
                        
                        await online_llm.close()
                        return True
                    else:
                        print("  ❌ Groq failed to generate proper response")
                        
                except Exception as e:
                    print(f"  ❌ Generation error: {e}")
                    import traceback
                    traceback.print_exc()
        
        await online_llm.close()
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return False

async def test_env_file():
    """Test .env file configuration"""
    print("\n📁 Testing .env File Configuration:")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("  ❌ .env file not found")
        print("  Create .env file with your Groq API key:")
        print("  GROQ_API_KEY=gsk-your-actual-key-here")
        return False
    
    print("  ✅ .env file exists")
    
    # Read and check content
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        if 'GROQ_API_KEY=' in content:
            print("  ✅ GROQ_API_KEY found in .env")
            
            # Extract the key
            for line in content.split('\n'):
                if line.startswith('GROQ_API_KEY='):
                    key_value = line.split('=', 1)[1].strip()
                    if key_value in ['', 'your_groq_api_key_here', 'gsk-your_groq_api_key_here']:
                        print("  ❌ GROQ_API_KEY is placeholder/empty")
                        return False
                    elif key_value.startswith('gsk-'):
                        print("  ✅ GROQ_API_KEY format looks correct")
                        return True
                    else:
                        print("  ⚠️ GROQ_API_KEY doesn't start with 'gsk-'")
                        return True  # Still might work
        else:
            print("  ❌ GROQ_API_KEY not found in .env")
            return False
            
    except Exception as e:
        print(f"  ❌ Error reading .env file: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    try:
        print("🚀 Pascal Groq Integration Test Suite")
        print("=" * 50)
        
        # Test environment
        env_ok = asyncio.run(test_env_file())
        
        if not env_ok:
            print("\n❌ Environment configuration issues found")
            print("\nTo fix:")
            print("1. Create or edit .env file: nano .env")
            print("2. Add your Groq API key: GROQ_API_KEY=gsk-your-actual-key")
            print("3. Get API key from: https://console.groq.com/")
            return 1
        
        # Run main test
        result = asyncio.run(test_groq_debug())
        
        print("\n" + "=" * 50)
        if result:
            print("🎉 GROQ API WORKING PERFECTLY!")
            print("\nYour Groq integration is fully functional with current models.")
            print("Pascal will use Groq as the primary online provider.")
            print("\nRun Pascal with: ./run.sh")
        else:
            print("❌ GROQ API ISSUES DETECTED")
            print("\nPossible issues:")
            print("1. API key might be invalid or expired")
            print("2. Account might be rate limited or out of credits")
            print("3. Network connectivity issues")
            print("4. Models might have changed (this test uses 2024 models)")
            print("\nTroubleshooting steps:")
            print("1. Check your Groq console: https://console.groq.com/")
            print("2. Verify your API key is active and has credits")
            print("3. Try regenerating your API key")
            print("4. Check your internet connection")
            print("5. If issues persist, Pascal will fall back to other providers")
        
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
