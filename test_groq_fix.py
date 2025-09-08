#!/usr/bin/env python3
"""
Enhanced debug test script for Groq API integration with current models
Updated for 2024 Groq API models and endpoints - FIXED for API key compatibility
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
        
        # Check settings - FIXED: Check both GROQ and GROK for compatibility
        print("\n📋 Settings Check:")
        groq_key = getattr(settings, 'groq_api_key', None)
        print(f"  groq_api_key exists: {groq_key is not None}")
        
        # Check environment variables directly
        env_groq = os.getenv("GROQ_API_KEY")
        env_grok = os.getenv("GROK_API_KEY")  # Legacy support
        
        print(f"  GROQ_API_KEY in env: {env_groq is not None}")
        print(f"  GROK_API_KEY in env: {env_grok is not None}")
        
        if groq_key:
            print(f"  groq_api_key starts with: {groq_key[:15]}...")
            print(f"  groq_api_key length: {len(groq_key)}")
            
            # Check for common invalid values
            invalid_values = ['', 'your_groq_api_key_here', 'gsk-your_groq_api_key_here', 'your_grok_api_key_here']
            if groq_key in invalid_values:
                print("  ❌ API key appears to be placeholder/invalid")
                return False
            elif not groq_key.startswith('gsk-'):
                print("  ⚠️ API key doesn't start with 'gsk-' (expected Groq format)")
                print(f"  Actual start: {groq_key[:10]}...")
            else:
                print("  ✅ API key format looks correct")
        else:
            print("  ❌ No Groq API key found")
            print("\n🔧 API Key Setup:")
            print("  1. Get API key from: https://console.groq.com/")
            print("  2. Add to .env file: GROQ_API_KEY=gsk-your-actual-key")
            print("  3. Make sure .env file is in the pascal directory")
            return False
        
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
                
                # Test current info query that should route to Groq
                print(f"\n🧪 Testing Current Info Query:")
                current_info_payload = {
                    "model": working_model,
                    "messages": [
                        {"role": "user", "content": "What day is today?"}
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
                print("\n🔧 Troubleshooting:")
                print("  1. Verify API key at: https://console.groq.com/")
                print("  2. Check if account has quota/credits")
                print("  3. Try regenerating the API key")
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
                print(f"  Groq config exists: True")
                print(f"  Groq models: {groq_config.get('models', [])}")
                print(f"  Groq default model: {groq_config.get('default_model', 'None')}")
                print(f"  Groq API key configured: {groq_config.get('api_key') is not None}")
        
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
        
        # Check for both GROQ and GROK (legacy support)
        has_groq = 'GROQ_API_KEY=' in content
        has_grok = 'GROK_API_KEY=' in content
        
        if has_groq:
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
                        print(f"  Key starts with: {key_value[:10]}...")
                        return True  # Still might work
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
                        return True
        else:
            print("  ❌ Neither GROQ_API_KEY nor GROK_API_KEY found in .env")
            return False
            
    except Exception as e:
        print(f"  ❌ Error reading .env file: {e}")
        return False
    
    return True

async def test_pascal_router():
    """Test Pascal's routing logic"""
    print("\n🧭 Testing Pascal Router Logic:")
    
    try:
        from modules.router import LightningRouter
        from modules.personality import PersonalityManager
        from modules.memory import MemoryManager
        
        # Create router components
        personality_manager = PersonalityManager()
        memory_manager = MemoryManager()
        router = LightningRouter(personality_manager, memory_manager)
        
        # Initialize
        await router._check_llm_availability()
        
        print(f"  Offline available: {router.offline_available}")
        print(f"  Online available: {router.online_available}")
        print(f"  Router mode: {router.mode.value}")
        
        # Test routing decisions
        test_queries = [
            ("Hello, how are you?", "Should route to offline (simple)"),
            ("What day is today?", "Should route to online (current info)"),
            ("Explain quantum computing", "Should route based on complexity"),
            ("Who is the current Prime Minister?", "Should route to online (current info)")
        ]
        
        for query, expected in test_queries:
            decision = router._decide_route(query)
            route_type = "offline" if decision.use_offline else "online"
            print(f"  Query: '{query}'")
            print(f"    Decision: {route_type} ({decision.reason})")
            print(f"    Expected: {expected}")
            print()
        
        await router.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Router test error: {e}")
        return False

def main():
    """Main test function"""
    try:
        print("🚀 Pascal Groq Integration Test Suite (Fixed)")
        print("=" * 60)
        
        # Test environment
        env_ok = asyncio.run(test_env_file())
        
        if not env_ok:
            print("\n❌ Environment configuration issues found")
            print("\nTo fix:")
            print("1. Create or edit .env file: nano .env")
            print("2. Add your Groq API key: GROQ_API_KEY=gsk-your-actual-key")
            print("3. Get API key from: https://console.groq.com/")
            print("4. If you have GROK_API_KEY, rename it to GROQ_API_KEY")
            return 1
        
        # Test router logic
        router_ok = asyncio.run(test_pascal_router())
        
        # Run main test
        result = asyncio.run(test_groq_debug())
        
        print("\n" + "=" * 60)
        if result:
            print("🎉 GROQ API WORKING PERFECTLY!")
            print("\nYour Groq integration is fully functional with current models.")
            print("Pascal will use Groq as the primary online provider.")
            print("Current information queries will automatically route to Groq.")
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
            print("5. Make sure you're using GROQ_API_KEY not GROK_API_KEY")
            print("6. If issues persist, Pascal will fall back to other providers")
        
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
