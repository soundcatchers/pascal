#!/usr/bin/env python3
"""
Ollama Connection Diagnostic Script
Tests Ollama service and model availability
"""

import asyncio
import sys
import os
import subprocess
import json
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

async def test_ollama_service():
    """Test Ollama service step by step"""
    print("🔧 Ollama Service Diagnostic")
    print("=" * 40)
    
    # Test 1: Check if Ollama service is running
    print("\n1. 🔍 Checking Ollama Service Status:")
    print("-" * 30)
    
    try:
        result = subprocess.run(['systemctl', 'is-active', 'ollama'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and 'active' in result.stdout:
            print("✅ Ollama service is running")
        else:
            print("❌ Ollama command failed")
            print("💡 Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Ollama command timed out")
        return False
    except FileNotFoundError:
        print("❌ Ollama command not found")
        print("💡 Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    except Exception as e:
        print(f"❌ Error running ollama command: {e}")
        return False
    
    # Test 3: Check HTTP API availability
    print("\n3. 🌐 Testing Ollama HTTP API:")
    print("-" * 30)
    
    if not AIOHTTP_AVAILABLE:
        print("❌ aiohttp not available - install with: pip install aiohttp")
        return False
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get('http://localhost:11434/api/version') as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Ollama API responding: version {data.get('version', 'unknown')}")
                else:
                    print(f"❌ Ollama API error: HTTP {response.status}")
                    return False
    except asyncio.TimeoutError:
        print("❌ Ollama API timeout")
        print("💡 Check if Ollama is running: sudo systemctl status ollama")
        return False
    except Exception as e:
        print(f"❌ Ollama API connection error: {e}")
        print("💡 Check if Ollama is running: sudo systemctl status ollama")
        return False
    
    # Test 4: List available models
    print("\n4. 📦 Checking Available Models:")
    print("-" * 30)
    
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get('http://localhost:11434/api/tags') as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    
                    if models:
                        print(f"✅ Found {len(models)} model(s):")
                        for model in models:
                            name = model.get('name', 'unknown')
                            size = model.get('size', 0)
                            size_gb = size / (1024**3) if size > 0 else 0
                            modified = model.get('modified', 'unknown')
                            print(f"  • {name} ({size_gb:.1f}GB) - {modified}")
                        
                        # Check for recommended models
                        model_names = [m.get('name', '') for m in models]
                        recommended = ['nemotron-mini:4b-instruct-q4_K_M', 'nemotron-mini', 'qwen2.5:3b', 'phi3:mini']
                        
                        found_recommended = []
                        for rec in recommended:
                            for model_name in model_names:
                                if rec in model_name or model_name in rec:
                                    found_recommended.append(model_name)
                                    break
                        
                        if found_recommended:
                            print(f"✅ Recommended models available: {found_recommended}")
                        else:
                            print("⚠️ No recommended models found")
                            print("💡 Download Nemotron: ollama pull nemotron-mini:4b-instruct-q4_K_M")
                        
                        return found_recommended  # Return the recommended models found
                    else:
                        print("❌ No models found")
                        print("💡 Download a model: ollama pull nemotron-mini:4b-instruct-q4_K_M")
                        return False
                else:
                    print(f"❌ Failed to list models: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False

async def test_model_loading(model_name: str):
    """Test loading and using a specific model"""
    print(f"\n5. 🧪 Testing Model: {model_name}")
    print("-" * 30)
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            
            # Test model with a simple prompt
            payload = {
                "model": model_name,
                "prompt": "Hello! Please respond with just 'Hi' and nothing else.",
                "stream": False,
                "options": {
                    "num_predict": 5,  # Very short response
                    "temperature": 0.1,
                    "num_ctx": 256
                }
            }
            
            print(f"  Loading model and generating test response...")
            
            async with session.post('http://localhost:11434/api/generate', json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '').strip()
                    
                    if response_text:
                        print(f"  ✅ Model working! Response: '{response_text}'")
                        
                        # Get model info
                        eval_count = data.get('eval_count', 0)
                        eval_duration = data.get('eval_duration', 0)
                        
                        if eval_count > 0 and eval_duration > 0:
                            tokens_per_second = eval_count / (eval_duration / 1e9)
                            print(f"  📊 Performance: {tokens_per_second:.1f} tokens/second")
                        
                        return True
                    else:
                        print(f"  ❌ Model loaded but gave empty response")
                        return False
                else:
                    error_text = await response.text()
                    print(f"  ❌ Model test failed: HTTP {response.status}")
                    print(f"     Error: {error_text[:200]}")
                    return False
                    
    except asyncio.TimeoutError:
        print(f"  ❌ Model test timed out (this can happen on first load)")
        print(f"     The model may still work in Pascal - this is normal for first use")
        return False
    except Exception as e:
        print(f"  ❌ Model test error: {e}")
        return False

async def test_pascal_integration():
    """Test Pascal's offline LLM integration"""
    print(f"\n6. 🔗 Testing Pascal Integration:")
    print("-" * 30)
    
    try:
        # Import Pascal's offline LLM
        from modules.offline_llm import LightningOfflineLLM
        print("  ✅ LightningOfflineLLM imported successfully")
        
        # Create instance
        llm = LightningOfflineLLM()
        print("  ✅ LightningOfflineLLM instance created")
        
        # Initialize
        print("  🔄 Initializing LightningOfflineLLM...")
        success = await llm.initialize()
        
        if success:
            print("  ✅ LightningOfflineLLM initialized successfully")
            
            # Get status
            status = llm.get_status()
            print(f"  📊 Status:")
            print(f"     Available: {status['available']}")
            print(f"     Model loaded: {status['model_loaded']}")
            print(f"     Current model: {status['current_model']}")
            print(f"     Profile: {status['performance_profile']}")
            
            # Test a simple response
            print("  🧪 Testing simple response...")
            response = await llm.generate_response(
                "Say hello in one word only", 
                "Be brief", 
                ""
            )
            print(f"  ✅ Response: '{response[:50]}{'...' if len(response) > 50 else ''}'")
            
            await llm.close()
            return True
        else:
            print("  ❌ LightningOfflineLLM initialization failed")
            print(f"     Last error: {llm.last_error}")
            return False
            
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main diagnostic function"""
    print("🤖 Ollama + Pascal Diagnostic")
    print("=" * 50)
    
    # Test Ollama service
    models_available = await test_ollama_service()
    
    if not models_available:
        print("\n" + "=" * 50)
        print("❌ OLLAMA SERVICE ISSUES DETECTED")
        print("=" * 50)
        print("\n🔧 Common fixes:")
        print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Start service: sudo systemctl start ollama")
        print("3. Enable auto-start: sudo systemctl enable ollama")
        print("4. Download model: ollama pull nemotron-mini:4b-instruct-q4_K_M")
        print("5. Check status: systemctl status ollama")
        return 1
    
    # Test model loading with the first recommended model found
    if isinstance(models_available, list) and models_available:
        model_working = await test_model_loading(models_available[0])
    else:
        model_working = False
    
    # Test Pascal integration
    pascal_working = await test_pascal_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    checks = [
        ("Ollama Service", models_available is not False),
        ("Models Available", bool(models_available)),
        ("Model Loading", model_working),
        ("Pascal Integration", pascal_working)
    ]
    
    passed = 0
    for check_name, result in checks:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    overall_health = (passed / len(checks)) * 100
    print(f"\nOverall Health: {passed}/{len(checks)} ({overall_health:.0f}%)")
    
    if overall_health >= 75:
        print("\n🎉 OLLAMA IS WORKING!")
        print("Your offline LLM should work in Pascal.")
        print("\n🚀 Next steps:")
        print("1. Run Pascal: ./run.sh")
        print("2. Test with: 'Hello Pascal'")
        print("3. Check routing with: 'status' command")
    elif overall_health >= 50:
        print("\n⚡ OLLAMA IS PARTIALLY WORKING")
        print("Some issues detected but may still work.")
    else:
        print("\n⚠️ OLLAMA NEEDS ATTENTION")
        print("Multiple issues detected - check output above.")
        print("\n🔧 Quick fix commands:")
        print("sudo systemctl start ollama")
        print("ollama pull nemotron-mini:4b-instruct-q4_K_M")
    
    return 0 if overall_health >= 50 else 1

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n⏹️ Diagnostic interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1) service is not running")
            print("💡 Try: sudo systemctl start ollama")
            print("💡 Enable auto-start: sudo systemctl enable ollama")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️ Systemctl command timed out")
    except FileNotFoundError:
        print("⚠️ systemctl not found (may not be a systemd system)")
    except Exception as e:
        print(f"⚠️ Error checking service: {e}")
    
    # Test 2: Check if Ollama command is available
    print("\n2. 🔍 Checking Ollama Command:")
    print("-" * 30)
    
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ollama command available: {version}")
        else:
            print("❌ Ollama
