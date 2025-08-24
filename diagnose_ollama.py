#!/usr/bin/env python3
"""
Pascal AI Assistant - Ollama Diagnostic Script
Diagnoses connection and model issues
"""

import asyncio
import aiohttp
import json
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

async def check_ollama_service():
    """Check if Ollama service is running"""
    print("🔍 Checking Ollama service status...")
    
    # Check systemctl status
    import subprocess
    try:
        result = subprocess.run(['systemctl', 'status', 'ollama'], 
                              capture_output=True, text=True, timeout=5)
        if 'active (running)' in result.stdout:
            print("✅ Ollama service is running")
            return True
        else:
            print("❌ Ollama service is not running")
            print("   Run: sudo systemctl start ollama")
            return False
    except Exception as e:
        print(f"⚠️ Could not check service status: {e}")
        return None

async def check_ollama_api():
    """Check if Ollama API is responding"""
    print("\n🔍 Checking Ollama API connection...")
    
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    async with aiohttp.ClientSession() as session:
        # Check version endpoint
        try:
            async with session.get(f"{ollama_host}/api/version") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Ollama API is responding")
                    print(f"   Version: {data.get('version', 'unknown')}")
                    return True
                else:
                    print(f"❌ Ollama API returned status {response.status}")
                    return False
        except aiohttp.ClientError as e:
            print(f"❌ Cannot connect to Ollama API at {ollama_host}")
            print(f"   Error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

async def check_models():
    """Check available models"""
    print("\n🔍 Checking available models...")
    
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{ollama_host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    
                    if not models:
                        print("❌ No models found!")
                        print("   Run: ./download_models.sh")
                        return []
                    
                    print(f"✅ Found {len(models)} model(s):")
                    
                    # Check for recommended models
                    recommended = [
                        "nemotron-mini:4b-instruct-q4_K_M",
                        "qwen3:4b-instruct",
                        "gemma3:4b-it-q4_K_M",
                        "phi3:mini",
                        "llama3.2:3b",
                        "gemma2:2b"
                    ]
                    
                    found_models = []
                    for model in models:
                        name = model.get('name', '')
                        size = model.get('size', 0)
                        size_gb = size / (1024**3)
                        found_models.append(name)
                        
                        is_recommended = any(rec in name for rec in recommended)
                        marker = "⭐" if is_recommended else "  "
                        print(f"   {marker} {name} ({size_gb:.1f}GB)")
                    
                    # Check if any recommended models are missing
                    missing_recommended = []
                    for rec in recommended[:3]:  # Check top 3 recommendations
                        if not any(rec in model for model in found_models):
                            missing_recommended.append(rec)
                    
                    if missing_recommended:
                        print("\n⚠️ Recommended models not found:")
                        for model in missing_recommended:
                            print(f"   • {model}")
                        print("   Install with: ollama pull [model-name]")
                    
                    return found_models
                else:
                    print(f"❌ Failed to get models: status {response.status}")
                    return []
        except Exception as e:
            print(f"❌ Error checking models: {e}")
            return []

async def test_model_loading(model_name: str):
    """Test loading a specific model"""
    print(f"\n🔍 Testing model loading: {model_name}")
    
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    async with aiohttp.ClientSession() as session:
        try:
            payload = {
                "model": model_name,
                "prompt": "",
                "stream": False,
                "keep_alive": "1m"
            }
            
            print(f"   Loading model...")
            async with session.post(
                f"{ollama_host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    print(f"✅ Model {model_name} loaded successfully")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to load model: {error_text[:200]}")
                    return False
        except asyncio.TimeoutError:
            print(f"❌ Timeout loading model (>30s)")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

async def test_generation(model_name: str):
    """Test generating a response"""
    print(f"\n🔍 Testing response generation with {model_name}")
    
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    async with aiohttp.ClientSession() as session:
        try:
            payload = {
                "model": model_name,
                "prompt": "Say 'Hello, Pascal is working!' in exactly 5 words.",
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_predict": 20
                }
            }
            
            print(f"   Generating response...")
            start_time = asyncio.get_event_loop().time()
            
            async with session.post(
                f"{ollama_host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    elapsed = asyncio.get_event_loop().time() - start_time
                    
                    response_text = data.get('response', '')
                    print(f"✅ Response generated in {elapsed:.2f}s")
                    print(f"   Response: {response_text[:100]}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Generation failed: {error_text[:200]}")
                    return False
        except asyncio.TimeoutError:
            print(f"❌ Generation timeout (>30s)")
            return False
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return False

async def check_environment():
    """Check environment configuration"""
    print("\n🔍 Checking environment configuration...")
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file exists")
        
        # Check for API keys
        with open(env_file, 'r') as f:
            content = f.read()
            
        has_grok = "GROK_API_KEY=" in content and "your_grok_api_key_here" not in content
        has_openai = "OPENAI_API_KEY=" in content and "your_openai_api_key_here" not in content
        has_anthropic = "ANTHROPIC_API_KEY=" in content and "your_anthropic_api_key_here" not in content
        
        if has_grok or has_openai or has_anthropic:
            print("✅ Online API keys configured:")
            if has_grok:
                print("   • Grok API")
            if has_openai:
                print("   • OpenAI API")
            if has_anthropic:
                print("   • Anthropic API")
        else:
            print("ℹ️ No online API keys configured (offline-only mode)")
    else:
        print("⚠️ .env file not found")
        print("   Copy .env.example to .env and configure if needed")
    
    # Check Ollama host
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    print(f"\nOllama host: {ollama_host}")

async def run_diagnostics():
    """Run all diagnostic checks"""
    print("=" * 60)
    print("🤖 Pascal AI - Ollama Diagnostics")
    print("=" * 60)
    
    # Check environment
    await check_environment()
    
    # Check Ollama service
    service_ok = await check_ollama_service()
    
    # Check API connection
    api_ok = await check_ollama_api()
    
    if not api_ok:
        print("\n❌ Cannot proceed without Ollama API connection")
        print("\nTroubleshooting steps:")
        print("1. Start Ollama: sudo systemctl start ollama")
        print("2. Check status: sudo systemctl status ollama")
        print("3. Check logs: sudo journalctl -u ollama -n 50")
        return False
    
    # Check models
    models = await check_models()
    
    if not models:
        print("\n❌ No models available")
        print("\nInstall models with:")
        print("1. Run installer: ./download_models.sh")
        print("2. Or manually: ollama pull phi3:mini")
        return False
    
    # Test first available model
    test_model = models[0] if models else None
    
    if test_model:
        # Test loading
        load_ok = await test_model_loading(test_model)
        
        if load_ok:
            # Test generation
            gen_ok = await test_generation(test_model)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Diagnostic Summary:")
    print("=" * 60)
    
    all_ok = True
    
    if service_ok:
        print("✅ Ollama service: Running")
    else:
        print("❌ Ollama service: Not running")
        all_ok = False
    
    if api_ok:
        print("✅ Ollama API: Connected")
    else:
        print("❌ Ollama API: Not connected")
        all_ok = False
    
    if models:
        print(f"✅ Models: {len(models)} available")
    else:
        print("❌ Models: None available")
        all_ok = False
    
    if all_ok:
        print("\n✅ Everything looks good! Pascal should work.")
        print("\nRun Pascal with: ./run.sh")
    else:
        print("\n❌ Some issues need to be fixed.")
        print("\nFor help, check:")
        print("• README.md for setup instructions")
        print("• GitHub issues for common problems")
    
    return all_ok

async def main():
    """Main entry point"""
    try:
        success = await run_diagnostics()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDiagnostics interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Diagnostic error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
