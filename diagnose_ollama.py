#!/usr/bin/env python3
"""
Pascal AI Assistant - Ollama Diagnostic Script (Standalone)
Diagnoses connection and model issues without external dependencies
"""

import subprocess
import json
import sys
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

def check_ollama_service():
    """Check if Ollama service is running"""
    print("🔍 Checking Ollama service status...")
    
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
    except subprocess.TimeoutExpired:
        print("⚠️ Timeout checking service status")
        return None
    except FileNotFoundError:
        print("⚠️ systemctl not found - checking if Ollama is running via API")
        return None
    except Exception as e:
        print(f"⚠️ Could not check service status: {e}")
        return None

def check_ollama_api():
    """Check if Ollama API is responding"""
    print("\n🔍 Checking Ollama API connection...")
    
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        # Test version endpoint
        url = f"{ollama_host}/api/version"
        req = urllib.request.Request(url)
        
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                print(f"✅ Ollama API is responding")
                print(f"   Version: {data.get('version', 'unknown')}")
                return True
            else:
                print(f"❌ Ollama API returned status {response.status}")
                return False
                
    except urllib.error.URLError as e:
        print(f"❌ Cannot connect to Ollama API at {ollama_host}")
        print(f"   Error: {e}")
        print("\n   Possible fixes:")
        print("   1. Start Ollama: sudo systemctl start ollama")
        print("   2. Check if Ollama is installed: ollama --version")
        print("   3. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_models():
    """Check available models using ollama CLI"""
    print("\n🔍 Checking available models...")
    
    # First try using ollama CLI
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Skip header line
                models = []
                print(f"✅ Found models via CLI:")
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            models.append(model_name)
                            # Check if it's a recommended model
                            recommended = ["phi", "llama", "gemma", "qwen", "nemotron"]
                            is_rec = any(rec in model_name.lower() for rec in recommended)
                            marker = "⭐" if is_rec else "  "
                            print(f"   {marker} {line}")
                
                if not models:
                    print("❌ No models found!")
                    print("   Install models with: ./download_models.sh")
                    print("   Or manually: ollama pull phi3:mini")
                    
                return models
            else:
                print("❌ No models found via CLI")
                
    except FileNotFoundError:
        print("⚠️ Ollama CLI not found - trying API")
    except subprocess.TimeoutExpired:
        print("⚠️ Timeout getting models via CLI - trying API")
    except Exception as e:
        print(f"⚠️ Error using ollama CLI: {e}")
    
    # Fallback to API
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        url = f"{ollama_host}/api/tags"
        req = urllib.request.Request(url)
        
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                models = data.get('models', [])
                
                if not models:
                    print("❌ No models found via API!")
                    print("   Install models with: ./download_models.sh")
                    print("   Or manually: ollama pull phi3:mini")
                    return []
                
                print(f"✅ Found {len(models)} model(s) via API:")
                
                found_models = []
                for model in models:
                    name = model.get('name', '')
                    size = model.get('size', 0)
                    size_gb = size / (1024**3) if size > 0 else 0
                    found_models.append(name)
                    
                    # Check if recommended
                    recommended = ["phi", "llama", "gemma", "qwen", "nemotron"]
                    is_rec = any(rec in name.lower() for rec in recommended)
                    marker = "⭐" if is_rec else "  "
                    print(f"   {marker} {name} ({size_gb:.1f}GB)")
                
                return found_models
            else:
                print(f"❌ Failed to get models: status {response.status}")
                return []
                
    except Exception as e:
        print(f"❌ Error checking models via API: {e}")
        return []

def test_model_generation(model_name=None):
    """Test generating a response using ollama CLI"""
    if not model_name:
        # Try to get first available model
        models = check_models()
        if models:
            model_name = models[0]
        else:
            print("\n❌ No models available to test")
            return False
    
    print(f"\n🔍 Testing response generation with {model_name}")
    
    try:
        # Use ollama run with a simple prompt
        test_prompt = "Say hello in 5 words or less"
        print(f"   Sending test prompt: '{test_prompt}'")
        
        start_time = time.time()
        
        # Use echo to pipe the prompt to ollama
        result = subprocess.run(
            f'echo "{test_prompt}" | ollama run {model_name}',
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0 and result.stdout:
            response = result.stdout.strip()
            print(f"✅ Response generated in {elapsed:.2f}s")
            print(f"   Response: {response[:100]}")
            return True
        else:
            print(f"❌ Generation failed")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Generation timeout (>30s)")
        print("   Model may be too large for your system")
        return False
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def check_environment():
    """Check environment configuration"""
    print("\n🔍 Checking environment configuration...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"⚠️ Python {python_version.major}.{python_version.minor} (3.8+ recommended)")
    
    # Check if we're in pascal directory
    current_dir = Path.cwd()
    if current_dir.name == 'pascal':
        print("✅ In pascal directory")
    else:
        print(f"⚠️ Not in pascal directory (current: {current_dir})")
    
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
        print("   Copy .env.example to .env if needed")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("ℹ️ Not in virtual environment")
        print("   Activate with: source venv/bin/activate")
    
    # Check Ollama host setting
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    print(f"\nOllama host: {ollama_host}")

def check_system_resources():
    """Check system resources"""
    print("\n🔍 Checking system resources...")
    
    # Check RAM
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    total_kb = int(line.split()[1])
                    total_gb = total_kb / (1024 * 1024)
                    
                    if total_gb >= 8:
                        print(f"✅ RAM: {total_gb:.1f}GB (Good for most models)")
                    elif total_gb >= 4:
                        print(f"⚠️ RAM: {total_gb:.1f}GB (Use smaller models)")
                    else:
                        print(f"❌ RAM: {total_gb:.1f}GB (May struggle with LLMs)")
                    break
    except Exception as e:
        print(f"⚠️ Could not check RAM: {e}")
    
    # Check disk space
    try:
        result = subprocess.run(['df', '-h', '.'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 4:
                    available = parts[3]
                    print(f"✅ Disk space available: {available}")
    except Exception as e:
        print(f"⚠️ Could not check disk space: {e}")
    
    # Check CPU
    try:
        cpu_count = os.cpu_count()
        print(f"✅ CPU cores: {cpu_count}")
    except Exception:
        pass

def check_ollama_installation():
    """Check if Ollama is installed"""
    print("\n🔍 Checking Ollama installation...")
    
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Ollama is installed")
            print(f"   {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama not properly installed")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found in PATH")
        print("\nTo install Ollama:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    except Exception as e:
        print(f"⚠️ Error checking Ollama: {e}")
        return False

def run_diagnostics():
    """Run all diagnostic checks"""
    print("=" * 60)
    print("🤖 Pascal AI - Ollama Diagnostics (Standalone)")
    print("=" * 60)
    
    # Check environment first
    check_environment()
    
    # Check system resources
    check_system_resources()
    
    # Check Ollama installation
    ollama_installed = check_ollama_installation()
    
    if not ollama_installed:
        print("\n❌ Ollama is not installed!")
        print("\nInstall Ollama first:")
        print("1. Run: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Then run this diagnostic again")
        return False
    
    # Check Ollama service
    service_ok = check_ollama_service()
    
    # Check API connection
    api_ok = check_ollama_api()
    
    if not api_ok:
        print("\n❌ Cannot connect to Ollama API")
        print("\nTroubleshooting steps:")
        print("1. Start Ollama: sudo systemctl start ollama")
        print("2. Enable auto-start: sudo systemctl enable ollama")
        print("3. Check status: sudo systemctl status ollama")
        print("4. View logs: sudo journalctl -u ollama -n 50")
        return False
    
    # Check models
    models = check_models()
    
    if not models:
        print("\n❌ No models available")
        print("\nInstall models:")
        print("1. Quick: ollama pull phi3:mini")
        print("2. Or run full installer: ./download_models.sh")
        return False
    
    # Test generation with first model
    gen_ok = test_model_generation(models[0] if models else None)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Diagnostic Summary:")
    print("=" * 60)
    
    all_ok = True
    
    if ollama_installed:
        print("✅ Ollama: Installed")
    else:
        print("❌ Ollama: Not installed")
        all_ok = False
    
    if service_ok:
        print("✅ Service: Running")
    elif service_ok is None:
        print("⚠️ Service: Unknown (check via API)")
    else:
        print("❌ Service: Not running")
        all_ok = False
    
    if api_ok:
        print("✅ API: Connected")
    else:
        print("❌ API: Not connected")
        all_ok = False
    
    if models:
        print(f"✅ Models: {len(models)} available")
    else:
        print("❌ Models: None available")
        all_ok = False
    
    if gen_ok:
        print("✅ Generation: Working")
    else:
        print("⚠️ Generation: Not tested or failed")
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print("\n✅ Everything looks good! Pascal should work.")
        print("\nNext steps:")
        print("1. Activate virtual environment: source venv/bin/activate")
        print("2. Install Python dependencies: pip install -r requirements.txt")
        print("3. Run Pascal: ./run.sh")
    else:
        print("\n❌ Some issues need to be fixed (see above)")
        print("\nQuick fix attempts:")
        print("1. Start Ollama: sudo systemctl start ollama")
        print("2. Install a model: ollama pull phi3:mini")
        print("3. Then try: ./run.sh")
    
    return all_ok

def main():
    """Main entry point"""
    try:
        success = run_diagnostics()
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
    main()
