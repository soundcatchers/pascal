#!/usr/bin/env python3
"""
Pascal AI Assistant - Complete System Diagnostic
Tests all components: virtual environment, dependencies, Ollama, and online APIs
"""

import sys
import subprocess
import os
from pathlib import Path
import asyncio

def check_virtual_environment():
    """Check if we're in a virtual environment"""
    print("🐍 Virtual Environment Check:")
    print("-" * 30)
    
    # Check if in virtual environment
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if in_venv:
        print("✅ Running in virtual environment")
        print(f"   Virtual env path: {sys.prefix}")
        print(f"   Python executable: {sys.executable}")
    else:
        print("❌ NOT in virtual environment")
        print("   Run: source venv/bin/activate")
        return False
    
    # Check Python version
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"❌ Python version too old: {version.major}.{version.minor}.{version.micro}")
        return False
    
    return True

def check_dependencies():
    """Check critical Python dependencies"""
    print("\n📦 Dependencies Check:")
    print("-" * 30)
    
    critical_deps = {
        'aiohttp': 'Online LLM functionality',
        'requests': 'HTTP requests',
        'openai': 'OpenAI API',
        'anthropic': 'Anthropic API', 
        'rich': 'Console formatting',
        'colorama': 'Terminal colors',
        'asyncio': 'Async operations (built-in)'
    }
    
    missing_deps = []
    
    for dep, description in critical_deps.items():
        try:
            if dep == 'asyncio':
                import asyncio
                version = 'built-in'
            else:
                module = __import__(dep)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✅ {dep}: {version} ({description})")
        except ImportError:
            print(f"❌ {dep}: MISSING ({description})")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Fix with: pip install -r requirements.txt")
        return False
    
    return True

def check_project_structure():
    """Check if all required files exist"""
    print("\n📁 Project Structure Check:")
    print("-" * 30)
    
    required_files = {
        'main.py': 'Main entry point',
        'config/settings.py': 'Configuration',
        'modules/online_llm.py': 'Online LLM module',
        'modules/offline_llm.py': 'Offline LLM module',
        'modules/router.py': 'Router module',
        'modules/personality.py': 'Personality system',
        'modules/memory.py': 'Memory system',
        'requirements.txt': 'Dependencies list',
        'run.sh': 'Startup script',
        '.env.example': 'Environment template'
    }
    
    missing_files = []
    
    for file, description in required_files.items():
        if Path(file).exists():
            print(f"✅ {file}: Found ({description})")
        else:
            print(f"❌ {file}: Missing ({description})")
            missing_files.append(file)
    
    # Check directories
    required_dirs = ['data', 'data/models', 'data/memory', 'data/cache', 'config/personalities']
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/: Found")
        else:
            print(f"⚠️ {directory}/: Missing (will be created)")
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_environment_config():
    """Check .env configuration"""
    print("\n🔧 Environment Configuration Check:")
    print("-" * 30)
    
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found")
        print("   Copy .env.example to .env and configure API keys")
        return False
    
    print("✅ .env file exists")
    
    # Read .env and check API keys
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        api_keys = {
            'GROQ_API_KEY': ('Groq API (Primary)', 'gsk-'),
            'GEMINI_API_KEY': ('Gemini API (Secondary)', None),
            'OPENAI_API_KEY': ('OpenAI API (Fallback)', 'sk-')
        }
        
        configured_keys = 0
        
        for key, (description, prefix) in api_keys.items():
            if f'{key}=' in content:
                # Extract value
                for line in content.split('\n'):
                    if line.strip().startswith(f'{key}='):
                        value = line.split('=', 1)[1].strip()
                        
                        invalid_values = ['', f'your_{key.lower()}_here', f'{prefix}your_{key.lower()}_here']
                        if key == 'GROQ_API_KEY':
                            invalid_values.extend(['your_groq_api_key_here', 'gsk-your_groq_api_key_here'])
                        
                        if value in invalid_values:
                            print(f"⚠️ {key}: Placeholder value ({description})")
                        elif prefix and not value.startswith(prefix):
                            print(f"⚠️ {key}: Wrong format, should start with '{prefix}' ({description})")
                        else:
                            print(f"✅ {key}: Configured ({description})")
                            configured_keys += 1
                        break
            else:
                print(f"❌ {key}: Not found ({description})")
        
        if configured_keys == 0:
            print("\n❌ No API keys configured!")
            print("   Pascal will run offline-only mode")
            print("   Add at least one API key for online functionality")
            return False
        else:
            print(f"\n✅ {configured_keys} API key(s) configured")
            return True
            
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False

async def test_online_llm():
    """Test online LLM functionality"""
    print("\n🌐 Online LLM Test:")
    print("-" * 30)
    
    try:
        # Add project to path
        sys.path.append(str(Path(__file__).parent))
        
        from modules.online_llm import OnlineLLM
        
        online_llm = OnlineLLM()
        success = await online_llm.initialize()
        
        if success:
            print("✅ Online LLM initialized successfully")
            
            stats = online_llm.get_provider_stats()
            available = stats['available_providers']
            preferred = stats['preferred_provider']
            
            print(f"   Available providers: {available}")
            print(f"   Preferred provider: {preferred}")
            
            # Test generation if any provider is available
            if available:
                try:
                    response = await online_llm.generate_response(
                        "Say 'online test successful'",
                        "You are a test assistant",
                        ""
                    )
                    
                    if "successful" in response.lower():
                        print("✅ Online response generation working")
                    else:
                        print(f"⚠️ Unexpected response: {response[:50]}...")
                        
                except Exception as e:
                    print(f"❌ Response generation failed: {e}")
            
            await online_llm.close()
            return True
        else:
            print("❌ Online LLM initialization failed")
            if hasattr(online_llm, 'last_error') and online_llm.last_error:
                print(f"   Error: {online_llm.last_error}")
            return False
            
    except Exception as e:
        print(f"❌ Online LLM test error: {e}")
        return False

def test_ollama():
    """Test Ollama availability"""
    print("\n🦙 Ollama Test:")
    print("-" * 30)
    
    try:
        # Check if ollama command exists
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ Ollama is installed")
            print(f"   Version: {result.stdout.strip()}")
        else:
            print("❌ Ollama installation issue")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama not found")
        print("   Install with: curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Ollama command timeout")
        return False
    except Exception as e:
        print(f"❌ Ollama check error: {e}")
        return False
    
    # Check if Ollama service is running
    try:
        result = subprocess.run(['systemctl', 'is-active', 'ollama'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.stdout.strip() == 'active':
            print("✅ Ollama service is running")
        else:
            print("⚠️ Ollama service not running")
            print("   Start with: sudo systemctl start ollama")
            
    except Exception:
        print("⚠️ Could not check Ollama service status")
    
    # Check for available models
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # More than just header
                model_count = len(lines) - 1
                print(f"✅ Found {model_count} Ollama model(s)")
                
                # Show first few models
                for line in lines[1:4]:  # Show up to 3 models
                    if line.strip():
                        model_name = line.split()[0]
                        print(f"   • {model_name}")
                
                if model_count > 3:
                    print(f"   ... and {model_count - 3} more")
                    
                return True
            else:
                print("⚠️ No Ollama models found")
                print("   Download models with: ./download_models.sh")
                return False
        else:
            print("❌ Could not list Ollama models")
            return False
            
    except Exception as e:
        print(f"⚠️ Could not check Ollama models: {e}")
        return False

async def test_pascal_imports():
    """Test Pascal module imports"""
    print("\n🤖 Pascal Module Import Test:")
    print("-" * 30)
    
    # Add project to path
    sys.path.append(str(Path(__file__).parent))
    
    modules_to_test = {
        'config.settings': 'Configuration system',
        'modules.online_llm': 'Online LLM module',
        'modules.offline_llm': 'Offline LLM module', 
        'modules.router': 'Router module',
        'modules.personality': 'Personality system',
        'modules.memory': 'Memory system'
    }
    
    failed_imports = []
    
    for module, description in modules_to_test.items():
        try:
            __import__(module)
            print(f"✅ {module}: Imported successfully ({description})")
        except ImportError as e:
            print(f"❌ {module}: Import failed ({description})")
            print(f"   Error: {e}")
            failed_imports.append(module)
        except Exception as e:
            print(f"⚠️ {module}: Import error ({description})")
            print(f"   Error: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        return False
    
    # Test settings specifically
    try:
        from config.settings import settings
        print(f"\n✅ Settings loaded successfully")
        print(f"   Pascal version: {settings.version}")
        print(f"   Debug mode: {settings.debug_mode}")
        print(f"   Performance mode: {settings.performance_mode}")
        
        hw_info = settings.get_hardware_info()
        print(f"   Hardware: {hw_info.get('pi_model', 'Unknown')} with {hw_info.get('available_ram_gb', 'Unknown')}GB RAM")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False

def check_permissions():
    """Check file permissions"""
    print("\n🔒 Permissions Check:")
    print("-" * 30)
    
    executable_files = [
        'run.sh',
        'download_models.sh', 
        'test_performance.py',
        'diagnose_ollama.py',
        'diagnose_online_apis.py',
        'test_groq_fix.py'
    ]
    
    permission_issues = []
    
    for file in executable_files:
        file_path = Path(file)
        if file_path.exists():
            if os.access(file_path, os.X_OK):
                print(f"✅ {file}: Executable")
            else:
                print(f"❌ {file}: Not executable")
                permission_issues.append(file)
        else:
            print(f"⚠️ {file}: Not found")
    
    if permission_issues:
        print(f"\n⚠️ Fix permissions with:")
        print(f"   chmod +x {' '.join(permission_issues)}")
    
    return len(permission_issues) == 0

def show_quick_fixes():
    """Show common quick fixes"""
    print("\n🔧 Quick Fixes:")
    print("-" * 30)
    
    print("If you have issues, try these fixes in order:")
    print()
    print("1. Virtual Environment Issues:")
    print("   source venv/bin/activate")
    print("   pip install --upgrade pip")
    print("   pip install -r requirements.txt")
    print()
    print("2. Missing Dependencies:")
    print("   pip install aiohttp requests openai anthropic rich colorama")
    print()
    print("3. API Configuration:")
    print("   cp .env.example .env")
    print("   nano .env  # Add your API keys")
    print()
    print("4. Ollama Issues:")
    print("   sudo systemctl start ollama")
    print("   ./download_models.sh")
    print()
    print("5. Permissions:")
    print("   chmod +x run.sh download_models.sh *.py")
    print()
    print("6. Test Specific Components:")
    print("   python test_groq_fix.py  # Test Groq API")
    print("   python diagnose_ollama.py  # Test Ollama")
    print("   python diagnose_online_apis.py  # Test all APIs")

async def main():
    """Main diagnostic function"""
    print("🔍 Pascal AI Assistant - Complete System Diagnostic")
    print("=" * 60)
    
    # Track overall status
    all_checks = []
    
    # Run all checks
    all_checks.append(("Virtual Environment", check_virtual_environment()))
    all_checks.append(("Dependencies", check_dependencies()))
    all_checks.append(("Project Structure", check_project_structure()))
    all_checks.append(("Environment Config", check_environment_config()))
    all_checks.append(("Module Imports", await test_pascal_imports()))
    all_checks.append(("Online LLM", await test_online_llm()))
    all_checks.append(("Ollama", test_ollama()))
    all_checks.append(("Permissions", check_permissions()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for check_name, result in all_checks:
        if result:
            print(f"✅ {check_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {check_name}: FAILED")
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 ALL CHECKS PASSED!")
        print("Pascal should work perfectly. Run: ./run.sh")
    elif failed <= 2:
        print("\n⚠️ Minor issues detected - Pascal should still work")
        print("Run: ./run.sh to test, or fix issues above")
    else:
        print("\n❌ Multiple issues detected - fix required")
        print("Follow the quick fixes below, then run this diagnostic again")
    
    # Show fixes if needed
    if failed > 0:
        show_quick_fixes()
    
    print("\n" + "=" * 60)
    return failed == 0

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Diagnostic interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Diagnostic error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
