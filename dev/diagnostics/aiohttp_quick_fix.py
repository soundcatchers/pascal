#!/usr/bin/env python3
"""
aiohttp Quick Fix Script for Pascal
Specifically addresses aiohttp import and compatibility issues
"""

import sys
import subprocess
import os
from pathlib import Path

def print_status(message):
    print(f"[INFO] {message}")

def print_success(message):
    print(f"[SUCCESS] {message}")

def print_error(message):
    print(f"[ERROR] {message}")

def print_warning(message):
    print(f"[WARNING] {message}")

def check_virtual_environment():
    """Check if we're in a virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print_success(f"Virtual environment detected: {sys.prefix}")
        return True
    else:
        print_warning("Not in virtual environment")
        
        # Check if venv exists
        venv_path = Path("venv")
        if venv_path.exists():
            print_warning("Virtual environment exists but not activated")
            print("Run: source venv/bin/activate")
            return False
        else:
            print_error("No virtual environment found")
            print("Run: python3 -m venv venv && source venv/bin/activate")
            return False

def test_aiohttp_import():
    """Test aiohttp import and version"""
    print_status("Testing aiohttp import...")
    
    try:
        import aiohttp
        version = aiohttp.__version__
        print_success(f"aiohttp v{version} imported successfully")
        
        # Test basic functionality
        try:
            # Try creating a ClientTimeout (common failure point)
            timeout = aiohttp.ClientTimeout(total=10)
            print_success("aiohttp ClientTimeout creation works")
            
            # Try creating a TCPConnector
            connector = aiohttp.TCPConnector(limit=1)
            print_success("aiohttp TCPConnector creation works")
            
            return True, version
            
        except Exception as e:
            print_error(f"aiohttp functionality test failed: {e}")
            return False, version
            
    except ImportError as e:
        print_error(f"aiohttp import failed: {e}")
        return False, None

def fix_aiohttp_installation():
    """Fix aiohttp installation"""
    print_status("Fixing aiohttp installation...")
    
    # List of versions to try (in order of preference)
    versions_to_try = [
        "3.9.5",  # Latest stable
        "3.9.1",  # Alternative
        "3.8.6",  # Fallback
        "3.8.4"   # Last resort
    ]
    
    for version in versions_to_try:
        print_status(f"Trying aiohttp=={version}...")
        
        try:
            # Uninstall current version
            subprocess.run([
                sys.executable, '-m', 'pip', 'uninstall', 'aiohttp', '-y'
            ], capture_output=True, check=False)
            
            # Install specific version
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', f'aiohttp=={version}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print_success(f"aiohttp {version} installed successfully")
                
                # Test the installation
                success, installed_version = test_aiohttp_import()
                if success:
                    print_success(f"aiohttp {installed_version} is working!")
                    return True
                else:
                    print_warning(f"aiohttp {version} installed but not working properly")
            else:
                print_warning(f"Failed to install aiohttp {version}")
                print(f"Error: {result.stderr}")
                
        except Exception as e:
            print_error(f"Error installing aiohttp {version}: {e}")
    
    print_error("Failed to install working version of aiohttp")
    return False

def install_dependencies():
    """Install other required dependencies"""
    print_status("Installing other dependencies...")
    
    dependencies = [
        "aiofiles==23.2.0",
        "requests==2.31.0", 
        "python-dotenv==1.0.0",
        "rich==13.7.0",
        "colorama==0.4.6",
        "psutil==5.9.8"
    ]
    
    for dep in dependencies:
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', dep
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print_success(f"Installed {dep}")
            else:
                print_warning(f"Failed to install {dep}: {result.stderr}")
                
        except Exception as e:
            print_error(f"Error installing {dep}: {e}")

def test_pascal_imports():
    """Test Pascal module imports"""
    print_status("Testing Pascal module imports...")
    
    # Add current directory to path for testing
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    modules_to_test = [
        ('config.settings', 'Pascal settings'),
        ('modules.offline_llm', 'Offline LLM module'),
        ('modules.online_llm', 'Online LLM module'),
        ('modules.router', 'Router module'),
        ('modules.skills_manager', 'Skills manager'),
        ('modules.personality', 'Personality manager'),
        ('modules.memory', 'Memory manager')
    ]
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print_success(f"{description} imported successfully")
            success_count += 1
        except ImportError as e:
            print_error(f"{description} import failed: {e}")
        except Exception as e:
            print_error(f"{description} error: {e}")
    
    print_status(f"Pascal modules: {success_count}/{total_count} imported successfully")
    return success_count == total_count

def create_simple_test():
    """Create a simple test script"""
    test_content = '''#!/usr/bin/env python3
"""
Simple Pascal Test - Check if aiohttp fix worked
"""

import asyncio
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

async def test_basic_functionality():
    """Test basic Pascal functionality"""
    print("🧪 Testing basic Pascal functionality...")
    
    try:
        # Test aiohttp
        import aiohttp
        print(f"✅ aiohttp v{aiohttp.__version__} working")
        
        # Test basic session creation
        async with aiohttp.ClientSession() as session:
            print("✅ aiohttp session creation works")
        
        # Test Pascal imports
        from config.settings import settings
        print(f"✅ Pascal settings loaded (v{settings.version})")
        
        from modules.offline_llm import LightningOfflineLLM
        print("✅ LightningOfflineLLM imported")
        
        from modules.router import LightningRouter
        print("✅ LightningRouter imported")
        
        print("\\n🎉 Basic functionality test PASSED!")
        print("\\nNext steps:")
        print("1. Run full test: python test_current_info_fix.py")
        print("2. If that passes, run Pascal: ./run.sh")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test FAILED: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_basic_functionality())
    sys.exit(0 if result else 1)
'''
    
    with open('test_basic_pascal.py', 'w') as f:
        f.write(test_content)
    
    os.chmod('test_basic_pascal.py', 0o755)
    print_success("Created test_basic_pascal.py")

def main():
    """Main fix function"""
    print("🔧 Pascal aiohttp Quick Fix")
    print("=" * 40)
    
    # Check virtual environment
    if not check_virtual_environment():
        print_error("Please activate virtual environment first")
        return False
    
    # Test current aiohttp
    success, version = test_aiohttp_import()
    
    if success:
        print_success(f"aiohttp v{version} is already working!")
        
        # Still test Pascal imports
        if test_pascal_imports():
            print_success("Pascal modules are working!")
            create_simple_test()
            print_success("✅ No fixes needed - Pascal should work")
            print("Run: python test_basic_pascal.py")
            return True
        else:
            print_warning("Pascal modules have issues")
    
    # Fix aiohttp
    print_status("Attempting to fix aiohttp...")
    
    if fix_aiohttp_installation():
        print_success("aiohttp fixed successfully!")
        
        # Install other dependencies
        install_dependencies()
        
        # Test Pascal imports
        if test_pascal_imports():
            print_success("Pascal modules are now working!")
            create_simple_test()
            
            print_success("✅ Fix completed successfully!")
            print("\nNext steps:")
            print("1. Test basic functionality: python test_basic_pascal.py")
            print("2. Run full test: python test_current_info_fix.py")
            print("3. Start Pascal: ./run.sh")
            return True
        else:
            print_error("Pascal modules still have issues")
            return False
    else:
        print_error("Failed to fix aiohttp")
        print("\nManual fix options:")
        print("1. pip uninstall aiohttp")
        print("2. pip install aiohttp==3.9.5")
        print("3. Or try: pip install aiohttp==3.8.6")
        return False

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Fix interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fix failed: {e}")
        sys.exit(1)
