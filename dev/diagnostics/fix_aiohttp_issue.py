#!/usr/bin/env python3
"""
Pascal aiohttp Fix Script
Specifically addresses the "cannot access local variable 'aiohttp'" error
"""

import sys
import subprocess
from pathlib import Path

def fix_aiohttp_version():
    """Fix aiohttp version compatibility"""
    print("🔧 Fixing aiohttp compatibility issue...")
    
    try:
        # First, uninstall current aiohttp
        print("Uninstalling current aiohttp...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'aiohttp', '-y'], 
                      capture_output=True)
        
        # Install compatible version
        print("Installing compatible aiohttp version...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'aiohttp==3.9.5'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ aiohttp 3.9.5 installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            
            # Try fallback version
            print("Trying fallback version 3.8.6...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'aiohttp==3.8.6'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ aiohttp 3.8.6 installed successfully")
                return True
            else:
                print(f"❌ Fallback installation failed: {result.stderr}")
                return False
                
    except Exception as e:
        print(f"❌ aiohttp fix failed: {e}")
        return False

def test_aiohttp_import():
    """Test if aiohttp imports correctly"""
    print("🧪 Testing aiohttp import...")
    
    try:
        import aiohttp
        print(f"✅ aiohttp version {aiohttp.__version__} imported successfully")
        
        # Test basic functionality
        import asyncio
        
        async def test_session():
            try:
                async with aiohttp.ClientSession() as session:
                    print("✅ aiohttp ClientSession creation works")
                    return True
            except Exception as e:
                print(f"❌ aiohttp ClientSession failed: {e}")
                return False
        
        result = asyncio.run(test_session())
        return result
        
    except ImportError as e:
        print(f"❌ aiohttp import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ aiohttp test failed: {e}")
        return False

def test_pascal_offline_llm():
    """Test Pascal offline LLM with fixed aiohttp"""
    print("🧪 Testing Pascal offline LLM...")
    
    try:
        # Add current directory to path
        sys.path.append(str(Path(__file__).parent))
        
        from modules.offline_llm import LightningOfflineLLM
        print("✅ LightningOfflineLLM imported successfully")
        
        # Test instance creation
        llm = LightningOfflineLLM()
        print("✅ LightningOfflineLLM instance created")
        
        # This should not cause the aiohttp variable error anymore
        print("✅ No aiohttp variable scoping error detected")
        return True
        
    except NameError as e:
        if 'aiohttp' in str(e):
            print(f"❌ aiohttp variable scoping error still present: {e}")
            return False
        else:
            print(f"❌ Other NameError: {e}")
            return False
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def main():
    """Main fix function"""
    print("⚡ Pascal aiohttp Fix Tool")
    print("=" * 40)
    print("This fixes the 'cannot access local variable aiohttp' error")
    print("")
    
    # Step 1: Fix aiohttp version
    if not fix_aiohttp_version():
        print("\n❌ Failed to fix aiohttp version")
        print("Manual fix:")
        print("  pip uninstall aiohttp")
        print("  pip install aiohttp==3.9.5")
        return False
    
    # Step 2: Test aiohttp import
    if not test_aiohttp_import():
        print("\n❌ aiohttp still not working after fix")
        return False
    
    # Step 3: Test Pascal integration
    if not test_pascal_offline_llm():
        print("\n❌ Pascal offline LLM still has issues")
        print("The aiohttp variable scoping error may still be present")
        print("Check that you're using the fixed offline_llm.py module")
        return False
    
    print("\n✅ aiohttp fix completed successfully!")
    print("\nNext steps:")
    print("  1. Run: python quick_fix.py")
    print("  2. Test: ./run.sh")
    print("  3. Try: 'Hello Pascal'")
    
    return True

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
