#!/usr/bin/env python3
"""
Pascal Quick Fix Script
Rapidly diagnose and fix the most common Pascal issues
"""

import asyncio
import sys
import os
import subprocess
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

class QuickFix:
    """Quick diagnostic and fix for Pascal issues"""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
    
    def check_python_imports(self):
        """Check critical Python imports"""
        print("🔍 Checking Python imports...")
        
        critical_imports = [
            ('aiohttp', 'HTTP client for online functionality'),
            ('asyncio', 'Async support'),
            ('json', 'JSON processing'),
            ('pathlib', 'Path handling')
        ]
        
        import_issues = []
        
        for module_name, description in critical_imports:
            try:
                __import__(module_name)
                print(f"  ✅ {module_name}: Available")
            except ImportError:
                print(f"  ❌ {module_name}: Missing - {description}")
                import_issues.append(module_name)
        
        if import_issues:
            self.issues_found.append(f"Missing imports: {', '.join(import_issues)}")
            return False
        
        # Test aiohttp version specifically
        try:
            import aiohttp
            version = aiohttp.__version__
            version_parts = list(map(int, version.split('.')[:2]))
            
            if version_parts >= [3, 8]:
                print(f"  ✅ aiohttp version: {version} (compatible)")
                return True
            else:
                print(f"  ⚠️ aiohttp version: {version} (may have issues)")
                self.issues_found.append(f"aiohttp version {version} may be incompatible")
                return False
                
        except Exception as e:
            print(f"  ❌ aiohttp version check failed: {e}")
            self.issues_found.append("aiohttp version check failed")
            return False
    
    def check_pascal_modules(self):
        """Check Pascal module imports"""
        print("\n🔍 Checking Pascal modules...")
        
        pascal_modules = [
            ('config.settings', 'Pascal configuration'),
            ('modules.offline_llm', 'Offline LLM module'),
            ('modules.online_llm', 'Online LLM module'),
            ('modules.router', 'Router module')
        ]
        
        module_issues = []
        
        for module_path, description in pascal_modules:
            try:
                __import__(module_path)
                print(f"  ✅ {module_path}: Available")
            except ImportError as e:
                print(f"  ❌ {module_path}: Failed - {e}")
                module_issues.append(module_path)
        
        if module_issues:
            self.issues_found.append(f"Pascal module issues: {', '.join(module_issues)}")
            return False
        
        return True
    
    def check_ollama_service(self):
        """Check Ollama service status"""
        print("\n🔍 Checking Ollama service...")
        
        # Check if Ollama command exists
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"  ✅ Ollama command: {version}")
            else:
                print(f"  ❌ Ollama command failed")
                self.issues_found.append("Ollama command not working")
                return False
        except FileNotFoundError:
            print(f"  ❌ Ollama not installed")
            self.issues_found.append("Ollama not installed")
            return False
        except Exception as e:
            print(f"  ❌ Ollama check failed: {e}")
            self.issues_found.append(f"Ollama check error: {e}")
            return False
        
        # Check if service is running
        try:
            result = subprocess.run(['systemctl', 'is-active', 'ollama'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and 'active' in result.stdout:
                print(f"  ✅ Ollama service: Running")
                return True
            else:
                print(f"  ❌ Ollama service: Not running")
                self.issues_found.append("Ollama service not running")
                return False
        except Exception as e:
            print(f"  ⚠️ Service check failed: {e}")
            return True  # Don't fail on this, might work anyway
    
    async def check_llm_integration(self):
        """Test LLM integration"""
        print("\n🔍 Testing LLM integration...")
        
        try:
            from modules.offline_llm import LightningOfflineLLM
            print("  ✅ LightningOfflineLLM imported")
            
            llm = LightningOfflineLLM()
            print("  ✅ LLM instance created")
            
            # Quick initialization test
            success = await llm.initialize()
            
            if success:
                print("  ✅ LLM initialization successful")
                status = llm.get_status()
                print(f"    Model: {status.get('current_model', 'Unknown')}")
                print(f"    Available: {status.get('available', False)}")
                await llm.close()
                return True
            else:
                print("  ❌ LLM initialization failed")
                error = getattr(llm, 'last_error', 'Unknown error')
                print(f"    Error: {error}")
                self.issues_found.append(f"LLM initialization failed: {error}")
                return False
                
        except Exception as e:
            print(f"  ❌ LLM integration test failed: {e}")
            self.issues_found.append(f"LLM integration error: {e}")
            return False
    
    async def test_routing_system(self):
        """Test the routing system"""
        print("\n🔍 Testing routing system...")
        
        try:
            from modules.router import LightningRouter
            from modules.personality import PersonalityManager
            from modules.memory import MemoryManager
            
            print("  ✅ Router modules imported")
            
            # Create components
            personality_manager = PersonalityManager()
            memory_manager = MemoryManager()
            router = LightningRouter(personality_manager, memory_manager)
            
            print("  ✅ Router components created")
            
            # Test availability check
            await router._check_llm_availability()
            
            print(f"  ℹ️ Offline available: {router.offline_available}")
            print(f"  ℹ️ Online available: {router.online_available}")
            print(f"  ℹ️ Skills available: {router.skills_available}")
            
            # Test routing decision
            test_query = "Hello, how are you?"
            decision = router._decide_route_enhanced(test_query)
            print(f"  ✅ Routing test: '{test_query}' -> {decision.route_type}")
            
            await router.close()
            return True
            
        except Exception as e:
            print(f"  ❌ Routing system test failed: {e}")
            self.issues_found.append(f"Routing system error: {e}")
            return False
    
    def apply_fixes(self):
        """Apply automatic fixes for common issues"""
        print("\n🔧 Applying automatic fixes...")
        
        fixes_applied = 0
        
        # Fix 1: Install missing aiohttp
        if any('aiohttp' in issue for issue in self.issues_found):
            print("  🔧 Fixing aiohttp installation...")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'aiohttp==3.9.5'], 
                             check=True, capture_output=True)
                print("  ✅ aiohttp installed/updated")
                fixes_applied += 1
            except Exception as e:
                print(f"  ❌ aiohttp fix failed: {e}")
        
        # Fix 2: Start Ollama service
        if any('service not running' in issue for issue in self.issues_found):
            print("  🔧 Starting Ollama service...")
            try:
                subprocess.run(['sudo', 'systemctl', 'start', 'ollama'], 
                             check=True, capture_output=True)
                print("  ✅ Ollama service started")
                fixes_applied += 1
            except Exception as e:
                print(f"  ⚠️ Ollama service fix failed: {e}")
                print("    Try manually: sudo systemctl start ollama")
        
        # Fix 3: Install Ollama if missing
        if any('not installed' in issue for issue in self.issues_found):
            print("  ℹ️ Ollama installation required")
            print("    Run: curl -fsSL https://ollama.ai/install.sh | sh")
        
        self.fixes_applied = [f"Applied {fixes_applied} automatic fixes"]
        return fixes_applied > 0
    
    def generate_recommendations(self):
        """Generate specific recommendations based on issues found"""
        print("\n💡 Recommendations:")
        
        if not self.issues_found:
            print("  🎉 No major issues detected! Pascal should work.")
            print("  ➡️ Try running: ./run.sh")
            return
        
        # aiohttp issues
        if any('aiohttp' in issue for issue in self.issues_found):
            print("  🔧 aiohttp Issues:")
            print("    • pip install aiohttp==3.9.5")
            print("    • Or try: pip install aiohttp==3.8.6")
            print("    • Check: python -c 'import aiohttp; print(aiohttp.__version__)'")
        
        # Ollama issues
        if any('ollama' in issue.lower() for issue in self.issues_found):
            print("  🔧 Ollama Issues:")
            print("    • Install: curl -fsSL https://ollama.ai/install.sh | sh")
            print("    • Start: sudo systemctl start ollama")
            print("    • Enable: sudo systemctl enable ollama")
            print("    • Download model: ollama pull nemotron-mini:4b-instruct-q4_K_M")
        
        # Pascal module issues
        if any('modules' in issue for issue in self.issues_found):
            print("  🔧 Pascal Module Issues:")
            print("    • Check you're in the pascal directory")
            print("    • Verify all files are present")
            print("    • Check Python path")
        
        # LLM integration issues
        if any('LLM' in issue for issue in self.issues_found):
            print("  🔧 LLM Integration Issues:")
            print("    • Check aiohttp version compatibility")
            print("    • Verify Ollama is running and accessible")
            print("    • Test with: python ollama_diagnostic.py")
        
        # General recommendations
        print("  🔧 General Fixes:")
        print("    • Run full diagnostic: python complete_diagnostic.py")
        print("    • Check logs for detailed errors")
        print("    • Verify virtual environment: source venv/bin/activate")
    
    def create_repair_script(self):
        """Create a repair script for manual execution"""
        repair_script = """#!/bin/bash

# Pascal Repair Script
# Generated by quick_fix.py

echo "🔧 Pascal Repair Script"
echo "======================"

# Fix 1: Update aiohttp
echo "Updating aiohttp..."
pip install aiohttp==3.9.5

# Fix 2: Start Ollama
echo "Starting Ollama service..."
sudo systemctl start ollama
sudo systemctl enable ollama

# Fix 3: Download model if needed
echo "Checking for Nemotron model..."
if ! ollama list | grep -q "nemotron"; then
    echo "Downloading Nemotron model..."
    ollama pull nemotron-mini:4b-instruct-q4_K_M
fi

# Fix 4: Test installation
echo "Testing Pascal..."
python quick_fix.py

echo "✅ Repair script completed"
"""
        
        with open('repair_pascal.sh', 'w') as f:
            f.write(repair_script)
        
        os.chmod('repair_pascal.sh', 0o755)
        print("  📝 Created repair_pascal.sh script")

async def main():
    """Main quick fix function"""
    print("⚡ Pascal Quick Fix Tool")
    print("=" * 40)
    
    fixer = QuickFix()
    
    # Run diagnostics
    tests = [
        ("Python Imports", fixer.check_python_imports),
        ("Pascal Modules", fixer.check_pascal_modules),
        ("Ollama Service", fixer.check_ollama_service),
        ("LLM Integration", fixer.check_llm_integration),
        ("Routing System", fixer.test_routing_system)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"  ❌ {test_name} test error: {e}")
            fixer.issues_found.append(f"{test_name} test failed: {e}")
    
    # Summary
    health_score = (passed_tests / total_tests) * 100
    print(f"\n📊 Quick Diagnostic Summary")
    print("=" * 30)
    print(f"Tests passed: {passed_tests}/{total_tests} ({health_score:.0f}%)")
    
    if fixer.issues_found:
        print(f"Issues found: {len(fixer.issues_found)}")
        for i, issue in enumerate(fixer.issues_found, 1):
            print(f"  {i}. {issue}")
    
    # Apply fixes if issues found
    if fixer.issues_found:
        print(f"\n🔧 Attempting automatic fixes...")
        fixes_applied = fixer.apply_fixes()
        
        if fixes_applied:
            print(f"✅ Some fixes applied - rerun quick_fix.py to test")
        
        # Generate recommendations
        fixer.generate_recommendations()
        
        # Create repair script
        fixer.create_repair_script()
    else:
        print(f"\n🎉 No critical issues found!")
        print(f"Pascal should be ready to run.")
        print(f"\n➡️ Next steps:")
        print(f"  • Start Pascal: ./run.sh")
        print(f"  • Test with: 'Hello Pascal'")
    
    return health_score >= 80

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if result:
            print(f"\n✅ Quick fix completed successfully!")
        else:
            print(f"\n⚠️ Issues remain - check recommendations above")
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print(f"\n⏹️ Quick fix interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Quick fix failed: {e}")
        sys.exit(1)
