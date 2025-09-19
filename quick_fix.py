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
