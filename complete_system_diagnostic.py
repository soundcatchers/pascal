#!/usr/bin/env python3
"""
Pascal Complete System Diagnostic - FIXED VERSION
Comprehensive testing and debugging of all Pascal components
"""

import asyncio
import sys
import os
import subprocess
import time
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode
os.environ['DEBUG'] = 'true'

class Colors:
    """Color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.NC}")
    print(f"{Colors.WHITE}{text}{Colors.NC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.NC}")

def print_section(text: str):
    """Print section header"""
    print(f"\n{Colors.BLUE}{'='*20} {text} {'='*20}{Colors.NC}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.NC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.NC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️ {text}{Colors.NC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️ {text}{Colors.NC}")

class SystemDiagnostic:
    """Complete system diagnostic class"""
    
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
        
        if details:
            print(f"   {details}")
        
        self.results[test_name] = {'success': success, 'details': details}
    
    async def test_dependencies(self):
        """Test Python dependencies"""
        print_section("Python Dependencies")
        
        # Test critical imports
        dependencies = [
            ('aiohttp', 'HTTP client for online functionality'),
            ('asyncio', 'Async support'),
            ('json', 'JSON processing'),
            ('pathlib', 'Path handling'),
            ('rich', 'Console formatting'),
            ('psutil', 'System monitoring')
        ]
        
        for dep, description in dependencies:
            try:
                __import__(dep)
                self.log_test(f"{dep} import", True, description)
            except ImportError:
                self.log_test(f"{dep} import", False, f"Missing: {description}")
        
        # Test aiohttp version specifically
        try:
            import aiohttp
            version = aiohttp.__version__
            version_tuple = tuple(map(int, version.split('.')[:2]))
            
            if version_tuple >= (3, 8):
                self.log_test(f"aiohttp version {version}", True, "Compatible version")
            else:
                self.log_test(f"aiohttp version {version}", False, "May have compatibility issues")
                
        except Exception as e:
            self.log_test("aiohttp version check", False, f"Error: {e}")
    
    async def test_configuration(self):
        """Test configuration system"""
        print_section("Configuration System")
        
        # Test settings import
        try:
            from config.settings import settings
            self.log_test("Settings import", True, f"Pascal v{settings.version}")
            
            # Test API key configuration
            groq_configured = settings.groq_api_key and settings.validate_groq_api_key(settings.groq_api_key)
            self.log_test("Groq API key", groq_configured, 
                         "Configured" if groq_configured else "Not configured - online features limited")
            
            # Test hardware detection
            if settings.is_raspberry_pi:
                self.log_test("Hardware detection", True, f"{settings.pi_model}, {settings.available_ram_gb}GB RAM")
            else:
                self.log_test("Hardware detection", True, "Non-Pi hardware detected")
            
        except Exception as e:
            self.log_test("Settings configuration", False, f"Error: {e}")
    
    async def test_ollama_system(self):
        """Test Ollama system"""
        print_section("Ollama System")
        
        # Test Ollama service
        try:
            result = subprocess.run(['systemctl', 'is-active', 'ollama'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'active' in result.stdout:
                self.log_test("Ollama service", True, "Running")
            else:
                self.log_test("Ollama service", False, "Not running - sudo systemctl start ollama")
        except Exception as e:
            self.log_test("Ollama service", False, f"Error checking: {e}")
        
        # Test Ollama command
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.log_test("Ollama command", True, version)
            else:
                self.log_test("Ollama command", False, "Command failed")
        except FileNotFoundError:
            self.log_test("Ollama command", False, "Not installed - curl -fsSL https://ollama.ai/install.sh | sh")
        except Exception as e:
            self.log_test("Ollama command", False, f"Error: {e}")
        
        # Test HTTP API
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get('http://localhost:11434/api/version') as response:
                    if response.status == 200:
                        data = await response.json()
                        self.log_test("Ollama API", True, f"Version {data.get('version', 'unknown')}")
                    else:
                        self.log_test("Ollama API", False, f"HTTP {response.status}")
        except Exception as e:
            self.log_test("Ollama API", False, f"Connection error: {e}")
        
        # Test models
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get('http://localhost:11434/api/tags') as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        if models:
                            model_names = [m.get('name', 'unknown') for m in models]
                            self.log_test("Ollama models", True, f"Found {len(models)} models: {', '.join(model_names[:3])}")
                        else:
                            self.log_test("Ollama models", False, "No models found - ollama pull nemotron-mini:4b-instruct-q4_K_M")
                    else:
                        self.log_test("Ollama models", False, f"API error: {response.status}")
        except Exception as e:
            self.log_test("Ollama models", False, f"Error: {e}")
    
    async def test_llm_modules(self):
        """Test LLM modules"""
        print_section("LLM Modules")
        
        # Test offline LLM import
        try:
            from modules.offline_llm import LightningOfflineLLM
            self.log_test("Offline LLM import", True, "LightningOfflineLLM imported")
        except Exception as e:
            self.log_test("Offline LLM import", False, f"Import error: {e}")
            return
        
        # Test offline LLM initialization
        try:
            llm = LightningOfflineLLM()
            self.log_test("Offline LLM creation", True, "Instance created")
            
            # Test initialization
            success = await llm.initialize()
            if success:
                status = llm.get_status()
                self.log_test("Offline LLM init", True, 
                             f"Model: {status.get('current_model', 'unknown')}, Profile: {status.get('performance_profile', 'unknown')}")
                
                # Test simple response
                try:
                    response = await llm.generate_response("Hello", "", "")
                    if response and not response.startswith("Model error"):
                        self.log_test("Offline LLM response", True, f"Response: {response[:50]}...")
                    else:
                        self.log_test("Offline LLM response", False, f"Empty or error response: {response}")
                except Exception as e:
                    self.log_test("Offline LLM response", False, f"Response error: {e}")
                
                await llm.close()
            else:
                self.log_test("Offline LLM init", False, f"Error: {llm.last_error}")
                
        except Exception as e:
            self.log_test("Offline LLM initialization", False, f"Error: {e}")
        
        # Test online LLM
        try:
            from modules.online_llm import OnlineLLM
            self.log_test("Online LLM import", True, "OnlineLLM imported")
            
            online_llm = OnlineLLM()
            success = await online_llm.initialize()
            if success:
                self.log_test("Online LLM init", True, "Groq API configured and working")
                await online_llm.close()
            else:
                self.log_test("Online LLM init", False, "Groq API not configured or not working")
                
        except Exception as e:
            self.log_test("Online LLM", False, f"Error: {e}")
    
    async def test_router_system(self):
        """Test router system"""
        print_section("Router System")
        
        try:
            from modules.router import LightningRouter
            from modules.personality import PersonalityManager
            from modules.memory import MemoryManager
            
            self.log_test("Router imports", True, "All router components imported")
            
            # Create components
            personality_manager = PersonalityManager()
            memory_manager = MemoryManager()
            router = LightningRouter(personality_manager, memory_manager)
            
            self.log_test("Router creation", True, "Router and components created")
            
            # Initialize router
            await router._check_llm_availability()
            
            # Check availability
            systems_available = {
                'offline': router.offline_available,
                'online': router.online_available,
                'skills': router.skills_available
            }
            
            available_count = sum(systems_available.values())
            if available_count > 0:
                available_systems = [k for k, v in systems_available.items() if v]
                self.log_test("Router systems", True, f"Available: {', '.join(available_systems)}")
            else:
                self.log_test("Router systems", False, "No systems available")
            
            # Test routing decisions
            test_queries = [
                ("Hello, how are you?", "Should route to available system"),
                ("What day is today?", "Should detect current info need"),
                ("15 + 23", "Should route to skills if available"),
                ("Explain Python", "Should prefer offline if available")
            ]
            
            routing_working = True
            for query, description in test_queries:
                try:
                    decision = router._decide_route_enhanced(query)
                    if decision.route_type != 'fallback' or available_count == 0:
                        self.log_test(f"Routing: '{query[:20]}...'", True, 
                                     f"{decision.route_type} - {decision.reason}")
                    else:
                        self.log_test(f"Routing: '{query[:20]}...'", False, 
                                     f"Unexpected fallback: {decision.reason}")
                        routing_working = False
                except Exception as e:
                    self.log_test(f"Routing: '{query[:20]}...'", False, f"Error: {e}")
                    routing_working = False
            
            # Test actual response if any system is available
            if available_count > 0:
                try:
                    test_query = "Hello Pascal"
                    response = await router.get_response(test_query)
                    if response and len(response.strip()) > 5:
                        self.log_test("Router response", True, f"Response: {response[:50]}...")
                    else:
                        self.log_test("Router response", False, f"Empty response: {response}")
                except Exception as e:
                    self.log_test("Router response", False, f"Error: {e}")
            
            await router.close()
            
        except Exception as e:
            self.log_test("Router system", False, f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_skills_system(self):
        """Test skills system"""
        print_section("Skills System")
        
        try:
            from modules.skills_manager import EnhancedSkillsManager
            self.log_test("Skills import", True, "EnhancedSkillsManager imported")
            
            skills_manager = EnhancedSkillsManager()
            api_status = await skills_manager.initialize()
            
            # Check basic skills availability
            basic_skills_working = True
            
            # Test datetime skill
            try:
                result = await skills_manager.execute_skill("What time is it?", "datetime")
                if result and result.success:
                    self.log_test("DateTime skill", True, f"Response: {result.response}")
                else:
                    self.log_test("DateTime skill", False, "Skill execution failed")
                    basic_skills_working = False
            except Exception as e:
                self.log_test("DateTime skill", False, f"Error: {e}")
                basic_skills_working = False
            
            # Test calculator skill
            try:
                result = await skills_manager.execute_skill("15 + 23", "calculator")
                if result and result.success and "38" in result.response:
                    self.log_test("Calculator skill", True, f"Response: {result.response}")
                else:
                    self.log_test("Calculator skill", False, "Calculation failed or incorrect")
                    basic_skills_working = False
            except Exception as e:
                self.log_test("Calculator skill", False, f"Error: {e}")
                basic_skills_working = False
            
            # Test API-based skills
            for service, status in api_status.items():
                self.log_test(f"{service.title()} API", status['available'], status['message'])
            
            await skills_manager.close()
            
        except Exception as e:
            self.log_test("Skills system", False, f"Error: {e}")
    
    async def test_integration(self):
        """Test full system integration"""
        print_section("Integration Test")
        
        try:
            # Import main components
            from config.settings import settings
            from modules.router import LightningRouter
            from modules.personality import PersonalityManager
            from modules.memory import MemoryManager
            
            # Create full system
            personality_manager = PersonalityManager()
            memory_manager = MemoryManager()
            
            # Load default personality and memory
            await personality_manager.load_personality("default")
            await memory_manager.load_session()
            
            self.log_test("Component initialization", True, "All components initialized")
            
            # Create router
            router = LightningRouter(personality_manager, memory_manager)
            await router._check_llm_availability()
            
            # Test end-to-end responses
            test_scenarios = [
                ("Hello Pascal", "Basic greeting"),
                ("What's 2+2?", "Simple calculation"),
                ("What day is today?", "Current info query")
            ]
            
            successful_responses = 0
            for query, description in test_scenarios:
                try:
                    response = await router.get_response(query)
                    if response and len(response.strip()) > 5 and not response.startswith("I'm sorry"):
                        self.log_test(f"Integration: {description}", True, f"Response: {response[:50]}...")
                        successful_responses += 1
                    else:
                        self.log_test(f"Integration: {description}", False, f"Poor response: {response}")
                except Exception as e:
                    self.log_test(f"Integration: {description}", False, f"Error: {e}")
            
            # Overall integration assessment
            if successful_responses == len(test_scenarios):
                self.log_test("Full integration", True, "All scenarios working")
            elif successful_responses > 0:
                self.log_test("Full integration", True, f"{successful_responses}/{len(test_scenarios)} scenarios working")
            else:
                self.log_test("Full integration", False, "No scenarios working properly")
            
            await router.close()
            
        except Exception as e:
            self.log_test("Integration test", False, f"Error: {e}")
    
    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        print_section("Recommendations")
        
        recommendations = []
        
        # Check for common issues
        if not self.results.get("aiohttp import", {}).get("success"):
            recommendations.append("Install aiohttp: pip install aiohttp==3.8.6")
        
        if not self.results.get("Ollama service", {}).get("success"):
            recommendations.append("Start Ollama service: sudo systemctl start ollama")
        
        if not self.results.get("Ollama models", {}).get("success"):
            recommendations.append("Download Nemotron model: ollama pull nemotron-mini:4b-instruct-q4_K_M")
        
        if not self.results.get("Groq API key", {}).get("success"):
            recommendations.append("Configure Groq API key in .env file for online features")
        
        if not self.results.get("Offline LLM init", {}).get("success"):
            recommendations.append("Run Ollama optimization: ./ollama_optimization.sh")
        
        if not self.results.get("Router systems", {}).get("success"):
            recommendations.append("Check system logs and run individual component tests")
        
        # Performance recommendations
        if self.results.get("Router systems", {}).get("success"):
            health_score = (self.passed_tests / self.total_tests) * 100
            if health_score < 80:
                recommendations.append("Run performance optimization scripts")
        
        if not recommendations:
            recommendations.append("System appears healthy - run Pascal with ./run.sh")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{Colors.YELLOW}{i}.{Colors.NC} {rec}")
        
        return recommendations
    
    def print_summary(self):
        """Print diagnostic summary"""
        print_header("DIAGNOSTIC SUMMARY")
        
        health_score = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"\n{Colors.WHITE}Tests Passed: {self.passed_tests}/{self.total_tests} ({health_score:.1f}%){Colors.NC}")
        
        if health_score >= 90:
            print(f"{Colors.GREEN}🎉 EXCELLENT HEALTH - Pascal is ready to use!{Colors.NC}")
            status = "Excellent"
        elif health_score >= 75:
            print(f"{Colors.GREEN}✅ GOOD HEALTH - Pascal should work well{Colors.NC}")
            status = "Good"
        elif health_score >= 60:
            print(f"{Colors.YELLOW}⚠️ FAIR HEALTH - Pascal will work with limitations{Colors.NC}")
            status = "Fair"
        elif health_score >= 40:
            print(f"{Colors.YELLOW}⚠️ POOR HEALTH - Pascal has significant issues{Colors.NC}")
            status = "Poor"
        else:
            print(f"{Colors.RED}❌ CRITICAL ISSUES - Pascal may not work properly{Colors.NC}")
            status = "Critical"
        
        # Show component status
        print(f"\n{Colors.CYAN}Component Status:{Colors.NC}")
        components = {
            "Dependencies": self.results.get("aiohttp import", {}).get("success", False),
            "Configuration": self.results.get("Settings import", {}).get("success", False),
            "Ollama Service": self.results.get("Ollama service", {}).get("success", False),
            "Ollama Models": self.results.get("Ollama models", {}).get("success", False),
            "Offline LLM": self.results.get("Offline LLM init", {}).get("success", False),
            "Online LLM": self.results.get("Online LLM init", {}).get("success", False),
            "Router System": self.results.get("Router systems", {}).get("success", False),
            "Skills System": self.results.get("DateTime skill", {}).get("success", False),
        }
        
        for component, status in components.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component}")
        
        return status

async def main():
    """Main diagnostic function"""
    print_header("PASCAL COMPLETE SYSTEM DIAGNOSTIC")
    print_info("This diagnostic will test all Pascal components and identify issues")
    
    diagnostic = SystemDiagnostic()
    
    # Run all diagnostic tests
    await diagnostic.test_dependencies()
    await diagnostic.test_configuration()
    await diagnostic.test_ollama_system()
    await diagnostic.test_llm_modules()
    await diagnostic.test_router_system()
    await diagnostic.test_skills_system()
    await diagnostic.test_integration()
    
    # Generate summary and recommendations
    status = diagnostic.print_summary()
    recommendations = diagnostic.generate_recommendations()
    
    # Next steps
    print_section("Next Steps")
    if status in ["Excellent", "Good"]:
        print_info("Start Pascal: ./run.sh")
        print_info("Test with queries like: 'Hello Pascal', 'What time is it?', 'What day is today?'")
    elif status == "Fair":
        print_info("Address recommendations above, then start Pascal: ./run.sh")
        print_info("Pascal will work with reduced functionality")
    else:
        print_warning("Address critical issues above before running Pascal")
        print_info("Run specific diagnostic scripts:")
        print_info("  python ollama_diagnostic.py")
        print_info("  python skills_diagnostic.py")
    
    return status in ["Excellent", "Good", "Fair"]

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⏹️ Diagnostic interrupted{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Fatal error: {e}{Colors.NC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
