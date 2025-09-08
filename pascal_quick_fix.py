#!/usr/bin/env python3
"""
Pascal Quick Fix Tool - Enhanced diagnostics and fixes
"""

import sys
import os
from pathlib import Path

def quick_diagnosis():
    """Run quick diagnosis of common issues"""
    print("🔍 Pascal Quick Diagnosis")
    print("=" * 40)
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file exists")
        
        # Check for Groq API key
        with open(env_file, 'r') as f:
            content = f.read()
        
        if 'GROQ_API_KEY=' in content:
            for line in content.split('\n'):
                if line.startswith('GROQ_API_KEY='):
                    key = line.split('=', 1)[1].strip()
                    if key and key not in ['', 'your_groq_api_key_here', 'gsk-your_groq_api_key_here']:
                        print(f"✅ GROQ_API_KEY configured: {key[:15]}...")
                    else:
                        print("❌ GROQ_API_KEY is placeholder/empty")
                    break
        else:
            print("❌ GROQ_API_KEY not found in .env")
    else:
        print("❌ .env file not found")
    
    # Try to import settings
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from config.settings import settings
        print(f"✅ Settings imported successfully")
        print(f"   Debug mode: {settings.debug_mode}")
        print(f"   Groq API available: {settings.groq_api_key is not None}")
    except Exception as e:
        print(f"❌ Settings import failed: {e}")
    
    # Check aiohttp
    try:
        import aiohttp
        print(f"✅ aiohttp installed: {aiohttp.__version__}")
    except ImportError:
        print("❌ aiohttp not installed")

def fix_env_file():
    """Fix common .env file issues"""
    print("🔧 Fixing .env file...")
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("Creating .env file from template...")
        example_file = Path(".env.example")
        if example_file.exists():
            with open(example_file, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print("✅ Created .env file from .env.example")
            print("⚠️ Please add your actual API keys to .env")
        else:
            # Create minimal .env
            with open(env_file, 'w') as f:
                f.write("# Pascal AI Environment Variables\n")
                f.write("GROQ_API_KEY=\n")
                f.write("DEBUG=false\n")
            print("✅ Created minimal .env file")
            print("⚠️ Please add your Groq API key to .env")
    else:
        # Fix common issues in existing .env
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        fixed = False
        new_lines = []
        has_groq = False
        
        for line in lines:
            # Fix GROK -> GROQ
            if line.startswith('GROK_API_KEY='):
                key_value = line.split('=', 1)[1].strip()
                new_lines.append(f'GROQ_API_KEY={key_value}\n')
                print("✅ Fixed: GROK_API_KEY renamed to GROQ_API_KEY")
                fixed = True
                has_groq = True
            elif line.startswith('GROQ_API_KEY='):
                new_lines.append(line)
                has_groq = True
            else:
                new_lines.append(line)
        
        # Add GROQ_API_KEY if missing
        if not has_groq:
            new_lines.append('\n# Groq API (Primary - fastest)\n')
            new_lines.append('GROQ_API_KEY=\n')
            print("✅ Added GROQ_API_KEY placeholder")
            fixed = True
        
        if fixed:
            with open(env_file, 'w') as f:
                f.writelines(new_lines)
            print("✅ .env file updated")
        else:
            print("✅ .env file looks good")

def test_current_info_flow():
    """Test the current info query flow"""
    print("🧪 Testing Current Info Query Flow:")
    
    try:
        import asyncio
        sys.path.insert(0, str(Path(__file__).parent))
        
        async def test():
            from modules.router import LightningRouter
            from modules.personality import PersonalityManager
            from modules.memory import MemoryManager
            
            personality_manager = PersonalityManager()
            memory_manager = MemoryManager()
            router = LightningRouter(personality_manager, memory_manager)
            
            # Test routing decision
            test_queries = [
                "What day is today?",
                "Hello, how are you?",
                "What's the current date?",
                "Who is the current president?"
            ]
            
            for query in test_queries:
                decision = router._decide_route(query)
                needs_current = router._needs_current_information(query)
                print(f"Query: '{query}'")
                print(f"  Needs current info: {needs_current}")
                print(f"  Route decision: {'offline' if decision.use_offline else 'online'}")
                print(f"  Reason: {decision.reason}")
                print()
            
            await router.close()
        
        asyncio.run(test())
        print("✅ Current info flow test passed")
        
    except Exception as e:
        print(f"❌ Current info flow test failed: {e}")
        import traceback
        traceback.print_exc()

def test_groq_direct():
    """Test Groq API directly"""
    print("🧪 Testing Groq API Directly:")
    
    try:
        import asyncio
        import aiohttp
        
        async def test():
            # Get API key from environment
            groq_key = os.getenv("GROQ_API_KEY")
            
            if not groq_key or groq_key in ['', 'your_groq_api_key_here', 'gsk-your_groq_api_key_here']:
                print("❌ No valid GROQ_API_KEY found")
                return False
            
            print(f"Using API key: {groq_key[:20]}...")
            
            session = aiohttp.ClientSession()
            try:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {groq_key}'
                }
                
                payload = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "user", "content": "Say 'test successful' and nothing else"}
                    ],
                    "max_tokens": 10,
                    "temperature": 0.1
                }
                
                async with session.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    print(f"Response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        if 'choices' in data and data['choices']:
                            content = data['choices'][0]['message']['content']
                            print(f"✅ Groq API works! Response: {content}")
                            return True
                    else:
                        error_text = await response.text()
                        print(f"❌ API error {response.status}: {error_text[:200]}")
                        return False
                        
            finally:
                await session.close()
        
        result = asyncio.run(test())
        return result
        
    except Exception as e:
        print(f"❌ Groq test failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Pascal Quick Fix Tool")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'fix-env':
            fix_env_file()
        elif command == 'test-current':
            test_current_info_flow()
        elif command == 'test-groq':
            test_groq_direct()
        elif command == 'diagnose':
            quick_diagnosis()
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  diagnose - Run quick diagnosis")
            print("  fix-env - Fix .env file issues")
            print("  test-current - Test current info routing")
            print("  test-groq - Test Groq API directly")
    else:
        # Default: run diagnosis
        quick_diagnosis()
        print("=" * 50)
        print("🔧 Quick Fixes:")
        print("1. For API key issues: python pascal_quick_fix.py fix-env")
        print("2. To test current info: python pascal_quick_fix.py test-current")
        print("3. To test Groq directly: python pascal_quick_fix.py test-groq")
        print("4. Full Groq test: python test_groq_fix.py")
        print("5. Start Pascal: ./run.sh")

if __name__ == "__main__":
    main()
