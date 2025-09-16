#!/usr/bin/env python3
"""
Pascal Skills Manager Diagnostic Script
Debug and test the enhanced skills system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Enable debug mode
os.environ['DEBUG'] = 'true'

async def test_skills_system():
    """Test the enhanced skills system step by step"""
    print("🔧 Pascal Enhanced Skills Diagnostic")
    print("=" * 50)
    
    try:
        # Test 1: Import skills manager
        print("\n📦 Testing Skills Manager Import:")
        print("-" * 30)
        
        try:
            from modules.skills_manager import EnhancedSkillsManager
            print("✅ EnhancedSkillsManager imported successfully")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False
        
        # Test 2: Create skills manager instance
        print("\n🔧 Creating Skills Manager Instance:")
        print("-" * 30)
        
        try:
            skills_manager = EnhancedSkillsManager()
            print("✅ EnhancedSkillsManager instance created")
        except Exception as e:
            print(f"❌ Instance creation failed: {e}")
            return False
        
        # Test 3: Check environment variables
        print("\n🔑 Checking API Keys:")
        print("-" * 30)
        
        weather_key = os.getenv('OPENWEATHER_API_KEY')
        news_key = os.getenv('NEWS_API_KEY')
        
        print(f"OpenWeather API Key: {'✅ Set' if weather_key else '❌ Missing'}")
        print(f"News API Key: {'✅ Set' if news_key else '❌ Missing'}")
        
        if weather_key:
            print(f"  Weather key length: {len(weather_key)} chars")
        if news_key:
            print(f"  News key length: {len(news_key)} chars")
        
        # Test 4: Initialize skills manager
        print("\n🚀 Initializing Skills Manager:")
        print("-" * 30)
        
        try:
            api_status = await skills_manager.initialize()
            print("✅ Skills manager initialization completed")
            
            for service, status in api_status.items():
                status_icon = "✅" if status['available'] else "❌"
                print(f"  {status_icon} {service}: {status['message']}")
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 5: Test individual skills
        print("\n🧪 Testing Individual Skills:")
        print("-" * 30)
        
        test_queries = [
            ("What time is it?", "datetime"),
            ("What day is today?", "datetime"),
            ("15 + 23", "calculator"),
            ("20% of 150", "calculator"),
            ("Weather in London", "weather"),
            ("Latest news", "news")
        ]
        
        for query, expected_skill in test_queries:
            print(f"\nTesting: '{query}'")
            
            # Test skill detection
            detected_skill = skills_manager.can_handle_directly(query)
            if detected_skill:
                print(f"  ✅ Detected skill: {detected_skill}")
                if detected_skill == expected_skill:
                    print(f"  ✅ Correct skill detected")
                else:
                    print(f"  ⚠️ Expected {expected_skill}, got {detected_skill}")
                
                # Test skill execution
                try:
                    result = await skills_manager.execute_skill(query, detected_skill)
                    if result and result.success:
                        print(f"  ✅ Execution successful: {result.response[:100]}...")
                        print(f"  ⏱️ Execution time: {result.execution_time:.3f}s")
                    else:
                        print(f"  ❌ Execution failed: {result.response if result else 'No result'}")
                except Exception as e:
                    print(f"  ❌ Execution error: {e}")
            else:
                print(f"  ❌ No skill detected (expected: {expected_skill})")
        
        # Test 6: Test router integration
        print("\n🚦 Testing Router Integration:")
        print("-" * 30)
        
        try:
            from config.settings import settings
            from modules.router import EnhancedRouter
            from modules.personality import PersonalityManager
            from modules.memory import MemoryManager
            
            personality_manager = PersonalityManager()
            memory_manager = MemoryManager()
            router = EnhancedRouter(personality_manager, memory_manager)
            
            print("✅ Router components created")
            
            # Test router initialization
            await router._check_system_availability()
            
            print(f"✅ Router initialized")
            print(f"  Skills available: {router.skills_available}")
            print(f"  Skills manager: {router.skills_manager is not None}")
            
            if router.skills_available and router.skills_manager:
                # Test routing decisions
                for query, expected_skill in test_queries:
                    decision = router._decide_route(query)
                    print(f"  Query: '{query}' -> {decision.route_type} ({decision.reason})")
                    if decision.use_skill:
                        print(f"    Skill: {decision.skill_name}")
            
            await router.close()
            
        except Exception as e:
            print(f"❌ Router integration failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 7: Performance summary
        print("\n📊 Performance Summary:")
        print("-" * 30)
        
        skills_stats = skills_manager.get_skill_stats()
        if skills_stats:
            for skill_name, stats in skills_stats.items():
                print(f"  {skill_name.title()}:")
                print(f"    Executions: {stats['executions']}")
                print(f"    Success rate: {stats['success_rate']}")
                print(f"    Avg time: {stats['avg_execution_time']}")
        else:
            print("  No performance data available")
        
        # Clean up
        await skills_manager.close()
        
        print("\n" + "=" * 50)
        print("🎉 SKILLS DIAGNOSTIC COMPLETE")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main diagnostic function"""
    try:
        result = asyncio.run(test_skills_system())
        if result:
            print("\n✅ Skills system appears to be working!")
            print("\n💡 Next steps:")
            print("1. Run Pascal: ./run.sh")
            print("2. Test skills with: 'What time is it?'")
            print("3. Check routing with debug mode: 'debug' command")
        else:
            print("\n❌ Skills system has issues - check output above")
            print("\n🔧 Common fixes:")
            print("1. Install aiohttp: pip install aiohttp")
            print("2. Add API keys to .env file")
            print("3. Check Python path and imports")
        return 0 if result else 1
    except KeyboardInterrupt:
        print("\n⏹️ Diagnostic interrupted")
        return 1
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
