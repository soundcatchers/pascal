"""
Pascal AI Assistant - FIXED Main Entry Point (v4.0)
Features: Enhanced Skills + Nemotron + Groq with 3-tier routing
"""

import asyncio
import signal
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import enhanced components
from config.settings import settings
from modules.router import EnhancedRouter as LightningRouter
from modules.personality import PersonalityManager
from modules.memory import MemoryManager

class Pascal:
    """FIXED Enhanced Pascal AI Assistant with 3-tier system"""
    
    def __init__(self):
        self.console = Console()
        self.running = False
        
        # Core components
        self.router = None
        self.personality_manager = None
        self.memory_manager = None
        
        # Enhanced performance tracking
        self.session_stats = {
            'queries': 0,
            'skill_queries': 0,
            'current_info_queries': 0,
            'offline_queries': 0,
            'online_queries': 0,
            'start_time': None,
            'skills_time_saved': 0.0
        }
    
    async def initialize(self):
        """FIXED: Initialize Pascal's enhanced components"""
        try:
            self.console.print(Panel.fit(
                Text("⚡ Pascal AI Assistant v4.0 - Enhanced Skills Edition", style="bold cyan"),
                border_style="cyan"
            ))
            
            # Initialize personality and memory
            self.personality_manager = PersonalityManager()
            self.memory_manager = MemoryManager()
            
            # Load default personality
            await self.personality_manager.load_personality("default")
            await self.memory_manager.load_session()
            
            # FIXED: Initialize enhanced router with skills
            self.router = LightningRouter(self.personality_manager, self.memory_manager)
            
            # FIXED: Check enhanced system availability
            await self.router._check_system_availability()
            
            # Verify at least one system is available
            if not any([self.router.skills_available, self.router.offline_available, self.router.online_available]):
                self.console.print("❌ Failed to initialize - no systems available", style="red")
                self.console.print("\n🔧 Quick fixes:", style="yellow")
                self.console.print("• For enhanced skills: Add API keys to .env file")
                self.console.print("• For offline: sudo systemctl start ollama")
                self.console.print("• For online: Add GROQ_API_KEY to .env")
                return False
            
            # Show system health
            health = self.router.get_system_health()
            health_color = "green" if health['overall_health_score'] >= 80 else "yellow" if health['overall_health_score'] >= 60 else "red"
            
            self.console.print(f"🎯 System Health: {health['overall_health_score']}/100 ({health['system_status']})", style=health_color)
            
            self.console.print("⚡ Pascal enhanced system ready for ultra-fast responses!", style="green")
            return True
            
        except Exception as e:
            self.console.print(f"❌ Initialization failed: {e}", style="red")
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
            return False
    
    def display_status(self):
        """FIXED: Display Pascal's enhanced status with performance metrics"""
        config = settings.get_config_summary()
        router_stats = self.router.get_router_stats() if self.router else {}
        
        # Create enhanced status table
        status_table = Table(title="⚡ Pascal Enhanced Status", border_style="blue")
        status_table.add_column("Component", style="cyan", no_wrap=True)
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="white")
        
        # Basic info
        status_table.add_row("Version", f"v{config['pascal_version']}", "Enhanced Skills Edition")
        status_table.add_row("Personality", config['personality'], "Active")
        status_table.add_row("Memory", "✅ Enabled" if config['memory_enabled'] else "❌ Disabled", "")
        status_table.add_row("Streaming", "⚡ Enabled" if config['streaming_enabled'] else "❌ Disabled", "Real-time responses")
        
        # Enhanced system status
        if router_stats.get('system_status'):
            system_status = router_stats['system_status']
            
            # Enhanced Skills status
            if system_status.get('enhanced_skills'):
                skills_info = self.router.get_available_skills() if self.router else []
                api_skills = [s for s in skills_info if s.get('api_required', False)]
                configured_apis = [s for s in api_skills if s.get('api_configured', False)]
                
                status_table.add_row(
                    "Enhanced Skills", 
                    "✅ Available", 
                    f"{len(configured_apis)}/{len(api_skills)} APIs configured"
                )
                
                # Show individual skills
                for skill in skills_info[:4]:  # Show top 4 skills
                    skill_status = "✅" if not skill.get('api_required') or skill.get('api_configured') else "⚠️"
                    status_table.add_row(
                        f"  └─ {skill['name'].title()}", 
                        f"{skill_status} {skill['confidence']}", 
                        skill['speed']
                    )
            else:
                status_table.add_row("Enhanced Skills", "❌ Not available", "Check API keys")
            
            # Offline LLM status
            if system_status.get('offline_llm'):
                status_table.add_row("Offline (Nemotron)", "✅ Ready", "Local AI via Ollama")
            else:
                status_table.add_row("Offline (Nemotron)", "❌ Not available", "Check Ollama")
            
            # Online LLM status  
            if system_status.get('online_llm'):
                status_table.add_row("Online (Groq)", "✅ Ready", "Current info enabled")
            else:
                status_table.add_row("Online (Groq)", "❌ Not available", "Check API key")
        
        # Enhanced performance stats
        if router_stats.get('performance_stats'):
            stats = router_stats['performance_stats']
            
            total_requests = stats.get('total_requests', 0)
            if total_requests > 0:
                status_table.add_row("Total Requests", str(total_requests), "This session")
                status_table.add_row("Skills Used", stats.get('skill_percentage', '0%'), f"Avg: {stats.get('skill_avg_time', 0):.3f}s")
                status_table.add_row("Offline Used", stats.get('offline_percentage', '0%'), f"Avg: {stats.get('offline_avg_time', 0):.3f}s")
                status_table.add_row("Online Used", stats.get('online_percentage', '0%'), f"Avg: {stats.get('online_avg_time', 0):.3f}s")
                status_table.add_row("Time Saved", stats.get('total_time_saved', '0s'), "Via enhanced skills")
                status_table.add_row("Efficiency", stats.get('efficiency_ratio', '1x'), "Skills vs LLMs")
        
        self.console.print(status_table)
        
        # Enhanced performance tips
        perf_text = """[bold]⚡ Enhanced Performance Features:[/bold]
  • Instant responses: Date/Time, Calculator (0.001s)
  • Fast API responses: Weather, News (0.5-2s) 
  • Local AI: Nemotron for general queries (0.5-2s)
  • Online AI: Groq for complex current info (2-4s)
  • Smart routing automatically chooses fastest method
  
[bold]Quick Commands:[/bold]
  • help/status - Show this information
  • skills - List all available enhanced skills
  • health - Show detailed system health
  • clear - Clear conversation history
  • debug - Toggle debug mode
  • quit/exit - Stop Pascal
  
[bold]Example Queries:[/bold]
  • "What time is it?" → Instant datetime skill
  • "20% of 150" → Instant calculator skill  
  • "Weather in London" → Fast weather API
  • "Latest news" → Fast news API
  • "Hello, how are you?" → Local Nemotron
  • "What's happening today?" → Online Groq"""
        
        self.console.print(Panel(perf_text, title="Enhanced Features Guide", border_style="green"))
        
        # Show recommendations if any
        if router_stats.get('recommendations'):
            rec_text = "\n".join([f"• {rec['message']}" for rec in router_stats['recommendations'][:3]])
            self.console.print(Panel(rec_text, title="💡 Recommendations", border_style="yellow"))
    
    async def process_command(self, user_input: str) -> bool:
        """FIXED: Process special commands including enhanced skills commands"""
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'bye']:
            return False
        
        elif command in ['help', 'status']:
            self.display_status()
        
        elif command == 'skills':
            # Show enhanced skills information
            if self.router and self.router.skills_manager:
                skills_info = self.router.get_available_skills()
                
                skills_table = Table(title="🚀 Enhanced Skills Available", border_style="cyan")
                skills_table.add_column("Skill", style="cyan")
                skills_table.add_column("Status", style="green")
                skills_table.add_column("Speed", style="yellow")
                skills_table.add_column("Examples", style="white")
                
                for skill in skills_info:
                    status = "✅ Ready" if not skill.get('api_required') or skill.get('api_configured') else "⚠️ API needed"
                    examples = ", ".join(skill['examples'][:2])  # Show first 2 examples
                    
                    skills_table.add_row(
                        skill['name'].title(),
                        status,
                        skill['speed'],
                        examples + "..."
                    )
                
                self.console.print(skills_table)
                
                # Show detailed stats
                stats = self.router.skills_manager.get_skill_stats()
                if stats:
                    self.console.print(f"\n📊 Skills Performance: {stats}", style="dim")
            else:
                self.console.print("❌ Enhanced skills not available", style="red")
        
        elif command == 'health':
            # Show detailed system health
            if self.router:
                health = self.router.get_system_health()
                
                health_table = Table(title="🏥 System Health Report", border_style="blue")
                health_table.add_column("Component", style="cyan")
                health_table.add_column("Status", style="green")
                health_table.add_column("Details", style="white")
                
                health_table.add_row("Overall Health", f"{health['overall_health_score']}/100", health['system_status'])
                
                for component, status in health['components'].items():
                    component_name = component.replace('_', ' ').title()
                    health_table.add_row(component_name, status, "")
                
                self.console.print(health_table)
                
                # Show performance summary
                perf = health['performance_summary']
                self.console.print(f"\n📈 Performance: {perf['total_requests']} requests, {perf['skills_used_percentage']} via skills, {perf['total_time_saved']} saved", style="dim")
                
                # Show recommendations
                if health['recommendations']:
                    rec_text = "\n".join([f"• {rec['message']}" for rec in health['recommendations']])
                    self.console.print(Panel(rec_text, title="Recommendations", border_style="yellow"))
            else:
                self.console.print("❌ Router not available", style="red")
        
        elif command == 'clear':
            await self.memory_manager.clear_session()
            self.console.print("✅ Conversation cleared", style="green")
        
        elif command == 'debug':
            settings.debug_mode = not settings.debug_mode
            self.console.print(f"Debug mode: {'ON' if settings.debug_mode else 'OFF'}", style="yellow")
        
        elif command == 'stats':
            # Show detailed enhanced statistics
            if self.router:
                stats = self.router.get_router_stats()
                self.console.print(f"Enhanced Router Stats: {stats}", style="dim")
            
            # Show session statistics
            import time
            if self.session_stats['start_time']:
                session_duration = time.time() - self.session_stats['start_time']
                session_info = {
                    'session_duration_minutes': f"{session_duration/60:.1f}",
                    'queries_per_minute': f"{self.session_stats['queries']/(session_duration/60):.1f}" if session_duration > 0 else "0",
                    **self.session_stats
                }
                self.console.print(f"Session Stats: {session_info}", style="dim")
        
        else:
            # Not a command, process as normal input
            return True
        
        return True  # Continue running
    
    async def chat_loop(self):
        """FIXED: Enhanced chat interaction loop with 3-tier routing"""
        import time
        self.session_stats['start_time'] = time.time()
        
        self.console.print("\n💬 Chat with Pascal Enhanced Edition\n", style="cyan")
        self.console.print("Type 'help' for commands. Responses optimized for maximum speed!\n", style="dim")
        
        # Show initial tip based on available systems
        if self.router:
            if self.router.skills_available:
                self.console.print("💡 Try: 'What time is it?', '20% of 150', 'Weather in London', or 'Latest news'\n", style="green")
            elif self.router.offline_available:
                self.console.print("💡 Local AI ready. Try any question!\n", style="green")
            elif self.router.online_available:
                self.console.print("💡 Online AI ready. Try current info queries!\n", style="green")
        
        while self.running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Process commands
                should_continue = await self.process_command(user_input)
                if not should_continue:
                    break
                
                # Skip if it was a command
                command_keywords = ['help', 'status', 'skills', 'health', 'clear', 'debug', 'stats']
                if any(user_input.lower().startswith(cmd) for cmd in command_keywords):
                    continue
                
                # Update session stats
                self.session_stats['queries'] += 1
                
                # Predict routing for stats
                if self.router:
                    decision = self.router._decide_route(user_input)
                    if decision.use_skill:
                        self.session_stats['skill_queries'] += 1
                    elif decision.use_offline:
                        self.session_stats['offline_queries'] += 1
                    elif decision.use_online:
                        self.session_stats['online_queries'] += 1
                    
                    if decision.is_current_info:
                        self.session_stats['current_info_queries'] += 1
                
                # Stream Pascal's response
                self.console.print("Pascal: ", style="bold magenta", end="")
                
                try:
                    # Stream response with enhanced routing
                    response_text = ""
                    response_start = time.time()
                    
                    async for chunk in self.router.get_streaming_response(user_input):
                        print(chunk, end="", flush=True)
                        response_text += chunk
                    
                    response_time = time.time() - response_start
                    print()  # New line after streaming
                    
                    # Store complete response in memory
                    if response_text:
                        await self.memory_manager.add_interaction(user_input, response_text)
                    
                    # Track time saved if skills were used
                    if self.router and self.router.last_decision and self.router.last_decision.use_skill:
                        estimated_llm_time = 2.0  # Average LLM time
                        time_saved = max(0, estimated_llm_time - response_time)
                        self.session_stats['skills_time_saved'] += time_saved
                    
                    # Show performance info in debug mode
                    if settings.debug_mode and self.router and self.router.last_decision:
                        decision = self.router.last_decision
                        route_type = "SKILL" if decision.use_skill else "NEMOTRON" if decision.use_offline else "GROQ"
                        self.console.print(f"[DEBUG] Route: {route_type}, Time: {response_time:.3f}s, Confidence: {decision.confidence:.2f}", style="dim")
                
                except Exception as e:
                    self.console.print(f"\nSorry, I encountered an error: {e}", style="red")
                    if settings.debug_mode:
                        import traceback
                        traceback.print_exc()
                
                print()  # Add spacing
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"\nUnexpected error: {e}", style="red")
                if settings.debug_mode:
                    import traceback
                    traceback.print_exc()
    
    async def shutdown(self):
        """Enhanced graceful shutdown"""
        self.console.print("\n🔄 Shutting down Pascal Enhanced Edition...", style="yellow")
        
        try:
            # Save memory
            if self.memory_manager:
                await self.memory_manager.save_session()
            
            # Close enhanced router
            if self.router:
                await self.router.close()
            
            # Show enhanced session summary
            if self.session_stats['queries'] > 0:
                import time
                session_duration = time.time() - self.session_stats['start_time']
                
                summary_table = Table(title="📊 Enhanced Session Summary", border_style="blue")
                summary_table.add_column("Metric", style="cyan")
                summary_table.add_column("Value", style="white")
                
                summary_table.add_row("Total Queries", str(self.session_stats['queries']))
                summary_table.add_row("Skills Used", str(self.session_stats['skill_queries']))
                summary_table.add_row("Current Info Queries", str(self.session_stats['current_info_queries']))
                summary_table.add_row("Offline Queries", str(self.session_stats['offline_queries']))
                summary_table.add_row("Online Queries", str(self.session_stats['online_queries']))
                summary_table.add_row("Session Duration", f"{session_duration/60:.1f} minutes")
                summary_table.add_row("Time Saved by Skills", f"{self.session_stats['skills_time_saved']:.1f}s")
                
                if session_duration > 0:
                    qpm = self.session_stats['queries'] / (session_duration / 60)
                    summary_table.add_row("Queries per Minute", f"{qpm:.1f}")
                    
                    skills_percentage = (self.session_stats['skill_queries'] / self.session_stats['queries']) * 100
                    summary_table.add_row("Skills Usage", f"{skills_percentage:.1f}%")
                
                self.console.print(summary_table)
            
            self.running = False
            self.console.print("👋 Thank you for using Pascal Enhanced Edition!", style="cyan")
            
        except Exception as e:
            self.console.print(f"Error during shutdown: {e}", style="red")
    
    async def run(self):
        """Main run method for enhanced Pascal"""
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize enhanced system
        if not await self.initialize():
            return
        
        self.running = True
        
        # Show initial enhanced status
        self.display_status()
        
        # Start enhanced chat loop
        await self.chat_loop()
        
        # Enhanced shutdown
        await self.shutdown()

async def main():
    """Main entry point for Pascal Enhanced Edition"""
    pascal = Pascal()
    await pascal.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        if hasattr(settings, 'debug_mode') and settings.debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)
