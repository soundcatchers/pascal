"""
Pascal AI Assistant - COMPLETE Main Entry Point (Simplified)
Streamlined for Pi 5 with Groq + Ollama only
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

# Import components (keeping existing structure)
from config.settings import settings
from modules.router import LightningRouter
from modules.personality import PersonalityManager
from modules.memory import MemoryManager

class Pascal:
    """Main Pascal AI Assistant class - Simplified"""
    
    def __init__(self):
        self.console = Console()
        self.running = False
        
        # Core components
        self.router = None
        self.personality_manager = None
        self.memory_manager = None
        
        # Performance tracking
        self.session_stats = {
            'queries': 0,
            'current_info_queries': 0,
            'offline_queries': 0,
            'online_queries': 0,
            'start_time': None
        }
    
    async def initialize(self):
        """Initialize Pascal's components"""
        try:
            self.console.print(Panel.fit(
                Text("⚡ Pascal AI Assistant v3.0 - Simplified & Fast", style="bold cyan"),
                border_style="cyan"
            ))
            
            # Initialize personality and memory
            self.personality_manager = PersonalityManager()
            self.memory_manager = MemoryManager()
            
            # Load default personality
            await self.personality_manager.load_personality("default")
            await self.memory_manager.load_session()
            
            # Initialize router (it will handle LLM initialization)
            self.router = LightningRouter(self.personality_manager, self.memory_manager)
            
            # Check LLM availability
            await self.router._check_llm_availability()
            
            if not self.router.offline_available and not self.router.online_available:
                self.console.print("❌ Failed to initialize - no LLMs available", style="red")
                return False
            
            self.console.print("⚡ Pascal initialized - ready for lightning-fast responses!", style="green")
            return True
            
        except Exception as e:
            self.console.print(f"❌ Initialization failed: {e}", style="red")
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
            return False
    
    def display_status(self):
        """Display Pascal's current status with performance metrics"""
        config = settings.get_config_summary()
        router_stats = self.router.get_router_stats() if self.router else {}
        
        # Create status table
        status_table = Table(title="⚡ Pascal Status (Simplified)", border_style="blue")
        status_table.add_column("Component", style="cyan", no_wrap=True)
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="white")
        
        # Basic info
        status_table.add_row("Version", f"v{config['pascal_version']}", "Simplified & Fast")
        status_table.add_row("Personality", config['personality'], "Active")
        status_table.add_row("Memory", "✅ Enabled" if config['memory_enabled'] else "❌ Disabled", "")
        status_table.add_row("Streaming", "⚡ Enabled" if config['streaming_enabled'] else "❌ Disabled", "Lightning responses")
        status_table.add_row("Target Response", f"{config['target_response_time']}s", "Maximum time")
        
        # LLM status
        if router_stats.get('online_available'):
            status_table.add_row("Online (Groq)", "✅ Ready", "Current info enabled")
        else:
            status_table.add_row("Online (Groq)", "❌ Not available", "Check API key")
        
        if router_stats.get('offline_available'):
            # Get Ollama-specific status
            if self.router and self.router.offline_llm:
                try:
                    offline_stats = self.router.offline_llm.get_performance_stats()
                    current_model = offline_stats.get('current_model', 'Unknown')
                    status_table.add_row("Offline (Ollama)", "✅ Ready", f"Model: {current_model}")
                except:
                    status_table.add_row("Offline (Ollama)", "✅ Ready", "Model loaded")
            else:
                status_table.add_row("Offline (Ollama)", "✅ Ready", "Connected")
        else:
            status_table.add_row("Offline (Ollama)", "❌ Not available", "Check Ollama")
        
        # Performance stats
        if router_stats.get('stats'):
            stats = router_stats['stats']
            status_table.add_row("Offline Requests", str(stats.get('offline_requests', 0)), 
                               f"Avg: {stats.get('offline_avg_time', 0):.2f}s")
            status_table.add_row("Online Requests", str(stats.get('online_requests', 0)), 
                               f"Avg: {stats.get('online_avg_time', 0):.2f}s")
            status_table.add_row("Current Info", str(stats.get('current_info_requests', 0)), 
                               "Auto-routed online")
        
        # Session stats
        status_table.add_row("Session Queries", str(self.session_stats['queries']), "This session")
        
        self.console.print(status_table)
        
        # Performance tips
        perf_text = """[bold]⚡ Performance Tips:[/bold]
  • Current info (date, news, weather) → Groq (1-3s)
  • General questions → Ollama (0.5-2s)
  • Streaming enabled for instant feedback
  • Binary routing for maximum speed
  
[bold]Quick Commands:[/bold]
  • help/status - Show this information
  • clear - Clear conversation history
  • debug - Toggle debug mode
  • stats - Show detailed statistics
  • quit/exit - Stop Pascal"""
        
        self.console.print(Panel(perf_text, title="Quick Help", border_style="green"))
    
    async def process_command(self, user_input: str) -> bool:
        """Process special commands"""
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'bye']:
            return False
        
        elif command in ['help', 'status']:
            self.display_status()
        
        elif command == 'clear':
            await self.memory_manager.clear_session()
            self.console.print("✅ Conversation cleared", style="green")
        
        elif command == 'debug':
            settings.debug_mode = not settings.debug_mode
            self.console.print(f"Debug mode: {'ON' if settings.debug_mode else 'OFF'}", style="yellow")
        
        elif command == 'stats':
            # Show detailed statistics
            if self.router:
                stats = self.router.get_router_stats()
                self.console.print(f"Router Stats: {stats}", style="dim")
            
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
        """Main chat interaction loop with streaming support"""
        import time
        self.session_stats['start_time'] = time.time()
        
        self.console.print("\n💬 Chat with Pascal (⚡ Simplified & Fast)\n", style="cyan")
        self.console.print("Type 'help' for commands. Responses stream instantly!\n", style="dim")
        
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
                command_keywords = ['help', 'status', 'clear', 'debug', 'stats']
                if any(user_input.lower().startswith(cmd) for cmd in command_keywords):
                    continue
                
                # Update session stats
                self.session_stats['queries'] += 1
                
                # Check if current info query for stats
                if self.router:
                    is_current_info = self.router._needs_current_information(user_input)
                    if is_current_info:
                        self.session_stats['current_info_queries'] += 1
                
                # Stream Pascal's response
                self.console.print("Pascal: ", style="bold magenta", end="")
                
                try:
                    # Stream response with instant feedback
                    response_text = ""
                    async for chunk in self.router.get_streaming_response(user_input):
                        print(chunk, end="", flush=True)
                        response_text += chunk
                    print()  # New line after streaming
                    
                    # Store complete response in memory
                    if response_text:
                        await self.memory_manager.add_interaction(user_input, response_text)
                    
                    # Update routing stats
                    if self.router and self.router.last_decision:
                        if self.router.last_decision.use_online:
                            self.session_stats['online_queries'] += 1
                        else:
                            self.session_stats['offline_queries'] += 1
                
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
        """Graceful shutdown"""
        self.console.print("\n🔄 Shutting down Pascal...", style="yellow")
        
        try:
            # Save memory
            if self.memory_manager:
                await self.memory_manager.save_session()
            
            # Close router
            if self.router:
                await self.router.close()
            
            # Show session summary
            if self.session_stats['queries'] > 0:
                import time
                session_duration = time.time() - self.session_stats['start_time']
                
                summary_table = Table(title="📊 Session Summary", border_style="blue")
                summary_table.add_column("Metric", style="cyan")
                summary_table.add_column("Value", style="white")
                
                summary_table.add_row("Total Queries", str(self.session_stats['queries']))
                summary_table.add_row("Current Info Queries", str(self.session_stats['current_info_queries']))
                summary_table.add_row("Offline Queries", str(self.session_stats['offline_queries']))
                summary_table.add_row("Online Queries", str(self.session_stats['online_queries']))
                summary_table.add_row("Session Duration", f"{session_duration/60:.1f} minutes")
                
                if session_duration > 0:
                    qpm = self.session_stats['queries'] / (session_duration / 60)
                    summary_table.add_row("Queries per Minute", f"{qpm:.1f}")
                
                self.console.print(summary_table)
            
            self.running = False
            self.console.print("👋 Goodbye!", style="cyan")
            
        except Exception as e:
            self.console.print(f"Error during shutdown: {e}", style="red")
    
    async def run(self):
        """Main run method"""
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize
        if not await self.initialize():
            return
        
        self.running = True
        
        # Show initial status
        self.display_status()
        
        # Start chat loop
        await self.chat_loop()
        
        # Shutdown
        await self.shutdown()

async def main():
    """Main entry point"""
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
