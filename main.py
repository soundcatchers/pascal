"""
Pascal AI Assistant - FIXED Main Entry Point (v4.0)
Features: Simplified Nemotron + Groq with fast routing
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

# Import simplified components
from config.settings import settings
from modules.router import LightningRouter
from modules.personality import PersonalityManager
from modules.memory import MemoryManager

class Pascal:
    """FIXED Pascal AI Assistant with simplified routing"""
    
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
            'offline_queries': 0,
            'online_queries': 0,
            'start_time': None,
        }
    
    async def initialize(self):
        """FIXED: Initialize Pascal's simplified components"""
        try:
            self.console.print(Panel.fit(
                Text("⚡ Pascal AI Assistant v4.0 - Simplified Edition", style="bold cyan"),
                border_style="cyan"
            ))
            
            # Initialize personality and memory
            self.personality_manager = PersonalityManager()
            self.memory_manager = MemoryManager()
            
            # Load default personality
            await self.personality_manager.load_personality("default")
            await self.memory_manager.load_session()
            
            # FIXED: Initialize simplified router
            self.router = LightningRouter(self.personality_manager, self.memory_manager)
            
            # FIXED: Check system availability using correct method
            await self.router._check_llm_availability()
            
            # Verify at least one system is available
            if not (self.router.offline_available or self.router.online_available):
                self.console.print("❌ Failed to initialize - no systems available", style="red")
                self.console.print("\n🔧 Quick fixes:", style="yellow")
                self.console.print("• For offline: sudo systemctl start ollama")
                self.console.print("• For online: Add GROQ_API_KEY to .env")
                return False
            
            # Show system status
            health = self.router.get_system_health()
            health_color = "green" if health['overall_health_score'] >= 80 else "yellow" if health['overall_health_score'] >= 60 else "red"
            
            self.console.print(f"🎯 System Health: {health['overall_health_score']}/100 ({health['system_status']})", style=health_color)
            
            self.console.print("⚡ Pascal simplified system ready for fast responses!", style="green")
            return True
            
        except Exception as e:
            self.console.print(f"❌ Initialization failed: {e}", style="red")
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
            return False
    
    def display_status(self):
        """FIXED: Display Pascal's simplified status"""
        config = settings.get_config_summary()
        router_stats = self.router.get_router_stats() if self.router else {}
        
        # Create status table
        status_table = Table(title="⚡ Pascal Simplified Status", border_style="blue")
        status_table.add_column("Component", style="cyan", no_wrap=True)
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="white")
        
        # Basic info
        status_table.add_row("Version", f"v{config['pascal_version']}", "Simplified Edition")
        status_table.add_row("Personality", config['personality'], "Active")
        status_table.add_row("Streaming", "⚡ Enabled" if config['streaming_enabled'] else "❌ Disabled", "Real-time responses")
        
        # System status
        if router_stats.get('system_status'):
            system_status = router_stats['system_status']
            
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
        
        # Performance stats
        if router_stats.get('performance_stats'):
            stats = router_stats['performance_stats']
            
            total_requests = stats.get('total_requests', 0)
            if total_requests > 0:
                status_table.add_row("Total Requests", str(total_requests), "This session")
                status_table.add_row("Offline Used", stats.get('offline_percentage', '0%'), f"Avg: {stats.get('offline_avg_time', '0s')}")
                status_table.add_row("Online Used", stats.get('online_percentage', '0%'), f"Avg: {stats.get('online_avg_time', '0s')}")
        
        self.console.print(status_table)
        
        # Performance tips
        perf_text = """[bold]⚡ Simplified Performance Features:[/bold]
  • Local AI: Nemotron for general queries (1-3s)
  • Online AI: Groq for current info queries (2-4s)
  • Smart routing automatically chooses fastest method
  
[bold]Quick Commands:[/bold]
  • help/status - Show this information
  • clear - Clear conversation history
  • debug - Toggle debug mode
  • quit/exit - Stop Pascal
  
[bold]Example Queries:[/bold]
  • "Hello, how are you?" → Local Nemotron
  • "What day is today?" → Online Groq (current info)
  • "Explain Python" → Local Nemotron
  • "Latest news" → Online Groq (current info)"""
        
        self.console.print(Panel(perf_text, title="Simplified Features Guide", border_style="green"))
        
        # Show recommendations if any
        if router_stats.get('recommendations'):
            rec_text = "\n".join([f"• {rec}" for rec in router_stats['recommendations'][:3]])
            self.console.print(Panel(rec_text, title="💡 Recommendations", border_style="yellow"))
    
    async def process_command(self, user_input: str) -> bool:
        """FIXED: Process special commands"""
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'bye']:
            return False
        
        elif command in ['help', 'status']:
            self.display_status()
        
        elif command == 'health':
            # Show detailed system health
            if self.router:
                health = self.router.get_system_health()
                
                health_table = Table(title="🏥 System Health Report", border_style="blue")
                health_table.add_column("Component", style="cyan")
                health_table.add_column("Status", style="green")
                
                health_table.add_row("Overall Health", f"{health['overall_health_score']}/100")
                
                for component, status in health['components'].items():
                    component_name = component.replace('_', ' ').title()
                    health_table.add_row(component_name, status)
                
                self.console.print(health_table)
                
                # Show recommendations
                if health['recommendations']:
                    rec_text = "\n".join([f"• {rec}" for rec in health['recommendations']])
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
        """FIXED: Simplified chat interaction loop"""
        import time
        self.session_stats['start_time'] = time.time()
        
        self.console.print("\n💬 Chat with Pascal Simplified Edition\n", style="cyan")
        self.console.print("Type 'help' for commands. Optimized for speed!\n", style="dim")
        
        # Show initial tip based on available systems
        if self.router:
            if self.router.offline_available:
                self.console.print("💡 Local AI ready. Try: 'Hello Pascal' or 'Explain Python'\n", style="green")
            if self.router.online_available:
                self.console.print("💡 Online AI ready. Try: 'What day is today?' or 'Latest news'\n", style="green")
        
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
                command_keywords = ['help', 'status', 'health', 'clear', 'debug', 'stats']
                if any(user_input.lower().startswith(cmd) for cmd in command_keywords):
                    continue
                
                # Update session stats
                self.session_stats['queries'] += 1
                
                # Predict routing for stats
                if self.router:
                    decision = self.router._decide_route(user_input)
                    if decision.use_offline:
                        self.session_stats['offline_queries'] += 1
                    elif decision.use_online:
                        self.session_stats['online_queries'] += 1
                
                # Stream Pascal's response
                self.console.print("Pascal: ", style="bold magenta", end="")
                
                try:
                    # Stream response with simplified routing
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
                    
                    # Show performance info in debug mode
                    if settings.debug_mode and self.router and self.router.last_decision:
                        decision = self.router.last_decision
                        route_type = "NEMOTRON" if decision.use_offline else "GROQ"
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
        """Simplified graceful shutdown"""
        self.console.print("\n🔄 Shutting down Pascal Simplified Edition...", style="yellow")
        
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
                summary_table.add_row("Offline Queries", str(self.session_stats['offline_queries']))
                summary_table.add_row("Online Queries", str(self.session_stats['online_queries']))
                summary_table.add_row("Session Duration", f"{session_duration/60:.1f} minutes")
                
                if session_duration > 0:
                    qpm = self.session_stats['queries'] / (session_duration / 60)
                    summary_table.add_row("Queries per Minute", f"{qpm:.1f}")
                
                self.console.print(summary_table)
            
            self.running = False
            self.console.print("👋 Thank you for using Pascal Simplified Edition!", style="cyan")
            
        except Exception as e:
            self.console.print(f"Error during shutdown: {e}", style="red")
    
    async def run(self):
        """Main run method for simplified Pascal"""
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize simplified system
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
    """Main entry point for Pascal Simplified Edition"""
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
