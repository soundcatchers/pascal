"""
Pascal AI Assistant - FIXED Main Entry Point (v4.0)
Features: Simplified Nemotron + Groq with fast routing + Enhanced Conversational Context
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

# Import components
from config.settings import settings
from modules.enhanced_context_router import EnhancedIntelligentRouter
from modules.personality import PersonalityManager
from modules.memory import MemoryManager

class Pascal:
    """FIXED Pascal AI Assistant with enhanced conversational context"""
    
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
            'follow_up_queries': 0,
            'context_enhanced_queries': 0,
            'start_time': None,
        }
    
    async def initialize(self):
        """FIXED: Initialize Pascal's enhanced conversational components"""
        try:
            self.console.print(Panel.fit(
                Text("⚡ Pascal AI Assistant v4.0 - Enhanced Conversational Edition", style="bold cyan"),
                border_style="cyan"
            ))
            
            # Initialize personality and memory
            self.personality_manager = PersonalityManager()
            self.memory_manager = MemoryManager()
            
            # Load default personality
            await self.personality_manager.load_personality("default")
            await self.memory_manager.load_session()
            
            # FIXED: Initialize enhanced intelligent router
            self.router = EnhancedIntelligentRouter.create(self.personality_manager, self.memory_manager)
            
            # FIXED: Check system availability using correct method
            await self.router._check_llm_availability()
            
            # Verify at least one system is available
            if not (self.router.offline_available or self.router.online_available):
                self.console.print("❌ Failed to initialize - no systems available", style="red")
                self.console.print("\n🔧 Quick fixes:", style="yellow")
                self.console.print("• For offline: sudo systemctl start ollama")
                self.console.print("• For online: Add GROQ_API_KEY to .env")
                return False
            
            # FIXED: Get system health properly
            health_score = self._calculate_system_health()
            health_color = "green" if health_score >= 80 else "yellow" if health_score >= 60 else "red"
            
            self.console.print(f"🎯 System Health: {health_score}/100", style=health_color)
            
            self.console.print("⚡ Pascal enhanced conversational system ready!", style="green")
            return True
            
        except Exception as e:
            self.console.print(f"❌ Initialization failed: {e}", style="red")
            if settings.debug_mode:
                import traceback
                traceback.print_exc()
            return False
    
    def _calculate_system_health(self) -> int:
        """Calculate overall system health score"""
        health = 0
        
        if self.router.offline_available:
            health += 30
        if self.router.online_available:
            health += 40
        if self.router.skills_available:
            health += 20
        
        health += 10  # Router always available
        return health
    
    async def chat_loop(self):
        """FIXED: Enhanced conversational chat interaction loop"""
        import time
        self.session_stats['start_time'] = time.time()
        
        self.console.print("\n💬 Chat with Pascal Enhanced Conversational Edition\n", style="cyan")
        self.console.print("Type 'help' for commands. Now with enhanced context awareness!\n", style="dim")
        
        # Show initial tip based on available systems
        if self.router:
            if self.router.offline_available and self.router.online_available:
                self.console.print("💡 Try: 'Who is the UK PM?' then 'What about their deputy?'\n", style="green")
            elif self.router.online_available:
                self.console.print("💡 Online AI ready for current information and follow-ups!\n", style="green")
        
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
                command_keywords = ['help', 'status', 'health', 'clear', 'debug', 'stats', 'context']
                if any(user_input.lower().startswith(cmd) for cmd in command_keywords):
                    continue
                
                # Update session stats
                self.session_stats['queries'] += 1
                
                # Stream Pascal's response using enhanced router
                self.console.print("Pascal: ", style="bold magenta", end="")
                
                try:
                    response_text = ""
                    response_start = time.time()
                    
                    # Use enhanced streaming response
                    async for chunk in self.router.get_enhanced_streaming_response(user_input, session_id=self.memory_manager.session_id):
                        print(chunk, end="", flush=True)
                        response_text += chunk
                    
                    response_time = time.time() - response_start
                    print()  # New line after streaming
                    
                    # Update stats based on routing decision
                    if self.router.last_decision:
                        decision = self.router.last_decision
                        if decision.use_offline:
                            self.session_stats['offline_queries'] += 1
                        elif decision.use_online:
                            self.session_stats['online_queries'] += 1
                        
                        # Track follow-ups and context enhancements
                        if 'follow-up' in decision.reason.lower():
                            self.session_stats['follow_up_queries'] += 1
                        if 'context' in decision.reason.lower():
                            self.session_stats['context_enhanced_queries'] += 1
                    
                    # Show debug info
                    if settings.debug_mode and self.router.last_decision:
                        decision = self.router.last_decision
                        route_type = "NEMOTRON" if decision.use_offline else "GROQ" if decision.use_online else "SKILL"
                        extra_info = []
                        if 'follow-up' in decision.reason.lower():
                            extra_info.append("Follow-up")
                        if 'context' in decision.reason.lower():
                            extra_info.append("Context-Enhanced")
                        extra_str = f" [{', '.join(extra_info)}]" if extra_info else ""
                        self.console.print(f"[DEBUG] Route: {route_type}, Time: {response_time:.3f}s, Confidence: {decision.confidence:.2f}{extra_str}", style="dim")
                
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
    
    async def process_command(self, user_input: str) -> bool:
        """FIXED: Process special commands"""
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'bye']:
            return False
        
        elif command in ['help', 'status']:
            self.display_status()
        
        elif command == 'clear':
            await self.memory_manager.clear_session()
            # Reset router context
            if hasattr(self.router, 'conversation_context'):
                self.router.conversation_context = {
                    'recent_topics': [],
                    'last_intent': None,
                    'source_memory': {},
                    'follow_up_chain': [],
                    'entities_mentioned': [],
                    'current_topic': None,
                    'topic_history': []
                }
            self.console.print("✅ Conversation and context cleared", style="green")
        
        elif command == 'context':
            # Show current conversation context
            if hasattr(self.router, 'get_context_summary'):
                context_summary = self.router.get_context_summary()
                self.console.print(Panel(context_summary, title="🧠 Current Context", border_style="blue"))
            else:
                self.console.print("❌ Context not available", style="red")
        
        elif command == 'debug':
            settings.debug_mode = not settings.debug_mode
            self.console.print(f"Debug mode: {'ON' if settings.debug_mode else 'OFF'}", style="yellow")
        
        else:
            # Not a command, process as normal input
            return True
        
        return True  # Continue running
    
    def display_status(self):
        """FIXED: Display Pascal's enhanced conversational status"""
        config = settings.get_config_summary()
        
        # Create status table
        status_table = Table(title="⚡ Pascal Enhanced Conversational Status", border_style="blue")
        status_table.add_column("Component", style="cyan", no_wrap=True)
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="white")
        
        # Basic info
        status_table.add_row("Version", f"v{config['pascal_version']}", "Enhanced Conversational Edition")
        status_table.add_row("Context Awareness", "✅ Enhanced", "Maintains conversation flow")
        status_table.add_row("Follow-up Detection", "✅ Advanced", "Understands pronouns and references")
        status_table.add_row("Search Enhancement", "✅ Intelligent", "Context-aware query optimization")
        status_table.add_row("Streaming", "⚡ Enabled" if config['streaming_enabled'] else "❌ Disabled", "Real-time responses")
        
        # System status
        system_status = self._get_system_status()
        
        if system_status.get('offline_llm'):
            status_table.add_row("Offline (Nemotron)", "✅ Ready", "Local AI via Ollama")
        else:
            status_table.add_row("Offline (Nemotron)", "❌ Not available", "Check Ollama")
        
        if system_status.get('online_llm'):
            status_table.add_row("Online (Groq)", "✅ Ready", "Current info + enhanced search")
        else:
            status_table.add_row("Online (Groq)", "❌ Not available", "Check API key")
        
        if system_status.get('skills_manager'):
            status_table.add_row("Skills", "✅ Ready", "Enhanced capabilities")
        else:
            status_table.add_row("Skills", "❌ Not available", "Check API keys")
        
        self.console.print(status_table)
        
        # Show conversation context if available
        if self.router and hasattr(self.router, 'conversation_context'):
            ctx = self.router.conversation_context
            if ctx.get('current_topic') or ctx.get('entities_mentioned'):
                context_info = []
                if ctx.get('current_topic'):
                    context_info.append(f"Current Topic: {ctx['current_topic']}")
                if ctx.get('entities_mentioned'):
                    context_info.append(f"Entities: {', '.join(ctx['entities_mentioned'][:3])}")
                if ctx.get('topic_history'):
                    context_info.append(f"Recent Topics: {', '.join(ctx['topic_history'][-3:])}")
                
                if context_info:
                    context_text = "\n".join(context_info)
                    self.console.print(Panel(context_text, title="🧠 Current Context", border_style="yellow"))
        
        # Enhanced conversational features
        perf_text = """[bold]⚡ Enhanced Conversational Features:[/bold]
  • Context Awareness: Remembers conversation topics and sources
  • Smart Follow-ups: Understands "who came second?" after F1 questions
  • Search Enhancement: Transforms "he" to "Keir Starmer" in follow-ups
  • Topic Flow: Tracks conversation progression for better responses
  
[bold]Conversational Examples:[/bold]
  • "Who is the UK Prime Minister?" → Gets current info
  • "What about their deputy?" → Enhances to "UK Deputy Prime Minister"
  • "Didn't he get a new one today?" → "Keir Starmer UK Deputy PM new today"
  
[bold]Quick Commands:[/bold]
  • help/status - Show this information
  • context - Show current conversation context
  • clear - Clear conversation history and context
  • debug - Toggle debug mode
  • quit/exit - Stop Pascal"""
        
        self.console.print(Panel(perf_text, title="Enhanced Conversational Features", border_style="green"))
        
        # Show session stats if available
        if self.session_stats['queries'] > 0:
            stats_text = f"""Queries: {self.session_stats['queries']} | Follow-ups: {self.session_stats['follow_up_queries']} | Context-Enhanced: {self.session_stats['context_enhanced_queries']}
Offline: {self.session_stats['offline_queries']} | Online: {self.session_stats['online_queries']}"""
            self.console.print(Panel(stats_text, title="📊 Session Stats", border_style="blue"))
    
    def _get_system_status(self) -> dict:
        """Get detailed system status"""
        return {
            'offline_llm': self.router.offline_available if self.router else False,
            'online_llm': self.router.online_available if self.router else False,
            'skills_manager': self.router.skills_available if self.router else False,
        }
    
    async def shutdown(self):
        """Enhanced graceful shutdown"""
        self.console.print("\n🔄 Shutting down Pascal Enhanced Conversational Edition...", style="yellow")
        
        try:
            # Save memory
            if self.memory_manager:
                await self.memory_manager.save_session()
            
            # Close router connections properly
            if self.router and hasattr(self.router, 'close'):
                await self.router.close()
            
            # Show enhanced session summary
            if self.session_stats['queries'] > 0:
                import time
                session_duration = time.time() - self.session_stats['start_time']
                
                summary_table = Table(title="📊 Enhanced Session Summary", border_style="blue")
                summary_table.add_column("Metric", style="cyan")
                summary_table.add_column("Value", style="white")
                
                summary_table.add_row("Total Queries", str(self.session_stats['queries']))
                summary_table.add_row("Follow-up Queries", str(self.session_stats['follow_up_queries']))
                summary_table.add_row("Context-Enhanced", str(self.session_stats['context_enhanced_queries']))
                summary_table.add_row("Offline Queries", str(self.session_stats['offline_queries']))
                summary_table.add_row("Online Queries", str(self.session_stats['online_queries']))
                summary_table.add_row("Session Duration", f"{session_duration/60:.1f} minutes")
                
                if self.session_stats['queries'] > 0:
                    follow_up_rate = (self.session_stats['follow_up_queries'] / self.session_stats['queries']) * 100
                    context_rate = (self.session_stats['context_enhanced_queries'] / self.session_stats['queries']) * 100
                    summary_table.add_row("Follow-up Rate", f"{follow_up_rate:.1f}%")
                    summary_table.add_row("Context Enhancement Rate", f"{context_rate:.1f}%")
                
                self.console.print(summary_table)
            
            self.running = False
            self.console.print("👋 Thank you for using Pascal Enhanced Conversational Edition!", style="cyan")
            
        except Exception as e:
            self.console.print(f"Error during shutdown: {e}", style="red")
    
    async def run(self):
        """Main run method for enhanced conversational Pascal"""
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize enhanced system
        if not await self.initialize():
            return
        
        self.running = True
        
        # Show initial status
        self.display_status()
        
        # Start enhanced chat loop
        await self.chat_loop()
        
        # Shutdown
        await self.shutdown()

async def main():
    """Main entry point for Pascal Enhanced Conversational Edition"""
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
