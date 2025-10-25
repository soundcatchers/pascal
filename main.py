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

# Import simplified components
from config.settings import settings
from modules.intelligent_router import IntelligentRouter
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
            'start_time': None,
        }
        
        # Conversational context
        self.conversation_context = {
            'last_topic': None,
            'last_sources': None,
            'conversation_flow': []
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
            
            # FIXED: Initialize IntelligentRouter (not LightningRouter)
            self.router = IntelligentRouter(self.personality_manager, self.memory_manager)
            
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
    
    def _get_system_status(self) -> dict:
        """Get detailed system status"""
        return {
            'offline_llm': self.router.offline_available if self.router else False,
            'online_llm': self.router.online_available if self.router else False,
            'skills_manager': self.router.skills_available if self.router else False,
        }
    
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
        status_table.add_row("Personality", config['personality'], "Active")
        status_table.add_row("Context Awareness", "✅ Enhanced", "Maintains conversation flow")
        status_table.add_row("Streaming", "⚡ Enabled" if config['streaming_enabled'] else "❌ Disabled", "Real-time responses")
        
        # System status
        system_status = self._get_system_status()
        
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
        
        # Skills status
        if system_status.get('skills_manager'):
            status_table.add_row("Skills", "✅ Ready", "Enhanced capabilities")
        else:
            status_table.add_row("Skills", "❌ Not available", "Check API keys")
        
        self.console.print(status_table)
        
        # Enhanced conversational features
        perf_text = """[bold]⚡ Enhanced Conversational Features:[/bold]
  • Context Awareness: Remembers conversation topics and sources
  • Smart Follow-ups: Understands "who came second?" after F1 questions
  • Source Continuity: Maintains reference to previous search results
  • Topic Flow: Tracks conversation progression for better responses
  
[bold]Conversational Examples:[/bold]
  • "Who won the last F1 grand prix?" → Gets current info
  • "Who came second?" → Understands context, continues with same topic
  • "What about third place?" → Maintains F1 context throughout
  
[bold]Quick Commands:[/bold]
  • help/status - Show this information
  • clear - Clear conversation history
  • debug - Toggle debug mode
  • quit/exit - Stop Pascal"""
        
        self.console.print(Panel(perf_text, title="Enhanced Conversational Features", border_style="green"))
    
    def _analyze_query_context(self, user_input: str) -> dict:
        """Analyze query for conversational context clues"""
        query_lower = user_input.lower().strip()
        
        # Context indicators
        follow_up_indicators = [
            'who came', 'what about', 'and then', 'what happened', 'tell me more',
            'who was', 'what was', 'where was', 'when was', 'how about',
            'second', 'third', 'next', 'after that', 'also', 'too'
        ]
        
        pronouns = ['that', 'it', 'they', 'them', 'he', 'she', 'him', 'her', 'this']
        
        is_follow_up = (
            any(indicator in query_lower for indicator in follow_up_indicators) or
            any(pronoun in query_lower.split() for pronoun in pronouns) or
            len(query_lower.split()) <= 5
        )
        
        return {
            'is_follow_up': is_follow_up,
            'query_type': 'follow_up' if is_follow_up else 'new_topic',
            'has_pronouns': any(pronoun in query_lower.split() for pronoun in pronouns),
            'word_count': len(query_lower.split())
        }
    
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
                health_score = self._calculate_system_health()
                system_status = self._get_system_status()
                
                health_table = Table(title="🏥 System Health Report", border_style="blue")
                health_table.add_column("Component", style="cyan")
                health_table.add_column("Status", style="green")
                
                health_table.add_row("Overall Health", f"{health_score}/100")
                health_table.add_row("Offline LLM", "✅ Available" if system_status['offline_llm'] else "❌ Unavailable")
                health_table.add_row("Online LLM", "✅ Available" if system_status['online_llm'] else "❌ Unavailable")
                health_table.add_row("Skills Manager", "✅ Available" if system_status['skills_manager'] else "❌ Unavailable")
                
                self.console.print(health_table)
                
                # Show recommendations
                recommendations = []
                if not system_status['offline_llm']:
                    recommendations.append("Enable offline: sudo systemctl start ollama")
                if not system_status['online_llm']:
                    recommendations.append("Configure Groq API key in .env")
                
                if recommendations:
                    rec_text = "\n".join([f"• {rec}" for rec in recommendations])
                    self.console.print(Panel(rec_text, title="Recommendations", border_style="yellow"))
            else:
                self.console.print("❌ Router not available", style="red")
        
        elif command == 'clear':
            await self.memory_manager.clear_session()
            self.conversation_context = {
                'last_topic': None,
                'last_sources': None,
                'conversation_flow': []
            }
            self.console.print("✅ Conversation cleared", style="green")
        
        elif command == 'debug':
            settings.debug_mode = not settings.debug_mode
            self.console.print(f"Debug mode: {'ON' if settings.debug_mode else 'OFF'}", style="yellow")
        
        elif command == 'context':
            # Show current conversation context
            context_info = f"""Current Context:
• Last Topic: {self.conversation_context.get('last_topic', 'None')}
• Has Sources: {bool(self.conversation_context.get('last_sources'))}
• Flow Length: {len(self.conversation_context.get('conversation_flow', []))}"""
            self.console.print(Panel(context_info, title="Conversation Context", border_style="blue"))
        
        elif command == 'stats':
            # Show detailed statistics
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
        """FIXED: Enhanced conversational chat interaction loop"""
        import time
        self.session_stats['start_time'] = time.time()
        
        self.console.print("\n💬 Chat with Pascal Enhanced Conversational Edition\n", style="cyan")
        self.console.print("Type 'help' for commands. Now with enhanced context awareness!\n", style="dim")
        
        # Show initial tip based on available systems
        if self.router:
            if self.router.offline_available:
                self.console.print("💡 Local AI ready. Try conversational follow-ups!\n", style="green")
            if self.router.online_available:
                self.console.print("💡 Online AI ready. Ask about current events, then follow up!\n", style="green")
        
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
                
                # Analyze conversational context
                context_analysis = self._analyze_query_context(user_input)
                
                # Update session stats
                self.session_stats['queries'] += 1
                
                # Update conversation flow
                self.conversation_context['conversation_flow'].append({
                    'user_input': user_input,
                    'timestamp': time.time(),
                    'context_analysis': context_analysis
                })
                
                # Keep conversation flow manageable
                if len(self.conversation_context['conversation_flow']) > 10:
                    self.conversation_context['conversation_flow'] = self.conversation_context['conversation_flow'][-10:]
                
                # Predict routing for stats
                if self.router:
                    decision = await self.router.make_intelligent_decision(user_input, session_id=self.memory_manager.session_id)
                    if decision.use_offline:
                        self.session_stats['offline_queries'] += 1
                    elif decision.use_online:
                        self.session_stats['online_queries'] += 1
                
                # Stream Pascal's response
                self.console.print("Pascal: ", style="bold magenta", end="")
                
                try:
                    # Stream response with enhanced routing and context
                    response_text = ""
                    response_start = time.time()
                    
                    async for chunk in self.router.get_streaming_response(user_input, session_id=self.memory_manager.session_id):
                        print(chunk, end="", flush=True)
                        response_text += chunk
                    
                    response_time = time.time() - response_start
                    print()  # New line after streaming
                    
                    # Update conversation context
                    if response_text:
                        # Extract topic from the response for context
                        self.conversation_context['last_topic'] = self._extract_topic(user_input, response_text)
                        
                        # Get sources from memory for future follow-ups
                        try:
                            last_sources = await self.memory_manager.get_last_assistant_sources()
                            if last_sources:
                                self.conversation_context['last_sources'] = last_sources
                        except:
                            pass
                    
                    # Show performance info in debug mode
                    if settings.debug_mode and self.router and self.router.last_decision:
                        decision = self.router.last_decision
                        route_type = "NEMOTRON" if decision.use_offline else "GROQ" if decision.use_online else "SKILL"
                        context_info = f"Follow-up: {context_analysis['is_follow_up']}" if context_analysis['is_follow_up'] else ""
                        self.console.print(f"[DEBUG] Route: {route_type}, Time: {response_time:.3f}s, Confidence: {decision.confidence:.2f} {context_info}", style="dim")
                
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
    
    def _extract_topic(self, user_input: str, response: str) -> str:
        """Extract main topic from user input and response for context tracking"""
        # Simple topic extraction - look for key terms
        topics = []
        
        # Common topic indicators
        topic_keywords = [
            'f1', 'formula', 'grand prix', 'race', 'racing',
            'weather', 'news', 'sports', 'politics', 'technology',
            'python', 'programming', 'code', 'software'
        ]
        
        text_to_analyze = (user_input + " " + response).lower()
        
        for keyword in topic_keywords:
            if keyword in text_to_analyze:
                topics.append(keyword)
        
        # Return the first found topic or a generic one
        return topics[0] if topics else 'general'
    
    async def shutdown(self):
        """Enhanced graceful shutdown"""
        self.console.print("\n🔄 Shutting down Pascal Enhanced Conversational Edition...", style="yellow")
        
        try:
            # Save memory
            if self.memory_manager:
                await self.memory_manager.save_session()
            
            # Close router
            if self.router and hasattr(self.router, 'close'):
                await self.router.close()
            
            # Show session summary
            if self.session_stats['queries'] > 0:
                import time
                session_duration = time.time() - self.session_stats['start_time']
                
                summary_table = Table(title="📊 Enhanced Session Summary", border_style="blue")
                summary_table.add_column("Metric", style="cyan")
                summary_table.add_column("Value", style="white")
                
                summary_table.add_row("Total Queries", str(self.session_stats['queries']))
                summary_table.add_row("Offline Queries", str(self.session_stats['offline_queries']))
                summary_table.add_row("Online Queries", str(self.session_stats['online_queries']))
                summary_table.add_row("Conversation Flow", str(len(self.conversation_context.get('conversation_flow', []))))
                summary_table.add_row("Session Duration", f"{session_duration/60:.1f} minutes")
                
                if session_duration > 0:
                    qpm = self.session_stats['queries'] / (session_duration / 60)
                    summary_table.add_row("Queries per Minute", f"{qpm:.1f}")
                
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
