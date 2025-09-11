"""
Pascal AI Assistant - Main Entry Point (Fixed)
A lightning-fast, streaming AI assistant for Raspberry Pi 5
FIXED: Corrected async issues and module imports
"""

import sys
import asyncio
import signal
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from config.settings import settings
from modules.router import LightningRouter
from modules.personality import PersonalityManager
from modules.memory import MemoryManager

class Pascal:
    """Main Pascal AI Assistant class with lightning-fast streaming"""
    
    def __init__(self):
        self.console = Console()
        self.running = False
        self.router = None
        self.personality_manager = None
        self.memory_manager = None
    
    async def initialize(self):
        """Initialize Pascal's core systems"""
        try:
            self.console.print(Panel.fit(
                Text("⚡ Initializing Pascal AI Assistant - Lightning Version", style="bold cyan"),
                border_style="cyan"
            ))
            
            # Initialize components
            self.personality_manager = PersonalityManager()
            self.memory_manager = MemoryManager()
            self.router = LightningRouter(self.personality_manager, self.memory_manager)

            # Wait for router to check LLM availability
            await self.router._check_llm_availability()
            
            # Load default personality
            await self.personality_manager.load_personality(settings.default_personality)
            
            # Load memory
            await self.memory_manager.load_session()
            
            self.console.print("⚡ Pascal initialized - Ready for lightning-fast responses!", style="green")
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
        
        # Create status table
        status_table = Table(title="⚡ Pascal Lightning Status", border_style="blue")
        status_table.add_column("Component", style="cyan", no_wrap=True)
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="white")
        
        # Basic info - FIXED: Changed display text to "Lightning with Groq"
        status_table.add_row("Version", f"v{config['pascal_version']}", "Lightning with Groq")
        status_table.add_row("Personality", config['personality'], "Active")
        status_table.add_row("Memory", "✅ Enabled" if config['memory_enabled'] else "❌ Disabled", "")
        status_table.add_row("Streaming", "⚡ Enabled" if config['streaming_enabled'] else "❌ Disabled", "Lightning responses")
        status_table.add_row("Target Response", f"{config['target_response_time']}s", "Maximum time")
        
        # LLM status - FIXED: Changed from grok_configured to groq_configured
        if config.get('groq_configured'):
            status_table.add_row("Groq API", "✅ Configured", "Primary online")
        if config.get('gemini_configured'):
            status_table.add_row("Gemini API", "✅ Configured", "Secondary online")
        if config.get('openai_configured'):
            status_table.add_row("OpenAI API", "✅ Configured", "Fallback online")
        status_table.add_row("Online APIs", "✅ Available" if config['online_apis_configured'] else "❌ Not configured", "")
        
        # Get Ollama-specific status
        if self.router and self.router.offline_llm:
            try:
                offline_stats = self.router.offline_llm.get_performance_stats()
                if offline_stats.get('ollama_enabled'):
                    current_model = offline_stats.get('current_model', 'None')
                    status_table.add_row("Ollama", "⚡ Connected", f"Keep-alive: {offline_stats.get('keep_alive_active', False)}")
                    status_table.add_row("Current Model", current_model, f"Priority: {offline_stats.get('model_priority', 'N/A')}")
                    
                    # Performance metrics
                    if 'avg_inference_time' in offline_stats:
                        status_table.add_row("Avg Response", offline_stats['avg_inference_time'], "")
                    if 'avg_first_token_time' in offline_stats:
                        status_table.add_row("First Token", offline_stats['avg_first_token_time'], "⚡ Speed metric")
                else:
                    status_table.add_row("Ollama", "❌ Not available", "Run ./download_models.sh")
            except Exception as e:
                status_table.add_row("Ollama", "❌ Error checking", str(e)[:50])
        else:
            status_table.add_row("Ollama", "❌ Not initialized", "")
        
        # Preferred models
        if config.get('preferred_models'):
            models_str = ", ".join(config['preferred_models'][:2])
            status_table.add_row("Preferred Models", models_str[:40] + "...", "Lightning optimized")
        
        # Hardware info
        hw_info = config.get('hardware_info', {})
        if hw_info:
            pi_model = hw_info.get('pi_model', 'Unknown')
            ram_gb = hw_info.get('available_ram_gb', 'Unknown')
            status_table.add_row("Hardware", pi_model, f"RAM: {ram_gb}GB")
        
        self.console.print(status_table)
        
        # Performance tips
        perf_text = """[bold]⚡ Lightning Performance Tips:[/bold]
  • Primary model: nemotron-mini:4b-instruct-q4_K_M (fastest)
  • Fallback 1: qwen2.5:3b
  • Fallback 2: phi3:mini
  • Target: 1-3 second responses
  • Streaming enabled for instant feedback
  
[bold]Online API Priority:[/bold]
  • Groq: Lightning-fast inference (primary)
  • Gemini: Free tier available (secondary)
  • OpenAI: Reliable fallback"""
        
        self.console.print(Panel(perf_text, title="Performance", border_style="yellow"))
        
        # Commands help
        commands_text = """[bold]Available Commands:[/bold]
  • help/status - Show this information
  • personality [name] - Switch personality
  • model [name] - Switch Ollama model
  • models - List available models
  • profile [speed|balanced|quality] - Set performance profile
  • download [model] - Download new model via Ollama
  • remove [model] - Remove model
  • clear - Clear conversation history
  • quit/exit - Stop Pascal"""
        
        self.console.print(Panel(commands_text, title="Commands", border_style="green"))
    
    async def process_command(self, user_input: str) -> bool:
        """Process special commands"""
        command = user_input.lower().strip()
        parts = command.split()
        
        if command in ['quit', 'exit', 'bye']:
            return False
        
        elif command in ['help', 'status']:
            self.display_status()
        
        elif command.startswith('personality '):
            if len(parts) >= 2:
                personality_name = parts[1]
                try:
                    await self.personality_manager.load_personality(personality_name)
                    self.console.print(f"✅ Switched to {personality_name} personality", style="green")
                except Exception as e:
                    self.console.print(f"❌ Failed to switch personality: {e}", style="red")
            else:
                available = self.personality_manager.list_available_personalities()
                self.console.print(f"Available personalities: {', '.join(available)}", style="cyan")
        
        elif command == 'models':
            await self._show_available_models()
        
        elif command.startswith('model '):
            if len(parts) >= 2:
                model_name = ' '.join(parts[1:])  # Handle model names with spaces
                await self._switch_model(model_name)
            else:
                self.console.print("Usage: model [model_name]", style="yellow")
                await self._show_available_models()
        
        elif command.startswith('profile '):
            if len(parts) >= 2:
                profile = parts[1]
                if profile in ['speed', 'balanced', 'quality']:
                    self.router.set_performance_preference(profile)
                    self.console.print(f"⚡ Set performance profile to {profile}", style="green")
                else:
                    self.console.print("Available profiles: speed, balanced, quality", style="yellow")
            else:
                current_profile = settings.performance_mode
                self.console.print(f"Current profile: {current_profile}", style="cyan")
                self.console.print("Available profiles: speed, balanced, quality", style="cyan")
        
        elif command.startswith('download '):
            if len(parts) >= 2:
                model_name = ' '.join(parts[1:])
                await self._download_model(model_name)
            else:
                self.console.print("Usage: download [model_name]", style="yellow")
                self.console.print("Recommended: nemotron-mini:4b-instruct-q4_K_M", style="cyan")
        
        elif command.startswith('remove '):
            if len(parts) >= 2:
                model_name = ' '.join(parts[1:])
                await self._remove_model(model_name)
            else:
                self.console.print("Usage: remove [model_name]", style="yellow")
        
        elif command == 'clear':
            await self.memory_manager.clear_session()
            self.console.print("✅ Conversation cleared", style="green")
        
        else:
            # Not a command, process as normal input
            return True
        
        return True  # Continue running
    
    async def _show_available_models(self):
        """Show available Ollama models"""
        if not self.router or not self.router.offline_llm:
            self.console.print("❌ Ollama not available", style="red")
            return
        
        try:
            models = self.router.offline_llm.list_available_models()
            
            if not models:
                self.console.print("No models available. Download with: download [model_name]", style="yellow")
                self.console.print("Recommended: download nemotron-mini:4b-instruct-q4_K_M", style="cyan")
                return
            
            # Create models table
            models_table = Table(title="⚡ Available Models", border_style="green")
            models_table.add_column("Model", style="cyan", no_wrap=True)
            models_table.add_column("Size", style="white")
            models_table.add_column("Priority", style="yellow")
            models_table.add_column("Type", style="blue")
            models_table.add_column("Status", style="magenta")
            
            for model in models:
                status = "⚡ LOADED" if model['loaded'] else "⚪ Available"
                model_type = "Primary" if model['is_primary'] else ("Fallback" if model['is_fallback'] else "Other")
                models_table.add_row(
                    model['name'],
                    model['size'],
                    str(model['priority']),
                    model_type,
                    status
                )
            
            self.console.print(models_table)
        except Exception as e:
            self.console.print(f"❌ Error listing models: {e}", style="red")
    
    async def _switch_model(self, model_name: str):
        """Switch to a different Ollama model"""
        if not self.router or not self.router.offline_llm:
            self.console.print("❌ Ollama not available", style="red")
            return
        
        self.console.print(f"🔄 Switching to model: {model_name}...", style="yellow")
        
        try:
            success = await self.router.offline_llm.switch_model(model_name)
            if success:
                self.console.print(f"⚡ Switched to model: {model_name}", style="green")
            else:
                self.console.print(f"❌ Failed to switch to model: {model_name}", style="red")
                self.console.print("Use 'models' command to see available models", style="cyan")
        except Exception as e:
            self.console.print(f"❌ Error switching model: {e}", style="red")
    
    async def _download_model(self, model_name: str):
        """Download a new model via Ollama"""
        if not self.router or not self.router.offline_llm:
            self.console.print("❌ Ollama not available", style="red")
            return
        
        self.console.print(f"📥 Downloading model: {model_name}...", style="yellow")
        self.console.print("This may take several minutes depending on model size", style="cyan")
        
        try:
            success = await self.router.offline_llm.pull_model(model_name)
            if success:
                self.console.print(f"✅ Successfully downloaded: {model_name}", style="green")
                # Auto-switch if it's a preferred model
                if model_name in settings.preferred_models:
                    await self._switch_model(model_name)
            else:
                self.console.print(f"❌ Failed to download: {model_name}", style="red")
        except Exception as e:
            self.console.print(f"❌ Download error: {e}", style="red")
    
    async def _remove_model(self, model_name: str):
        """Remove a model via Ollama"""
        if not self.router or not self.router.offline_llm:
            self.console.print("❌ Ollama not available", style="red")
            return
        
        # Confirm removal
        confirm = input(f"Are you sure you want to remove {model_name}? (y/N): ")
        if confirm.lower() != 'y':
            self.console.print("Cancelled", style="yellow")
            return
        
        self.console.print(f"🗑️ Removing model: {model_name}...", style="yellow")
        
        try:
            success = await self.router.offline_llm.remove_model(model_name)
            if success:
                self.console.print(f"✅ Successfully removed: {model_name}", style="green")
            else:
                self.console.print(f"❌ Failed to remove: {model_name}", style="red")
        except Exception as e:
            self.console.print(f"❌ Remove error: {e}", style="red")
    
    async def chat_loop(self):
        """Main chat interaction loop with streaming support"""
        self.console.print("\n💬 Chat with Pascal (⚡ Lightning Mode)\n", style="cyan")
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
                command_keywords = ['help', 'status', 'clear', 'models', 'model', 'personality', 'profile', 'download', 'remove']
                if any(user_input.lower().startswith(cmd) for cmd in command_keywords):
                    continue
                
                # Stream Pascal's response
                self.console.print("Pascal: ", style="bold magenta", end="")
                
                try:
                    # Stream response with instant feedback
                    if settings.streaming_enabled:
                        response_text = ""
                        async for chunk in self.router.get_streaming_response(user_input):
                            print(chunk, end="", flush=True)
                            response_text += chunk
                        print()  # New line after streaming
                        
                        # Store complete response in memory
                        if response_text:
                            await self.memory_manager.add_interaction(user_input, response_text)
                    else:
                        # Non-streaming fallback
                        response = await self.router.get_response(user_input)
                        self.console.print(response, style="white")
                        
                        # Store response in memory
                        if response:
                            await self.memory_manager.add_interaction(user_input, response)
                
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
            if self.memory_manager:
                await self.memory_manager.save_session()
            
            if self.router:
                await self.router.close()
            
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
        if settings.debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)
