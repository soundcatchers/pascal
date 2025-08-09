"""
Pascal AI Assistant - Main Entry Point (Ollama Version)
A modular, offline-first AI assistant for Raspberry Pi 5 with Ollama integration
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
from modules.router import Router
from modules.personality import PersonalityManager
from modules.memory import MemoryManager

class Pascal:
    """Main Pascal AI Assistant class with Ollama support"""
    
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
                Text("🤖 Initializing Pascal AI Assistant (Ollama Powered)", style="bold cyan"),
                border_style="cyan"
            ))
            
            # Initialize components
            self.personality_manager = PersonalityManager()
            self.memory_manager = MemoryManager()
            self.router = Router(self.personality_manager, self.memory_manager)

            # Wait for router to check LLM availability
            await self.router._check_llm_availability()
            
            # Load default personality
            await self.personality_manager.load_personality(settings.default_personality)
            
            # Load memory
            await self.memory_manager.load_session()
            
            self.console.print("✅ Pascal initialized successfully!", style="green")
            return True
            
        except Exception as e:
            self.console.print(f"❌ Initialization failed: {e}", style="red")
            return False
    
    def display_status(self):
        """Display Pascal's current status with Ollama information"""
        config = settings.get_config_summary()
        
        # Create status table
        status_table = Table(title="Pascal System Status", border_style="blue")
        status_table.add_column("Component", style="cyan", no_wrap=True)
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="white")
        
        # Basic info
        status_table.add_row("Version", f"v{config['pascal_version']}", "Ollama-powered")
        status_table.add_row("Personality", config['personality'], "Active")
        status_table.add_row("Memory", "✅ Enabled" if config['memory_enabled'] else "❌ Disabled", "")
        status_table.add_row("Debug Mode", "✅ On" if config['debug_mode'] else "❌ Off", "")
        
        # LLM status
        status_table.add_row("Online APIs", "✅ Available" if config['online_apis_configured'] else "❌ Not configured", "")
        
        # Get Ollama-specific status
        if self.router and self.router.offline_llm:
            offline_stats = self.router.offline_llm.get_performance_stats()
            if offline_stats.get('ollama_enabled'):
                current_model = offline_stats.get('current_model', 'None')
                model_size = offline_stats.get('model_size', 'Unknown')
                status_table.add_row("Ollama", "✅ Connected", f"Host: {offline_stats.get('ollama_host', 'localhost:11434')}")
                status_table.add_row("Current Model", current_model, f"Size: {model_size}")
                status_table.add_row("Available Models", str(offline_stats.get('available_models', 0)), "")
                
                # Performance info
                avg_time = offline_stats.get('avg_inference_time', 'N/A')
                profile = offline_stats.get('performance_profile', 'balanced')
                status_table.add_row("Performance", f"{profile.title()} Profile", f"Avg: {avg_time}")
            else:
                status_table.add_row("Ollama", "❌ Not available", "Run ./download_models.sh to install")
        else:
            status_table.add_row("Ollama", "❌ Not initialized", "")
        
        # Hardware info
        hw_info = config.get('hardware_info', {})
        if hw_info:
            pi_model = hw_info.get('pi_model', 'Unknown')
            ram_gb = hw_info.get('available_ram_gb', 'Unknown')
            status_table.add_row("Hardware", pi_model, f"RAM: {ram_gb}GB")
        
        self.console.print(status_table)
        
        # Commands help
        commands_text = """[bold]Available Commands:[/bold]
  • help/status - Show this information
  • personality [name] - Switch personality (default, assistant)
  • model [name] - Switch Ollama model
  • models - List available models
  • profile [speed|balanced|quality] - Set performance profile
  • download [model] - Download new model via Ollama
  • remove [model] - Remove model via Ollama
  • clear - Clear conversation history
  • quit/exit - Stop Pascal"""
        
        self.console.print(Panel(commands_text, title="Commands", border_style="green"))
    
    async def process_command(self, user_input: str) -> bool:
        """Process special commands including Ollama model management"""
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
                model_name = parts[1]
                await self._switch_model(model_name)
            else:
                self.console.print("Usage: model [model_name]", style="yellow")
                await self._show_available_models()
        
        elif command.startswith('profile '):
            if len(parts) >= 2:
                profile = parts[1]
                if profile in ['speed', 'balanced', 'quality']:
                    self.router.set_performance_preference(profile)
                    self.console.print(f"✅ Set performance profile to {profile}", style="green")
                else:
                    self.console.print("Available profiles: speed, balanced, quality", style="yellow")
            else:
                current_profile = settings.performance_mode
                self.console.print(f"Current profile: {current_profile}", style="cyan")
                self.console.print("Available profiles: speed, balanced, quality", style="cyan")
        
        elif command.startswith('download '):
            if len(parts) >= 2:
                model_name = parts[1]
                await self._download_model(model_name)
            else:
                self.console.print("Usage: download [model_name]", style="yellow")
                self.console.print("Example: download phi3:mini", style="cyan")
        
        elif command.startswith('remove '):
            if len(parts) >= 2:
                model_name = parts[1]
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
        
        models = self.router.offline_llm.list_available_models()
        
        if not models:
            self.console.print("No models available. Download with: download [model_name]", style="yellow")
            return
        
        # Create models table
        models_table = Table(title="Available Ollama Models", border_style="green")
        models_table.add_column("Model", style="cyan", no_wrap=True)
        models_table.add_column("Size", style="white")
        models_table.add_column("Speed", style="green")
        models_table.add_column("Quality", style="blue")
        models_table.add_column("RAM", style="yellow")
        models_table.add_column("Status", style="magenta")
        
        for model in models:
            status = "🟢 LOADED" if model['loaded'] else "⚪ Available"
            models_table.add_row(
                model['name'],
                model['size'],
                model['speed_rating'],
                model['quality_rating'],
                model['ram_usage'],
                status
            )
        
        self.console.print(models_table)
    
    async def _switch_model(self, model_name: str):
        """Switch to a different Ollama model"""
        if not self.router or not self.router.offline_llm:
            self.console.print("❌ Ollama not available", style="red")
            return
        
        self.console.print(f"🔄 Switching to model: {model_name}...", style="yellow")
        
        try:
            success = await self.router.offline_llm.switch_model(model_name)
            if success:
                self.console.print(f"✅ Switched to model: {model_name}", style="green")
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
        """Main chat interaction loop"""
        self.console.print("\n💬 Chat with Pascal (type 'help' for commands)\n", style="cyan")
        
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
                
                # Get Pascal's response
                self.console.print("Pascal: ", style="bold magenta", end="")
                
                try:
                    response = await self.router.get_response(user_input)
                    self.console.print(response, style="white")
                
                except Exception as e:
                    self.console.print(f"Sorry, I encountered an error: {e}", style="red")
                    if settings.debug_mode:
                        import traceback
                        traceback.print_exc()
                
                print()  # Add spacing
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.console.print("\n🔄 Shutting down Pascal...", style="yellow")
        
        if self.memory_manager:
            await self.memory_manager.save_session()
        
        if self.router and self.router.offline_llm:
            await self.router.offline_llm.close()
        
        if self.router and self.router.online_llm:
            await self.router.online_llm.close()
        
        self.running = False
        self.console.print("👋 Goodbye!", style="cyan")
    
    async def run(self):
        """Main run method"""
        # Setup signal handlers
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
