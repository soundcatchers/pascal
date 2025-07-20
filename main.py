"""
Pascal AI Assistant - Main Entry Point
A modular, offline-first AI assistant for Raspberry Pi 5
"""

import sys
import asyncio
import signal
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from config.settings import settings
from modules.router import Router
from modules.personality import PersonalityManager
from modules.memory import MemoryManager

class Pascal:
    """Main Pascal AI Assistant class"""
    
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
                Text("🤖 Initializing Pascal AI Assistant", style="bold cyan"),
                border_style="cyan"
            ))
            
            # Initialize components
            self.personality_manager = PersonalityManager()
            self.memory_manager = MemoryManager()
            self.router = Router(self.personality_manager, self.memory_manager)
            
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
        """Display Pascal's current status"""
        config = settings.get_config_summary()
        
        status_text = f"""[bold]Pascal v{config['pascal_version']}[/bold]
        
🔧 Configuration:
  • Personality: {config['personality']}
  • Online APIs: {'✅' if config['online_apis_configured'] else '❌'}
  • Local Model: {'✅' if config['local_model_available'] else '❌'}
  • Prefer Offline: {'✅' if config['prefer_offline'] else '❌'}
  • Memory: {'✅' if config['memory_enabled'] else '❌'}
  • Debug Mode: {'✅' if config['debug_mode'] else '❌'}

💡 Available Commands:
  • help - Show this help
  • status - Show system status
  • personality [name] - Switch personality
  • clear - Clear conversation
  • quit/exit - Stop Pascal
"""
        
        self.console.print(Panel(status_text, title="Pascal Status", border_style="blue"))
    
    async def process_command(self, user_input: str) -> bool:
        """Process special commands"""
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'bye']:
            return False
        
        elif command == 'help':
            self.display_status()
        
        elif command == 'status':
            self.display_status()
        
        elif command.startswith('personality '):
            personality_name = command.split(' ', 1)[1]
            try:
                await self.personality_manager.load_personality(personality_name)
                self.console.print(f"✅ Switched to {personality_name} personality", style="green")
            except Exception as e:
                self.console.print(f"❌ Failed to switch personality: {e}", style="red")
        
        elif command == 'clear':
            await self.memory_manager.clear_session()
            self.console.print("✅ Conversation cleared", style="green")
        
        else:
            # Not a command, process as normal input
            return True
        
        return True  # Continue running
    
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
                if user_input.lower() in ['help', 'status', 'clear'] or user_input.lower().startswith('personality '):
                    continue
                
                # Get Pascal's response
                self.console.print("Pascal: ", style="bold magenta", end="")
                
                try:
                    response = await self.router.get_response(user_input)
                    self.console.print(response, style="white")
                
                except Exception as e:
                    self.console.print(f"Sorry, I encountered an error: {e}", style="red")
                
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
        sys.exit(1)
