"""
Pascal AI Assistant - Main Entry Point (Lightning Version)
A lightning-fast, streaming AI assistant for Raspberry Pi 5
"""

import sys
import asyncio
import signal
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live

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
            return False
    
    def display_status(self):
        """Display Pascal's current status with performance metrics"""
        config = settings.get_config_summary()
        
        # Create status table
        status_table = Table(title="⚡ Pascal Lightning Status", border_style="blue")
        status_table.add_column("Component", style="cyan", no_wrap=True)
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="white")
        
        # Basic info
        status_table.add_row("Version", f"v{config['pascal_version']}", "Lightning with Grok")
        status_table.add_row("Personality", config['personality'], "Active")
        status_table.add_row("Memory", "✅ Enabled" if config['memory_enabled'] else "❌ Disabled", "")
        status_table.add_row("Streaming", "⚡ Enabled" if config['streaming_enabled'] else "❌ Disabled", "Lightning responses")
        status_table.add_row("Target Response", f"{config['target_response_time']}s", "Maximum time")
        
        # LLM status
        if config['grok_configured']:
            status_table.add_row("Grok API", "✅ Configured", "Primary online")
        status_table.add_row("Online APIs", "✅ Available" if config['online_apis_configured'] else "❌ Not configured", "")
        
        # Get Ollama-specific status
        if self.router and self.router.offline_llm:
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
  • Fallback 1: qwen3:4b-instruct
  • Fallback 2: gemma3:4b-it-q4_K_M
  • Target: 1-3 second responses
  • Streaming enabled for instant feedback"""
        
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
