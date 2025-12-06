"""
Pascal AI Assistant - Main Entry Point (v4.3)
Features: Simplified Nemotron + Groq with fast routing + Enhanced Conversational Context + Voice Input
Voice-safe exit commands: quit, stop, goodbye, done, etc.
Fixes: Noise word filtering (the, a, i), reliable exit via _stop_requested flag
"""

import asyncio
import signal
import sys
import time
import argparse
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

# Voice input (optional)
try:
    from modules.speech_input import SpeechInputManager
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Homophone fixer for voice-safe exit commands
try:
    from modules.homophone_fixer import HomophoneFixer
    HOMOPHONE_FIXER_AVAILABLE = True
except ImportError:
    HOMOPHONE_FIXER_AVAILABLE = False

class Pascal:
    """FIXED Pascal AI Assistant with enhanced conversational context + Voice Input"""
    
    def __init__(self, voice_mode: bool = False, list_audio_devices: bool = False, debug_audio: bool = False):
        self.console = Console()
        self.running = False
        self.voice_mode = voice_mode
        self.list_audio_devices = list_audio_devices
        self.debug_audio = debug_audio
        
        # Core components
        self.router = None
        self.personality_manager = None
        self.memory_manager = None
        self.speech_manager = None
        
        # Voice-safe exit command detection
        self.exit_detector = HomophoneFixer() if HOMOPHONE_FIXER_AVAILABLE else None
        
        # Voice input state
        self.current_voice_input = ""
        self.voice_is_final = False
        
        # Performance tracking
        self.session_stats = {
            'queries': 0,
            'offline_queries': 0,
            'online_queries': 0,
            'follow_up_queries': 0,
            'context_enhanced_queries': 0,
            'greeting_queries': 0,
            'forced_offline_queries': 0,
            'forced_online_queries': 0,
            'voice_queries': 0,
            'start_time': None,
        }
    
    async def initialize(self):
        """FIXED: Initialize Pascal's enhanced conversational components + Voice Input"""
        try:
            voice_status = " + Voice Input" if self.voice_mode else ""
            self.console.print(Panel.fit(
                Text(f"⚡ Pascal AI Assistant v4.1 - Enhanced Conversational Edition{voice_status}", style="bold cyan"),
                border_style="cyan"
            ))
            
            # Initialize personality and memory
            self.personality_manager = PersonalityManager()
            self.memory_manager = MemoryManager()
            
            # Load default personality (Pascal)
            await self.personality_manager.load_personality("pascal")
            await self.memory_manager.load_session()
            
            # CRITICAL: Clear short-term memory on startup (new session = fresh context)
            # Old memories in short-term cause pollution: "weather tomorrow" → "day after" → pulls CHEESE conversation!
            # Move any existing short-term to long-term (preserve history), then start fresh
            if hasattr(self.memory_manager, 'short_term_memory') and self.memory_manager.short_term_memory:
                # Archive old short-term to long-term before clearing
                for interaction in self.memory_manager.short_term_memory:
                    self.memory_manager.long_term_memory.append(interaction)
                self.memory_manager.short_term_memory.clear()
                await self.memory_manager.save_session()
                if settings.debug_mode:
                    self.console.print(f"[DEBUG] Archived old session memories, starting fresh", style="dim")
            
            # FIXED: Initialize enhanced intelligent router
            self.router = EnhancedIntelligentRouter.create(self.personality_manager, self.memory_manager)
            
            # Clear any stale context from previous sessions
            if hasattr(self.router, 'clear_context'):
                self.router.clear_context()
            
            # FIXED: Eager initialization at startup (not on first query)
            await self.router._check_llm_availability()
            self.router._initialized = True  # Mark as initialized to prevent re-init on first query
            
            # Initialize voice input if enabled
            if self.voice_mode or self.list_audio_devices:
                if not VOICE_AVAILABLE:
                    self.console.print("❌ Voice input not available. Install with:", style="red")
                    self.console.print("   pip install vosk pyaudio", style="yellow")
                    if self.voice_mode:
                        return False
                else:
                    self.speech_manager = SpeechInputManager(debug_audio=self.debug_audio)
                    
                    if self.list_audio_devices:
                        self.speech_manager.initialize()
                        self.speech_manager.list_devices()
                        return False
                    
                    if self.speech_manager.initialize():
                        device_info = self.speech_manager.get_device_info()
                        if device_info['available']:
                            marker = "🎙️  ReSpeaker" if device_info.get('is_respeaker') else "🎤"
                            self.console.print(f"{marker} Voice input ready: {device_info['name']}", style="green")
                        else:
                            self.console.print("⚠️  Voice input initialized but no device detected", style="yellow")
                    else:
                        self.console.print("❌ Failed to initialize voice input", style="red")
                        if self.voice_mode:
                            return False
            
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
    
    async def check_personality_switch(self, user_input: str) -> bool:
        """
        Check if user is trying to switch personalities
        Pattern: "[Name] (are) you out there" (voice-safe with flexible matching)
        
        Matches:
        - "Jarvis are you out there"
        - "Jarvis you out there" (missing "are" - common with voice)
        - "Hey Jarvis are you out there"
        - "Rick you out there"
        
        Returns True if personality switch was attempted (success or failure)
        """
        import re
        
        # ULTRA-FLEXIBLE voice-safe pattern: "are" is now optional
        # Pattern: [optional hey/hello] [NAME] [optional "are"] you out there
        # This catches voice recognition errors where "are" gets dropped
        pattern = r'(?:(?:hey|hello)[,\s]+)?([a-zA-Z][a-zA-Z\s.]*?)\s*,?\s+(?:are\s+)?you\s+out\s+there.*$'
        match = re.match(pattern, user_input.strip(), re.IGNORECASE)
        
        if not match:
            return False
        
        # Extract personality name (clean up spaces, dots, etc.)
        raw_name = match.group(1).strip()
        
        # Remove honorifics and clean name
        honorifics = ['dr', 'mr', 'mrs', 'ms', 'miss', 'sir', 'madam']
        stopwords = ['hey', 'hello', 'are', 'you', 'the', 'a', 'an']  # Prevent "are" from being captured as name
        name_parts = raw_name.lower().split()
        
        # Filter out honorifics, stopwords, and dots
        clean_parts = [part.rstrip('.') for part in name_parts 
                      if part.rstrip('.').lower() not in honorifics 
                      and part.rstrip('.').lower() not in stopwords]
        
        # Validate we have a real name (not empty or just stopwords)
        if not clean_parts:
            if settings.debug_mode:
                self.console.print(f"[DEBUG] No valid personality name found in: {user_input}", style="dim")
            return False
        
        # Join remaining parts (handles multi-word names like "iron man" or single names like "jarvis")
        requested_personality = '_'.join(clean_parts) if len(clean_parts) > 1 else clean_parts[0]
        
        # Get current personality name (or "pascal" as default)
        current_personality_name = self.personality_manager.current_personality or "pascal"
        
        # Try to load the requested personality
        success = await self.personality_manager.load_personality(requested_personality)
        
        if success:
            # Successfully switched! Get greeting from new personality
            greeting = await self.personality_manager.get_greeting()
            self.console.print(f"\n{requested_personality.upper()}: {greeting}\n", style="bold cyan")
            
            # Update router with new personality
            self.router.personality_manager = self.personality_manager
            
            # CRITICAL: Clear short-term memory to prevent context pollution across personalities
            # But PRESERVE the conversation history by moving it to long-term memory first!
            if hasattr(self.memory_manager, 'short_term_memory') and self.memory_manager.short_term_memory is not None:
                # Move ALL short-term interactions to long-term memory (preserve history)
                if self.memory_manager.short_term_memory:
                    if settings.debug_mode:
                        self.console.print(f"[DEBUG] Moving {len(self.memory_manager.short_term_memory)} interactions to long-term memory", style="dim")
                    
                    # Archive current conversation before switching personalities
                    for interaction in self.memory_manager.short_term_memory:
                        self.memory_manager.long_term_memory.append(interaction)
                    
                    # Now clear short-term (context reset for new personality)
                    self.memory_manager.short_term_memory.clear()
                    
                    # Save to disk immediately to preserve this conversation
                    await self.memory_manager.save_session()
                    
                    if settings.debug_mode:
                        self.console.print(f"[DEBUG] Cleared conversation context for personality switch (history preserved)", style="dim")
            
            # Also clear router's context tracking
            if hasattr(self.router, 'clear_context'):
                self.router.clear_context()
            
            if settings.debug_mode:
                self.console.print(f"[DEBUG] Switched from {current_personality_name} to {requested_personality}", style="dim")
        else:
            # Failed to load personality - respond with fallback
            fallback_message = f"{requested_personality.capitalize()} is not here right now."
            self.console.print(f"\n{current_personality_name.upper()}: {fallback_message}\n", style="bold yellow")
            
            if settings.debug_mode:
                self.console.print(f"[DEBUG] Failed to load personality: {requested_personality}", style="dim")
        
        return True
    
    def _on_voice_input(self, text: str, is_final: bool):
        """Callback for voice input from speech recognition"""
        if is_final:
            self.current_voice_input = text
            self.voice_is_final = True
            print(f"\r🎤 You (voice): {text}")
        else:
            print(f"\r🎤 Listening: {text}...", end="", flush=True)
    
    async def chat_loop(self):
        """FIXED: Enhanced conversational chat interaction loop + Voice Input"""
        self.session_stats['start_time'] = time.time()
        
        mode_str = "Voice + Text" if self.voice_mode else "Text"
        self.console.print(f"\n💬 Chat with Pascal Enhanced Conversational Edition ({mode_str} Mode)\n", style="cyan")
        self.console.print("Type 'help' for commands. Now with enhanced context awareness!\n", style="dim")
        
        if self.voice_mode:
            self.console.print("🎙️  Voice mode active - speak naturally or type text", style="green")
            self.console.print("💡 Say 'exit' or 'quit' to stop\n", style="dim")
        
        if self.router:
            if self.router.offline_available and self.router.online_available:
                self.console.print("💡 Try: 'Hello Pascal' (offline) then 'Who won the last F1 race?' (online) then 'Who came second?' (enhanced follow-up)\n", style="green")
            elif self.router.online_available:
                self.console.print("💡 Online AI ready for current information and follow-ups!\n", style="green")
        
        if self.voice_mode and self.speech_manager:
            self.speech_manager.start_listening(self._on_voice_input)
        
        while self.running:
            try:
                user_input = ""
                
                # Handle voice input
                if self.voice_mode and self.voice_is_final:
                    user_input = self.current_voice_input
                    self.voice_is_final = False
                    self.current_voice_input = ""
                    self.session_stats['voice_queries'] += 1
                elif self.voice_mode:
                    # In voice mode, poll for voice input (non-blocking)
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # Text-only mode: wait for keyboard input
                    try:
                        user_input_raw = await asyncio.get_event_loop().run_in_executor(
                            None, input, "You: "
                        )
                        user_input = user_input_raw.strip()
                    except EOFError:
                        await asyncio.sleep(0.1)
                        continue
                
                # Skip if no input
                if not user_input:
                    continue
                
                # CRITICAL: Check for active authentication challenge
                # If active, route input to auth system instead of normal processing
                if self.memory_manager.auth_manager.is_active():
                    response = await self.memory_manager.process_auth_response(user_input)
                    self.console.print(f"\nPascal: {response}\n", style="bold yellow")
                    continue
                
                # Check for personality switching: "[Name] are you out there"
                personality_switch = await self.check_personality_switch(user_input)
                if personality_switch:
                    continue  # Already handled, skip to next input
                
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
                
                # Check if it's a greeting for stats
                greeting_patterns = ['hello', 'hi', 'hey', 'how are you', 'good morning']
                if any(pattern in user_input.lower() for pattern in greeting_patterns):
                    self.session_stats['greeting_queries'] += 1
                
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
                        
                        # Track enhanced routing decisions
                        if 'enhanced context' in decision.reason.lower():
                            if 'forcing offline' in decision.reason.lower():
                                self.session_stats['forced_offline_queries'] += 1
                            elif 'forcing online' in decision.reason.lower():
                                self.session_stats['forced_online_queries'] += 1
                        
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
                        if 'enhanced context' in decision.reason.lower():
                            extra_info.append("Enhanced-Override")
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
        """FIXED: Process special commands with voice-safe exit detection"""
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'bye']:
            self.console.print("\n👋 Goodbye! Shutting down Pascal...", style="cyan")
            if self.speech_manager:
                self.speech_manager.stop_listening()
            return False
        
        if self.exit_detector and self.exit_detector.is_exit_command(command):
            self.console.print(f"\n👋 Exit command detected: '{command}'. Shutting down Pascal...", style="cyan")
            if self.speech_manager:
                self.speech_manager.stop_listening()
            return False
        
        elif command in ['help', 'status']:
            self.display_status()
        
        elif command == 'clear':
            await self.memory_manager.clear_session()
            # Reset router context
            if hasattr(self.router, 'clear_context'):
                self.router.clear_context()
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
        
        elif command == 'test':
            # Test routing for different query types
            test_queries = [
                ("Hello Pascal", "Should route to OFFLINE"),
                ("What day is today?", "Should route to ONLINE"),
                ("Who won the last F1 race?", "Should route to ONLINE"),
                ("Who came second?", "Should be enhanced follow-up to ONLINE")
            ]
            
            for query, expected in test_queries:
                try:
                    decision, context_info = await self.router.make_enhanced_intelligent_decision(query, session_id=self.memory_manager.session_id)
                    route_type = "OFFLINE" if decision.use_offline else "ONLINE" if decision.use_online else "SKILL"
                    context_desc = "Greeting" if context_info['is_greeting'] else "Follow-up" if context_info['is_follow_up'] else "Standalone"
                    self.console.print(f"'{query}' -> {route_type} ({context_desc}) - {expected}", style="dim")
                except Exception as e:
                    self.console.print(f"'{query}' -> ERROR: {e}", style="red")
        
        # NEW: Memory management commands
        elif command.startswith('forget '):
            # Extract individual name
            name = user_input[7:].strip()  # Remove "forget " prefix
            if name:
                response = await self.memory_manager.forget_individual(name)
                self.console.print(response, style="yellow")
            else:
                self.console.print("❌ Please specify who to forget: 'forget [name]'", style="red")
        
        elif command == 'complete memory wipe' or command == 'memory wipe':
            response = await self.memory_manager.complete_memory_wipe()
            self.console.print(response, style="bold red")
        
        elif command.startswith('track '):
            # Simple individual tracking: "track John age=25 location=London"
            parts = user_input[6:].strip().split()
            if len(parts) >= 1:
                name = parts[0]
                attributes = {}
                for part in parts[1:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        attributes[key] = value
                self.memory_manager.add_individual(name, **attributes)
                self.console.print(f"✅ Tracked {name} with {len(attributes)} attributes", style="green")
            else:
                self.console.print("❌ Usage: track [name] [key=value] ...", style="red")
        
        elif command == 'memories' or command == 'individuals':
            # List tracked individuals
            if self.memory_manager.individuals:
                self.console.print("\n📋 Tracked Individuals:", style="bold")
                for name, data in self.memory_manager.individuals.items():
                    attrs = data.get('attributes', {})
                    attr_str = ', '.join([f"{k}={v}" for k, v in attrs.items()])
                    self.console.print(f"  • {data['name']}: {attr_str}", style="dim")
            else:
                self.console.print("No individuals tracked yet.", style="dim")
        
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
        status_table.add_row("Context Awareness", "✅ Enhanced", "Overrides base router decisions")
        status_table.add_row("Follow-up Detection", "✅ Advanced", "Understands pronouns and references")
        status_table.add_row("Search Enhancement", "✅ Intelligent", "Context-aware query optimization")
        status_table.add_row("Greeting Detection", "✅ Perfect", "Forces offline for personality")
        status_table.add_row("Streaming", "⚡ Enabled" if config['streaming_enabled'] else "❌ Disabled", "Real-time responses")
        
        # System status
        system_status = self._get_system_status()
        
        if system_status.get('offline_llm'):
            status_table.add_row("Offline (Nemotron)", "✅ Ready", "Local AI via Ollama - Greetings & Chat")
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
        
        # FIXED: Simplified session context calculation
        session_duration = 0
        if hasattr(self.router, 'conversation_context') and self.router.conversation_context.get('session_start_time'):
            session_duration = time.time() - self.router.conversation_context['session_start_time']
        
        # Enhanced conversational features
        perf_text = f"""[bold]⚡ Enhanced Conversational Features:[/bold]
  • Smart Routing Override: Greetings always go to offline for personality
  • Context-Aware Follow-ups: "Who came second?" after F1 questions
  • Search Enhancement: Improves F1 queries with better date ranges
  • Entity Tracking: Remembers F1, politics, weather topics
  • Session Context: Current session started {session_duration:.0f}s ago
  
[bold]Fixed Issues:[/bold]
  • ✅ Greetings now route to offline (Nemotron) for better personality
  • ✅ F1 searches improved with better date handling (2024-2025)
  • ✅ Follow-ups maintain context and enhance search queries
  • ✅ Memory leaks reduced with proper session cleanup
  
[bold]Conversational Examples:[/bold]
  • "Hello Pascal" → OFFLINE (Nemotron) for friendly personality
  • "Who won the last F1 race?" → ONLINE (Groq) with enhanced search
  • "Who came second?" → ONLINE with F1 context enhancement
  
[bold]Quick Commands:[/bold]
  • help/status - Show this information
  • context - Show current conversation context
  • test - Test routing for different query types
  • clear - Clear conversation history and context
  • debug - Toggle debug mode
  • quit/exit/stop/goodbye/done - Stop Pascal (voice-safe variants supported)"""
        
        self.console.print(Panel(perf_text, title="Enhanced Conversational Features", border_style="green"))
        
        # Show session stats if available
        if self.session_stats['queries'] > 0:
            stats_text = f"""Queries: {self.session_stats['queries']} | Greetings: {self.session_stats['greeting_queries']} | Follow-ups: {self.session_stats['follow_up_queries']} | Enhanced: {self.session_stats['context_enhanced_queries']}
Offline: {self.session_stats['offline_queries']} | Online: {self.session_stats['online_queries']} | Forced Offline: {self.session_stats['forced_offline_queries']} | Forced Online: {self.session_stats['forced_online_queries']}"""
            self.console.print(Panel(stats_text, title="📊 Session Stats", border_style="blue"))
    
    def _get_system_status(self) -> dict:
        """Get detailed system status"""
        return {
            'offline_llm': self.router.offline_available if self.router else False,
            'online_llm': self.router.online_available if self.router else False,
            'skills_manager': self.router.skills_available if self.router else False,
        }
    
    async def shutdown(self):
        """Enhanced graceful shutdown with proper cleanup + Voice Input"""
        self.console.print("\n🔄 Shutting down Pascal Enhanced Conversational Edition...", style="yellow")
        
        try:
            # Stop voice input if active
            if self.speech_manager:
                self.speech_manager.stop_listening()
                self.speech_manager.close()
            
            # Save memory
            if self.memory_manager:
                await self.memory_manager.save_session()
            
            # Close router connections properly to prevent memory leaks
            if self.router and hasattr(self.router, 'close'):
                await self.router.close()
            
            # Show enhanced session summary
            if self.session_stats['queries'] > 0:
                session_duration = time.time() - self.session_stats['start_time']
                
                summary_table = Table(title="📊 Enhanced Session Summary", border_style="blue")
                summary_table.add_column("Metric", style="cyan")
                summary_table.add_column("Value", style="white")
                
                summary_table.add_row("Total Queries", str(self.session_stats['queries']))
                summary_table.add_row("Greetings", str(self.session_stats['greeting_queries']))
                summary_table.add_row("Follow-up Queries", str(self.session_stats['follow_up_queries']))
                summary_table.add_row("Context-Enhanced", str(self.session_stats['context_enhanced_queries']))
                summary_table.add_row("Offline Queries", str(self.session_stats['offline_queries']))
                summary_table.add_row("Online Queries", str(self.session_stats['online_queries']))
                summary_table.add_row("Forced Offline", str(self.session_stats['forced_offline_queries']))
                summary_table.add_row("Forced Online", str(self.session_stats['forced_online_queries']))
                summary_table.add_row("Session Duration", f"{session_duration/60:.1f} minutes")
                
                if self.session_stats['queries'] > 0:
                    follow_up_rate = (self.session_stats['follow_up_queries'] / self.session_stats['queries']) * 100
                    context_rate = (self.session_stats['context_enhanced_queries'] / self.session_stats['queries']) * 100
                    greeting_rate = (self.session_stats['greeting_queries'] / self.session_stats['queries']) * 100
                    override_rate = ((self.session_stats['forced_offline_queries'] + self.session_stats['forced_online_queries']) / self.session_stats['queries']) * 100
                    summary_table.add_row("Greeting Rate", f"{greeting_rate:.1f}%")
                    summary_table.add_row("Follow-up Rate", f"{follow_up_rate:.1f}%")
                    summary_table.add_row("Context Enhancement Rate", f"{context_rate:.1f}%")
                    summary_table.add_row("Enhanced Override Rate", f"{override_rate:.1f}%")
                
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
    """Main entry point for Pascal Enhanced Conversational Edition + Voice Input"""
    parser = argparse.ArgumentParser(description='Pascal AI Assistant - Your intelligent voice & text assistant')
    parser.add_argument('--voice', action='store_true', help='Enable voice input mode (requires Vosk + microphone)')
    parser.add_argument('--list-devices', action='store_true', help='List available audio input devices and exit')
    parser.add_argument('--debug-audio', action='store_true', help='Show ALSA debug messages (for troubleshooting audio issues)')
    
    args = parser.parse_args()
    
    pascal = Pascal(voice_mode=args.voice, list_audio_devices=args.list_devices, debug_audio=args.debug_audio)
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
