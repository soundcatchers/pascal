Pascal AI Assistant (Ollama-Powered)
An intelligent, offline-first AI assistant designed for Raspberry Pi 5, featuring Ollama integration, modular architecture, personality management, and seamless offline/online switching.

🤖 About Pascal
Pascal is a comprehensive AI assistant that runs locally on Raspberry Pi 5 using Ollama for superior model management while maintaining the ability to leverage online AI services when needed. Named after the programming language, Pascal combines local intelligence with cloud capabilities to provide fast, reliable assistance.

✨ Why Ollama?
Major advantages over direct GGUF/llama-cpp-python approach:

✅ No compilation needed - Much faster installation (minutes vs hours)
✅ Better ARM optimization - Specifically optimized for Raspberry Pi 5
✅ Automatic model management - Download, switch, and remove models easily
✅ Reliable downloads - No more 404 errors or broken model files
✅ Built-in quantization - Optimal model formats automatically selected
✅ Easy scaling - Add more models without manual configuration
✅ Better resource management - Automatic memory optimization
🧠 Infinite Memory System (NEW!)
Pascal now remembers everything - making it a true conversational companion, not just a chatbot!

Key Features
📅 10-Year Retention: Stores up to 50,000 interactions (vs. previous 30-day limit)
👥 Individual Tracking: Remember people and their details forever
🔒 Voice-Safe Deletion: 3-stage authentication prevents accidental data loss
💾 Automatic Backups: Every deletion creates a timestamped backup
⚡ Auto-Save: Memory saved every 5 minutes
Why This Matters
Instead of forgetting conversations after 30 days, Pascal now:

Remembers your friends, family, and colleagues indefinitely
Tracks important details: birthdays, preferences, locations
Maintains conversation context across weeks and months
Never loses track of ongoing projects or discussions
Example Usage
# Track someone you mentioned
You: My friend Sarah is 28, lives in NYC and loves cooking
Pascal: [Responds naturally]
You: track Sarah age=28 location=NYC loves=cooking
Pascal: ✅ Tracked Sarah with 3 attributes
# Weeks later...
You: What did I tell you about Sarah?
Pascal: [Recalls: Sarah is 28, lives in NYC, loves cooking]
# View all tracked people
You: memories
Pascal: 📋 Tracked Individuals:
        • Sarah: age=28, location=NYC, loves=cooking
        • John: age=35, location=London, job=engineer
# Safe deletion with 3-stage authentication
You: forget John
Pascal: ⚠️ WARNING: Stage 1/3: Say 'confirm forget'
You: confirm forget
Pascal: 🔒 Stage 2/3: Say 'delete confirmed'
You: delete confirmed
Pascal: 🔒 Stage 3/3: Say 'permanently delete'
You: permanently delete
Pascal: ✅ Successfully forgot 'John'. Backup created.

✨ Features
Phase 1 - Core System (Current)
✅ Ollama Integration - Professional local model management
✅ Smart Routing - Intelligent offline/online LLM switching
✅ Infinite Memory - 10-year retention with 50,000 interaction limit
✅ Individual Tracking - Remember people and their details indefinitely
✅ 3-Stage Authentication - Voice-safe deletion with automatic backups
✅ Personality System - Customizable and switchable personalities
✅ Performance Profiles - Speed/Balanced/Quality modes optimized for Pi 5
✅ Model Management - Download, switch, and remove models via commands
✅ Web Search Integration - Brave Search API for current information
✅ Virtual Environment - Isolated dependency management
✅ Auto-installer - One-command setup with Ollama integration
Phase 2 - Enhanced Features (Planned)
🔄 Voice Input - Speech-to-text using Whisper
🔄 Text-to-Speech - Natural voice responses
🔄 Interrupt Handling - Listen while speaking
🔄 Visual Display - Mission Impossible-style reactive graphics
Phase 3 - Skills & Integration (Planned)
🔄 Google Maps - Navigation and location services
🔄 Messaging - SMS/WhatsApp integration
🔄 Music Control - Spotify and local music playback
🔄 Smart Home - IoT device control
🔄 Weather - Real-time weather information
🚀 Quick Start
Prerequisites
Raspberry Pi 5 with 8GB+ RAM (16GB recommended)
1TB NVMe SSD (or 32GB+ SD card)
Internet connection (for Ollama installation and online features)
Installation
Clone the repository:

git clone https://github.com/soundcatchers/pascal.git
cd pascal

Run the installer (much faster with Ollama!):

chmod +x install.sh
./install.sh

Install Ollama and download models:

./download_models.sh

Start Pascal:

./run.sh

🤖 Recommended Models for Pi 5
Model	Size	Speed	Quality	Best For
phi3:mini	2.3GB	⭐⭐⭐⭐⭐	⭐⭐⭐⭐	Quick responses, general chat
llama3.2:3b	2.0GB	⭐⭐⭐⭐	⭐⭐⭐⭐⭐	Balanced performance
gemma2:2b	1.6GB	⭐⭐⭐⭐⭐	⭐⭐⭐	Minimal resources
qwen2.5:7b	4.4GB	⭐⭐⭐	⭐⭐⭐⭐⭐	Complex reasoning
💬 Usage
Basic Commands
help - Show system status and commands
status - Display detailed system information
models - List available Ollama models
model [name] - Switch to different model
download [model] - Download new model via Ollama
remove [model] - Remove model to free space
profile [speed|balanced|quality] - Set performance profile
personality [name] - Switch personality
clear - Clear conversation history
quit or exit - Stop Pascal
Memory Commands (NEW!)
track [name] [key=value]... - Track an individual with attributes
Example: track John age=25 location=London likes=pizza
memories or individuals - List all tracked individuals
forget [name] - Start 3-stage authentication to forget someone
complete memory wipe - Start 3-stage authentication to wipe all memories
cancel - Cancel any active authentication process
Example Conversation
You: Hello Pascal
Pascal: Hello! I'm Pascal, powered by Ollama. How can I help you today?
You: My friend Sarah is 28, lives in NYC and loves cooking
Pascal: [Responds naturally and remembers this in context]
You: track Sarah age=28 location=NYC loves=cooking
Pascal: ✅ Tracked Sarah with 3 attributes
You: memories
Pascal: 📋 Tracked Individuals:
        • Sarah: age=28, location=NYC, loves=cooking
You: models
Pascal: [Shows table of available models with sizes and performance ratings]
You: Who won the last F1 race?
Pascal: [Searches web and provides current information]
You: Who came second?
Pascal: [Understands this is a follow-up and searches with F1 context]

Model Management
# In Pascal chat:
download llama3.2:3b     # Download new model
model llama3.2:3b        # Switch to model
remove old-model         # Remove unused model
models                   # List all models
# Command line (alternative):
ollama pull phi3:mini    # Download model
ollama list              # List models  
ollama rm old-model      # Remove model

🏗️ Architecture
Module Structure
├── main.py                    # Entry point & orchestration (~50 lines)
├── requirements.txt           # Python dependencies
├── install.sh                # One-command installer with venv setup
├── run.sh                    # Activation script for easy startup
├── README.md                 # Auto-updating documentation
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore (includes venv/)
├── venv/                     # Virtual environment (git ignored)
├── config/
│   ├── settings.py           # Global configuration
│   ├── personalities/        # JSON personality definitions
│   │   ├── default.json
│   │   ├── assistant.json
│   │   └── custom.json
│   └── skills_config.json    # Skills settings & API keys
├── modules/
│   ├── personality.py        # Pascal's personality management & switching
│   ├── offline_llm.py        # Local model handling (llama.cpp/Ollama)
│   ├── online_llm.py         # API calls (OpenAI, Anthropic, Gemini)
│   ├── router.py             # Smart offline/online routing
│   ├── memory.py             # Pascal's short/long-term memory system
│   ├── voice_input.py        # Speech-to-text (Whisper)
│   ├── voice_output.py       # Pascal's text-to-speech (Coqui TTS)
│   ├── visual_display.py     # Pascal's MI-style reactive graphics
│   ├── skills_manager.py     # Pascal's skills orchestration
│   ├── interrupt_manager.py  # Interrupt detection & handling
│   ├── audio_processor.py    # VAD & noise cancellation
│   └── threading_manager.py  # Thread coordination & management
├── skills/
│   ├── maps.py               # Google Maps integration
│   ├── messaging.py          # SMS/WhatsApp APIs
│   ├── music.py              # Spotify/local music control
│   ├── weather.py            # Weather data
│   ├── smart_home.py         # IoT device control
│   └── custom_skills/        # User-added skills directory
├── utils/
│   ├── installer.py          # Auto-setup functions
│   ├── helpers.py            # Common utility functions
│   └── audio_analysis.py     # FFT & audio processing for visuals
├── data/
│   ├── models/               # Local LLM storage
│   ├── memory/               # Pascal's conversation history & learning
│   ├── personalities/        # Pascal's personality-specific data
│   └── cache/                # API response caching
└── tests/
    ├── test_modules.py       # Unit tests for each module
    └── integration_tests.py  # Full system tests

Smart Routing Logic
Pascal intelligently decides between offline (Ollama) and online processing based on:

Query complexity and type
Current information requirements
Available local models
Performance profile settings
Resource availability
🔧 Configuration
Performance Profiles
Speed Profile: Fast responses (1-2s) using optimized models
Balanced Profile: Good quality and speed (2-4s)
Quality Profile: Best responses (3-6s) using larger models
Personalities
Create custom personalities by adding JSON files to config/personalities/:

{
  "name": "Custom Pascal",
  "description": "A customized version of Pascal",
  "traits": {
    "helpfulness": 0.9,
    "curiosity": 0.8,
    "formality": 0.3,
    "humor": 0.7
  },
  "speaking_style": {
    "tone": "friendly and casual",
    "verbosity": "concise but informative"
  },
  "system_prompt": "You are Pascal, a helpful AI assistant with a custom personality..."
}

🛠️ Development
Adding New Models
# From Pascal chat:
download mistral:7b          # Download any Ollama model
model mistral:7b            # Switch to new model
# Models are automatically integrated

Testing Performance
# Test all performance profiles
python3 test_performance.py
# Quick test
python3 test_performance.py quick
# Stress test
python3 test_performance.py stress 60

📊 Performance
Hardware Requirements
Minimum: Raspberry Pi 5 8GB, 32GB storage
Recommended: Raspberry Pi 5 16GB, 1TB NVMe SSD
Optimal: Above + active cooling for sustained performance
Pi 5 Benchmarks (Ollama vs Direct GGUF)
Metric	Ollama	Direct GGUF	Improvement
Installation Time	5 minutes	45+ minutes	9x faster
Model Loading	10-15 seconds	30-60 seconds	3x faster
Memory Usage	Optimized	Manual tuning	Better
Model Switching	Instant	Restart required	Much better
Reliability	High	Variable	More stable
Response Times on Pi 5
Model	Speed Profile	Balanced Profile	Quality Profile
phi3:mini	1.5s	2.5s	3.5s
llama3.2:3b	2.0s	3.0s	4.5s
gemma2:2b	1.2s	2.0s	3.0s
qwen2.5:7b	3.0s	4.5s	6.0s
🤝 Contributing
Fork the repository
Create a feature branch
Make your changes
Add tests
Submit a pull request
Development Guidelines
Follow PEP 8 style guidelines
Add docstrings to all functions
Include type hints
Write tests for new features
Update documentation
📝 Changelog
Version 4.3.0 (Current - Infinite Memory Edition)
🎉 INFINITE LONG-TERM MEMORY - 10-year retention (50,000 interactions)
👥 Individual Tracking - Remember people and their details forever
🔒 3-Stage Authentication - Voice-safe deletion with automatic backups
🐛 Stricter Follow-up Detection - Prevents false positives on topic shifts
🔍 Enhanced Web Search - Brave Search API integration
⚡ Generic Context Enhancement - Works across all domains, not just F1
Automatic backup before destructive operations
2-minute timeout with cancel option for auth challenges
Version 1.1.0 (Previous - Ollama)
Major upgrade to Ollama integration
9x faster installation (no compilation needed)
Improved model management with download/switch/remove commands
Better ARM optimization for Pi 5
More reliable model downloads
Enhanced performance monitoring
Improved error handling and fallbacks
Version 1.0.0 (Initial - Direct GGUF)
Initial release with llama-cpp-python
Basic offline/online routing
Personality management system
Basic memory management
🔒 Privacy & Security
Local Processing: Sensitive data stays on-device with Ollama
Infinite Local Memory: 10-year conversation history stored locally only
Individual Privacy: Tracked people data never leaves your device
3-Stage Authentication: Prevents accidental deletion from voice errors
Automatic Backups: All deletions create timestamped backups
API Key Security: Environment variable protection
Model Isolation: Ollama manages model sandboxing
Network Control: Choose when to use online services
Memory Safety Features
Voice-Safe Deletion: 3 different confirmation phrases required
Timeout Protection: 2-minute timeout per authentication stage
Cancel Anytime: Type "cancel" at any stage to abort
Backup Location: data/memory/backup_[session]_[timestamp].json
No Accidental Loss: Impossible to delete data by accident
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Ollama Team for excellent local LLM infrastructure
Raspberry Pi Foundation for amazing ARM hardware
Open-source AI/ML ecosystem
Built for the Pi enthusiast community
📞 Support
Issues: Use GitHub Issues for bug reports
Discussions: Use GitHub Discussions for questions
Documentation: Check the wiki for detailed guides
Models: Browse Ollama Model Library
🔗 Useful Commands
# System management
sudo systemctl status ollama    # Check Ollama service
vcgencmd measure_temp          # Monitor Pi temperature
htop                          # Monitor system resources
# Ollama management  
ollama list                   # List downloaded models
ollama show [model]           # Show model details
ollama ps                     # Show running models

Made with ❤️ for Raspberry Pi enthusiasts and AI developers
Powered by 🦙 Ollama for the best local AI experience

Configure API Keys (Optional but Recommended) Get free API keys for enhanced functionality: Weather API (OpenWeatherMap)

Visit: https://openweathermap.org/api Sign up for free account Get API key (1,000 calls/day free) Add to .env: OPENWEATHER_API_KEY=your_key_here

News API (NewsAPI)

Visit: https://newsapi.org Sign up for free account Get API key (100 requests/day free) Add to .env: NEWS_API_KEY=your_key_here

Groq API (Required for current info)

Visit: https://console.groq.com/ Get free API key Add to .env: GROQ_API_KEY=gsk_your_key_here
