# Pascal AI Assistant

An intelligent, offline-first AI assistant designed for Raspberry Pi 5, featuring modular architecture, personality management, and seamless offline/online switching.

## 🤖 About Pascal

Pascal is a comprehensive AI assistant that runs locally on Raspberry Pi 5 while maintaining the ability to leverage online AI services when needed. Named after the programming language, Pascal combines local intelligence with cloud capabilities to provide fast, reliable assistance.

## ✨ Features

### Phase 1 - Core System (Current)
- ✅ **Modular Architecture** - Clean, maintainable code structure
- ✅ **Smart Routing** - Intelligent offline/online LLM switching
- ✅ **Personality System** - Customizable and switchable personalities
- ✅ **Memory Management** - Short-term and long-term conversation memory
- ✅ **Virtual Environment** - Isolated dependency management
- ✅ **Auto-installer** - One-command setup on multiple Pi devices

### Phase 2 - Enhanced Features (Planned)
- 🔄 **Voice Input** - Speech-to-text using Whisper
- 🔄 **Text-to-Speech** - Natural voice responses
- 🔄 **Interrupt Handling** - Listen while speaking
- 🔄 **Visual Display** - Mission Impossible-style reactive graphics

### Phase 3 - Skills & Integration (Planned)
- 🔄 **Google Maps** - Navigation and location services
- 🔄 **Messaging** - SMS/WhatsApp integration
- 🔄 **Music Control** - Spotify and local music playback
- 🔄 **Smart Home** - IoT device control
- 🔄 **Weather** - Real-time weather information

## 🚀 Quick Start

### Prerequisites
- Raspberry Pi 5 with 16GB RAM
- 1TB NVMe SSD
- USB microphone and speaker
- Internet connection (for initial setup and online features)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pascal.git
   cd pascal
   ```

2. **Run the installer:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Start Pascal:**
   ```bash
   ./run.sh
   ```

### Configuration

1. **API Keys (Optional):**
   Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   nano .env
   ```

2. **Local Model (Optional):**
   Download a compatible GGUF model to `data/models/local_model.gguf`

## 💬 Usage

### Basic Commands
- `help` - Show available commands
- `status` - Display system status
- `personality [name]` - Switch personality
- `clear` - Clear conversation history
- `quit` or `exit` - Stop Pascal

### Example Conversation
```
You: Hello Pascal
Pascal: Hello! I'm Pascal. How can I help you today?

You: What's the weather like?
Pascal: I'd be happy to help with weather information, but I need to set up 
        weather services first. You can configure this in the skills settings.

You: personality assistant
Pascal: ✅ Switched to assistant personality

You: status
Pascal: [Shows detailed system status including available LLMs, memory usage, etc.]
```

## 🏗️ Architecture

### Module Structure
```
pascal/
├── main.py                    # Entry point
├── config/
│   ├── settings.py           # Global configuration
│   └── personalities/        # Personality definitions
├── modules/
│   ├── router.py             # Smart LLM routing
│   ├── personality.py        # Personality management
│   ├── memory.py             # Memory system
│   ├── offline_llm.py        # Local model handling
│   ├── online_llm.py         # API integration
│   └── ...                   # Additional modules
├── skills/                   # Extensible skills
├── utils/                    # Helper utilities
└── data/                     # Models, memory, cache
```

### Smart Routing Logic
Pascal intelligently decides between offline and online processing based on:
- Query complexity
- Current information requirements
- Available resources
- User preferences
- Performance metrics

## 🔧 Configuration

### Personalities
Create custom personalities by adding JSON files to `config/personalities/`:

```json
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
```

### Memory System
- **Short-term**: Recent conversation context (configurable limit)
- **Long-term**: Persistent conversation history
- **Learning**: User preferences and facts
- **Auto-save**: Periodic memory persistence

## 🛠️ Development

### Adding New Skills
1. Create a new file in `skills/`
2. Implement the skill interface
3. Register in `config/skills_config.json`
4. Test and document

### Testing
```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
python -m pytest tests/

# Test specific module
python -m pytest tests/test_modules.py::TestRouter
```

### Debugging
Enable debug mode by setting `DEBUG=true` in your `.env` file or:
```bash
export DEBUG=true
./run.sh
```

## 📊 Performance

### Hardware Requirements
- **Minimum**: Raspberry Pi 5 8GB, 32GB SD card
- **Recommended**: Raspberry Pi 5 16GB, 1TB NVMe SSD
- **Optimal**: Above + AI accelerator (future support)

### Benchmarks
| Configuration | Response Time | Memory Usage |
|---------------|---------------|--------------|
| Offline Only  | 2-5 seconds   | 4-8GB RAM    |
| Online Only   | 1-3 seconds   | 1-2GB RAM    |
| Hybrid        | 1-5 seconds   | 3-6GB RAM    |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints
- Write tests for new features
- Update documentation

## 📝 Changelog

### Version 1.0.0 (Current)
- Initial release with core functionality
- Smart offline/online routing
- Personality management system
- Memory management
- Modular architecture

## 🔒 Privacy & Security

- **Local Processing**: Sensitive data can stay on-device
- **No Persistent Logging**: Conversations stored locally only
- **API Key Security**: Environment variable protection
- **Memory Encryption**: Optional memory encryption (planned)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by advanced AI assistants and voice interfaces
- Built for the Raspberry Pi community
- Thanks to the open-source AI/ML ecosystem

## 📞 Support

- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the wiki for detailed guides

---

**Made with ❤️ for Raspberry Pi enthusiasts and AI developers**
