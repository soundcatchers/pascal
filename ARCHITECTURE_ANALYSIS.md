# Pascal AI Assistant - Architecture Analysis

## Project Overview

Pascal is an AI assistant designed for Raspberry Pi 5 that combines local (offline) AI processing via Ollama with online AI capabilities via Groq. The system features intelligent routing between offline and online models, personality management, conversation memory, and real-time information retrieval through Google Search.

**Version:** 4.3.0 (Ultra-speed-optimized)
**Primary Language:** Python
**Target Hardware:** Raspberry Pi 5 (optimized for 8GB-16GB RAM)

---

## Core Architecture

### 1. Entry Point: `main.py`

The main entry point orchestrates the entire Pascal system through the `Pascal` class:

**Key Components:**
- **Initialization**: Sets up router, personality manager, and memory manager
- **Chat Loop**: Handles user interaction and command processing
- **Streaming Responses**: Real-time response generation
- **Performance Tracking**: Monitors queries, response times, and routing decisions
- **Graceful Shutdown**: Saves memory and closes connections

**Flow:**
```
Initialize → Check System Availability → Display Status → Start Chat Loop → Process Input → 
Route Query → Generate Response → Update Memory → Save Session → Shutdown
```

---

## Module Breakdown

### 2. Configuration (`config/settings.py`)

**Purpose:** Central configuration management for all system settings

**Key Features:**
- Hardware detection (Raspberry Pi model, RAM, CPU cores)
- Performance optimization (ultra-speed settings for Pi 5)
- API key management (Groq)
- Ollama configuration (host, timeout, threading)
- Memory limits and context windows
- Performance profiles (speed, balanced, quality)

**Important Settings:**
- `target_response_time`: 1.5s (ultra-aggressive)
- `max_context_length`: 128-256 tokens (ultra-minimal for speed)
- `max_response_tokens`: 40-60 tokens (very short responses)
- `ollama_timeout`: 5-6 seconds
- `streaming_enabled`: True (real-time responses)

**Hardware Optimization:**
- Pi 5 16GB: Most aggressive (1.3s target, 256 context)
- Pi 5 8GB: Balanced ultra-speed (1.5s target, 128 context)
- Pi 4: Conservative (3.0s target, 64 context)

---

### 3. Intelligent Router (`modules/intelligent_router.py`)

**Purpose:** Smart routing between offline (Ollama) and online (Groq) LLMs

**Key Classes:**
- `IntelligentRouteDecision`: Routing decision with reasoning
- `PerformanceTracker`: Tracks system performance metrics
- `LightningRouter`: Main routing logic

**Routing Logic:**
1. **Query Analysis**: Uses `EnhancedQueryAnalyzer` to analyze query
2. **Current Info Detection**: Identifies queries needing real-time data
3. **System Health Check**: Evaluates offline/online system availability
4. **Performance Analysis**: Considers past performance metrics
5. **Route Decision**: Chooses optimal path (offline/online/fallback)

**Decision Factors:**
- Query complexity and intent
- Temporal indicators (today, latest, recent, etc.)
- System availability and health
- Performance history
- Follow-up query detection

**Routing Rules:**
- **Offline (Nemotron)**: General queries, explanations, coding help
- **Online (Groq)**: Current events, recent news, time-sensitive info, sports results
- **Fallback**: When primary system fails

---

### 4. Offline LLM (`modules/offline_llm.py`)

**Purpose:** Local AI processing via Ollama

**Key Class:** `LightningOfflineLLM`

**Features:**
- Ollama integration (HTTP API)
- Model management (automatic selection from preferred models)
- Performance profiling (speed, balanced, quality)
- Keep-alive management (maintains loaded model)
- Streaming and non-streaming responses

**Preferred Models (in priority order):**
1. `nemotron-mini:4b-instruct-q4_K_M` (primary)
2. `nemotron-fast` (speed-optimized)
3. `qwen2.5:3b-instruct`
4. `phi3:mini`
5. `gemma2:2b`

**Performance Profiles:**
- **Speed**: 80 tokens, 0.3 temp, 512 context → 2-4s target
- **Balanced**: 150 tokens, 0.5 temp, 1024 context → 3-6s target
- **Quality**: 300 tokens, 0.7 temp, 2048 context → 5-10s target

**Connection Management:**
- HTTP session with connection pooling
- TCP connector with keep-alive
- Automatic retry on connection errors
- Health monitoring and consecutive error tracking

---

### 5. Online LLM (`modules/online_llm.py`)

**Purpose:** Cloud-based AI via Groq API with Google Search integration

**Key Class:** `OnlineLLM`

**Features:**
- Groq API integration (llama-3.1-8b-instant model)
- Google Custom Search API integration
- Intelligent search query optimization
- News vs. general search routing
- Temporal query detection
- Source citation in responses

**Search Capabilities:**
- **General Search**: Sports, facts, current information
- **News Search**: Breaking news, political events, recent developments
- **Query Optimization**: Automatic query refinement for better results
- **Sports-Specific**: F1, BTCC, race results optimization

**Search Detection Patterns:**
- Temporal indicators: "today", "latest", "recent", "who won"
- Sports queries: "last F1 race", "who won the game"
- Current info: "current prime minister", "what's happening"
- News events: "breaking news", "latest news", "what happened"

**Response Enhancement:**
- Real-time datetime injection
- Search result formatting
- Source attribution
- Fallback search on no results

---

### 6. Personality Manager (`modules/personality.py`)

**Purpose:** Manages Pascal's personality traits and conversation style

**Key Class:** `PersonalityManager`

**Features:**
- Personality loading from JSON configs
- Personality switching at runtime
- System prompt generation
- Conversational phrase management
- Response style adjustment

**Personality Configuration:**
```json
{
  "name": "Default Pascal",
  "description": "Friendly and helpful AI assistant",
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
  "system_prompt": "You are Pascal...",
  "conversation_style": {
    "greeting": "Hello! I'm Pascal...",
    "thinking": "Let me think about that...",
    "clarification": "Could you help me understand...",
    "completion": "I hope that helps!",
    "error": "I'm having trouble with that..."
  }
}
```

**Personality Aspects:**
- Greeting phrases
- Thinking indicators
- Clarification requests
- Completion phrases
- Error handling phrases
- Response tone adjustment

---

### 7. Memory Manager (`modules/memory.py`)

**Purpose:** Conversation history and context management

**Key Classes:**
- `MemoryInteraction`: Single interaction storage
- `MemoryManager`: Memory management system

**Memory Types:**
- **Short-term Memory**: Recent conversation (last 50 interactions)
- **Long-term Memory**: Historical data (optional, 30-day retention)
- **User Preferences**: Learned likes, dislikes, habits
- **Learned Facts**: User-specific information

**Features:**
- Automatic learning from conversations
- Preference extraction (likes, loves, hates, prefers)
- Fact extraction (name, job, location)
- Context generation for LLM prompts
- Memory search and retrieval
- Session import/export
- Auto-save (every 5 minutes)

**Source Tracking:**
- Stores metadata with responses
- Tracks search result sources
- Enables follow-up query detection
- Provides citation information

**Memory Cleanup:**
- Removes interactions older than 30 days
- Limits long-term memory to 1000 interactions
- Maintains most recent data priority

---

### 8. Query Analyzer (`modules/query_analyzer.py`)

**Purpose:** Advanced query analysis for intelligent routing

**Key Class:** `EnhancedQueryAnalyzer`

**Analysis Dimensions:**
- **Query Complexity**: Simple, moderate, complex, very complex
- **Query Intent**: Information, explanation, current info, creative, skill
- **Temporal Context**: Current, recent, historical, specific time
- **Domain Detection**: Sports, news, politics, tech, science, etc.
- **Follow-up Detection**: Identifies continuations of previous queries

**Features:**
- Multi-pattern matching
- Confidence scoring
- Real-time info requirement detection
- Sports result detection
- News event detection
- Source citation needs

---

### 9. Current Info Module (`modules/current_info.py`)

**Purpose:** Real-time information detection and handling

**Features:**
- Temporal keyword detection
- Current event patterns
- Time-sensitive query identification
- Weather, news, stock detection

---

### 10. Prompt Builder (`modules/prompt_builder.py`)

**Purpose:** Construct optimized prompts for LLMs

**Features:**
- Context injection (personality, memory)
- Token optimization
- System message formatting
- User query structuring

---

## Data Flow

### Typical Query Flow:

```
User Input
    ↓
Command Processing (help, status, clear, etc.)
    ↓
Query Analysis (EnhancedQueryAnalyzer)
    ├─ Complexity Analysis
    ├─ Intent Detection
    ├─ Temporal Context
    ├─ Domain Detection
    └─ Follow-up Detection
    ↓
Routing Decision (LightningRouter)
    ├─ Check System Availability
    ├─ Analyze Query Requirements
    ├─ Evaluate Performance History
    └─ Choose Route (Offline/Online/Fallback)
    ↓
Response Generation
    ├─ Offline Path (Ollama)
    │   ├─ Load Personality Context
    │   ├─ Load Memory Context
    │   ├─ Build Optimized Prompt
    │   ├─ Generate via Nemotron
    │   └─ Stream Response
    │
    └─ Online Path (Groq)
        ├─ Detect Search Needs
        ├─ Execute Google Search (if needed)
        ├─ Inject Current DateTime
        ├─ Build Enhanced Prompt
        ├─ Generate via Groq
        └─ Stream Response with Sources
    ↓
Memory Storage
    ├─ Store Interaction
    ├─ Extract Learnings
    ├─ Update Preferences
    └─ Auto-save (if interval elapsed)
    ↓
Performance Tracking
    ├─ Record Response Time
    ├─ Update Success Rate
    ├─ Track Query Type Performance
    └─ Calculate Health Metrics
```

---

## File Relationships

### Core Dependencies:

```
main.py
├── config/settings.py (configuration)
├── modules/intelligent_router.py (routing)
│   ├── modules/query_analyzer.py (analysis)
│   ├── modules/offline_llm.py (local AI)
│   ├── modules/online_llm.py (cloud AI)
│   ├── modules/current_info.py (temporal detection)
│   └── modules/prompt_builder.py (prompt construction)
├── modules/personality.py (personality)
└── modules/memory.py (conversation memory)
```

### Configuration Flow:

```
.env (environment variables)
    ↓
config/settings.py (loads and validates)
    ↓
All Modules (consume settings)
```

### Data Persistence:

```
data/
├── memory/
│   └── default_memory.json (conversation history)
└── cache/
    └── (temporary cache files)

config/
└── personalities/
    ├── default.json
    ├── assistant.json
    └── custom.json
```

---

## External Integrations

### 1. Ollama (Local AI)
- **Endpoint**: http://localhost:11434
- **Models**: Nemotron, Qwen, Phi3, Gemma
- **Protocol**: HTTP REST API
- **Features**: Model management, streaming, keep-alive

### 2. Groq (Cloud AI)
- **API**: https://api.groq.com/openai/v1/chat/completions
- **Model**: llama-3.1-8b-instant
- **Auth**: API key (gsk_xxx)
- **Features**: Fast inference, streaming, function calling

### 3. Google Custom Search
- **API**: https://www.googleapis.com/customsearch/v1
- **Auth**: API key + Search Engine ID
- **Features**: General search, news search, result ranking
- **Usage**: Current information, sports results, news events

---

## Performance Optimization

### Speed Optimizations:

1. **Minimal Context Windows**: 64-256 tokens (vs. typical 2048+)
2. **Ultra-Short Responses**: 40-60 tokens (vs. typical 200+)
3. **Aggressive Timeouts**: 5-6s (vs. typical 30s)
4. **Single Connection Pool**: Reduced overhead
5. **Skip Context for Short Queries**: ≤8 words
6. **Keep-Alive Management**: Model stays loaded
7. **Streaming**: Real-time token delivery
8. **Performance Profiling**: Adaptive based on history

### Hardware-Specific:

- **Pi 5 Optimizations**: 4-core threading, ARM-optimized builds
- **Memory Management**: Minimal memory limits, no long-term storage by default
- **Connection Pooling**: Limited to 1-2 connections
- **DNS Caching**: Reduces lookup overhead

---

## Error Handling

### Fallback Mechanisms:

1. **Offline Unavailable → Online**: Automatic fallback
2. **Online Unavailable → Offline**: Reverse fallback
3. **Both Unavailable → Error Message**: Graceful degradation
4. **Model Load Failure → Try Next Model**: Automatic model selection
5. **Search Failure → Continue Without Results**: No blocking
6. **Timeout → Retry with Shorter Context**: Adaptive retries

### Health Monitoring:

- **Consecutive Error Tracking**: After 3 failures, reduce confidence
- **Success Rate Calculation**: Real-time metrics
- **Response Time Tracking**: Performance degradation detection
- **System Availability Checks**: Pre-query validation

---

## Security Considerations

1. **API Key Management**: Environment variables, validation
2. **Local Processing**: Sensitive data stays on-device
3. **No Persistent Logging**: Privacy by default
4. **Input Validation**: Command parsing safety
5. **Connection Security**: HTTPS for external APIs

---

## Command System

### User Commands:

- `help` / `status` → Display system status
- `health` → Show detailed health report
- `clear` → Clear conversation history
- `debug` → Toggle debug mode
- `stats` → Show detailed statistics
- `quit` / `exit` → Graceful shutdown

### Command Processing:

```python
async def process_command(self, user_input: str) -> bool:
    command = user_input.lower().strip()
    if command in ['quit', 'exit', 'bye']:
        return False  # Signal shutdown
    elif command in ['help', 'status']:
        self.display_status()
    # ... more commands
    return True  # Continue running
```

---

## Testing & Diagnostics

### Test Files:

```
dev/tests/
├── performance_test.py (response time testing)
├── test_routing_intelligence.py (routing logic tests)
├── skills_diagnostic.py (skills system tests)
└── fast_nemotron_test.py (offline model tests)

dev/diagnostics/
├── complete_diagnostic.py (full system diagnostic)
├── ollama_diagnostic.py (Ollama connectivity)
├── aiohttp_quick_fix.py (dependency fixes)
└── fix_aiohttp_issue.py (aiohttp troubleshooting)
```

---

## Installation & Setup

### Dependencies (`requirements.txt`):

**Critical:**
- `aiohttp==3.9.5` (HTTP client for async operations)
- `requests==2.31.0` (HTTP requests)
- `python-dotenv==1.0.0` (environment variables)

**UI/Display:**
- `rich==13.7.0` (terminal formatting)
- `colorama==0.4.6` (color support)

**System:**
- `psutil==5.9.8` (system monitoring)
- `pyyaml==6.0.1` (configuration files)

**Performance:**
- `uvloop>=0.19.0` (async event loop optimization, Unix only)

### Installation Scripts:

- `install.sh`: Main installer (sets up venv, installs deps)
- `download_models.sh`: Ollama installation and model download
- `run.sh`: Activation script for easy startup

---

## Future Enhancements (Planned)

### Phase 2:
- Voice Input (Whisper speech-to-text)
- Text-to-Speech (Coqui TTS)
- Interrupt Handling (listen while speaking)
- Visual Display (Mission Impossible-style graphics)

### Phase 3:
- Google Maps integration
- SMS/WhatsApp messaging
- Music control (Spotify)
- Smart home IoT control
- Weather integration

---

## Key Design Decisions

### 1. Why Ollama over Direct GGUF?
- 9x faster installation (no compilation)
- Better ARM optimization for Pi
- Automatic model management
- Reliable downloads
- Built-in quantization

### 2. Why Groq over OpenAI/Anthropic?
- Faster inference (2-4s vs 5-10s)
- Lower latency
- Cost-effective for personal use
- Good model quality (Llama 3.1)

### 3. Why Minimal Context?
- Speed: Smaller context = faster processing
- Memory: Lower RAM usage on Pi
- Focus: Forces concise, relevant responses

### 4. Why Streaming?
- Better UX: User sees progress immediately
- Perceived speed: Feels faster than batch
- Interruptible: Can stop if needed

---

## Performance Metrics

### Target Performance:
- **Offline Queries**: 2-4 seconds (speed profile)
- **Online Queries**: 2-4 seconds (with search)
- **Overall Target**: <2 seconds average
- **Success Rate**: >95%

### Actual Performance (Pi 5 8GB):
- **Nemotron (offline)**: 2-3 seconds
- **Groq (online, no search)**: 1-2 seconds
- **Groq (online, with search)**: 3-4 seconds
- **Success Rate**: ~98%

---

## Code Quality & Maintainability

### Strengths:
- Clear module separation
- Extensive documentation
- Error handling throughout
- Performance tracking built-in
- Graceful degradation

### Areas for Improvement:
- Test coverage (basic tests exist)
- Type hints (partial coverage)
- Documentation generation (manual)
- CI/CD pipeline (not implemented)

---

## Summary

Pascal is a well-architected AI assistant that intelligently balances local and cloud processing to provide fast, accurate responses on resource-constrained hardware. The system demonstrates:

1. **Smart Routing**: Multi-factor decision making for optimal performance
2. **Performance Focus**: Ultra-aggressive optimizations for Pi 5
3. **Graceful Degradation**: Multiple fallback mechanisms
4. **User Experience**: Streaming responses, personality, memory
5. **Extensibility**: Modular design for future enhancements
6. **Real-time Data**: Google Search integration for current information

The architecture successfully addresses the challenges of running AI assistants on edge devices while maintaining responsiveness and quality.

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-24  
**Author:** AI Analysis (GitHub Copilot)
