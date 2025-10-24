# Pascal AI Assistant - File Relationships & Dependencies

## Table of Contents
1. [Module Dependency Graph](#module-dependency-graph)
2. [Import Analysis](#import-analysis)
3. [Data Flow Between Files](#data-flow-between-files)
4. [Configuration Propagation](#configuration-propagation)
5. [Shared State & Communication](#shared-state--communication)
6. [File Purpose Matrix](#file-purpose-matrix)

---

## Module Dependency Graph

### Level 0: Core Dependencies (External)
```
aiohttp (HTTP client)
requests (HTTP requests)
rich (Terminal UI)
python-dotenv (Environment variables)
psutil (System monitoring)
```

### Level 1: Configuration Layer
```
.env
  └── config/settings.py (Settings class)
      └── Consumed by ALL modules
```

### Level 2: Data Models & Utilities
```
modules/query_analyzer.py
  ├── QueryAnalysis (dataclass)
  ├── QueryComplexity (enum)
  ├── QueryIntent (enum)
  └── EnhancedQueryAnalyzer (class)

modules/current_info.py
  └── CurrentInfoDetector (class)

modules/prompt_builder.py
  └── build_prompt() function

utils/helpers.py
  └── Utility functions

utils/search_helpers.py
  └── Search-related utilities
```

### Level 3: Core Services
```
modules/personality.py
  └── PersonalityManager
      ├── Reads: config/personalities/*.json
      └── Used by: modules/intelligent_router.py, main.py

modules/memory.py
  └── MemoryManager
      ├── Reads/Writes: data/memory/*.json
      └── Used by: modules/intelligent_router.py, main.py

modules/offline_llm.py
  └── LightningOfflineLLM
      ├── Depends on: aiohttp, config/settings.py
      ├── Connects to: Ollama (http://localhost:11434)
      └── Used by: modules/intelligent_router.py

modules/online_llm.py
  └── OnlineLLM
      ├── Depends on: aiohttp, config/settings.py
      ├── Connects to: Groq API, Google Search API
      └── Used by: modules/intelligent_router.py
```

### Level 4: Orchestration Layer
```
modules/intelligent_router.py
  └── LightningRouter
      ├── Uses: query_analyzer.py
      ├── Uses: prompt_builder.py
      ├── Uses: current_info.py
      ├── Uses: offline_llm.py
      ├── Uses: online_llm.py
      ├── Uses: personality.py (via parameter)
      ├── Uses: memory.py (via parameter)
      └── Used by: main.py
```

### Level 5: Application Layer
```
main.py
  └── Pascal (class)
      ├── Uses: intelligent_router.py (LightningRouter)
      ├── Uses: personality.py (PersonalityManager)
      ├── Uses: memory.py (MemoryManager)
      ├── Uses: config/settings.py (Settings)
      └── Entry point for application
```

---

## Import Analysis

### main.py Imports
```python
import asyncio                          # Async operations
import signal                           # Signal handling
import sys                              # System operations
from pathlib import Path                # Path manipulation
from rich.console import Console        # Terminal formatting
from rich.panel import Panel            # UI panels
from rich.table import Table            # UI tables
from rich.text import Text              # Text formatting

from config.settings import settings    # ← Configuration
from modules.router import LightningRouter  # ← Core routing (NOTE: Should be intelligent_router)
from modules.personality import PersonalityManager  # ← Personality
from modules.memory import MemoryManager  # ← Memory
```

**Note:** There's a discrepancy - `main.py` imports `LightningRouter` from `modules.router` but the actual file is `modules/intelligent_router.py`. This suggests there may be a `__init__.py` or the file was renamed.

### config/settings.py Imports
```python
import os                               # Environment variables
import json                             # JSON handling
from pathlib import Path                # Path operations
from typing import Dict, Any, Optional, List  # Type hints
from dotenv import load_dotenv          # Environment variable loading
```
**No internal dependencies** - This is the base configuration layer.

### modules/intelligent_router.py Imports
```python
import asyncio                          # Async operations
import time                             # Timing
import json                             # JSON handling
from enum import Enum                   # Enumerations
from dataclasses import dataclass, asdict  # Data structures
from typing import Dict, List, Optional, AsyncGenerator, Tuple, Any  # Type hints
from pathlib import Path                # Path operations

from modules.query_analyzer import (    # ← Query analysis
    EnhancedQueryAnalyzer, QueryAnalysis, QueryComplexity, QueryIntent
)
from config.settings import settings    # ← Configuration
from modules.prompt_builder import build_prompt  # ← Prompt construction
```

### modules/offline_llm.py Imports
```python
import asyncio                          # Async operations
import json                             # JSON handling
import time                             # Timing
import re                               # Regular expressions
from typing import Optional, AsyncGenerator, Dict, Any, List  # Type hints

try:
    import aiohttp                      # HTTP client (conditional)
except ImportError:
    aiohttp = None

# Lazy import from config
from config.settings import settings    # ← Configuration
```

### modules/online_llm.py Imports
```python
import asyncio                          # Async operations
import json                             # JSON handling
import time                             # Timing
import re                               # Regular expressions
import os                               # Environment variables
from typing import Optional, AsyncGenerator, Dict, Any, List  # Type hints
from datetime import datetime, timezone  # Date/time handling

try:
    import aiohttp                      # HTTP client (conditional)
except ImportError:
    aiohttp = None

from config.settings import settings    # ← Configuration
```

### modules/personality.py Imports
```python
import json                             # JSON handling
import asyncio                          # Async operations
from typing import Dict, Any, Optional, List  # Type hints
from pathlib import Path                # Path operations

from config.settings import settings    # ← Configuration
```

### modules/memory.py Imports
```python
import json                             # JSON handling
import asyncio                          # Async operations
import time                             # Timing
from typing import Dict, Any, List, Optional  # Type hints
from pathlib import Path                # Path operations
from datetime import datetime, timedelta  # Date/time handling

from config.settings import settings    # ← Configuration
```

### modules/query_analyzer.py Imports
```python
import re                               # Regular expressions
from typing import List, Dict, Any, Optional, Set  # Type hints
from enum import Enum                   # Enumerations
from dataclasses import dataclass       # Data structures

from config.settings import settings    # ← Configuration
```

---

## Data Flow Between Files

### Configuration Flow
```
.env (environment variables)
  ↓
config/settings.py (loads and validates)
  ↓
settings object (singleton instance)
  ↓
Imported by ALL modules
  ├── main.py
  ├── modules/intelligent_router.py
  ├── modules/offline_llm.py
  ├── modules/online_llm.py
  ├── modules/personality.py
  ├── modules/memory.py
  └── modules/query_analyzer.py
```

### User Query Flow
```
User Input (main.py)
  ↓
Command Check (main.py)
  ├── If command: Execute command → Display result
  └── If query: Continue
  ↓
Query Analysis (intelligent_router.py)
  ↓ calls
EnhancedQueryAnalyzer.analyze() (query_analyzer.py)
  ↓ returns
QueryAnalysis object
  ↓
Routing Decision (intelligent_router.py)
  ├── Check offline_llm availability
  ├── Check online_llm availability
  ├── Evaluate query requirements
  └── Make routing decision
  ↓
Route to appropriate service
  ├── Offline: LightningOfflineLLM (offline_llm.py)
  │   ├── Get personality context (personality.py)
  │   ├── Get memory context (memory.py)
  │   ├── Build prompt (prompt_builder.py)
  │   ├── Call Ollama API
  │   └── Stream response
  │
  └── Online: OnlineLLM (online_llm.py)
      ├── Detect search needs
      ├── Execute Google Search (if needed)
      ├── Get personality context (personality.py)
      ├── Get memory context (memory.py)
      ├── Build enhanced prompt
      ├── Call Groq API
      └── Stream response with sources
  ↓
Response Processing (main.py)
  ├── Display streamed tokens
  └── Collect complete response
  ↓
Memory Update (memory.py)
  ├── Store interaction
  ├── Extract learnings
  └── Auto-save if needed
  ↓
Performance Tracking (intelligent_router.py)
  ├── Record response time
  ├── Update success rate
  └── Calculate health metrics
```

### Personality Context Flow
```
config/personalities/*.json (files)
  ↓
PersonalityManager.load_personality() (personality.py)
  ↓
personality_data (dict)
  ↓
PersonalityManager.get_system_prompt() (personality.py)
  ↓
system_prompt (string)
  ↓
Used by:
  ├── LightningOfflineLLM._build_prompt() (offline_llm.py)
  └── OnlineLLM.generate_response_stream() (online_llm.py)
```

### Memory Context Flow
```
data/memory/*.json (files)
  ↓
MemoryManager.load_session() (memory.py)
  ↓
short_term_memory (list of MemoryInteraction)
long_term_memory (list of MemoryInteraction)
user_preferences (dict)
learned_facts (dict)
  ↓
MemoryManager.get_context() (memory.py)
  ↓
context_string (string)
  ↓
Used by:
  ├── LightningOfflineLLM._build_prompt() (offline_llm.py)
  └── OnlineLLM.generate_response_stream() (online_llm.py)
```

### Performance Data Flow
```
System Response
  ↓
PerformanceTracker.record_request() (intelligent_router.py)
  ↓
SystemPerformance objects updated
  ├── total_requests++
  ├── successful_requests++ or failed_requests++
  ├── total_response_time += time
  ├── avg_response_time recalculated
  └── success_rate recalculated
  ↓
Used for routing decisions
  ├── LightningRouter._decide_route()
  └── LightningRouter.get_system_health()
```

---

## Configuration Propagation

### How Settings Reach Each Module

#### 1. Via Direct Import
Most modules import `settings` directly:
```python
from config.settings import settings

# Then access properties
settings.debug_mode
settings.ollama_host
settings.groq_api_key
```

**Modules using direct import:**
- `main.py`
- `modules/intelligent_router.py`
- `modules/offline_llm.py`
- `modules/online_llm.py`
- `modules/personality.py`
- `modules/memory.py`
- `modules/query_analyzer.py`

#### 2. Settings Usage by Module

**main.py:**
- `settings.debug_mode` - Debug output control
- `settings.get_config_summary()` - Status display

**modules/intelligent_router.py:**
- `settings.debug_mode` - Logging
- `settings.auto_route_current_info` - Routing behavior
- `settings.force_online_current_info` - Force online for current info
- `settings.current_info_confidence_threshold` - Confidence threshold

**modules/offline_llm.py:**
- `settings.debug_mode` - Logging
- `settings.ollama_host` - Ollama endpoint
- `settings.ollama_timeout` - Timeout settings
- `settings.ollama_keep_alive` - Keep-alive duration

**modules/online_llm.py:**
- `settings.debug_mode` - Logging
- `settings.groq_api_key` - API authentication
- `os.getenv('GOOGLE_SEARCH_API_KEY')` - Google API key
- `os.getenv('GOOGLE_SEARCH_ENGINE_ID')` - Search engine ID

**modules/personality.py:**
- `settings.debug_mode` - Logging
- `settings.config_dir` - Personality file directory
- `settings.get_personality_path()` - Path helper

**modules/memory.py:**
- `settings.debug_mode` - Logging
- `settings.short_term_memory_limit` - Memory limit
- `settings.long_term_memory_enabled` - Long-term feature flag
- `settings.memory_save_interval` - Auto-save interval
- `settings.get_memory_path()` - Path helper

**modules/query_analyzer.py:**
- `settings.debug_mode` - Logging
- (Potentially more in future)

---

## Shared State & Communication

### 1. Settings (Global, Read-Only)
```
config/settings.py
  └── settings (singleton)
      └── Shared across all modules (read-only after init)
```

### 2. Router State (Managed by LightningRouter)
```
LightningRouter instance (created in main.py)
  ├── offline_llm (LightningOfflineLLM instance)
  ├── online_llm (OnlineLLM instance)
  ├── query_analyzer (EnhancedQueryAnalyzer instance)
  ├── performance_tracker (PerformanceTracker instance)
  └── last_decision (IntelligentRouteDecision)

Shared via:
  - main.py holds the router instance
  - Router coordinates between offline_llm and online_llm
```

### 3. Personality State (Managed by PersonalityManager)
```
PersonalityManager instance (created in main.py)
  ├── current_personality (string)
  ├── personality_data (dict)
  └── personality_cache (dict)

Passed to:
  - LightningRouter (via constructor)
  - Used by offline_llm and online_llm via router
```

### 4. Memory State (Managed by MemoryManager)
```
MemoryManager instance (created in main.py)
  ├── short_term_memory (list)
  ├── long_term_memory (list)
  ├── user_preferences (dict)
  └── learned_facts (dict)

Passed to:
  - LightningRouter (via constructor)
  - Used by offline_llm and online_llm via router
  - Updated after each interaction
```

### 5. Performance State (Managed by PerformanceTracker)
```
PerformanceTracker instance (in LightningRouter)
  └── systems (dict of SystemPerformance)
      ├── 'offline' → SystemPerformance
      ├── 'online' → SystemPerformance
      └── 'skills' → SystemPerformance

Updated by:
  - LightningRouter after each request
  
Read by:
  - LightningRouter for routing decisions
  - main.py for status display
```

### Communication Patterns

#### 1. Synchronous Parameter Passing
```python
# main.py creates instances
personality_manager = PersonalityManager()
memory_manager = MemoryManager()
router = LightningRouter(personality_manager, memory_manager)

# Router passes to LLM services
async def get_response(query):
    personality_context = await self.personality_manager.get_system_prompt()
    memory_context = await self.memory_manager.get_context()
    
    if use_offline:
        return await self.offline_llm.generate_response(
            query, personality_context, memory_context
        )
```

#### 2. Asynchronous Callbacks
```python
# Streaming responses
async for chunk in self.router.get_streaming_response(query):
    print(chunk, end="", flush=True)
    response_text += chunk
```

#### 3. Event-Based Updates
```python
# Memory auto-save
if time.time() - self.last_save_time > self.save_interval:
    await self.save_session()
```

---

## File Purpose Matrix

| File | Primary Purpose | Depends On | Used By | State |
|------|----------------|------------|---------|-------|
| `main.py` | Application entry point, orchestration | router, personality, memory, settings | (none) | Router, personality, memory, session stats |
| `config/settings.py` | Central configuration | .env, OS | All modules | Settings object (singleton) |
| `modules/intelligent_router.py` | Smart routing logic | query_analyzer, offline_llm, online_llm, prompt_builder, settings | main.py | Performance tracker, routing decisions |
| `modules/offline_llm.py` | Local AI via Ollama | aiohttp, settings | intelligent_router | Session, model state, performance stats |
| `modules/online_llm.py` | Cloud AI via Groq + Search | aiohttp, settings | intelligent_router | Session, search state, performance stats |
| `modules/personality.py` | Personality management | settings | main.py, intelligent_router | Personality data, cache |
| `modules/memory.py` | Conversation memory | settings | main.py, intelligent_router | Short/long-term memory, preferences, facts |
| `modules/query_analyzer.py` | Query analysis | settings | intelligent_router | (stateless) |
| `modules/prompt_builder.py` | Prompt construction | (none) | intelligent_router | (stateless utility) |
| `modules/current_info.py` | Temporal detection | (none) | intelligent_router | (stateless utility) |
| `utils/helpers.py` | Utility functions | (varies) | (varies) | (stateless utilities) |
| `utils/search_helpers.py` | Search utilities | (varies) | online_llm | (stateless utilities) |

---

## Critical Dependencies

### External Services
1. **Ollama** (http://localhost:11434)
   - Required for: offline_llm.py
   - Fallback: online_llm.py
   - Failure mode: Falls back to online or error

2. **Groq API** (https://api.groq.com)
   - Required for: online_llm.py
   - Fallback: offline_llm.py
   - Failure mode: Falls back to offline or error

3. **Google Custom Search API** (https://www.googleapis.com)
   - Required for: online_llm.py (for current info queries)
   - Fallback: Continue without search
   - Failure mode: Generates response without real-time data

### Internal Dependencies
1. **config/settings.py**
   - Critical: YES
   - Required by: All modules
   - Failure mode: Application won't start

2. **modules/query_analyzer.py**
   - Critical: YES
   - Required by: intelligent_router.py
   - Failure mode: Router can't make intelligent decisions

3. **modules/personality.py**
   - Critical: NO
   - Required by: main.py, intelligent_router.py
   - Failure mode: Uses default personality

4. **modules/memory.py**
   - Critical: NO
   - Required by: main.py, intelligent_router.py
   - Failure mode: Operates without conversation history

---

## File Interaction Summary

### Most Connected Files (Hub Files)
1. **config/settings.py** - Used by ALL modules (configuration hub)
2. **modules/intelligent_router.py** - Coordinates offline_llm, online_llm, query analysis (orchestration hub)
3. **main.py** - Initializes and coordinates all major components (application hub)

### Least Connected Files (Leaf Files)
1. **modules/prompt_builder.py** - Pure utility, no dependencies
2. **modules/current_info.py** - Standalone detection logic
3. **utils/helpers.py** - Utility functions
4. **utils/search_helpers.py** - Search utilities

### Potential Circular Dependencies
None detected. The architecture follows a clear hierarchy:
```
Config Layer → Utility Layer → Service Layer → Orchestration Layer → Application Layer
```

---

## Recommendations for File Organization

### Current Strengths:
- Clear separation of concerns
- Minimal circular dependencies
- Centralized configuration
- Modular design

### Potential Improvements:
1. **Clarify router naming** - `modules/router.py` vs `modules/intelligent_router.py`
2. **Add __init__.py files** - Explicit module exports
3. **Type stub files** - For better IDE support
4. **Interface definitions** - Abstract base classes for LLM services
5. **Dependency injection** - More explicit parameter passing

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-24  
**Coverage:** All core Python modules analyzed
