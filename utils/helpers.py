"""
Pascal AI Assistant - Helper Utilities
Common utility functions used across modules
"""

import os
import time
import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

def format_timestamp(timestamp: float = None) -> str:
    """Format timestamp as readable string"""
    if timestamp is None:
        timestamp = time.time()
    
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_file_hash(file_path: Union[str, Path]) -> str:
    """Get MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def safe_filename(filename: str) -> str:
    """Convert string to safe filename"""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename.strip()

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and control characters"""
    import re
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text (simple implementation)"""
    import re
    
    # Convert to lowercase and extract words
    text = text.lower()
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may',
        'might', 'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Filter out stop words and count frequency
    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]

def is_question(text: str) -> bool:
    """Check if text appears to be a question"""
    text = text.strip()
    
    # Check for question mark
    if text.endswith('?'):
        return True
    
    # Check for question words at the beginning
    question_words = ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', 'did']
    first_word = text.split()[0].lower() if text.split() else ''
    
    return first_word in question_words

def estimate_reading_time(text: str, words_per_minute: int = 200) -> float:
    """Estimate reading time in minutes"""
    word_count = len(text.split())
    return word_count / words_per_minute

def parse_time_duration(duration_str: str) -> Optional[float]:
    """Parse duration string to seconds (e.g., '5m', '30s', '1h')"""
    import re
    
    pattern = r'(\d+(?:\.\d+)?)\s*([smhd]?)'
    match = re.match(pattern, duration_str.lower().strip())
    
    if not match:
        return None
    
    value, unit = match.groups()
    value = float(value)
    
    multipliers = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        '': 1  # Default to seconds
    }
    
    return value * multipliers.get(unit, 1)

async def retry_async(func, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry an async function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            await asyncio.sleep(delay)
            delay *= backoff

def load_json_file(file_path: Union[str, Path], default: Any = None) -> Any:
    """Safely load JSON file with fallback"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        return default

def save_json_file(file_path: Union[str, Path], data: Any, indent: int = 2) -> bool:
    """Safely save data to JSON file"""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception:
        return False

def get_system_info() -> Dict[str, Any]:
    """Get basic system information"""
    import platform
    import psutil
    
    try:
        cpu_count = os.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': cpu_count,
            'memory_total': memory.total,
            'memory_available': memory.available,
            'memory_percent': memory.percent,
            'disk_total': disk.total,
            'disk_free': disk.free,
            'disk_percent': (disk.used / disk.total) * 100
        }
    except ImportError:
        # psutil not available
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count()
        }

def validate_config(config: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """Validate configuration and return list of missing fields"""
    missing_fields = []
    
    for field in required_fields:
        if '.' in field:
            # Nested field
            parts = field.split('.')
            current = config
            
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    missing_fields.append(field)
                    break
        else:
            # Top-level field
            if field not in config:
                missing_fields.append(field)
    
    return missing_fields

def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def format_error_message(error: Exception, context: str = None) -> str:
    """Format error message for user display"""
    error_type = type(error).__name__
    error_msg = str(error)
    
    if context:
        return f"{context}: {error_type} - {error_msg}"
    else:
        return f"{error_type}: {error_msg}"

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a text-based progress bar"""
    if total == 0:
        return "[" + "=" * width + "] 100%"
    
    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + "-" * (width - filled)
    percentage = int(progress * 100)
    
    return f"[{bar}] {percentage}%"

class Timer:
    """Simple context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0
    
    def __str__(self) -> str:
        return f"{self.description}: {self.elapsed:.2f}s"

class AsyncTimer:
    """Async context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0
    
    def __str__(self) -> str:
        return f"{self.description}: {self.elapsed:.2f}s"
