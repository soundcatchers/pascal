"""
Pascal AI Assistant - Enhanced Query Analyzer
Intelligent query classification and routing optimization
"""

import re
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime

class QueryComplexity(Enum):
    """Query complexity levels"""
    INSTANT = "instant"        # <0.5s - Skills only
    SIMPLE = "simple"          # <2s - Prefer offline
    MODERATE = "moderate"      # 2-4s - Balanced routing  
    COMPLEX = "complex"        # 4-8s - Quality routing
    CURRENT = "current"        # Variable - Must be online if available

class QueryIntent(Enum):
    """Query intent classification"""
    GREETING = "greeting"
    CALCULATION = "calculation"
    TIME_QUERY = "time_query"
    DATE_QUERY = "date_query"
    WEATHER = "weather"
    NEWS = "news"
    EXPLANATION = "explanation"
    CREATION = "creation"
    COMPARISON = "comparison"
    CURRENT_INFO = "current_info"
    TROUBLESHOOTING = "troubleshooting"
    CASUAL_CHAT = "casual_chat"
    PROGRAMMING = "programming"

@dataclass
class QueryAnalysis:
    """Complete query analysis result"""
    original_query: str
    complexity: QueryComplexity
    intent: QueryIntent
    current_info_score: float
    temporal_indicators: List[str]
    confidence: float
    processing_time: float
    metadata: Dict[str, any]

class MultiLayerDetection:
    """Multi-layer current information detection system"""
    
    def __init__(self):
        self._compile_patterns()
        self._load_keywords()
    
    def _compile_patterns(self):
        """Compile optimized regex patterns"""
        
        # Layer 1: High-confidence current info patterns
        self.high_confidence_patterns = [
            # Datetime with current indicators
            re.compile(r'\b(?:what|tell me|show me)\s+(?:day|date)\s+(?:is\s+)?(?:it\s+)?today\b', re.I),
            re.compile(r'\btoday\'?s?\s+date\b', re.I),
            re.compile(r'\bcurrent\s+date\b', re.I),
            re.compile(r'\bwhat\s+day\s+is\s+(?:it\s+)?today\b', re.I),
            
            # Political current info
            re.compile(r'\b(?:current|who\s+is\s+(?:the\s+)?current)\s+(?:president|pm|prime\s+minister)\b', re.I),
            re.compile(r'\bwho\s+is\s+(?:the\s+)?(?:current\s+)?(?:us\s+|american\s+)?president\b', re.I),
            
            # News and events
            re.compile(r'\b(?:latest|breaking|recent|today\'?s?)\s+(?:news|headlines)\b', re.I),
            re.compile(r'\bwhat\'?s\s+(?:happening|in\s+the\s+news)\s+(?:today|now|currently)?\b', re.I),
            re.compile(r'\bcurrent\s+events\b', re.I),
            
            # Weather current
            re.compile(r'\b(?:current|today\'?s?|now)\s+weather\b', re.I),
            re.compile(r'\bweather\s+(?:today|now|currently)\b', re.I),
            
            # Sports current
            re.compile(r'\b(?:latest|current|today\'?s?)\s+(?:scores?|results?)\b', re.I),
            re.compile(r'\bwho\s+won\s+(?:today|yesterday|last\s+night)\b', re.I),
        ]
        
        # Layer 2: Medium-confidence patterns
        self.medium_confidence_patterns = [
            re.compile(r'\bweather\s+in\s+\w+\b', re.I),
            re.compile(r'\b(?:stock|share)\s+price\b', re.I),
            re.compile(r'\bexchange\s+rate\b', re.I),
            re.compile(r'\bnews\s+about\b', re.I),
        ]
        
        # Layer 3: Temporal indicator patterns
        self.temporal_patterns = [
            re.compile(r'\b(?:today|now|currently|right\s+now|at\s+the\s+moment)\b', re.I),
            re.compile(r'\b(?:latest|recent|breaking|fresh|new)\b', re.I),
            re.compile(r'\b(?:this\s+(?:morning|afternoon|evening|week|month|year))\b', re.I),
            re.compile(r'\b(?:up\s+to\s+date|real\s+time|live)\b', re.I),
        ]
    
    def _load_keywords(self):
        """Load keyword sets for analysis"""
        
        # Strong current info indicators
        self.strong_current_keywords = {
            'today', 'now', 'current', 'currently', 'latest', 'recent', 
            'breaking', 'live', 'real-time', 'up-to-date', 'fresh'
        }
        
        # Current info topics
        self.current_topics = {
            'news', 'headlines', 'weather', 'temperature', 'forecast',
            'president', 'election', 'politics', 'stocks', 'prices',
            'scores', 'results', 'events', 'happening'
        }
        
        # Non-current indicators (reduce current info score)
        self.non_current_keywords = {
            'explain', 'definition', 'what is', 'how does', 'tutorial',
            'history', 'past', 'ancient', 'historical', 'traditional'
        }
    
    def analyze(self, query: str) -> float:
        """Multi-layer current information analysis
        Returns: Score from 0.0 (not current) to 1.0 (definitely current)
        """
        query_lower = query.lower().strip()
        score = 0.0
        
        # Layer 1: High-confidence patterns (strong indicators)
        for pattern in self.high_confidence_patterns:
            if pattern.search(query_lower):
                score += 0.4  # Strong boost
                break
        
        # Layer 2: Medium-confidence patterns
        for pattern in self.medium_confidence_patterns:
            if pattern.search(query_lower):
                score += 0.2
                break
        
        # Layer 3: Temporal indicators
        temporal_count = 0
        for pattern in self.temporal_patterns:
            if pattern.search(query_lower):
                temporal_count += 1
        
        if temporal_count > 0:
            score += min(0.3, temporal_count * 0.1)
        
        # Layer 4: Keyword analysis
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Strong current keywords
        current_word_count = len(words.intersection(self.strong_current_keywords))
        if current_word_count > 0:
            score += min(0.3, current_word_count * 0.15)
        
        # Current topics
        topic_word_count = len(words.intersection(self.current_topics))
        if topic_word_count > 0:
            score += min(0.2, topic_word_count * 0.1)
        
        # Non-current indicators (penalty)
        non_current_count = len(words.intersection(self.non_current_keywords))
        if non_current_count > 0:
            score -= min(0.3, non_current_count * 0.1)
        
        # Layer 5: Context analysis
        # Question format analysis
        if query.strip().endswith('?'):
            if any(q in query_lower for q in ['what', 'when', 'who', 'where']):
                # Interrogative questions more likely to need current info
                score += 0.1
        
        # Ensure score stays in bounds
        return max(0.0, min(1.0, score))
    
    def get_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators from query"""
        indicators = []
        query_lower = query.lower()
        
        for pattern in self.temporal_patterns:
            matches = pattern.findall(query_lower)
            indicators.extend(matches)
        
        return list(set(indicators))  # Remove duplicates

class QueryClassifier:
    """Intelligent query classification system"""
    
    def __init__(self):
        self._compile_classification_patterns()
    
    def _compile_classification_patterns(self):
        """Compile patterns for intent and complexity classification"""
        
        # Intent patterns
        self.intent_patterns = {
            QueryIntent.GREETING: [
                re.compile(r'\b(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))\b', re.I),
                re.compile(r'\bhow\s+are\s+you\b', re.I),
            ],
            
            QueryIntent.CALCULATION: [
                re.compile(r'\b\d+\s*[\+\-\*\/\%]\s*\d+\b'),
                re.compile(r'\bwhat\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\b', re.I),
                re.compile(r'\bcalculate\b', re.I),
                re.compile(r'\b\d+\s+percent\s+of\s+\d+\b', re.I),
            ],
            
            QueryIntent.TIME_QUERY: [
                re.compile(r'\bwhat\s+time\s+is\s+it\b', re.I),
                re.compile(r'\bcurrent\s+time\b', re.I),
                re.compile(r'^time\??$', re.I),
            ],
            
            QueryIntent.DATE_QUERY: [
                re.compile(r'\bwhat\s+day\s+is\s+(?:it\s+)?today\b', re.I),
                re.compile(r'\btoday\'?s?\s+date\b', re.I),
                re.compile(r'\bcurrent\s+date\b', re.I),
            ],
            
            QueryIntent.WEATHER: [
                re.compile(r'\bweather\b', re.I),
                re.compile(r'\btemperature\b', re.I),
                re.compile(r'\bforecast\b', re.I),
                re.compile(r'\b(?:raining|snowing|sunny|cloudy)\b', re.I),
            ],
            
            QueryIntent.NEWS: [
                re.compile(r'\bnews\b', re.I),
                re.compile(r'\bheadlines\b', re.I),
                re.compile(r'\bbreaking\b', re.I),
                re.compile(r'\bevents\b', re.I),
            ],
            
            QueryIntent.EXPLANATION: [
                re.compile(r'\bexplain\b', re.I),
                re.compile(r'\bwhat\s+is\b', re.I),
                re.compile(r'\bhow\s+does\b', re.I),
                re.compile(r'\bwhy\s+(?:is|does|do)\b', re.I),
            ],
            
            QueryIntent.CREATION: [
                re.compile(r'\bwrite\s+(?:a|me|some)\b', re.I),
                re.compile(r'\bcreate\b', re.I),
                re.compile(r'\bgenerate\b', re.I),
                re.compile(r'\bmake\s+(?:a|me)\b', re.I),
            ],
            
            QueryIntent.PROGRAMMING: [
                re.compile(r'\bpython\s+(?:code|function|script)\b', re.I),
                re.compile(r'\bcode\s+(?:for|to|that)\b', re.I),
                re.compile(r'\bfunction\s+(?:to|that|for)\b', re.I),
                re.compile(r'\bdebug\b', re.I),
            ],
        }
        
        # Complexity indicators
        self.complexity_keywords = {
            QueryComplexity.INSTANT: {
                'hi', 'hello', 'time', 'what time',
            },
            QueryComplexity.SIMPLE: {
                'what is', 'how are you', 'calculate', 'convert',
            },
            QueryComplexity.MODERATE: {
                'explain', 'describe', 'how does', 'why',
            },
            QueryComplexity.COMPLEX: {
                'analyze', 'compare', 'evaluate', 'comprehensive',
                'detailed', 'in-depth', 'thorough', 'complete'
            },
        }
    
    def get_intent(self, query: str) -> QueryIntent:
        """Classify query intent"""
        query_lower = query.lower().strip()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    return intent
        
        # Default classification based on query characteristics
        if '?' in query and len(query.split()) <= 5:
            return QueryIntent.CASUAL_CHAT
        elif len(query.split()) > 20:
            return QueryIntent.EXPLANATION
        else:
            return QueryIntent.CASUAL_CHAT
    
    def get_complexity(self, query: str) -> QueryComplexity:
        """Classify query complexity"""
        query_lower = query.lower().strip()
        word_count = len(query.split())
        
        # Check for complexity keywords
        for complexity, keywords in self.complexity_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return complexity
        
        # Fallback to word count analysis
        if word_count <= 3:
            return QueryComplexity.INSTANT
        elif word_count <= 8:
            return QueryComplexity.SIMPLE
        elif word_count <= 15:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX

class EnhancedQueryAnalyzer:
    """Main query analyzer with multi-layer intelligence"""
    
    def __init__(self):
        self.detection_system = MultiLayerDetection()
        self.classifier = QueryClassifier()
        
        # Performance tracking
        self.analysis_count = 0
        self.total_analysis_time = 0.0
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Comprehensive query analysis"""
        start_time = time.time()
        
        # Multi-layer current info detection
        current_info_score = self.detection_system.analyze(query)
        temporal_indicators = self.detection_system.get_temporal_indicators(query)
        
        # Intent and complexity classification
        intent = self.classifier.get_intent(query)
        complexity = self.classifier.get_complexity(query)
        
        # Override complexity for current info queries
        if current_info_score >= 0.7:
            complexity = QueryComplexity.CURRENT
        
        # Calculate confidence based on various factors
        confidence = self._calculate_confidence(
            query, current_info_score, intent, complexity
        )
        
        processing_time = time.time() - start_time
        
        # Update performance tracking
        self.analysis_count += 1
        self.total_analysis_time += processing_time
        
        # Create metadata
        metadata = {
            'word_count': len(query.split()),
            'char_count': len(query),
            'has_question_mark': '?' in query,
            'analysis_number': self.analysis_count,
            'is_short_query': len(query.split()) <= 3,
            'is_long_query': len(query.split()) > 20,
        }
        
        return QueryAnalysis(
            original_query=query,
            complexity=complexity,
            intent=intent,
            current_info_score=current_info_score,
            temporal_indicators=temporal_indicators,
            confidence=confidence,
            processing_time=processing_time,
            metadata=metadata
        )
    
    def _calculate_confidence(self, query: str, current_info_score: float, 
                            intent: QueryIntent, complexity: QueryComplexity) -> float:
        """Calculate analysis confidence score"""
        confidence = 0.5  # Base confidence
        
        # High confidence for clear patterns
        if current_info_score >= 0.8:
            confidence += 0.3
        elif current_info_score >= 0.6:
            confidence += 0.2
        elif current_info_score <= 0.2:
            confidence += 0.2  # High confidence it's NOT current info
        
        # Intent-based confidence
        high_confidence_intents = {
            QueryIntent.GREETING, QueryIntent.CALCULATION, 
            QueryIntent.TIME_QUERY, QueryIntent.DATE_QUERY
        }
        if intent in high_confidence_intents:
            confidence += 0.2
        
        # Query length confidence
        word_count = len(query.split())
        if word_count <= 5 or word_count >= 15:
            confidence += 0.1  # Very short or long queries are easier to classify
        
        return min(1.0, confidence)
    
    def get_analysis_stats(self) -> Dict[str, any]:
        """Get analyzer performance statistics"""
        avg_time = self.total_analysis_time / max(self.analysis_count, 1)
        
        return {
            'total_analyses': self.analysis_count,
            'average_analysis_time': f"{avg_time:.4f}s",
            'total_analysis_time': f"{self.total_analysis_time:.4f}s",
            'analyses_per_second': f"{1/avg_time:.1f}" if avg_time > 0 else "N/A"
        }

# Test function for development
async def test_analyzer():
    """Test the enhanced query analyzer"""
    analyzer = EnhancedQueryAnalyzer()
    
    test_queries = [
        # Current info queries
        "What day is today?",
        "Who is the current president?", 
        "Latest news headlines",
        "Current weather in London",
        "Today's date",
        
        # Non-current queries
        "Hello, how are you?",
        "What is 2+2?",
        "Explain Python programming",
        "Write a function to sort a list",
        "What is the capital of France?",
        
        # Edge cases
        "What time is it?",  # Should be instant/skills
        "Weather forecast",   # Could be current or general
        "News about AI",     # Could be current or general
    ]
    
    print("🧪 Testing Enhanced Query Analyzer")
    print("=" * 50)
    
    for query in test_queries:
        analysis = await analyzer.analyze_query(query)
        
        print(f"\nQuery: '{query}'")
        print(f"  Intent: {analysis.intent.value}")
        print(f"  Complexity: {analysis.complexity.value}")
        print(f"  Current Info Score: {analysis.current_info_score:.2f}")
        print(f"  Confidence: {analysis.confidence:.2f}")
        print(f"  Temporal Indicators: {analysis.temporal_indicators}")
        print(f"  Processing Time: {analysis.processing_time:.4f}s")
    
    print(f"\n📊 Analysis Statistics:")
    stats = analyzer.get_analysis_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_analyzer())
