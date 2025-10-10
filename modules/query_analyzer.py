"""
Pascal AI Assistant - Enhanced Query Analyzer - FIXED "CURRENT" DETECTION
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
        """Compile optimized regex patterns - FIXED for generic "current" queries"""
        
        # Layer 1: High-confidence current info patterns - EXPANDED
        self.high_confidence_patterns = [
            # DateTime patterns
            re.compile(r'\bwhat\s+(?:day|date)\s+(?:is\s+)?(?:it\s+)?(?:today|now)?\b', re.I),
            re.compile(r'\bwhat\s+day\s+is\s+it\b', re.I),
            re.compile(r'\bwhat\s+day\s+is\s+today\b', re.I),
            re.compile(r'\btoday\'?s?\s+(?:date|day)\b', re.I),
            re.compile(r'\b(?:current|what)\s+date\b', re.I),
            re.compile(r'\bwhat\s+is\s+(?:the\s+)?date\s+today\b', re.I),
            re.compile(r'\btell\s+me\s+(?:the\s+)?(?:date|day)\b', re.I),
            
            # Political patterns
            re.compile(r'\b(?:who\s+is|who\'?s)\s+(?:the\s+)?current\s+(?:\w+\s+)?(?:president|prime\s+minister|pm|leader)\b', re.I),
            re.compile(r'\bcurrent\s+(?:\w+\s+)?(?:president|prime\s+minister|pm|leader)\b', re.I),
            re.compile(r'\b(?:who\s+is|who\'?s)\s+(?:the\s+)?(?:us|uk|french|german|canadian|australian|indian|japanese|chinese)?\s*(?:president|prime\s+minister|pm)\b', re.I),
            re.compile(r'\b(?:who\s+is|who\'?s)\s+(?:president|prime\s+minister|pm)\s+(?:of|in)\s+\w+\b', re.I),
            
            # CRITICAL FIX: Generic "current" patterns for ANY fact/record/information
            re.compile(r'\bcurrent\s+(?:world|land|speed|temperature|stock|price|exchange|population)\s+record\b', re.I),
            re.compile(r'\bcurrent\s+(?:\w+\s+)?(?:record|price|rate|value|status|standing|ranking|leader|champion)\b', re.I),
            re.compile(r'\bwhat\s+is\s+(?:the\s+)?current\s+\w+', re.I),  # "what is the current [anything]"
            re.compile(r'\bwhat\'?s\s+(?:the\s+)?current\s+\w+', re.I),  # "what's the current [anything]"
            
            # CRITICAL FIX: Temporal phrases "as of [date/year]"
            re.compile(r'\bas\s+of\s+(?:today|now|this\s+year|\d{4}|january|february|march|april|may|june|july|august|september|october|november|december)', re.I),
            re.compile(r'\bin\s+\d{4}\b', re.I),  # "in 2025"
            re.compile(r'\bthis\s+(?:year|month|week)\b', re.I),
            
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
            
            # Market/financial current
            re.compile(r'\bcurrent\s+(?:stock|bitcoin|crypto|market)\s+price\b', re.I),
            re.compile(r'\b(?:stock|bitcoin)\s+price\s+(?:today|now)\b', re.I),
        ]
        
        # Layer 2: Medium-confidence patterns
        self.medium_confidence_patterns = [
            re.compile(r'\bweather\s+in\s+\w+\b', re.I),
            re.compile(r'\b(?:stock|share)\s+price\b', re.I),
            re.compile(r'\bexchange\s+rate\b', re.I),
            re.compile(r'\bnews\s+about\b', re.I),
            re.compile(r'\b(?:prime\s+minister|president|leader)\s+of\s+(?:the\s+)?\w+\b', re.I),
            # ADDED: "latest" with any noun
            re.compile(r'\blatest\s+\w+\b', re.I),
        ]
        
        # Layer 3: Temporal indicator patterns - EXPANDED
        self.temporal_patterns = [
            re.compile(r'\b(?:today|now|currently|right\s+now|at\s+the\s+moment)\b', re.I),
            re.compile(r'\b(?:latest|recent|breaking|fresh|new)\b', re.I),
            re.compile(r'\b(?:this\s+(?:morning|afternoon|evening|week|month|year))\b', re.I),
            re.compile(r'\b(?:up\s+to\s+date|real\s+time|live)\b', re.I),
            re.compile(r'\bcurrent\b', re.I),
            re.compile(r'\bas\s+of\b', re.I),  # ADDED
            re.compile(r'\bin\s+\d{4}\b', re.I),  # ADDED: "in 2025"
        ]
    
    def _load_keywords(self):
        """Load keyword sets for analysis"""
        
        # Strong current info indicators - EXPANDED
        self.strong_current_keywords = {
            'today', 'now', 'current', 'currently', 'latest', 'recent', 
            'breaking', 'live', 'real-time', 'up-to-date', 'fresh',
            'this', 'what', 'day', 'date', 'who', 'is',
            'as', 'of'  # ADDED for "as of"
        }
        
        # Current info topics - EXPANDED
        self.current_topics = {
            'news', 'headlines', 'weather', 'temperature', 'forecast',
            'president', 'election', 'politics', 'stocks', 'prices',
            'scores', 'results', 'events', 'happening',
            'day', 'date', 'time', 'prime', 'minister', 'pm', 'leader',
            'record', 'price', 'rate', 'value', 'status',  # ADDED
            'standing', 'ranking', 'champion', 'holder',  # ADDED
            'bitcoin', 'crypto', 'market', 'exchange'  # ADDED
        }
        
        # Non-current indicators (reduce current info score)
        self.non_current_keywords = {
            'explain', 'definition', 'how does', 'tutorial',
            'history', 'past', 'ancient', 'historical', 'traditional',
            'always', 'generally', 'typically', 'was', 'were', 'had been'
        }
    
    def analyze(self, query: str) -> float:
        """Multi-layer current information analysis - FIXED
        Returns: Score from 0.0 (not current) to 1.0 (definitely current)
        """
        query_lower = query.lower().strip()
        score = 0.0
        
        # Layer 1: High-confidence patterns (strong indicators)
        for pattern in self.high_confidence_patterns:
            if pattern.search(query_lower):
                score += 0.5
                
                # EXTRA BOOST: "current" + any noun
                if 'current' in query_lower:
                    # Check if "current" is followed by a noun (not just "currently")
                    if re.search(r'\bcurrent\s+\w+', query_lower):
                        score += 0.2  # Big boost for "current [thing]"
                
                # EXTRA BOOST: "as of" indicates time-specific query
                if 'as of' in query_lower:
                    score += 0.15
                
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
            score += min(0.3, temporal_count * 0.15)
        
        # Layer 4: Keyword analysis
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Strong current keywords
        current_word_count = len(words.intersection(self.strong_current_keywords))
        if current_word_count > 0:
            score += min(0.3, current_word_count * 0.15)
        
        # Current topics
        topic_word_count = len(words.intersection(self.current_topics))
        if topic_word_count > 0:
            score += min(0.25, topic_word_count * 0.12)
            
            # Extra boost for multiple topic words
            if topic_word_count >= 2:
                score += 0.1
        
        # Non-current indicators (penalty)
        non_current_count = len(words.intersection(self.non_current_keywords))
        if non_current_count > 0:
            score -= min(0.3, non_current_count * 0.1)
        
        # Layer 5: Context analysis
        # Question format analysis
        if query.strip().endswith('?'):
            if any(q in query_lower for q in ['what', 'when', 'who', 'where']):
                score += 0.1
        
        # "who is" questions are almost always current info
        if query_lower.startswith('who is') or query_lower.startswith("who's"):
            score += 0.15
        
        # CRITICAL: "what is the current" is DEFINITELY current info
        if re.search(r'\bwhat\s+is\s+(?:the\s+)?current\b', query_lower):
            score += 0.25
        
        # Ensure score stays in bounds
        return max(0.0, min(1.0, score))
    
    def get_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators from query"""
        indicators = []
        query_lower = query.lower()
        
        for pattern in self.temporal_patterns:
            matches = pattern.findall(query_lower)
            indicators.extend(matches)
        
        return list(set(indicators))

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
                re.compile(r'\bwhat\s+day\s+is\s+(?:it\s+)?(?:today|now)?\b', re.I),
                re.compile(r'\btoday\'?s?\s+date\b', re.I),
                re.compile(r'\bcurrent\s+date\b', re.I),
                re.compile(r'\bwhat\s+(?:is\s+)?(?:the\s+)?date\b', re.I),
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
        
        # Political queries are CURRENT_INFO
        if any(word in query_lower for word in ['president', 'prime minister', 'pm', 'leader']) and any(word in query_lower for word in ['who', 'current', 'is']):
            return QueryIntent.CURRENT_INFO
        
        # ADDED: "current [anything]" is CURRENT_INFO
        if re.search(r'\bcurrent\s+\w+', query_lower):
            return QueryIntent.CURRENT_INFO
        
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
            QueryIntent.TIME_QUERY, QueryIntent.DATE_QUERY,
            QueryIntent.CURRENT_INFO
        }
        if intent in high_confidence_intents:
            confidence += 0.2
        
        # Query length confidence
        word_count = len(query.split())
        if word_count <= 5 or word_count >= 15:
            confidence += 0.1
        
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

# Test function
async def test_analyzer():
    """Test the enhanced query analyzer with problematic queries"""
    analyzer = EnhancedQueryAnalyzer()
    
    test_queries = [
        # The problematic queries from user's debug output
        "what is the current land speed record",
        "what is the current land speed record as of today in 2025",
        
        # Other current queries
        "what day is it",
        "who is the current english pm",
        "what time is it",
        
        # Should be offline
        "explain how recursion works",
        "what is the capital of France",
    ]
    
    print("🧪 Testing Enhanced Query Analyzer (FIXED 'CURRENT' DETECTION)")
    print("=" * 70)
    
    for query in test_queries:
        analysis = await analyzer.analyze_query(query)
        
        print(f"\n📝 Query: '{query}'")
        print(f"   Current Info Score: {analysis.current_info_score:.2f}")
        print(f"   Intent: {analysis.intent.value}")
        print(f"   Complexity: {analysis.complexity.value}")
        print(f"   Confidence: {analysis.confidence:.2f}")
        print(f"   Temporal: {analysis.temporal_indicators}")
        
        # Show routing decision
        if analysis.current_info_score >= 0.7:
            print(f"   ✅ Should route to: ONLINE (current info)")
        elif analysis.complexity == QueryComplexity.INSTANT:
            print(f"   → Should route to: SKILL (instant)")
        else:
            print(f"   → Should route to: OFFLINE (general)")
    
    print(f"\n" + "=" * 70)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_analyzer())
