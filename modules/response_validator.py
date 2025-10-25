"""
Response Validator for Pascal
Ensures responses are complete and conversational
"""

import re
from typing import Optional

class ResponseValidator:
    """Validates and improves response quality"""
    
    def __init__(self):
        self.min_response_length = 10
        self.incomplete_patterns = [
            r'\.{3,}$',  # Ends with multiple dots
            r'\s+$',     # Ends with whitespace
            r'[,;:]$',   # Ends with comma/semicolon/colon
            r'\([^)]*$', # Unclosed parentheses
            r'"[^"]*$',  # Unclosed quotes
        ]
    
    def validate_response(self, response: str, query: str) -> dict:
        """Validate response completeness and quality"""
        response = response.strip()
        
        validation_result = {
            'is_complete': True,
            'is_relevant': True,
            'quality_score': 1.0,
            'issues': [],
            'suggestions': []
        }
        
        # Check minimum length
        if len(response) < self.min_response_length:
            validation_result['is_complete'] = False
            validation_result['issues'].append('Response too short')
            validation_result['quality_score'] -= 0.3
        
        # Check for incomplete patterns
        for pattern in self.incomplete_patterns:
            if re.search(pattern, response):
                validation_result['is_complete'] = False
                validation_result['issues'].append('Response appears incomplete')
                validation_result['quality_score'] -= 0.2
                break
        
        # Check relevance (basic keyword matching)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if len(query_words & response_words) == 0 and len(query.split()) > 2:
            validation_result['is_relevant'] = False
            validation_result['issues'].append('Response may not be relevant to query')
            validation_result['quality_score'] -= 0.4
        
        # Check for conversational elements
        conversational_indicators = [
            'i', 'you', 'your', 'let me', 'i can', 'would you like',
            'here is', 'here are', 'based on', 'according to'
        ]
        
        if not any(indicator in response.lower() for indicator in conversational_indicators):
            validation_result['suggestions'].append('Consider making response more conversational')
            validation_result['quality_score'] -= 0.1
        
        return validation_result
    
    def improve_response(self, response: str, query: str, validation_result: dict) -> str:
        """Attempt to improve response based on validation"""
        improved_response = response.strip()
        
        # If response is too short, add context
        if len(improved_response) < self.min_response_length:
            if 'who came second' in query.lower() or 'who was second' in query.lower():
                improved_response = f"Regarding your question about who came second: {improved_response}. Would you like more details about the race results?"
            elif any(word in query.lower() for word in ['what', 'who', 'where', 'when', 'how']):
                improved_response = f"To answer your question: {improved_response}. Let me know if you need more information!"
        
        # If response seems incomplete, add a completion
        if not validation_result['is_complete']:
            if not improved_response.endswith(('.', '!', '?')):
                improved_response += "."
            
            # Add helpful follow-up
            improved_response += " Is there anything specific about this topic you'd like to know more about?"
        
        # If response lacks conversational tone, improve it
        if 'Consider making response more conversational' in validation_result.get('suggestions', []):
            if not improved_response.startswith(('I', 'Based on', 'According to', 'Let me', 'Here')):
                improved_response = f"Based on the information available, {improved_response.lower()}"
        
        return improved_response
