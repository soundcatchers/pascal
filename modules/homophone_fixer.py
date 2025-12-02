"""
Homophone Fixer Module

Fast, rule-based correction for common voice recognition errors.
No AI required - instant corrections using context patterns.

Speed: <1ms (instant)
Cost: FREE (pure Python, no external calls)
"""

import re
from typing import Dict, List, Tuple, Optional


class HomophoneFixer:
    """Fast rule-based homophone and common voice error fixer"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._build_rules()
    
    def _build_rules(self):
        """Build correction rules with context patterns"""
        
        self.before_verb_words = {
            'going', 'doing', 'coming', 'leaving', 'trying', 'making',
            'taking', 'getting', 'having', 'being', 'saying', 'looking',
            'wanting', 'needing', 'running', 'walking', 'talking', 'working',
            'playing', 'eating', 'sleeping', 'waiting', 'sitting', 'standing',
            'are', 'is', 'was', 'were', 'been', 'be', 'not', 'always', 'never',
            'going', 'gonna', 'about', 'ready', 'able', 'welcome', 'right',
            'wrong', 'sure', 'happy', 'sad', 'angry', 'tired', 'sick', 'fine',
            'good', 'great', 'okay', 'here', 'late', 'early', 'busy', 'free'
        }
        
        self.after_subject_words = {'i', 'you', 'we', 'they', 'he', 'she', 'it', 'who'}
        
        self.purchase_context = {
            'a', 'the', 'some', 'any', 'this', 'that', 'new', 'used', 'cheap',
            'expensive', 'more', 'another', 'one', 'two', 'three', 'food',
            'groceries', 'tickets', 'clothes', 'stuff', 'things', 'something',
            'anything', 'nothing', 'everything', 'car', 'house', 'phone', 'book'
        }
        
        self.time_context = {
            'is', 'was', 'will', 'did', 'does', 'can', 'could', 'would',
            'should', 'might', 'i', 'you', 'we', 'they', 'he', 'she', 'it',
            'that', 'this', 'the', 'a', 'finally', 'just', 'already', 'soon'
        }
        
        self.rules: List[Tuple[str, str, callable]] = [
            ('there', "they're", self._should_be_theyre),
            ('their', "they're", self._should_be_theyre),
            ('your', "you're", self._should_be_youre),
            ('its', "it's", self._should_be_its_contraction),
            ('by', 'buy', self._should_be_buy),
            ('bye', 'buy', self._should_be_buy),
            ('no', 'know', self._should_be_know),
            ('know', 'no', self._should_be_no),
            ('to', 'too', self._should_be_too),
            ('to', 'two', self._should_be_two),
            ('hear', 'here', self._should_be_here),
            ('here', 'hear', self._should_be_hear),
            ('write', 'right', self._should_be_right),
            ('right', 'write', self._should_be_write),
            ('weather', 'whether', self._should_be_whether),
            ('than', 'then', self._should_be_then),
            ('then', 'than', self._should_be_than),
            ('are', 'our', self._should_be_our),
            ('hour', 'our', self._should_be_our),
            ('sea', 'see', self._should_be_see),
            ('eye', 'I', self._should_be_i),
            ('whims', 'when', self._should_be_when),
            ('brighten', 'Britain', self._should_be_britain),
        ]
        
        self.simple_replacements = {
            'wanna': 'want to',
            'gonna': 'going to',
            'gotta': 'got to',
            'kinda': 'kind of',
            'sorta': 'sort of',
            'coulda': 'could have',
            'shoulda': 'should have',
            'woulda': 'would have',
            'musta': 'must have',
            'oughta': 'ought to',
            'lemme': 'let me',
            'gimme': 'give me',
            'dunno': "don't know",
            'cuz': 'because',
            'cos': 'because',
            'ur': 'your',
            'u': 'you',
            'r': 'are',
            'n': 'and',
            'w': 'with',
            'wat': 'what',
            'wut': 'what',
            'da': 'the',
            'dis': 'this',
            'dat': 'that',
            'dey': 'they',
            'dem': 'them',
        }
    
    def _get_context(self, words: List[str], index: int) -> Tuple[Optional[str], Optional[str]]:
        """Get word before and after the target word"""
        before = words[index - 1].lower() if index > 0 else None
        after = words[index + 1].lower() if index < len(words) - 1 else None
        return before, after
    
    def _should_be_theyre(self, words: List[str], index: int) -> bool:
        """there/their → they're when followed by verb-like words"""
        _, after = self._get_context(words, index)
        return after in self.before_verb_words
    
    def _should_be_youre(self, words: List[str], index: int) -> bool:
        """your → you're when followed by verb-like words"""
        _, after = self._get_context(words, index)
        return after in self.before_verb_words
    
    def _should_be_its_contraction(self, words: List[str], index: int) -> bool:
        """its → it's when followed by verb-like words"""
        _, after = self._get_context(words, index)
        return after in self.before_verb_words
    
    def _should_be_buy(self, words: List[str], index: int) -> bool:
        """by/bye → buy when in purchase context"""
        before, after = self._get_context(words, index)
        if before in {'to', 'will', 'can', 'could', 'should', 'would', 'want', 'gonna', 'going'}:
            return True
        if after in self.purchase_context:
            return True
        return False
    
    def _should_be_know(self, words: List[str], index: int) -> bool:
        """no → know after subject pronouns"""
        before, after = self._get_context(words, index)
        if before in self.after_subject_words:
            if after in {'what', 'how', 'why', 'when', 'where', 'who', 'that', 'this', 'it', 'him', 'her', 'them', 'about', 'if'}:
                return True
        return False
    
    def _should_be_no(self, words: List[str], index: int) -> bool:
        """know → no at start or after certain words"""
        before, _ = self._get_context(words, index)
        if before is None:
            return False
        if before in {'say', 'said', 'says', 'have', 'has', 'had', 'got'}:
            return True
        return False
    
    def _should_be_too(self, words: List[str], index: int) -> bool:
        """to → too when meaning 'also' or 'excessively'"""
        before, after = self._get_context(words, index)
        if after in {'much', 'many', 'little', 'few', 'big', 'small', 'fast', 'slow', 'hot', 'cold', 'hard', 'easy', 'late', 'early', 'long', 'short', 'old', 'young', 'expensive', 'cheap', 'loud', 'quiet', 'far', 'close', 'high', 'low', 'tired', 'busy'}:
            return True
        if before and index == len(words) - 1:
            return True
        return False
    
    def _should_be_two(self, words: List[str], index: int) -> bool:
        """to → two when in numeric context"""
        before, after = self._get_context(words, index)
        if after in {'of', 'people', 'things', 'times', 'days', 'weeks', 'months', 'years', 'hours', 'minutes', 'seconds'}:
            return True
        if before in {'one', 'or', 'and', 'about', 'around', 'only', 'just', 'like'}:
            if after not in {'go', 'be', 'do', 'get', 'have', 'see', 'know', 'make', 'take'}:
                return True
        return False
    
    def _should_be_here(self, words: List[str], index: int) -> bool:
        """hear → here when referring to location"""
        before, after = self._get_context(words, index)
        if before in {'come', 'over', 'right', 'stay', 'wait', 'sit', 'stand', 'be', 'am', 'is', 'are'}:
            return True
        if after in {'is', 'are', 'we', 'i', 'you', 'it'}:
            return True
        return False
    
    def _should_be_hear(self, words: List[str], index: int) -> bool:
        """here → hear when referring to listening"""
        before, after = self._get_context(words, index)
        if before in {'can', 'could', 'cannot', "can't", 'did', "didn't", 'do', "don't", 'will', "won't", 'to', 'want'}:
            return True
        if after in {'me', 'you', 'him', 'her', 'them', 'us', 'it', 'that', 'this', 'what', 'anything', 'something', 'nothing'}:
            return True
        return False
    
    def _should_be_right(self, words: List[str], index: int) -> bool:
        """write → right when meaning correct or direction"""
        before, after = self._get_context(words, index)
        if before in {'is', 'are', 'am', 'was', 'were', 'be', 'been', "that's", "it's", "he's", "she's"}:
            return True
        if after in {'now', 'here', 'there', 'away', 'side', 'turn', 'hand', 'answer', 'thing', 'way'}:
            return True
        return False
    
    def _should_be_write(self, words: List[str], index: int) -> bool:
        """right → write when referring to writing"""
        before, after = self._get_context(words, index)
        if before in {'to', 'can', 'could', 'will', 'would', 'should', 'must', "let's", 'gonna', 'going'}:
            if after in {'a', 'the', 'this', 'that', 'it', 'down', 'something', 'anything', 'letter', 'email', 'message', 'note', 'book', 'code'}:
                return True
        return False
    
    def _should_be_whether(self, words: List[str], index: int) -> bool:
        """weather → whether when used as conjunction"""
        before, after = self._get_context(words, index)
        if after in {'or', 'to', 'it', 'he', 'she', 'they', 'we', 'you', 'i', 'the', 'this', 'that'}:
            if before not in {'the', 'bad', 'good', 'nice', 'terrible', 'cold', 'hot', 'rainy', 'sunny', 'forecast'}:
                return True
        return False
    
    def _should_be_then(self, words: List[str], index: int) -> bool:
        """than → then when referring to time sequence"""
        before, after = self._get_context(words, index)
        if after in {'i', 'we', 'you', 'he', 'she', 'they', 'it', 'the', 'a', 'go', 'come', 'what', 'why'}:
            return True
        if before in {'and', 'but', 'if', 'when', 'until', 'before', 'after', 'just', 'right', 'since', 'back'}:
            return True
        return False
    
    def _should_be_than(self, words: List[str], index: int) -> bool:
        """then → than when used for comparison"""
        before, _ = self._get_context(words, index)
        if before in {'more', 'less', 'better', 'worse', 'bigger', 'smaller', 'faster', 'slower', 'older', 'younger', 'higher', 'lower', 'rather', 'other', 'different', 'greater', 'larger', 'stronger', 'weaker', 'easier', 'harder', 'longer', 'shorter', 'earlier', 'later', 'sooner'}:
            return True
        return False
    
    def _should_be_our(self, words: List[str], index: int) -> bool:
        """are/hour → our when possessive"""
        _, after = self._get_context(words, index)
        if after in {'house', 'home', 'car', 'family', 'friends', 'team', 'group', 'company', 'country', 'city', 'town', 'school', 'office', 'work', 'place', 'plan', 'goal', 'future', 'past', 'time', 'way', 'life', 'world', 'children', 'kids', 'parents', 'dog', 'cat', 'pet', 'stuff', 'things', 'new', 'old', 'first', 'last', 'next', 'own'}:
            return True
        return False
    
    def _should_be_see(self, words: List[str], index: int) -> bool:
        """sea → see when referring to vision"""
        before, after = self._get_context(words, index)
        if before in {'can', 'could', 'cannot', "can't", 'did', "didn't", 'do', "don't", 'will', "won't", 'to', 'want', 'let', 'lets', "let's", 'i', 'we', 'you', 'they'}:
            return True
        if after in {'you', 'me', 'him', 'her', 'them', 'us', 'it', 'that', 'this', 'what', 'if', 'how', 'why', 'where', 'when'}:
            return True
        return False
    
    def _should_be_i(self, words: List[str], index: int) -> bool:
        """eye → I when used as pronoun"""
        before, after = self._get_context(words, index)
        if before is None:
            if after in {'am', 'was', 'have', 'had', 'will', 'would', 'could', 'should', 'can', 'cannot', "can't", 'do', "don't", 'did', "didn't", 'want', 'need', 'think', 'know', 'see', 'hear', 'feel', 'like', 'love', 'hate', 'hope', 'wish', 'believe', 'understand', 'remember', 'forget', 'mean', 'guess', 'suppose', 'wonder', 'agree', 'disagree'}:
                return True
        return False
    
    def _should_be_when(self, words: List[str], index: int) -> bool:
        """whims → when (common voice error)"""
        _, after = self._get_context(words, index)
        if after in {'can', 'will', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'should', 'would', 'could', 'i', 'you', 'we', 'they', 'he', 'she', 'it', 'the', 'a'}:
            return True
        return False
    
    def _should_be_britain(self, words: List[str], index: int) -> bool:
        """brighten → Britain when referring to country"""
        _, after = self._get_context(words, index)
        if after in {'is', 'was', 'has', 'had', 'will', 'would', 'could', 'should', 'and', 'or', 'the', 'a'}:
            return True
        return False
    
    def fix(self, text: str) -> str:
        """
        Fix homophones and common voice errors in text
        
        Args:
            text: Input text to fix
            
        Returns:
            Corrected text
        """
        if not self.enabled or not text:
            return text
        
        words = text.split()
        if len(words) < 2:
            return text
        
        result = []
        i = 0
        
        while i < len(words):
            word = words[i]
            word_lower = word.lower()
            word_clean = re.sub(r'[^\w]', '', word_lower)
            
            if word_clean in self.simple_replacements:
                punct = ''
                if word[-1] in '.,!?;:':
                    punct = word[-1]
                replacement = self.simple_replacements[word_clean]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement + punct)
                i += 1
                continue
            
            corrected = False
            for wrong, right, condition_fn in self.rules:
                if word_clean == wrong:
                    try:
                        if condition_fn(words, i):
                            punct = ''
                            if word[-1] in '.,!?;:':
                                punct = word[-1]
                            
                            if word[0].isupper():
                                new_word = right.capitalize() + punct
                            else:
                                new_word = right + punct
                            
                            result.append(new_word)
                            corrected = True
                            break
                    except (IndexError, AttributeError):
                        continue
            
            if not corrected:
                result.append(word)
            
            i += 1
        
        return ' '.join(result)
    
    def get_status(self) -> dict:
        """Get fixer status"""
        return {
            "enabled": self.enabled,
            "rules_count": len(self.rules),
            "simple_replacements_count": len(self.simple_replacements)
        }


def test_homophone_fixer():
    """Test the homophone fixer"""
    import time
    
    print("\n" + "=" * 60)
    print("  Homophone Fixer Test")
    print("  Speed: Instant (no AI)")
    print("=" * 60)
    
    fixer = HomophoneFixer(enabled=True)
    
    test_cases = [
        ("whims can i come over", "when can i come over"),
        ("i want to by a car", "i want to buy a car"),
        ("there going to the store", "they're going to the store"),
        ("your welcome", "you're welcome"),
        ("i no what your saying", "i know what you're saying"),
        ("its going to rain", "it's going to rain"),
        ("eye want to see you", "I want to see you"),
        ("this is to much", "this is too much"),
        ("i wanna go home", "i want to go home"),
        ("brighten is a country", "Britain is a country"),
    ]
    
    print("\nTesting corrections:")
    print("-" * 60)
    
    total_time = 0
    successes = 0
    
    for input_text, expected in test_cases:
        start = time.time()
        result = fixer.fix(input_text)
        elapsed = (time.time() - start) * 1000
        total_time += elapsed
        
        if result.lower() == expected.lower():
            status = "✅"
            successes += 1
        elif expected.split()[0] in result.lower() or expected.split()[-1] in result.lower():
            status = "🟡"
            successes += 0.5
        else:
            status = "❌"
        
        print(f"  {status} '{input_text}'")
        print(f"     → '{result}' ({elapsed:.2f}ms)")
        if result.lower() != expected.lower():
            print(f"     Expected: '{expected}'")
    
    avg_time = total_time / len(test_cases)
    
    print("\n" + "=" * 60)
    print(f"  Results: {successes}/{len(test_cases)} correct")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Speed improvement: ~{4000/max(avg_time, 0.01):.0f}x faster than AI")
    print("=" * 60)


if __name__ == "__main__":
    test_homophone_fixer()
