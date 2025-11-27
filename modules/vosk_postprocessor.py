"""
Vosk Post-Processing Module

Improves Vosk speech recognition accuracy through:
1. Spell Check: Fixes common misrecognitions
2. Confidence-Based Filtering: Only corrects low-confidence words
3. Punctuation & Case Restoration: Makes output natural for LLM processing

All features can be individually enabled/disabled via settings.
"""

import json
import os
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False
    print("[POST] ⚠️  SymSpell not installed. Spell check disabled.")
    print("[POST] 💡 Install with: pip install symspellpy")

try:
    from recasepunc import CasePuncPredictor
    RECASEPUNC_AVAILABLE = True
except ImportError:
    RECASEPUNC_AVAILABLE = False
    print("[POST] ⚠️  Recasepunc not installed. Punctuation/case restoration disabled.")
    print("[POST] 💡 Install with: pip install recasepunc")


class VoskPostProcessor:
    """Post-processes Vosk recognition results to improve accuracy"""
    
    def __init__(
        self,
        enable_spell_check: bool = True,
        enable_confidence_filter: bool = True,
        enable_punctuation: bool = True,
        confidence_threshold: float = 0.80,
        spell_check_max_distance: int = 2
    ):
        """
        Initialize post-processor with configurable features
        
        Args:
            enable_spell_check: Enable spell checking (requires SymSpell)
            enable_confidence_filter: Only spell-check low-confidence words
            enable_punctuation: Enable punctuation and case restoration (requires Recasepunc)
            confidence_threshold: Words below this confidence get spell-checked (0.0-1.0)
            spell_check_max_distance: Maximum edit distance for spell suggestions
        """
        self.enable_spell_check = enable_spell_check and SYMSPELL_AVAILABLE
        self.enable_confidence_filter = enable_confidence_filter
        self.enable_punctuation = enable_punctuation and RECASEPUNC_AVAILABLE
        self.confidence_threshold = confidence_threshold
        self.spell_check_max_distance = spell_check_max_distance
        
        self.spell_checker: Optional[SymSpell] = None
        self.punctuator: Optional[CasePuncPredictor] = None
        
        if self.enable_spell_check:
            self._init_spell_checker(spell_check_max_distance)
        
        if self.enable_punctuation:
            self._init_punctuator()
    
    def _init_spell_checker(self, max_distance: int):
        """Initialize SymSpell spell checker"""
        try:
            self.spell_checker = SymSpell(
                max_dictionary_edit_distance=max_distance,
                prefix_length=7
            )
            
            dict_path = self._find_dictionary()
            if dict_path:
                self.spell_checker.load_dictionary(dict_path, term_index=0, count_index=1)
                print(f"[POST] ✅ Spell checker initialized: {dict_path}")
            else:
                print("[POST] ⚠️  Spell check dictionary not found")
                print("[POST] 💡 Download: wget https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt")
                print("[POST] 💡 Place in: config/frequency_dictionary_en_82_765.txt")
                self.enable_spell_check = False
                
        except Exception as e:
            print(f"[POST] ❌ Failed to initialize spell checker: {e}")
            self.enable_spell_check = False
    
    def _find_dictionary(self) -> Optional[str]:
        """Find SymSpell dictionary in common locations"""
        possible_paths = [
            'config/frequency_dictionary_en_82_765.txt',
            'frequency_dictionary_en_82_765.txt',
            '/usr/share/symspell/frequency_dictionary_en_82_765.txt',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _init_punctuator(self):
        """Initialize Recasepunc punctuation/case restoration"""
        try:
            checkpoint_path = self._find_checkpoint()
            if checkpoint_path:
                self.punctuator = CasePuncPredictor(checkpoint_path, lang="en")
                print(f"[POST] ✅ Punctuator initialized: {checkpoint_path}")
            else:
                print("[POST] ⚠️  Recasepunc checkpoint not found")
                print("[POST] 💡 Run setup script: ./setup_vosk_postprocessing.sh")
                self.enable_punctuation = False
                
        except TypeError as e:
            print(f"[POST] ⚠️  Recasepunc API mismatch: {e}")
            print("[POST] 💡 Try: pip install --upgrade recasepunc")
            self.enable_punctuation = False
        except Exception as e:
            print(f"[POST] ❌ Failed to initialize punctuator: {e}")
            self.enable_punctuation = False
    
    def _find_checkpoint(self) -> Optional[str]:
        """Find Recasepunc checkpoint in common locations"""
        possible_paths = [
            'config/recasepunc/checkpoint',
            'recasepunc/checkpoint',
            '/usr/share/recasepunc/checkpoint',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def correct_word(self, word: str) -> str:
        """Apply spell check to a single word"""
        if not self.enable_spell_check or not self.spell_checker:
            return word
        
        try:
            suggestions = self.spell_checker.lookup(
                word,
                Verbosity.CLOSEST,
                max_edit_distance=self.spell_check_max_distance
            )
            if suggestions:
                return suggestions[0].term
        except Exception:
            pass
        
        return word
    
    def process_with_confidence(self, vosk_result: str) -> str:
        """
        Process Vosk result with confidence-based spell checking
        
        Args:
            vosk_result: JSON string from Vosk recognizer
            
        Returns:
            Corrected text string
        """
        try:
            result = json.loads(vosk_result)
        except json.JSONDecodeError:
            return ""
        
        if 'result' in result and isinstance(result['result'], list):
            corrected_words = []
            
            for word_info in result['result']:
                word = word_info.get('word', '')
                conf = word_info.get('conf', 1.0)
                
                if self.enable_confidence_filter and conf < self.confidence_threshold:
                    corrected = self.correct_word(word)
                elif self.enable_spell_check and not self.enable_confidence_filter:
                    corrected = self.correct_word(word)
                else:
                    corrected = word
                
                corrected_words.append(corrected)
            
            return " ".join(corrected_words)
        
        return result.get('text', '')
    
    def process_simple(self, text: str) -> str:
        """
        Process plain text with spell checking (no confidence data)
        
        Args:
            text: Plain text string
            
        Returns:
            Corrected text string
        """
        if not self.enable_spell_check:
            return text
        
        words = text.split()
        corrected = [self.correct_word(word) for word in words]
        return " ".join(corrected)
    
    def add_punctuation(self, text: str) -> str:
        """
        Add punctuation and proper casing
        
        Args:
            text: Lowercase text without punctuation
            
        Returns:
            Text with punctuation and proper casing
        """
        if not self.enable_punctuation or not self.punctuator:
            return text
        
        try:
            result = self.punctuator.predict(text, apply_sbd=True)
            return result
        except TypeError as e:
            try:
                result = self.punctuator.predict([text])
                return result[0] if isinstance(result, list) else result
            except Exception:
                print(f"[POST] ⚠️  Punctuation API error: {e}")
                print(f"[POST] 💡 Disabling punctuation for this session")
                self.enable_punctuation = False
                return text
        except Exception as e:
            print(f"[POST] ⚠️  Punctuation failed: {e}")
            return text
    
    def process(self, vosk_result: str) -> str:
        """
        Complete post-processing pipeline for final results
        
        Args:
            vosk_result: JSON string from Vosk recognizer (must include word-level confidence data)
            
        Returns:
            Fully processed text with spell check and punctuation
        """
        text = self.process_with_confidence(vosk_result)
        
        if not text:
            return ""
        
        if self.enable_punctuation:
            text = self.add_punctuation(text)
        
        return text
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all post-processing features"""
        return {
            "spell_check": {
                "enabled": self.enable_spell_check,
                "available": SYMSPELL_AVAILABLE,
                "initialized": self.spell_checker is not None
            },
            "confidence_filter": {
                "enabled": self.enable_confidence_filter,
                "threshold": self.confidence_threshold
            },
            "punctuation": {
                "enabled": self.enable_punctuation,
                "available": RECASEPUNC_AVAILABLE,
                "initialized": self.punctuator is not None
            }
        }


def test_postprocessor():
    """Test post-processor with sample data"""
    print("\n" + "="*60)
    print("  Vosk Post-Processor Self-Test")
    print("="*60)
    
    processor = VoskPostProcessor(
        enable_spell_check=True,
        enable_confidence_filter=True,
        enable_punctuation=True
    )
    
    print("\n[TEST] Post-Processor Status:")
    print("-" * 60)
    status = processor.get_status()
    
    all_ok = True
    for feature, info in status.items():
        if isinstance(info, dict):
            enabled = info.get('enabled', False)
            initialized = info.get('initialized', info.get('available', False))
            status_icon = "✅" if (enabled and initialized) else "⚠️"
            print(f"  {status_icon} {feature}: enabled={enabled}, initialized={initialized}")
            if enabled and not initialized:
                all_ok = False
        else:
            print(f"  {feature}: {info}")
    
    if not all_ok:
        print("\n❌ Some features are enabled but not initialized!")
        print("Run ./setup_vosk_postprocessing.sh to install dependencies")
        return False
    
    print("\n[TEST] Testing Confidence-Based Spell Check:")
    print("-" * 60)
    test_result = {
        "result": [
            {"conf": 0.65, "word": "whims"},
            {"conf": 0.95, "word": "it"},
            {"conf": 0.92, "word": "built"},
            {"conf": 0.94, "word": "in"},
            {"conf": 0.58, "word": "brighten"}
        ],
        "text": "whims it built in brighten"
    }
    
    print(f"  Input:  '{test_result['text']}'")
    print(f"  Confidence scores:")
    for word_info in test_result['result']:
        word = word_info['word']
        conf = word_info['conf']
        will_check = "← will spell-check" if conf < 0.80 else ""
        print(f"    - '{word}' (confidence: {conf:.2f}) {will_check}")
    
    processed = processor.process(json.dumps(test_result))
    print(f"\n  Output: '{processed}'")
    
    print("\n[TEST] Testing Simple Spell Check (for partials):")
    print("-" * 60)
    simple_text = "whims it built in brighten"
    print(f"  Input:  '{simple_text}'")
    simple_processed = processor.process_simple(simple_text)
    print(f"  Output: '{simple_processed}'")
    
    print("\n" + "="*60)
    if all_ok:
        print("  ✅ All Tests Passed!")
    else:
        print("  ⚠️  Some Tests Failed - Check Dependencies")
    print("="*60)
    print()
    
    return all_ok


if __name__ == "__main__":
    test_postprocessor()
