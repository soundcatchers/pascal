"""
Voice AI Corrector Module

Uses a small, fast LLM (Gemma2:2b) via Ollama to fix context-aware
misrecognitions that spell check can't handle.

Example:
  Input:  "whims can i come over"  (spell check won't fix "whims")
  Output: "when can i come over"   (AI understands context)

Speed: ~300-500ms on Pi 5 with Gemma2:2b
Cost: FREE (100% offline via Ollama)
"""

import asyncio
import aiohttp
import time
from typing import Optional, Dict, Any


class VoiceAICorrector:
    """Context-aware voice correction using local Ollama LLM"""
    
    def __init__(
        self,
        enabled: bool = True,
        ollama_host: str = "http://localhost:11434",
        model: str = "gemma2:2b",
        timeout: float = 5.0
    ):
        """
        Initialize AI corrector
        
        Args:
            enabled: Enable AI correction
            ollama_host: Ollama server URL
            model: Model to use (gemma2:2b recommended for speed)
            timeout: Request timeout in seconds (5s default for Pi 5)
        """
        self.enabled = enabled
        self.ollama_host = ollama_host.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._available: Optional[bool] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._warmed_up = False
    
    async def _get_session(self, override_timeout: float = None) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            # Use 10s timeout for session - individual calls can override
            timeout = aiohttp.ClientTimeout(total=override_timeout or 10.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def check_available(self) -> bool:
        """Check if Ollama and the model are available"""
        if not self.enabled:
            return False
        
        try:
            session = await self._get_session()
            async with session.get(f"{self.ollama_host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m.get('name', '') for m in data.get('models', [])]
                    model_base = self.model.split(':')[0]
                    self._available = any(model_base in m for m in models)
                    return self._available
        except Exception:
            pass
        
        self._available = False
        return False
    
    async def warmup(self) -> bool:
        """Pre-load the model to avoid cold start delays"""
        if self._warmed_up:
            return True
        
        if not self._available:
            await self.check_available()
        
        if not self._available:
            return False
        
        try:
            print(f"[AI] Warming up {self.model} (loading model into memory)...")
            start = time.time()
            
            payload = {
                "model": self.model,
                "prompt": "test",
                "stream": False,
                "options": {
                    "num_predict": 1
                }
            }
            
            # Use longer timeout for warmup (model loading can take 20-30s)
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as warmup_session:
                async with warmup_session.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        elapsed = time.time() - start
                        print(f"[AI] ✅ Model warmed up in {elapsed:.1f}s")
                        self._warmed_up = True
                        return True
                    else:
                        print(f"[AI] ⚠️ Warmup returned status {response.status}")
        except asyncio.TimeoutError:
            print(f"[AI] ⚠️ Warmup timed out (model may still be loading)")
        except Exception as e:
            print(f"[AI] ⚠️ Warmup failed: {e}")
        
        return False
    
    async def correct(self, text: str) -> str:
        """
        Correct misheard words using AI context understanding
        
        Args:
            text: Text from voice recognition (after spell check/punctuation)
            
        Returns:
            Corrected text, or original if correction fails/disabled
        """
        if not self.enabled or not text or len(text.strip()) < 3:
            return text
        
        if self._available is None:
            await self.check_available()
        
        if not self._available:
            return text
        
        try:
            start_time = time.time()
            corrected = await self._call_ollama(text)
            elapsed = time.time() - start_time
            
            if corrected and corrected != text:
                print(f"[AI] Corrected in {elapsed:.2f}s: '{text}' → '{corrected}'")
                return corrected
            
            return text
            
        except asyncio.TimeoutError:
            print("[AI] ⚠️ Correction timed out, using original")
            return text
        except Exception as e:
            print(f"[AI] ⚠️ Correction failed: {e}")
            return text
    
    async def _call_ollama(self, text: str) -> str:
        """Call Ollama API for correction"""
        prompt = self._build_prompt(text)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 50,
                "top_k": 10,
                "top_p": 0.9
            }
        }
        
        session = await self._get_session()
        async with session.post(
            f"{self.ollama_host}/api/generate",
            json=payload
        ) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_response(data.get('response', ''), text)
        
        return text
    
    def _build_prompt(self, text: str) -> str:
        """Build the correction prompt - ultra minimal for speed"""
        return f"""Fix misheard words: {text}
Corrected:"""
    
    def _parse_response(self, response: str, original: str) -> str:
        """Parse and validate the AI response"""
        if not response:
            return original
        
        cleaned = response.strip()
        
        for prefix in ['Output:', 'Corrected:', 'Fixed:']:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        cleaned = cleaned.strip('"\'')
        
        if len(cleaned) < 2 or len(cleaned) > len(original) * 2:
            return original
        
        if not any(c.isalpha() for c in cleaned):
            return original
        
        return cleaned
    
    def correct_sync(self, text: str) -> str:
        """Synchronous wrapper for correct()"""
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.correct(text))
                return future.result(timeout=self.timeout + 1.0)
        except concurrent.futures.TimeoutError:
            print(f"[AI] ⚠️ Sync timeout ({self.timeout}s)")
            return text
        except Exception as e:
            return text
    
    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get corrector status"""
        return {
            "enabled": self.enabled,
            "available": self._available,
            "model": self.model,
            "ollama_host": self.ollama_host,
            "timeout": self.timeout
        }


async def test_ai_corrector():
    """Test the AI corrector"""
    print("\n" + "=" * 60)
    print("  Voice AI Corrector Self-Test")
    print("=" * 60)
    
    corrector = VoiceAICorrector(
        enabled=True,
        model="gemma2:2b",
        timeout=5.0
    )
    
    print("\n[TEST] Checking Ollama availability...")
    available = await corrector.check_available()
    
    if not available:
        print("  ❌ Ollama not available or gemma2:2b not installed")
        print("\n  To install the model:")
        print("    ollama pull gemma2:2b")
        print("\n  Make sure Ollama is running:")
        print("    ollama serve")
        await corrector.close()
        return False
    
    print(f"  ✅ Ollama available with {corrector.model}")
    
    print("\n[TEST] Warming up model (loads into memory)...")
    await corrector.warmup()
    
    print("\n[TEST] Testing context-aware corrections:")
    print("-" * 60)
    
    test_cases = [
        ("whims can i come over", "when can i come over"),
        ("brighten is a good day", "Britain is a good day"),
        ("i want to by a car", "i want to buy a car"),
        ("there going to the store", "they're going to the store"),
    ]
    
    for input_text, expected_hint in test_cases:
        print(f"\n  Input:    '{input_text}'")
        print(f"  Expected: '{expected_hint}' (or similar)")
        
        start = time.time()
        result = await corrector.correct(input_text)
        elapsed = time.time() - start
        
        print(f"  Output:   '{result}' ({elapsed:.2f}s)")
        
        if result != input_text:
            print(f"  ✅ Corrected!")
        else:
            print(f"  ⚪ No change (may be correct or model disagreed)")
    
    await corrector.close()
    
    print("\n" + "=" * 60)
    print("  ✅ AI Corrector Test Complete!")
    print("=" * 60)
    print()
    
    return True


def test_sync():
    """Run test synchronously"""
    asyncio.run(test_ai_corrector())


if __name__ == "__main__":
    test_sync()
