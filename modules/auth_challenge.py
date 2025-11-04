"""
Pascal AI Assistant - 3-Stage Authentication Challenge Manager

Handles multi-stage authentication for sensitive operations like:
- Forgetting individual memories
- Complete memory wipes

Prevents accidental deletion from speech recognition errors.
"""

import time
from typing import Dict, Any, Optional
from enum import Enum

class ChallengeType(Enum):
    """Types of authentication challenges"""
    FORGET_INDIVIDUAL = "forget_individual"
    COMPLETE_WIPE = "complete_wipe"

class ChallengeStage(Enum):
    """Stages of authentication"""
    NONE = 0
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3
    COMPLETED = 4

class AuthChallengeManager:
    """Manages 3-stage authentication for sensitive operations"""
    
    def __init__(self):
        self.active_challenge: Optional[ChallengeType] = None
        self.current_stage: ChallengeStage = ChallengeStage.NONE
        self.target_data: Optional[str] = None  # e.g., individual name for forget
        self.challenge_start_time: float = 0
        self.timeout_seconds: int = 120  # 2 minutes timeout
        
        # Different confirmation phrases for each stage to prevent accidental triggers
        self.forget_individual_phrases = {
            ChallengeStage.STAGE_1: "confirm forget",
            ChallengeStage.STAGE_2: "delete confirmed",
            ChallengeStage.STAGE_3: "permanently delete"
        }
        
        self.complete_wipe_phrases = {
            ChallengeStage.STAGE_1: "confirm wipe",
            ChallengeStage.STAGE_2: "wipe everything now",
            ChallengeStage.STAGE_3: "permanently delete all memories"
        }
    
    def start_forget_individual(self, individual_name: str) -> str:
        """Start 3-stage authentication for forgetting an individual"""
        self.active_challenge = ChallengeType.FORGET_INDIVIDUAL
        self.current_stage = ChallengeStage.STAGE_1
        self.target_data = individual_name
        self.challenge_start_time = time.time()
        
        return (
            f"⚠️  WARNING: You're about to forget all information about '{individual_name}'.\n"
            f"This will remove all memories, facts, and preferences related to this person.\n\n"
            f"🔒 Stage 1/3: Are you sure?\n"
            f"Say '{self.forget_individual_phrases[ChallengeStage.STAGE_1]}' to proceed, or 'cancel' to abort."
        )
    
    def start_complete_wipe(self) -> str:
        """Start 3-stage authentication for complete memory wipe"""
        self.active_challenge = ChallengeType.COMPLETE_WIPE
        self.current_stage = ChallengeStage.STAGE_1
        self.target_data = None
        self.challenge_start_time = time.time()
        
        return (
            "🚨 CRITICAL WARNING: COMPLETE MEMORY WIPE\n\n"
            "This will erase:\n"
            "• ALL conversation history\n"
            "• ALL memories of individuals\n"
            "• ALL learned facts and preferences\n"
            "• EVERYTHING Pascal knows about you\n\n"
            "⚠️  THIS IS IRREVERSIBLE!\n\n"
            f"🔒 Stage 1/3: Are you absolutely sure?\n"
            f"Say '{self.complete_wipe_phrases[ChallengeStage.STAGE_1]}' to proceed, or 'cancel' to abort."
        )
    
    def process_response(self, user_input: str) -> Dict[str, Any]:
        """Process user response to authentication challenge"""
        # Check for timeout
        if self._is_timeout():
            return self._handle_timeout()
        
        # Check for cancellation
        if "cancel" in user_input.lower():
            return self._handle_cancel()
        
        user_lower = user_input.lower().strip()
        
        # Get expected phrase for current stage
        if self.active_challenge == ChallengeType.FORGET_INDIVIDUAL:
            expected_phrase = self.forget_individual_phrases.get(self.current_stage)
        elif self.active_challenge == ChallengeType.COMPLETE_WIPE:
            expected_phrase = self.complete_wipe_phrases.get(self.current_stage)
        else:
            return {"status": "error", "message": "No active challenge"}
        
        # Check if user provided correct phrase
        if expected_phrase and expected_phrase in user_lower:
            return self._advance_stage()
        else:
            return {
                "status": "invalid_response",
                "message": f"❌ Incorrect phrase. Please say '{expected_phrase}' to continue, or 'cancel' to abort."
            }
    
    def _advance_stage(self) -> Dict[str, Any]:
        """Advance to next stage of authentication"""
        if self.current_stage == ChallengeStage.STAGE_1:
            self.current_stage = ChallengeStage.STAGE_2
            return self._get_stage_2_prompt()
        
        elif self.current_stage == ChallengeStage.STAGE_2:
            self.current_stage = ChallengeStage.STAGE_3
            return self._get_stage_3_prompt()
        
        elif self.current_stage == ChallengeStage.STAGE_3:
            self.current_stage = ChallengeStage.COMPLETED
            return self._handle_completion()
        
        return {"status": "error", "message": "Invalid stage"}
    
    def _get_stage_2_prompt(self) -> Dict[str, Any]:
        """Get prompt for stage 2"""
        if self.active_challenge == ChallengeType.FORGET_INDIVIDUAL:
            phrase = self.forget_individual_phrases[ChallengeStage.STAGE_2]
            return {
                "status": "stage_2",
                "message": (
                    f"🔒 Stage 2/3: Second confirmation required.\n"
                    f"This will permanently delete all data about '{self.target_data}'.\n\n"
                    f"Say '{phrase}' to proceed, or 'cancel' to abort."
                )
            }
        else:  # COMPLETE_WIPE
            phrase = self.complete_wipe_phrases[ChallengeStage.STAGE_2]
            return {
                "status": "stage_2",
                "message": (
                    "🔒 Stage 2/3: Second confirmation required.\n"
                    "⚠️  THIS WILL ERASE EVERYTHING!\n\n"
                    f"Say '{phrase}' to proceed, or 'cancel' to abort."
                )
            }
    
    def _get_stage_3_prompt(self) -> Dict[str, Any]:
        """Get prompt for stage 3 (final confirmation)"""
        if self.active_challenge == ChallengeType.FORGET_INDIVIDUAL:
            phrase = self.forget_individual_phrases[ChallengeStage.STAGE_3]
            return {
                "status": "stage_3",
                "message": (
                    f"🔒 Stage 3/3: FINAL CONFIRMATION\n"
                    f"Last chance to abort!\n\n"
                    f"Say '{phrase}' to permanently forget '{self.target_data}', or 'cancel' to abort."
                )
            }
        else:  # COMPLETE_WIPE
            phrase = self.complete_wipe_phrases[ChallengeStage.STAGE_3]
            return {
                "status": "stage_3",
                "message": (
                    "🔒 Stage 3/3: FINAL CONFIRMATION\n"
                    "⚠️  POINT OF NO RETURN\n\n"
                    f"Type the exact phrase: '{phrase}'\n"
                    "This will permanently erase ALL memories."
                )
            }
    
    def _handle_completion(self) -> Dict[str, Any]:
        """Handle successful completion of all 3 stages"""
        result = {
            "status": "completed",
            "challenge_type": self.active_challenge.value,
            "target_data": self.target_data
        }
        self.reset()
        return result
    
    def _handle_cancel(self) -> Dict[str, Any]:
        """Handle cancellation"""
        challenge_type = self.active_challenge.value if self.active_challenge else "unknown"
        self.reset()
        return {
            "status": "cancelled",
            "message": f"✅ Operation cancelled. No changes made."
        }
    
    def _handle_timeout(self) -> Dict[str, Any]:
        """Handle timeout"""
        self.reset()
        return {
            "status": "timeout",
            "message": "⏱️  Authentication timeout. Operation cancelled for safety."
        }
    
    def _is_timeout(self) -> bool:
        """Check if challenge has timed out"""
        if self.current_stage == ChallengeStage.NONE:
            return False
        return time.time() - self.challenge_start_time > self.timeout_seconds
    
    def reset(self):
        """Reset authentication state"""
        self.active_challenge = None
        self.current_stage = ChallengeStage.NONE
        self.target_data = None
        self.challenge_start_time = 0
    
    def is_active(self) -> bool:
        """Check if there's an active challenge"""
        return self.current_stage != ChallengeStage.NONE
    
    def get_status(self) -> Dict[str, Any]:
        """Get current authentication status"""
        return {
            "active": self.is_active(),
            "challenge_type": self.active_challenge.value if self.active_challenge else None,
            "stage": self.current_stage.value,
            "target": self.target_data,
            "timeout_remaining": max(0, self.timeout_seconds - (time.time() - self.challenge_start_time)) if self.is_active() else 0
        }
