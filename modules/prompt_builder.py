"""
Prompt builder for Pascal multi-turn conversations (compatible with existing modules/memory.py).

This module provides an async build_prompt function that:
- uses the existing MemoryManager.get_context(include_long_term=True) to get a compact context string
- uses MemoryManager.get_conversation_summary() for a short recap
- uses personality_manager.get_system_prompt() (async) to get the system/personality prompt
- assembles a concise prompt suitable for both offline and online LLMs
- trims older context to fit within a provided max_chars budget

Integration notes:
- memory_manager must be an instance of the repo's modules.memory.MemoryManager (or equivalent).
- personality_manager must provide async get_system_prompt().
- After receiving the assistant response, call memory_manager.add_interaction(user_text, assistant_text)
"""

from typing import Optional
from datetime import datetime
import asyncio

DEFAULT_MAX_CHARS = 4000
CONTEXT_MAX_CHARS = 2500
SUMMARY_MAX_CHARS = 800
INSTRUCTION_BLOCK = (
    "INSTRUCTIONS: Use the provided CONTEXT and the conversation SUMMARY for follow-up resolution. "
    "If the user asks a follow-up, resolve pronouns and references using the CONTEXT. "
    "Cite sources in brackets if available (e.g., [source]). If you cannot assert a fact, say "
    "'I don't have a reliable source for that right now.' Keep answers concise."
)
SEPARATOR = "\n---\n"

def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    # Prefer keeping recent content: keep the last max_chars
    return text[-max_chars:]


async def build_prompt(
    session_id: str,
    user_query: str,
    memory_manager,
    personality_manager,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_long_term: bool = True
) -> str:
    """
    Build an LLM prompt for the given session and user query.

    Args:
      session_id: identifier for the conversation session (used by memory_manager).
      user_query: current user utterance (string).
      memory_manager: existing MemoryManager instance (from modules.memory).
      personality_manager: object that provides async get_system_prompt() -> str
      max_chars: approximate max characters for the whole prompt (best-effort).
      include_long_term: whether to include long-term memory summary.

    Returns:
      prompt (str) ready to send to an LLM generate API.
    """

    # 1) System / personality prompt (async)
    system_prompt = ""
    try:
        if hasattr(personality_manager, "get_system_prompt"):
            # assume coroutine
            system_prompt = await personality_manager.get_system_prompt()
        elif hasattr(personality_manager, "get_system_prompt_sync"):
            system_prompt = personality_manager.get_system_prompt_sync()
    except Exception:
        # graceful fallback
        system_prompt = "You are Pascal, a helpful assistant. Be concise and use context when answering follow-ups."

    # 2) Short-term context from memory manager (async)
    context_text = ""
    try:
        # memory_manager.get_context is async in existing code
        if asyncio.iscoroutinefunction(getattr(memory_manager, "get_context", None)):
            context_text = await memory_manager.get_context(include_long_term=include_long_term)
        else:
            # synchronous fallback
            context_text = memory_manager.get_context(include_long_term=include_long_term)
    except Exception:
        context_text = ""

    # 3) Conversation summary (short)
    summary_text = ""
    try:
        if asyncio.iscoroutinefunction(getattr(memory_manager, "get_conversation_summary", None)):
            summary_text = await memory_manager.get_conversation_summary()
        else:
            summary_text = memory_manager.get_conversation_summary()
    except Exception:
        summary_text = ""

    # 4) Compact and trim components
    # Reserve approx proportions of max_chars for parts
    sys_budget = int(max_chars * 0.12)
    ctx_budget = int(max_chars * 0.60)
    summary_budget = int(max_chars * 0.12)
    query_budget = max_chars - (sys_budget + ctx_budget + summary_budget + 200)  # leave room for instructions

    sys_block = _truncate(system_prompt.strip(), max(200, sys_budget))
    ctx_block = _truncate(context_text.strip(), max(200, ctx_budget))
    sum_block = _truncate(summary_text.strip(), max(0, summary_budget))
    user_block = _truncate(user_query.strip(), max(800, query_budget))

    # 5) Assemble prompt
    parts = []
    parts.append("SYSTEM PROMPT:")
    parts.append(sys_block or "You are Pascal, an assistant that answers questions concisely and uses conversation context.")
    parts.append(SEPARATOR)

    if ctx_block:
        parts.append("CONTEXT (recent conversation):")
        parts.append(ctx_block)
        parts.append(SEPARATOR)

    if sum_block:
        parts.append("CONVERSATION SUMMARY:")
        parts.append(sum_block)
        parts.append(SEPARATOR)

    parts.append("INSTRUCTIONS:")
    parts.append(INSTRUCTION_BLOCK)
    parts.append(SEPARATOR)

    parts.append("USER QUERY:")
    parts.append(user_block)
    parts.append(SEPARATOR)

    # Optionally include a short note about how many recent turns are included
    try:
        # if memory_manager exposes short_term_memory length, include it; otherwise skip
        cnt = None
        if hasattr(memory_manager, "short_term_memory"):
            cnt = len(getattr(memory_manager, "short_term_memory", []))
        if cnt is not None:
            parts.append(f"(Context includes {cnt} recent interaction(s))")
            parts.append(SEPARATOR)
    except Exception:
        pass

    prompt = "\n".join(parts)

    # If still too long, aggressively trim the context area
    if len(prompt) > max_chars:
        # reduce context size
        ctx_block = _truncate(ctx_block, int(len(ctx_block) * 0.6))
        # rebuild prompt minimally
        parts = []
        parts.append("SYSTEM PROMPT:")
        parts.append(_truncate(sys_block, sys_budget))
        parts.append(SEPARATOR)
        if ctx_block:
            parts.append("CONTEXT (recent conversation - trimmed):")
            parts.append(ctx_block)
            parts.append(SEPARATOR)
        if sum_block:
            parts.append("CONVERSATION SUMMARY:")
            parts.append(_truncate(sum_block, int(summary_budget*0.6)))
            parts.append(SEPARATOR)
        parts.append("INSTRUCTIONS:")
        parts.append(INSTRUCTION_BLOCK)
        parts.append(SEPARATOR)
        parts.append("USER QUERY:")
        parts.append(_truncate(user_block, query_budget))
        parts.append(SEPARATOR)
        prompt = "\n".join(parts)
        # final char-level trim
        if len(prompt) > max_chars:
            prompt = prompt[-max_chars:]

    return prompt
