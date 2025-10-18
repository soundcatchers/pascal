"""
Search helpers for Pascal — build precise search queries depending on intent.
Used to avoid poor query rewrites and to prefer targeted site searches for specific domains.
"""

from typing import Dict, Optional
from datetime import datetime

def normalize_timeframe_to_dates(entities: Dict) -> Optional[Dict]:
    """
    Convert entities like {'relative': 'this weekend'} into from/to datetimes.
    Minimal implementation — replace with your query_analyzer normalization if present.
    """
    now = datetime.utcnow()
    if not entities:
        return None
    tf = entities.get("timeframe") or entities.get("date_range") or entities.get("relative_time")
    # Example: if relative_time == 'this weekend', return next saturday/sunday
    if isinstance(tf, str):
        if "weekend" in tf.lower():
            days_ahead = (5 - now.weekday()) % 7
            sat = (now.replace(hour=0, minute=0, second=0, microsecond=0) + \
                   timedelta(days=days_ahead))
            sun = sat + timedelta(days=1)
            return {"from": sat, "to": sun}
    return None

def build_search_query(original_query: str, intent: str = "general", entities: Dict = None) -> str:
    """
    Build a focused search query string for web / Google / site search.
    - intent: 'sports', 'news', 'weather', 'general'
    - entities: may contain series, event, timeframe, location, subject
    """
    q = original_query.strip()
    if entities is None:
        entities = {}
    # If sports intent, include series and event explicitly and prefer terms like 'pole position', 'sprint'
    if intent == "sports":
        parts = []
        series = entities.get("series") or entities.get("sport") or ""
        event = entities.get("event") or ""
        target = entities.get("target") or ""  # e.g., 'pole', 'result'
        timeframe = entities.get("timeframe")
        if series:
            parts.append(series)
        if event:
            parts.append(event)
        if target:
            parts.append(target)
        # fallback to original query if parts empty
        if not parts:
            parts = [q]
        # if timeframe exists, add readable dates
        if timeframe and isinstance(timeframe, dict) and 'from' in timeframe and 'to' in timeframe:
            from_date = timeframe['from'].strftime("%Y-%m-%d")
            to_date = timeframe['to'].strftime("%Y-%m-%d")
            parts.append(f"{from_date}..{to_date}")
        return " ".join([p for p in parts if p])
    # News intent -> include 'news' token
    if intent == "news":
        return f"{q} news"
    # Default: return original query (but trimmed)
    return q
