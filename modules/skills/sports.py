"""
Sports skill for Pascal — provides up-to-date answers for sports queries (F1).
Attempts Ergast API for structured F1 data, falls back to scraping official pages.
Async design using aiohttp.
"""

import aiohttp
import asyncio
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import quote_plus

class SkillResult:
    def __init__(self, success: bool, response: str, source: Optional[str] = None):
        self.success = success
        self.response = response
        self.source = source

class SportsSkill:
    """Simple sports skill focused on Formula 1 queries (pole, sprint, location)."""

    def __init__(self, settings=None):
        # settings can include timeouts, preferred sources, API keys (if any)
        self.settings = settings
        self.session: Optional[aiohttp.ClientSession] = None
        self.ergast_base = "http://ergast.com/api/f1"  # Ergast provides JSON for F1

    async def initialize(self) -> Dict[str, Any]:
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=15)
            self.session = aiohttp.ClientSession(timeout=timeout)
        # Return availability info (mimic other skills)
        return {"sports": {"available": True, "message": "Sports skill initialized"}}

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def _fetch_json(self, url: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    try:
                        return json.loads(text)
                    except Exception:
                        return None
                else:
                    return None
        except Exception:
            return None

    async def _ergast_current_round(self) -> Optional[Dict]:
        """Return current season and next round info from Ergast (summary)"""
        url = f"{self.ergast_base}/current.json"
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data.get("MRData", {})
        except Exception:
            return None

    async def query_f1_pole_and_location(self, timeframe: Dict[str, datetime]) -> SkillResult:
        """
        Attempt to find pole-sitter for the sprint (or qualifying) and event location.
        timeframe: dict with optional 'from' and 'to' datetimes if normalized by analyzer.
        """
        try:
            cur = await self._fetch_json(f"{self.ergast_base}/current.json")
            if cur:
                races = cur.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                candidate = None
                if timeframe and "from" in timeframe and "to" in timeframe:
                    fr = timeframe["from"].date()
                    to = timeframe["to"].date()
                    for r in races:
                        try:
                            rd = datetime.strptime(r.get("date"), "%Y-%m-%d").date()
                        except Exception:
                            continue
                        if fr <= rd <= to:
                            candidate = r
                            break
                else:
                    today = datetime.utcnow().date()
                    for r in races:
                        try:
                            rd = datetime.strptime(r.get("date"), "%Y-%m-%d").date()
                        except Exception:
                            continue
                        if rd >= today:
                            candidate = r
                            break

                if candidate:
                    race_name = candidate.get("raceName")
                    race_round = candidate.get("round")
                    race_season = candidate.get("season")
                    circuit = candidate.get("Circuit", {}).get("circuitName", "unknown circuit")
                    location = candidate.get("Circuit", {}).get("Location", {})
                    locality = location.get("locality", "")
                    country = location.get("country", "")
                    location_str = f"{circuit}, {locality} ({country})" if locality else f"{circuit} ({country})"

                    season = race_season
                    round_num = race_round

                    # Try qualifying (best available for pole information)
                    qual_resp = await self._fetch_json(f"{self.ergast_base}/{season}/{round_num}/qualifying.json")
                    if qual_resp:
                        races_q = qual_resp.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                        if races_q:
                            try:
                                qres = races_q[0].get("QualifyingResults", [])
                                if qres:
                                    pole = qres[0]
                                    driver = pole.get("Driver", {})
                                    driver_name = f"{driver.get('givenName','')} {driver.get('familyName','')}".strip()
                                    response = f"Pole (qualifying) for {race_name} ({location_str}) appears to be {driver_name}."
                                    source = f"https://ergast.com/mrd/{season}/{round_num}/qualifying"
                                    return SkillResult(True, response, source)
                            except Exception:
                                pass

                    # If qualifying not available, attempt race results (may not be sprint)
                    results_resp = await self._fetch_json(f"{self.ergast_base}/{season}/{round_num}/results.json")
                    if results_resp:
                        # Provide location and direction to official F1 site as fallback
                        return SkillResult(False, f"I found {race_name} at {location_str}, but couldn't find a verified sprint-pole in Ergast. Try the official F1 site: https://www.formula1.com/", "https://www.formula1.com/")

                    return SkillResult(False, f"I see {race_name} at {location_str}, but I couldn't find confirmed sprint pole information in the public API. Try the official F1 site: https://www.formula1.com/", "https://www.formula1.com/")
        except Exception:
            pass

        return SkillResult(False, "I couldn't fetch race results from the structured F1 API. I can try a live web search of Formula1.com or ESPN if you want.", None)

    async def execute(self, query: str, entities: Dict[str, Any] = None) -> SkillResult:
        """
        Main entry: query is user text, entities may contain normalized fields from query analyzer:
        - intent (e.g., 'F1')
        - timeframe: {'from': datetime, 'to': datetime}
        """
        if not self.session:
            await self.initialize()

        timeframe = None
        if entities:
            timeframe = entities.get("timeframe") or entities.get("date_range")
        if not timeframe:
            now = datetime.utcnow()
            days_ahead = (5 - now.weekday()) % 7
            sat = (now + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)
            sun = sat + timedelta(days=1)
            timeframe = {"from": sat, "to": sun}

        query_lower = query.lower()
        if "pole" in query_lower or "sprint" in query_lower or "qualif" in query_lower:
            return await self.query_f1_pole_and_location(timeframe)

        return SkillResult(False, "Sports skill can handle F1 pole/qualifying/sprint queries. Try asking specifically 'who is on pole for the F1 sprint at [event]?'", None)
