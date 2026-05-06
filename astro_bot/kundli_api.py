"""
kundli_api.py — AstrologyAPI.com client + kundli data parser.

Calls western_horoscope endpoint and extracts the handful of facts
the astrologer persona actually needs:
  - Sun / Moon / Rising sign
  - Which houses are "activated" (planets sitting there)
  - Any hard aspects (squares, oppositions) for the Cosmic Scapegoat technique
  - Any strong placements (conjunctions, trines) for positive predictions

The raw API response is ~200 lines of JSON. We distill it to a short
AstrologerBrief that gets injected into predict/validate prompts as plain
Hinglish-friendly facts — no technical jargon passed to the LLM.
"""

import os
import requests
import json
from typing import Optional
from dataclasses import dataclass, field, asdict


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class KundliInput:
    """
    All the birth-data fields the API needs.
    Fields are None until the user provides them.
    """
    day:   Optional[int]   = None
    month: Optional[int]   = None
    year:  Optional[int]   = None
    hour:  Optional[int]   = None
    minute: Optional[int]  = None
    lat:   Optional[float] = None
    lon:   Optional[float] = None
    tzone: Optional[float] = None
    city:  Optional[str]   = None   # stored for display, converted to lat/lon externally
    house_type: str        = "placidus"
    is_asteroids: str      = "false"

    def is_complete(self) -> bool:
        """Returns True when all required API fields are present."""
        return all([
            self.day, self.month, self.year,
            self.hour is not None, self.minute is not None,
            self.lat is not None, self.lon is not None,
            self.tzone is not None,
        ])

    def missing_fields(self) -> list[str]:
        """Returns list of field names still needed."""
        missing = []
        if not self.day or not self.month or not self.year:
            missing.append("date_of_birth")
        if self.hour is None or self.minute is None:
            missing.append("time_of_birth")
        if self.lat is None or self.lon is None:
            missing.append("place_of_birth")
        if self.tzone is None:
            missing.append("timezone")
        return missing

    def to_api_payload(self) -> dict:
        return {
            "day":          self.day,
            "month":        self.month,
            "year":         self.year,
            "hour":         self.hour,
            "min":          self.minute,
            "lat":          self.lat,
            "lon":          self.lon,
            "tzone":        self.tzone,
            "house_type":   self.house_type,
            "is_asteroids": self.is_asteroids,
        }


@dataclass
class AstrologerBrief:
    """
    Distilled kundli facts for the LLM — plain language, no jargon.
    This is what gets injected into prompts.
    """
    sun_sign:     str = ""
    moon_sign:    str = ""
    rising_sign:  str = ""
    # Key strengths (trines, conjunctions to benefics)
    strengths:    list[str] = field(default_factory=list)
    # Key tensions (squares, oppositions, malefic conjunctions)
    tensions:     list[str] = field(default_factory=list)
    # House activations relevant to topic
    love_house_notes:    str = ""   # 5th, 7th house situation
    career_house_notes:  str = ""   # 10th, 6th house situation
    family_house_notes:  str = ""   # 4th house situation
    # Current transits / dashas if available
    current_influences:  list[str] = field(default_factory=list)
    # Raw summary for LLM injection
    brief_text:          str = ""

    def as_prompt_text(self) -> str:
        """Returns a short, plain-language paragraph for prompt injection."""
        if self.brief_text:
            return self.brief_text
        lines = []
        if self.sun_sign:
            lines.append(f"Sun sign: {self.sun_sign}")
        if self.moon_sign:
            lines.append(f"Moon sign: {self.moon_sign}")
        if self.rising_sign:
            lines.append(f"Rising: {self.rising_sign}")
        if self.strengths:
            lines.append("Strengths: " + "; ".join(self.strengths))
        if self.tensions:
            lines.append("Tensions: " + "; ".join(self.tensions))
        if self.love_house_notes:
            lines.append(f"Love/relationship planets: {self.love_house_notes}")
        if self.career_house_notes:
            lines.append(f"Career planets: {self.career_house_notes}")
        if self.current_influences:
            lines.append("Current influences: " + "; ".join(self.current_influences))
        return "\n".join(lines) if lines else "Kundli data available but no strong patterns detected."


# ──────────────────────────────────────────────────────────────
# API caller
# ──────────────────────────────────────────────────────────────

def call_astrology_api(kundli_input: KundliInput) -> Optional[dict]:
    """
    Calls the AstrologyAPI western_horoscope endpoint.
    Returns the raw JSON dict, or None on failure.
    """
    api_key = os.getenv("ASTROLOGY_API_KEY", "")
    api_url = "https://json.astrologyapi.com/v1/western_horoscope"

    if not api_key:
        raise ValueError("ASTROLOGY_API_KEY env variable is not set.")

    headers = {
        "x-astrologyapi-key": api_key,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=kundli_input.to_api_payload(),
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_detail = ""
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = str(e.response.json())
            except Exception:
                error_detail = e.response.text
        raise RuntimeError(f"Astrology API error: {e}. Detail: {error_detail}")


# ──────────────────────────────────────────────────────────────
# Response parser — converts raw API JSON → AstrologerBrief
# ──────────────────────────────────────────────────────────────

# Planets considered benefic / malefic for Cosmic Scapegoat
MALEFIC_PLANETS = {"Saturn", "Mars", "Rahu", "Ketu", "Pluto", "Uranus"}
BENEFIC_PLANETS = {"Jupiter", "Venus", "Moon", "Sun", "Mercury"}

HARD_ASPECTS = {"square", "opposition", "quincunx"}
SOFT_ASPECTS  = {"trine", "sextile", "conjunction"}   # conjunction can be both, handled by planet type

LOVE_HOUSES    = {5, 7}    # romance, partnership
CAREER_HOUSES  = {6, 10}   # work, profession
FAMILY_HOUSES  = {4}       # home, parents


def _sign_from_degree(degree: float) -> str:
    """Convert 0–360 ecliptic degree to zodiac sign name."""
    signs = [
        "Aries", "Taurus", "Gemini", "Cancer",
        "Leo", "Virgo", "Libra", "Scorpio",
        "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]
    return signs[int(degree / 30) % 12]


def parse_kundli_response(raw: dict) -> AstrologerBrief:
    """
    Parse the western_horoscope JSON into an AstrologerBrief.
    Handles both the standard shape and gracefully degrades if fields are missing.
    """
    brief = AstrologerBrief()

    # ── 1. Core signs ──
    planets = raw.get("planets", {})
    if isinstance(planets, list):
        # API sometimes returns a list of planet objects
        planets = {p.get("name", p.get("planet", "")): p for p in planets}

    def _get_sign(planet_name: str) -> str:
        p = planets.get(planet_name, {})
        if isinstance(p, dict):
            sign = p.get("sign") or p.get("sign_name") or ""
            if not sign and "full_degree" in p:
                sign = _sign_from_degree(float(p["full_degree"]))
            return sign
        return ""

    brief.sun_sign    = _get_sign("Sun")    or raw.get("sun_sign", "")
    brief.moon_sign   = _get_sign("Moon")   or raw.get("moon_sign", "")
    brief.rising_sign = raw.get("ascendant", {}).get("sign", "") or raw.get("rising_sign", "")

    # ── 2. Aspects → strengths and tensions ──
    aspects = raw.get("aspects", [])
    for aspect in aspects:
        p1 = aspect.get("aspecting_planet", aspect.get("planet1", ""))
        p2 = aspect.get("aspected_planet",  aspect.get("planet2", ""))
        aspect_type = aspect.get("type", aspect.get("aspect_type", "")).lower()

        if aspect_type in HARD_ASPECTS:
            # Only flag if at least one planet is malefic
            if p1 in MALEFIC_PLANETS or p2 in MALEFIC_PLANETS:
                brief.tensions.append(f"{p1}–{p2} {aspect_type}")
        elif aspect_type in SOFT_ASPECTS:
            if p1 in BENEFIC_PLANETS or p2 in BENEFIC_PLANETS:
                brief.strengths.append(f"{p1}–{p2} {aspect_type}")

    # ── 3. House activations ──
    houses = raw.get("houses", {})
    if isinstance(houses, list):
        houses = {h.get("house", h.get("number", i+1)): h for i, h in enumerate(houses)}

    house_planets: dict[int, list[str]] = {}
    for p_name, p_data in planets.items():
        if isinstance(p_data, dict):
            house_num = p_data.get("house") or p_data.get("house_number")
            if house_num:
                house_planets.setdefault(int(house_num), []).append(p_name)

    love_notes = []
    career_notes = []
    family_notes = []
    for house_num, planet_list in house_planets.items():
        if house_num in LOVE_HOUSES:
            love_notes.extend(planet_list)
        if house_num in CAREER_HOUSES:
            career_notes.extend(planet_list)
        if house_num in FAMILY_HOUSES:
            family_notes.extend(planet_list)

    brief.love_house_notes   = ", ".join(love_notes)   if love_notes   else ""
    brief.career_house_notes = ", ".join(career_notes) if career_notes else ""
    brief.family_house_notes = ", ".join(family_notes) if family_notes else ""

    # ── 4. Build the brief_text for prompt injection ──
    # Keep this SHORT and in plain language — the LLM will Hinglishify it
    parts = []
    if brief.sun_sign:
        parts.append(f"Sun is in {brief.sun_sign}")
    if brief.moon_sign:
        parts.append(f"Moon is in {brief.moon_sign}")
    if brief.rising_sign:
        parts.append(f"Rising sign is {brief.rising_sign}")
    if brief.tensions:
        # Translate planet names to the desi names the chatbot uses
        tension_str = "; ".join(brief.tensions[:3])  # max 3 tensions
        parts.append(f"Tension aspects: {tension_str} — use these as Cosmic Scapegoat material")
    if brief.strengths:
        strength_str = "; ".join(brief.strengths[:3])
        parts.append(f"Strong aspects: {strength_str} — use these as positive prediction anchors")
    if brief.love_house_notes:
        parts.append(f"Planets in love/relationship houses (5th/7th): {brief.love_house_notes}")
    if brief.career_house_notes:
        parts.append(f"Planets in career houses (6th/10th): {brief.career_house_notes}")

    brief.brief_text = "\n".join(parts)
    return brief


# ──────────────────────────────────────────────────────────────
# Geocoding helper — city name → (lat, lon, tzone)
# Uses a simple hardcoded lookup for major Indian cities first,
# then falls back to the nominatim API (no key required).
# ──────────────────────────────────────────────────────────────

INDIAN_CITY_COORDS: dict[str, tuple[float, float, float]] = {
    "mumbai":     (19.076,  72.8777, 5.5),
    "delhi":      (28.6139, 77.2090, 5.5),
    "new delhi":  (28.6139, 77.2090, 5.5),
    "bangalore":  (12.9716, 77.5946, 5.5),
    "bengaluru":  (12.9716, 77.5946, 5.5),
    "hyderabad":  (17.3850, 78.4867, 5.5),
    "chennai":    (13.0827, 80.2707, 5.5),
    "kolkata":    (22.5726, 88.3639, 5.5),
    "pune":       (18.5204, 73.8567, 5.5),
    "ahmedabad":  (23.0225, 72.5714, 5.5),
    "jaipur":     (26.9124, 75.7873, 5.5),
    "lucknow":    (26.8467, 80.9462, 5.5),
    "surat":      (21.1702, 72.8311, 5.5),
    "indore":     (22.7196, 75.8577, 5.5),
    "bhopal":     (23.2599, 77.4126, 5.5),
    "patna":      (25.5941, 85.1376, 5.5),
    "nagpur":     (21.1458, 79.0882, 5.5),
    "kanpur":     (26.4499, 80.3319, 5.5),
    "chandigarh": (30.7333, 76.7794, 5.5),
    "amritsar":   (31.6340, 74.8723, 5.5),
    "dehradun":   (30.3165, 78.0322, 5.5),
    "varanasi":   (25.3176, 82.9739, 5.5),
    "agra":       (27.1767, 78.0081, 5.5),
}


def city_to_coords(city: str) -> Optional[tuple[float, float, float]]:
    """
    Returns (lat, lon, tzone) for a city name.
    Tries hardcoded lookup first, then Nominatim geocoding.
    Returns None if not found.
    """
    key = city.strip().lower()

    # Fast path: hardcoded Indian cities
    if key in INDIAN_CITY_COORDS:
        return INDIAN_CITY_COORDS[key]

    # Fallback: OpenStreetMap Nominatim (free, no key)
    try:
        geo_url = "https://nominatim.openstreetmap.org/search"
        params = {"q": city, "format": "json", "limit": 1}
        headers = {"User-Agent": "AstroLokalBot/1.0"}
        resp = requests.get(geo_url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        results = resp.json()
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            # Estimate timezone from longitude (rough: each 15° = 1 hour)
            tzone = round(lon / 15 * 2) / 2   # nearest 0.5
            return lat, lon, tzone
    except Exception:
        pass

    return None
