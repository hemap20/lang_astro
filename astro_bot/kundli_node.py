"""
kundli_node.py — Birth data collection + API call node for AstroLokal.

Design goals:
  1. Collect birth data ONE field at a time — never ask for everything at once
  2. Max 3 probes per field — if user skips or can't answer, move on gracefully
  3. Probe language is casual Hinglish, not a form — "kab paida hue the aap?"
  4. After collection is complete, silently call the API and store the brief
  5. If API fails, set api_failed=True and let predict/validate use fallback mode

Field collection order (designed to feel most natural):
  1. Date of birth  — almost everyone knows this
  2. Birth city     — simpler than lat/lon; we geocode internally
  3. Birth time     — hardest; many people don't know; skip after 3 fails
  4. Timezone       — auto-derived from city; rarely asked explicitly

The node is re-entered on each user turn until collection_complete=True.
After that it short-circuits to wherever the graph was heading.
"""

import re
import json
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import AstroState, KundliData
from .kundli_api import (
    KundliInput, AstrologerBrief,
    call_astrology_api, parse_kundli_response, city_to_coords
)

import os


# ──────────────────────────────────────────────
# LLM helpers (local, no circular import)
# ──────────────────────────────────────────────

KUNDLI_PERSONA = """
You are a friendly Indian astrologer chatbot. You speak in casual, warm Hinglish.
Keep messages SHORT — max 7 words each.
You are collecting the user's birth details to read their kundli.
Be warm, not clinical. Never say "please provide your data."
"""

def _llm_call(prompt: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0.6,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    messages = [
        SystemMessage(content=KUNDLI_PERSONA),
        HumanMessage(content=prompt),
    ]
    return llm.invoke(messages).content.strip()


def _llm_json(prompt: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    messages = [
        SystemMessage(content="You are a JSON-only extractor. Return valid JSON only, no explanation."),
        HumanMessage(content=prompt),
    ]
    raw = llm.invoke(messages).content.strip()
    raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
    return json.loads(raw)


# ──────────────────────────────────────────────
# Casual probe messages — one per field
# Varied so the bot doesn't repeat the same line
# ──────────────────────────────────────────────

DOB_PROBES = [
    "Pehle aapki date of birth bata do — aur year bhi? 😊",
    "Date of birth? Din, mahina, aur saal — teeno bata do.",
    "Ek last baar — janmdin ki date kya hai aapki?",
]
TOB_PROBES = [
    "Birth time pata hai? Approximate bhi chalega.",
    "Kabhi suna hai ghar mein — kaunse waqt paida hue the?",
    "Time bilkul exact nahi chahiye — roughly bata do.",
]
POB_PROBES = [
    "Kaunse city mein paida hue the?",
    "Birth city? Sirf city ka naam kafi hai.",
    "Ek baar aur pooch raha hun — janam kahan hua tha?",
]

DOB_SKIP_MSG = "Koi baat nahi, bina date ke bhi dekh lete hain!"
TOB_SKIP_MSG = "Time nahi pata? No problem — approximate se bhi reading ho jaati hai."
POB_SKIP_MSG = "City nahi pata? Theek hai, general reading karunga."


# ──────────────────────────────────────────────
# Birth data extractor — pulls structured data from user message
# ──────────────────────────────────────────────

def _extract_birth_data(message: str) -> dict:
    """
    Extract any birth-data fields from the user's raw message.
    Returns a dict with only the fields that were found.
    """
    prompt = f"""
Extract birth data from this message. Return JSON with ONLY the fields present:
- day: int (1-31)
- month: int (1-12)
- year: int (4 digit year)
- hour: int (0-23, use 24h format)
- minute: int (0-59)
- city: string (city name exactly as mentioned)
- skip: boolean (true if user says they don't know / can't remember / want to skip)

Message: "{message}"

Rules:
- If month is written as name (e.g. "March"), convert to number.
- If time is like "2 baje" = hour 14 if afternoon context, 2 if night context. Use common sense.
- If "subah" = AM, "shaam/raat" = PM.
- If user says "pata nahi" / "nahi pata" / "skip" / "chhod do" = set skip: true
- Return {{}} if nothing found.
"""
    try:
        return _llm_json(prompt)
    except Exception:
        return {}


# ──────────────────────────────────────────────
# Core kundli collection node
# ──────────────────────────────────────────────

def kundli_collect_node(state: AstroState) -> dict:
    """
    Incrementally collects birth data and calls the API when complete.

    State transitions:
      - While collecting: returns a probe message + updated kundli state
      - When collection_complete: calls API, stores brief, routes to validate
      - On API failure: sets api_failed, routes to validate (graceful fallback)
    """
    kundli = dict(state.get("kundli", {}))
    msg = state["current_user_message"]
    phase = dict(state.get("phase", {}))

    # ── Try to extract birth data from user's current message ──
    extracted = _extract_birth_data(msg)
    user_skipped = extracted.get("skip", False)

    # Merge extracted data into kundli state
    for field in ("day", "month", "year", "hour", "minute", "city"):
        if field in extracted and extracted[field] is not None:
            kundli[field] = extracted[field]

    # If city was provided, geocode it to lat/lon/tzone
    if "city" in extracted and extracted["city"] and not kundli.get("lat"):
        city_name = extracted["city"]
        coords = city_to_coords(city_name)
        if coords:
            kundli["lat"], kundli["lon"], kundli["tzone"] = coords
            kundli["city"] = city_name

    # ── Check what's still missing ──
    has_dob  = all(kundli.get(f) for f in ("day", "month", "year"))
    has_city = kundli.get("lat") is not None   # lat set = city resolved
    has_tob  = kundli.get("hour") is not None and kundli.get("minute") is not None

    dob_skipped = kundli.get("dob_skipped", False)
    tob_skipped = kundli.get("tob_skipped", False)
    pob_skipped = kundli.get("pob_skipped", False)

    # ── Handle user skipping a field ──
    if user_skipped:
        if not has_dob and not dob_skipped:
            kundli["dob_skipped"] = True
            dob_skipped = True
        elif not has_city and not pob_skipped:
            kundli["pob_skipped"] = True
            pob_skipped = True
        elif not has_tob and not tob_skipped:
            kundli["tob_skipped"] = True
            tob_skipped = True

    # ── Determine if collection is done ──
    # Complete if: DOB present OR skipped, AND city present OR skipped
    # TOB is optional (many users don't know it)
    dob_done  = has_dob  or dob_skipped
    city_done = has_city or pob_skipped
    tob_done  = has_tob  or tob_skipped or kundli.get("tob_probes", 0) >= 3

    # Initialize probe counters
    for counter in ("dob_probes", "tob_probes", "pob_probes"):
        kundli.setdefault(counter, 0)

    # ── If collection is complete, call API ──
    if dob_done and city_done and not kundli.get("api_called", False):
        kundli["collection_complete"] = True
        response_text, kundli = _call_api_and_store(kundli)
        phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1
        return {
            "messages": [AIMessage(content=response_text)],
            "kundli": kundli,
            "phase": phase,
            "next_action": "validate",
            "vpr_stage": "V",
        }

    # ── Still collecting — figure out which field to ask for next ──
    probe_msg, kundli = _next_probe(kundli, has_dob, has_city, has_tob,
                                    dob_skipped, pob_skipped, tob_skipped)

    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1

    return {
        "messages": [AIMessage(content=probe_msg)],
        "kundli": kundli,
        "phase": phase,
        "next_action": "kundli_collect",   # stay in collection loop
        "vpr_stage": "V",
    }


def _next_probe(
    kundli: dict,
    has_dob: bool, has_city: bool, has_tob: bool,
    dob_skipped: bool, pob_skipped: bool, tob_skipped: bool,
) -> tuple[str, dict]:
    """
    Decide which field to ask for next and return the probe message.
    Priority: DOB → City → TOB → (done)
    """

    # ── DOB ──
    if not has_dob and not dob_skipped:
        probe_idx = kundli.get("dob_probes", 0)
        if probe_idx >= 3:
            kundli["dob_skipped"] = True
            return DOB_SKIP_MSG, kundli
        msg = DOB_PROBES[probe_idx]
        kundli["dob_probes"] = probe_idx + 1
        return msg, kundli

    # ── City / Place of birth ──
    if not has_city and not pob_skipped:
        probe_idx = kundli.get("pob_probes", 0)
        if probe_idx >= 3:
            kundli["pob_skipped"] = True
            return POB_SKIP_MSG, kundli
        msg = POB_PROBES[probe_idx]
        kundli["pob_probes"] = probe_idx + 1
        return msg, kundli

    # ── Time of birth (optional) ──
    if not has_tob and not tob_skipped:
        probe_idx = kundli.get("tob_probes", 0)
        if probe_idx >= 3:
            kundli["tob_skipped"] = True
            return TOB_SKIP_MSG, kundli
        msg = TOB_PROBES[probe_idx]
        kundli["tob_probes"] = probe_idx + 1
        return msg, kundli

    # Shouldn't reach here, but safe fallback
    return "Theek hai, let's continue!", kundli


def _call_api_and_store(kundli: dict) -> tuple[str, dict]:
    """
    Build KundliInput, call the API, parse the response, store the brief.
    Returns (user-facing message, updated kundli dict).
    """
    kundli["api_called"] = True

    # Build input — use defaults for missing optional fields
    ki = KundliInput(
        day=kundli.get("day"),
        month=kundli.get("month"),
        year=kundli.get("year"),
        hour=kundli.get("hour", 12),      # noon default if time unknown
        minute=kundli.get("minute", 0),
        lat=kundli.get("lat"),
        lon=kundli.get("lon"),
        tzone=kundli.get("tzone", 5.5),   # IST default
    )

    # If DOB was skipped entirely, skip API call too
    if kundli.get("dob_skipped") or not ki.is_complete():
        kundli["api_failed"] = True
        kundli["brief_text"] = ""
        return "Chalo, baat karte hain! Kundli ke bina bhi dekh lete hain 😊", kundli

    try:
        raw_response = call_astrology_api(ki)
        brief = parse_kundli_response(raw_response)

        # Store all brief fields in kundli state
        kundli["sun_sign"]           = brief.sun_sign
        kundli["moon_sign"]          = brief.moon_sign
        kundli["rising_sign"]        = brief.rising_sign
        kundli["strengths"]          = brief.strengths
        kundli["tensions"]           = brief.tensions
        kundli["love_house_notes"]   = brief.love_house_notes
        kundli["career_house_notes"] = brief.career_house_notes
        kundli["family_house_notes"] = brief.family_house_notes
        kundli["brief_text"]         = brief.brief_text
        kundli["api_failed"]         = False

        # Build a warm, casual acknowledgment — NOT a data dump
        sun  = brief.sun_sign  or "interesting"
        moon = brief.moon_sign or "deep"
        user_msg = _llm_call(
            f"User's Sun sign is {sun} and Moon sign is {moon}. "
            f"Write 1 very short Hinglish message (max 12 words) "
            f"acknowledging you've checked their kundli and are about to share insights. "
            f"Warm, curious tone. Don't list the signs — just tease what you're seeing."
        )
        return user_msg, kundli

    except Exception as e:
        kundli["api_failed"] = True
        kundli["brief_text"] = ""
        # Log the error silently, give user a smooth recovery message
        return "Kundli check ho gayi — kuch interesting dikh raha hai! Batata hun...", kundli


# ──────────────────────────────────────────────
# Helper: should we collect kundli data?
# Called from classify_node routing logic
# ──────────────────────────────────────────────

def kundli_needed(state: AstroState) -> bool:
    """
    Returns True if kundli collection should run next.
    Triggers after intake (turn 2+) if collection hasn't happened yet.
    """
    kundli = state.get("kundli", {})
    phase = state.get("phase", {})

    # Already done or skipped
    if kundli.get("collection_complete") or kundli.get("api_called"):
        return False

    # Only start collecting after the first topic is established (turn 2+)
    if phase.get("bot_turn_count", 0) < 1:
        return False

    # Don't interrupt recovery or session_open flows
    next_action = state.get("next_action", "")
    if next_action in ("recover", "session_open", "containment"):
        return False

    return True
