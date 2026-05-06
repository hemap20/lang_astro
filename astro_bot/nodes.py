"""
nodes.py — All LangGraph node functions for the AstroLokal chatbot.

Each node receives the full AstroState, does its work, and returns a partial
state dict with only the fields it modifies. LangGraph merges these automatically.

v1 Node map:
  intake_node         — warm greeting + binary data collection
  classify_node       — detect topic, emotion, engagement signals
  validate_node       — emotional mirror + cosmic scapegoat (V in VPR)
  predict_node        — personalized, info-drip prediction (P in VPR)
  remedy_node         — mechanism-explained remedy (R in VPR)
  hook_node           — return-visit hook (end of session)
  recover_node        — re-engagement when user is going passive
  session_open_node   — return session opener with progress narrative
  vague_input_node    — handles "mere baare mein batao" type messages
  profile_update_node — silent extractor, runs on every user turn

v2 additions:
  containment_node    — empathetic containment for grief/shock/crisis moments
  barnum_node         — builds early rapport via universal-personal observations
  intent_feeling_node — separates attraction from readiness (love topics)
  pattern_node        — connects cross-turn themes into a visible "cycle"
"""

import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from .state import (
    AstroState, UserProfile, EngagementTracker,
    ConversationPhase, RemedyState,
    SHORT_REPLY_TOKENS, REMEDY_UNLOCK_TURN, HIGH_STRESS_EMOTIONS
)
from .prompts import (
    SYSTEM_PROMPT, INTAKE_PROMPT, CLASSIFY_INTENT_PROMPT,
    VALIDATE_PROMPT, PREDICT_PROMPT, REMEDY_PROMPT,
    HOOK_PROMPT, RECOVERY_PROMPT, SESSION_OPEN_PROMPT,
    VAGUE_INPUT_RECOVERY_PROMPT,
    # v2 prompts
    CONTAINMENT_PROMPT, BARNUM_NODE_PROMPT,
    INTENT_VS_FEELING_PROMPT, PATTERN_REINFORCE_PROMPT,
)
# v3: kundli collection helper
from .kundli_node import kundli_needed


# ──────────────────────────────────────────────
# Shared LLM instance
# ──────────────────────────────────────────────

def get_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """Return a Gemini LLM instance. Model is configurable via env."""
    import os
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def call_llm(prompt: str, system: str = SYSTEM_PROMPT, temperature: float = 0.7) -> str:
    """Thin wrapper: build messages, call Gemini, return text."""
    llm = get_llm(temperature)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def call_llm_json(prompt: str, temperature: float = 0.1) -> dict:
    """Call LLM and parse JSON response. Used for classify_node."""
    raw = call_llm(
        prompt,
        system="You are a JSON-only classifier. Return valid JSON, nothing else.",
        temperature=temperature
    )
    raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
    return json.loads(raw)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _profile_str(profile: UserProfile) -> str:
    parts = []
    if profile.get("name"):
        parts.append(f"User name: {profile['name']}")
    if profile.get("partner_name"):
        parts.append(f"Partner name: {profile['partner_name']}")
    if profile.get("dob"):
        parts.append(f"User DOB: {profile['dob']}")
    if profile.get("partner_dob"):
        parts.append(f"Partner DOB: {profile['partner_dob']}")
    if profile.get("relationship_duration"):
        parts.append(f"Relationship duration: {profile['relationship_duration']}")
    if profile.get("relationship_status"):
        parts.append(f"Contact status: {profile['relationship_status']}")
    if profile.get("incidents_mentioned"):
        parts.append(f"Incidents: {', '.join(profile['incidents_mentioned'][-3:])}")
    if profile.get("personal_items_mentioned"):
        parts.append(f"Items mentioned: {', '.join(profile['personal_items_mentioned'])}")
    if profile.get("themes_mentioned"):
        parts.append(f"Recurring themes: {', '.join(profile['themes_mentioned'])}")
    return "\n".join(parts) if parts else "No profile collected yet."


def _is_short_reply(message: str) -> bool:
    tokens = set(message.lower().strip().split())
    return tokens.issubset(SHORT_REPLY_TOKENS) or len(message.strip()) <= 4


def _kundli_brief(state: AstroState) -> str:
    """Return the kundli brief text for prompt injection, or empty string if not ready."""
    kundli = state.get("kundli", {})
    return kundli.get("brief_text", "") or ""


def _extract_themes(message: str) -> list[str]:
    """Lightweight keyword-based theme extractor for pattern reinforcement."""
    theme_map = {
        "stress": ["stress", "stressed", "tension", "takleef", "problem"],
        "confusion": ["confused", "confuse", "samajh", "pata nahi", "unclear"],
        "loneliness": ["akela", "alone", "lonely", "miss", "yaad"],
        "anger": ["gussa", "angry", "frustrat", "irritat", "naraaz"],
        "hope": ["hope", "asha", "umeed", "lagta hai", "shayad"],
        "fear": ["dar", "afraid", "scared", "darta", "worried"],
        "exhaustion": ["thak", "tired", "haara", "bahut ho gaya", "bore"],
    }
    msg_lower = message.lower()
    found = []
    for theme, keywords in theme_map.items():
        if any(kw in msg_lower for kw in keywords):
            found.append(theme)
    return found


# ──────────────────────────────────────────────
# NODE: profile_update_node (silent — no message generated)
# ──────────────────────────────────────────────

def profile_update_node(state: AstroState) -> dict:
    """
    Extracts and stores user details (names, dates, incidents, items, themes).
    Runs as a passthrough — doesn't generate any bot message.
    """
    msg = state["current_user_message"]
    profile = dict(state.get("user_profile", {}))

    extraction_prompt = f"""
Extract any personal details from this message. Return JSON with ONLY the fields found:
- partner_name: string (name of person being asked about)
- name: string (user's own name if mentioned)
- dob: string (date of birth in any format)
- relationship_duration: string (how long they've been together)
- relationship_status: "in_contact" | "no_contact" | "blocked" | "unknown"
- personal_items: list of objects/items mentioned (gifts, clothes, etc.)
- incidents: list of events/incidents described

Message: "{msg}"

Return ONLY valid JSON. If nothing found, return {{}}
"""
    try:
        extracted = call_llm_json(extraction_prompt)
        for key, value in extracted.items():
            if value and key not in ("personal_items", "incidents"):
                profile[key] = value
        if extracted.get("personal_items"):
            existing = profile.get("personal_items_mentioned", [])
            profile["personal_items_mentioned"] = list(set(existing + extracted["personal_items"]))
        if extracted.get("incidents"):
            existing = profile.get("incidents_mentioned", [])
            profile["incidents_mentioned"] = (existing + extracted["incidents"])[-10:]
        # Store verbatim words for echoing
        words = profile.get("user_exact_words", [])
        words.append(msg[:100])
        profile["user_exact_words"] = words[-10:]
    except Exception:
        pass

    # Theme extraction (lightweight, no LLM call)
    new_themes = _extract_themes(msg)
    existing_themes = profile.get("themes_mentioned", [])
    merged = list(dict.fromkeys(existing_themes + new_themes))  # deduplicated, order preserved
    profile["themes_mentioned"] = merged[-8:]

    return {"user_profile": profile}


# ──────────────────────────────────────────────
# NODE: intake_node
# ──────────────────────────────────────────────

def intake_node(state: AstroState) -> dict:
    """Warm greeting + single binary question. No data dump."""
    profile = state.get("user_profile", {})
    msg = state["current_user_message"]

    prompt = INTAKE_PROMPT.format(
        user_profile=_profile_str(profile),
        current_message=msg
    )
    response = call_llm(prompt)

    phase = dict(state.get("phase", {}))
    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1
    phase["current_phase"] = "intake"

    engagement = dict(state.get("engagement", {}))
    engagement["consecutive_short_replies"] = 0

    return {
        "messages": [AIMessage(content=response)],
        "phase": phase,
        "engagement": engagement,
        "vpr_stage": "V",
        "next_action": "classify",
    }


# ──────────────────────────────────────────────
# NODE: classify_node
# ──────────────────────────────────────────────

def classify_node(state: AstroState) -> dict:
    """
    Classifies message, updates engagement tracker, determines routing.
    v2: also sets is_high_stress for containment routing.
    """
    msg = state["current_user_message"]
    last_topic = state.get("detected_topic", "general")
    engagement = dict(state.get("engagement", {}))
    phase = dict(state.get("phase", {}))

    prompt = CLASSIFY_INTENT_PROMPT.format(
        current_message=msg,
        last_topic=last_topic,
    )
    try:
        result = call_llm_json(prompt)
    except Exception:
        result = {
            "topic": last_topic,
            "emotion": "neutral",
            "is_vague": len(msg.strip()) < 10,
            "intent": "unknown",
            "is_short_reply": _is_short_reply(msg),
            "repeated_question": False,
            "last_question_topic": msg[:50],
            "is_high_stress": False,
        }

    # ── Engagement updates ──
    if result.get("is_short_reply", False) or _is_short_reply(msg):
        engagement["consecutive_short_replies"] = engagement.get("consecutive_short_replies", 0) + 1
    else:
        engagement["consecutive_short_replies"] = 0
        engagement["positive_signals_count"] = engagement.get("positive_signals_count", 0) + 1

    if result.get("repeated_question", False):
        engagement["repeated_question_count"] = engagement.get("repeated_question_count", 0) + 1
        engagement["last_question_topic"] = result.get("last_question_topic", "")
    else:
        engagement["repeated_question_count"] = 0

    # ── Routing decision ──
    short_count = engagement.get("consecutive_short_replies", 0)
    repeat_count = engagement.get("repeated_question_count", 0)
    detected_emotion = result.get("emotion", "neutral")
    is_high_stress = result.get("is_high_stress", False) or detected_emotion in HIGH_STRESS_EMOTIONS

    # Priority order: containment > recovery > session_open > kundli_collect > vague > barnum > validate
    if is_high_stress and not phase.get("high_stress_contained", False):
        next_action = "containment"
        engagement["engagement_level"] = "high"  # they're engaged, just stressed
    elif short_count >= 3 or repeat_count >= 2:
        engagement["engagement_level"] = "low"
        next_action = "recover"
    elif engagement.get("is_return_session", False) and phase.get("bot_turn_count", 0) == 0:
        next_action = "session_open"
        engagement["engagement_level"] = "high"
    elif kundli_needed(state):
        # v3: collect birth data before first real prediction
        next_action = "kundli_collect"
        engagement["engagement_level"] = "high"
    elif result.get("is_vague", False) and short_count < 3:
        next_action = "vague_input"
        engagement["engagement_level"] = "medium"
    else:
        engagement["engagement_level"] = "high" if short_count == 0 else "medium"
        # v2: route through barnum early in the conversation (turns 3-8)
        turn_count = phase.get("bot_turn_count", 0)
        barnum_used = phase.get("barnum_used", [])
        if 3 <= turn_count <= 8 and len(barnum_used) == 0:
            next_action = "barnum"
        else:
            next_action = "validate"

    # ── Phase advancement ──
    turn_count = phase.get("bot_turn_count", 0)
    if turn_count < 10:
        phase["current_phase"] = "intake"
    elif turn_count < 30:
        phase["current_phase"] = "build"
    elif turn_count < 60:
        phase["current_phase"] = "peak"
    else:
        phase["current_phase"] = "sustain"

    phase["remedy_turn_unlocked"] = turn_count >= REMEDY_UNLOCK_TURN

    return {
        "detected_topic": result.get("topic", last_topic),
        "detected_emotion": detected_emotion,
        "is_high_stress": is_high_stress,
        "is_vague_input": result.get("is_vague", False),
        "user_question_intent": result.get("intent", "unknown"),
        "engagement": engagement,
        "phase": phase,
        "next_action": next_action,
    }


# ──────────────────────────────────────────────
# NODE: containment_node (v2)
# ──────────────────────────────────────────────

def containment_node(state: AstroState) -> dict:
    """
    Empathetic containment for grief/shock/crisis moments.
    Validates pain BEFORE any prediction or direction.
    Runs once per session (high_stress_contained flag prevents re-triggering).
    """
    profile = state.get("user_profile", {})
    msg = state["current_user_message"]
    topic = state.get("detected_topic", "general")

    # Infer the stress event type from topic + message
    stress_event_map = {
        "career": "job loss or career crisis",
        "family": "family conflict or separation",
        "health": "health scare",
        "financial": "financial crisis",
        "love": "relationship breakdown or rejection",
        "general": "life crisis",
    }
    stress_event = stress_event_map.get(topic, "difficult life event")

    prompt = CONTAINMENT_PROMPT.format(
        topic=topic,
        current_message=msg,
        user_name=profile.get("name", "aap"),
        stress_event=stress_event,
    )
    response = call_llm(prompt)

    phase = dict(state.get("phase", {}))
    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1
    phase["high_stress_contained"] = True   # don't trigger containment again this session

    return {
        "messages": [AIMessage(content=response)],
        "phase": phase,
        "vpr_stage": "V",      # after containment, next turn follows normal VPR
        "next_action": "classify",
    }


# ──────────────────────────────────────────────
# NODE: barnum_node (v2)
# ──────────────────────────────────────────────

def barnum_node(state: AstroState) -> dict:
    """
    Delivers a Barnum statement + flip question to build early rapport.
    Runs once (turns 3-8) — after first use, routes directly to validate.
    """
    profile = state.get("user_profile", {})
    phase = dict(state.get("phase", {}))
    barnum_used = phase.get("barnum_used", [])

    prompt = BARNUM_NODE_PROMPT.format(
        user_profile=_profile_str(profile),
        topic=state.get("detected_topic", "general"),
        barnum_used=barnum_used,
    )
    response = call_llm(prompt)

    # Track which barnum was used (store first 60 chars as key)
    barnum_used.append(response[:60])
    phase["barnum_used"] = barnum_used
    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1

    return {
        "messages": [AIMessage(content=response)],
        "phase": phase,
        "vpr_stage": "V",
        "next_action": "validate",   # barnum → validate → predict (normal flow resumes)
    }


# ──────────────────────────────────────────────
# NODE: validate_node (V in VPR) — updated v2
# ──────────────────────────────────────────────

def validate_node(state: AstroState) -> dict:
    """
    Emotional mirror + cosmic scapegoat.
    v2: also includes Barnum observation + flip question, and intent-vs-feeling check.
    """
    profile = state.get("user_profile", {})
    phase = dict(state.get("phase", {}))
    msg = state["current_user_message"]
    topic = state.get("detected_topic", "general")

    prompt = VALIDATE_PROMPT.format(
        topic=topic,
        emotion=state.get("detected_emotion", "neutral"),
        current_message=msg,
        partner_name=profile.get("partner_name", "woh insaan"),
        user_words=profile.get("user_exact_words", [msg])[-3:],
        barnum_used=phase.get("barnum_used", []),
        partner_attraction_noted=profile.get("partner_attraction_noted", False),
        kundli_brief=_kundli_brief(state),
    )
    response = call_llm(prompt)

    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1

    # v2: for love topics, route through intent_feeling node before predict
    if topic == "love" and not profile.get("partner_attraction_noted", False):
        next_action = "intent_feeling"
        vpr_stage = "P_prep"
    else:
        next_action = "predict"
        vpr_stage = "P"

    return {
        "messages": [AIMessage(content=response)],
        "phase": phase,
        "vpr_stage": vpr_stage,
        "next_action": next_action,
    }


# ──────────────────────────────────────────────
# NODE: intent_feeling_node (v2)
# ──────────────────────────────────────────────

def intent_feeling_node(state: AstroState) -> dict:
    """
    Separates partner's attraction/feeling from readiness/commitment.
    Runs once per love conversation — after first use, validate goes straight to predict.
    """
    profile = dict(state.get("user_profile", {}))
    msg = state["current_user_message"]

    # Infer the user's core concern
    user_concern_map = {
        "in_contact": "whether partner will commit",
        "no_contact": "whether partner will come back",
        "blocked": "whether partner still has feelings",
        "unknown": "what partner truly feels",
    }
    relationship_status = profile.get("relationship_status", "unknown")
    user_concern = user_concern_map.get(relationship_status, "what partner truly feels")

    prompt = INTENT_VS_FEELING_PROMPT.format(
        partner_name=profile.get("partner_name", "woh insaan"),
        current_message=msg,
        user_concern=user_concern,
        relationship_status=relationship_status,
    )
    response = call_llm(prompt)

    # Mark that attraction has been noted so this node doesn't re-trigger
    profile["partner_attraction_noted"] = True
    phase = dict(state.get("phase", {}))
    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1

    return {
        "messages": [AIMessage(content=response)],
        "user_profile": profile,
        "phase": phase,
        "vpr_stage": "P",
        "next_action": "predict",
    }


# ──────────────────────────────────────────────
# NODE: predict_node (P in VPR) — updated v2
# ──────────────────────────────────────────────

def predict_node(state: AstroState) -> dict:
    """
    Personalized prediction with info-drip.
    v2: adds delay-vs-denial framing, urgency hint, observation benchmark,
        balanced outlook, diagnostic follow-up, and pattern reinforcement trigger.
    """
    profile = state.get("user_profile", {})
    phase = dict(state.get("phase", {}))
    pending_drip = phase.get("pending_drip")
    is_followup = pending_drip is not None
    pattern_threads = profile.get("themes_mentioned", [])

    kundli = state.get("kundli", {})
    prompt = PREDICT_PROMPT.format(
        user_profile=_profile_str(profile),
        topic=state.get("detected_topic", "general"),
        emotion=state.get("detected_emotion", "neutral"),
        phase=phase.get("current_phase", "intake"),
        partner_name=profile.get("partner_name", "woh insaan"),
        pending_drip=pending_drip or "None — first prediction chunk",
        is_followup=is_followup,
        urgency_planted=phase.get("urgency_planted", False),
        pattern_threads=pattern_threads,
        last_diagnostic_question=phase.get("last_diagnostic_question", "none"),
        kundli_brief=_kundli_brief(state),
        sun_sign=kundli.get("sun_sign", ""),
        tension_planet=kundli.get("tensions", ["Shani"])[0].split("–")[0] if kundli.get("tensions") else "Shani",
    )
    response = call_llm(prompt)

    phase["pending_drip"] = response
    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1

    # Track if urgency was delivered (prevent repeat)
    # Heuristic: if "critical" or "important period" is in response, mark as planted
    if any(kw in response.lower() for kw in ["critical", "important period", "sensitive", "decisive"]):
        phase["urgency_planted"] = True

    # Track the diagnostic question type used (rotate next time)
    diag_types = ["focus_vs_confidence", "desire_vs_fear", "action_vs_waiting", "feeling_vs_situation"]
    last_used = phase.get("last_diagnostic_question", "none")
    try:
        next_diag = diag_types[(diag_types.index(last_used) + 1) % len(diag_types)]
    except ValueError:
        next_diag = diag_types[0]
    phase["last_diagnostic_question"] = next_diag

    # v2: if 2+ themes exist and pattern node hasn't run yet this cycle, route there
    if len(pattern_threads) >= 2 and not phase.get("pattern_threads"):
        phase["pattern_threads"] = pattern_threads
        next_action = "pattern"
        vpr_stage = "P_pattern"
    else:
        # Decide remedy vs hook
        remedy_unlocked = phase.get("remedy_turn_unlocked", False)
        already_given = len(state.get("remedy_state", {}).get("remedies_given", []))
        if remedy_unlocked and already_given < 3:
            next_action = "remedy"
            vpr_stage = "R"
        else:
            next_action = "hook"
            vpr_stage = "done"

    return {
        "messages": [AIMessage(content=response)],
        "phase": phase,
        "vpr_stage": vpr_stage,
        "next_action": next_action,
    }


# ──────────────────────────────────────────────
# NODE: pattern_node (v2)
# ──────────────────────────────────────────────

def pattern_node(state: AstroState) -> dict:
    """
    Connects cross-turn themes into a meaningful "cycle" narrative.
    Runs once when 2+ emotional themes have been collected.
    """
    profile = state.get("user_profile", {})
    phase = dict(state.get("phase", {}))
    pattern_threads = profile.get("themes_mentioned", [])

    prompt = PATTERN_REINFORCE_PROMPT.format(
        pattern_threads=pattern_threads,
        partner_name=profile.get("partner_name", "woh insaan"),
        topic=state.get("detected_topic", "general"),
    )
    response = call_llm(prompt)

    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1
    # Mark pattern as shown so it doesn't re-run until new themes appear
    phase["pattern_threads"] = pattern_threads[:]

    remedy_unlocked = phase.get("remedy_turn_unlocked", False)
    already_given = len(state.get("remedy_state", {}).get("remedies_given", []))
    next_action = "remedy" if (remedy_unlocked and already_given < 3) else "hook"
    vpr_stage = "R" if next_action == "remedy" else "done"

    return {
        "messages": [AIMessage(content=response)],
        "phase": phase,
        "vpr_stage": vpr_stage,
        "next_action": next_action,
    }


# ──────────────────────────────────────────────
# NODE: remedy_node (R in VPR) — updated v2 (supportive agency)
# ──────────────────────────────────────────────

def remedy_node(state: AstroState) -> dict:
    """
    ONE remedy per VPR cycle with mechanism explained.
    v2: adds supportive-agency framing so the remedy empowers the user, not controls others.
    """
    profile = state.get("user_profile", {})
    remedy_state = dict(state.get("remedy_state", {}))

    planet_map = {
        "love": "Shukra (Venus)",
        "career": "Brihaspati (Jupiter)",
        "financial": "Kuber / Shani",
        "family": "Chandra (Moon)",
        "health": "Surya (Sun)",
        "general": "Shani",
    }
    topic = state.get("detected_topic", "general")
    relevant_planet = planet_map.get(topic, "Shani")

    prompt = REMEDY_PROMPT.format(
        user_profile=_profile_str(profile),
        topic=topic,
        relevant_planet=relevant_planet,
        remedies_given=remedy_state.get("remedies_given", []),
    )
    response = call_llm(prompt)

    remedies_given = remedy_state.get("remedies_given", [])
    remedies_given.append(response[:80])
    remedy_state["remedies_given"] = remedies_given

    phase = dict(state.get("phase", {}))
    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1
    phase["last_vpr_cycle"] = phase.get("bot_turn_count", 0)
    phase["pending_drip"] = None

    return {
        "messages": [AIMessage(content=response)],
        "remedy_state": remedy_state,
        "phase": phase,
        "vpr_stage": "done",
        "next_action": "hook",
    }


# ──────────────────────────────────────────────
# NODE: hook_node — updated v2 (urgency variant)
# ──────────────────────────────────────────────

def hook_node(state: AstroState) -> dict:
    """
    Plants return-visit seed. Never closes the loop.
    v2: includes critical-window urgency hook variant.
    """
    profile = state.get("user_profile", {})
    phase = dict(state.get("phase", {}))
    hooks_planted = phase.get("hooks_planted", [])

    prompt = HOOK_PROMPT.format(
        hooks_planted=hooks_planted,
        phase=phase.get("current_phase", "build"),
        topic=state.get("detected_topic", "general"),
        partner_name=profile.get("partner_name", "woh insaan"),
        urgency_planted=phase.get("urgency_planted", False),
    )
    response = call_llm(prompt)

    hooks_planted.append(response[:60])
    phase["hooks_planted"] = hooks_planted
    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1

    # Track if urgency was embedded in the hook
    if any(kw in response.lower() for kw in ["critical", "galat move", "careful", "sensitive"]):
        phase["urgency_planted"] = True

    return {
        "messages": [AIMessage(content=response)],
        "phase": phase,
        "vpr_stage": "V",
        "next_action": "classify",
    }


# ──────────────────────────────────────────────
# NODE: recover_node — updated v2 (pattern recovery)
# ──────────────────────────────────────────────

def recover_node(state: AstroState) -> dict:
    """
    Re-engagement. v2: adds pattern-reinforcement recovery tactic.
    """
    profile = state.get("user_profile", {})
    engagement = state.get("engagement", {})
    phase = dict(state.get("phase", {}))

    short_count = engagement.get("consecutive_short_replies", 0)
    repeat_count = engagement.get("repeated_question_count", 0)

    if repeat_count >= 2:
        trigger_type = "repeated_question"
    elif short_count >= 3:
        trigger_type = "short_replies"
    else:
        trigger_type = "stall"

    pattern_threads = profile.get("themes_mentioned", [])

    prompt = RECOVERY_PROMPT.format(
        trigger_type=trigger_type,
        short_reply_count=short_count,
        repeated_topic=engagement.get("last_question_topic", ""),
        user_profile=_profile_str(profile),
        topic=state.get("detected_topic", "general"),
        pattern_threads=pattern_threads,
    )
    response = call_llm(prompt)

    eng_updated = dict(engagement)
    eng_updated["consecutive_short_replies"] = 0
    eng_updated["repeated_question_count"] = 0
    eng_updated["engagement_level"] = "medium"

    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1

    return {
        "messages": [AIMessage(content=response)],
        "engagement": eng_updated,
        "phase": phase,
        "vpr_stage": "V",
        "next_action": "classify",
    }


# ──────────────────────────────────────────────
# NODE: session_open_node — updated v2 (pattern reference)
# ──────────────────────────────────────────────

def session_open_node(state: AstroState) -> dict:
    """Return session opener with progress narrative. v2: references pattern threads."""
    profile = state.get("user_profile", {})
    remedy_state = state.get("remedy_state", {})
    phase = dict(state.get("phase", {}))
    engagement = dict(state.get("engagement", {}))

    last_remedy = remedy_state.get("remedies_given", ["—"])[-1] if remedy_state.get("remedies_given") else "—"
    last_hook = phase.get("hooks_planted", ["—"])[-1] if phase.get("hooks_planted") else "—"

    prompt = SESSION_OPEN_PROMPT.format(
        session_count=engagement.get("session_count", 2),
        topic=state.get("detected_topic", "general"),
        last_remedy=last_remedy,
        last_hook=last_hook,
        partner_name=profile.get("partner_name", "woh insaan"),
        last_prediction=phase.get("pending_drip", "—"),
        pattern_threads=profile.get("themes_mentioned", []),
    )
    response = call_llm(prompt)

    engagement["session_count"] = engagement.get("session_count", 1) + 1
    engagement["is_return_session"] = True
    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1

    return {
        "messages": [AIMessage(content=response)],
        "engagement": engagement,
        "phase": phase,
        "vpr_stage": "V",
        "next_action": "classify",
    }


# ──────────────────────────────────────────────
# NODE: vague_input_node
# ──────────────────────────────────────────────

def vague_input_node(state: AstroState) -> dict:
    """Handles vague messages. Uses Barnum broad claim as tactic 2."""
    profile = state.get("user_profile", {})
    msg = state["current_user_message"]

    prompt = VAGUE_INPUT_RECOVERY_PROMPT.format(
        current_message=msg,
        topic=state.get("detected_topic", "general"),
        user_profile=_profile_str(profile),
    )
    response = call_llm(prompt)

    phase = dict(state.get("phase", {}))
    phase["bot_turn_count"] = phase.get("bot_turn_count", 0) + 1

    return {
        "messages": [AIMessage(content=response)],
        "phase": phase,
        "vpr_stage": "V",
        "next_action": "classify",
    }
