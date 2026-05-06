"""
state.py — AstroLokal LangGraph Conversation State

All conversation memory lives here. Every node reads from and writes to this state.
The state is designed to support the VPR (Validation → Prediction → Remedy) sandwich
pattern identified in high-engagement astrologer conversations.

v2 additions:
  - Barnum / pattern-reinforcement memory
  - Urgency / delay-framing tracking
  - Diagnostic follow-up cycling
  - High-stress containment flag
  - Intent-vs-feeling separation tracking
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# ──────────────────────────────────────────────
# Engagement signal constants
# ──────────────────────────────────────────────

SHORT_REPLY_TOKENS = {"ok", "ji", "ha", "haan", "acha", "accha", "theek", "hmm", "okay", "k", "👍"}

HIGH_STRESS_EMOTIONS = {"grief", "shock", "crisis", "panic"}   # triggers containment node
DISENGAGEMENT_THRESHOLD = 3          # consecutive short replies before recovery
REMEDY_UNLOCK_TURN = 15              # remedies only appear after this many bot turns
INFO_DRIP_CHUNK_SIZE = 0.3           # reveal 30% of prediction at a time


# ──────────────────────────────────────────────
# Core state
# ──────────────────────────────────────────────

class UserProfile(TypedDict, total=False):
    """Stored once, referenced in every prediction."""
    name: str
    dob: str                        # date of birth
    pob: str                        # place of birth
    primary_topic: str              # love | career | family | health | general
    partner_name: str
    partner_dob: str
    relationship_duration: str
    relationship_status: str        # in_contact | no_contact | blocked | unknown
    personal_items_mentioned: list[str]   # earphones, tshirt, gifts — used in remedies
    incidents_mentioned: list[str]        # specific events the user described
    user_exact_words: list[str]           # verbatim phrases to echo back
    # ── v2: pattern reinforcement memory ──
    themes_mentioned: list[str]           # stress, confusion, loss, hope — track across turns
    # ── v2: intent vs feeling tracking ──
    partner_attraction_noted: bool        # has attraction been established?
    partner_readiness_noted: bool         # has commitment readiness been assessed?


class EngagementTracker(TypedDict, total=False):
    """Real-time engagement health monitor."""
    consecutive_short_replies: int          # reset to 0 on any substantive reply
    repeated_question_count: int            # how many times user asked same thing
    last_question_topic: str
    engagement_level: str                   # high | medium | low | recovering
    positive_signals_count: int             # longer messages, unprompted details, follow-ups
    session_count: int                      # number of sessions (reconnects)
    is_return_session: bool


class ConversationPhase(TypedDict, total=False):
    """
    Tracks which phase of the conversation arc we're in.

    Phase progression:
      intake → build → peak → sustain → hook

    intake  : turns 1–10  — binary probes, first affirming prediction
    build   : turns 10–30 — deeper probing, user opens up
    peak    : turns 30–60 — user fully engaged, remedy introduction
    sustain : turns 60+   — remedy layers, return hooks
    hook    : last 2–3 msgs of session — curiosity/urgency plant
    """
    current_phase: str           # intake | build | peak | sustain | hook
    bot_turn_count: int
    remedy_turn_unlocked: bool   # True once bot_turn_count >= REMEDY_UNLOCK_TURN
    last_vpr_cycle: int          # bot turn when last full VPR sandwich completed
    pending_drip: Optional[str]     # the 70% of a prediction not yet revealed
    hooks_planted: list[str]     # list of hooks already used (avoid repetition)
    # ── v2 additions ──
    barnum_used: list[str]            # track which Barnum statements used (avoid repeats)
    urgency_planted: bool             # has a "critical period" urgency been introduced?
    last_diagnostic_question: str     # last pointed follow-up asked (cycle through types)
    high_stress_contained: bool       # has empathetic containment been delivered this session?
    pattern_threads: list[str]        # themes the bot has connected into a "pattern" narrative


class RemedyState(TypedDict, total=False):
    """Tracks remedy delivery to avoid dumping all at once."""
    remedies_given: list[str]         # descriptions of remedies already delivered
    remedy_mechanisms_explained: list[str]
    active_remedy_day: Optional[str]     # e.g. "Shukrawar" — forces return visit
    remedy_progress_reported: bool    # has user reported back on a remedy?
    pending_remedy: Optional[str]        # remedy teased but not yet fully explained


class KundliData(TypedDict, total=False):
    """
    Birth data collection state + API result.

    Collection happens incrementally — one field per turn, max 3 probes per field.
    Once complete, the API is called and the parsed brief is stored here.
    """
    # ── Raw birth data (collected from user) ──
    day:    int
    month:  int
    year:   int
    hour:   int
    minute: int
    lat:    float
    lon:    float
    tzone:  float
    city:   str          # stored for display; converted to lat/lon via geocoding

    # ── Collection state ──
    collection_complete: bool        # all API fields are present
    api_called: bool                 # True once the API has been called (avoid repeat calls)
    api_failed: bool                 # True if API call failed (use fallback predictions)

    # ── Retry tracking (per field, max 3 probes each) ──
    dob_probes:   int                # how many times we've asked for date of birth
    tob_probes:   int                # time of birth
    pob_probes:   int                # place of birth
    dob_skipped:  bool               # user declined / said "don't remember"
    tob_skipped:  bool
    pob_skipped:  bool

    # ── Parsed kundli brief (injected into predict/validate prompts) ──
    sun_sign:     str
    moon_sign:    str
    rising_sign:  str
    strengths:    list[str]          # positive planetary aspects
    tensions:     list[str]          # challenging planetary aspects
    love_house_notes:   str
    career_house_notes: str
    family_house_notes: str
    brief_text:   str                # short plain-language summary for LLM injection


class AstroState(TypedDict):
    """
    Master state object passed through every LangGraph node.
    """
    # ── Message history (LangGraph managed) ──
    messages: Annotated[list, add_messages]

    # ── Conversation memory ──
    user_profile: UserProfile
    engagement: EngagementTracker
    phase: ConversationPhase
    remedy_state: RemedyState
    kundli: KundliData               # v3: birth data + API result

    # ── Routing signals (set by classify_intent, read by router) ──
    next_action: str          # validate | predict | remedy | recover | hook | intake | session_open | kundli_collect
    vpr_stage: str            # which part of VPR we're currently in: V | P | R | done

    # ── Per-turn context ──
    current_user_message: str
    detected_topic: str       # love | career | family | financial | health | general
    detected_emotion: str     # distress | hope | confusion | anger | grief | shock | neutral
    is_vague_input: bool
    user_question_intent: str   # specific_question | venting | follow_up | new_topic | unknown
    # ── v2: high-stress routing flag ──
    is_high_stress: bool      # True when emotion is grief/shock/crisis → routes to containment
