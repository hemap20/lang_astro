"""
graph.py — LangGraph conversation graph for AstroLokal chatbot.

Graph structure (v2):
  Every user turn flows through:
    1. profile_update   — extract + store user details (silent)
    2. classify         — detect topic, emotion, engagement signals
    3. router           — conditional edge based on classify output:
         containment    → for grief/shock/crisis moments (v2)
         barnum         → early rapport node, turns 3-8 (v2)
         validate       → V in VPR sandwich (default)
         intent_feeling → love topic: attraction vs. readiness split (v2)
         predict        → P in VPR sandwich
         pattern        → connects cross-turn themes into a cycle (v2)
         remedy         → R in VPR (only after turn 15)
         hook           → return-visit seed
         recover        → re-engagement (3+ short replies / repeated Q)
         session_open   → return session with progress narrative
         vague_input    → handles "mere baare mein batao"
         intake         → first-turn greeting

The graph is compiled once and invoked with a persistent checkpointer.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AstroState
from .nodes import (
    intake_node,
    classify_node,
    validate_node,
    predict_node,
    remedy_node,
    hook_node,
    recover_node,
    session_open_node,
    vague_input_node,
    profile_update_node,
    # v2 nodes
    containment_node,
    barnum_node,
    intent_feeling_node,
    pattern_node,
)
# v3: kundli collection node
from .kundli_node import kundli_collect_node


# ──────────────────────────────────────────────
# Routing functions
# ──────────────────────────────────────────────

def route_after_classify(state: AstroState) -> str:
    """
    Priority order (v3):
      1. containment    — grief/shock/crisis (before any prediction)
      2. recover        — 3+ short replies or repeated question
      3. session_open   — return session first turn
      4. kundli_collect — birth data collection (v3)
      5. barnum         — early rapport (turns 3-8, once)
      6. vague_input    — message too vague
      7. validate       — default VPR start
      8. intake         — very first turn
    """
    next_action = state.get("next_action", "validate")
    valid_routes = {
        "containment", "recover", "session_open", "kundli_collect",
        "barnum", "vague_input", "validate", "intake",
    }
    return next_action if next_action in valid_routes else "validate"


def route_after_kundli(state: AstroState) -> str:
    """
    After kundli_collect_node:
      - If still collecting: loop back (END → wait for next user message)
      - If collection complete + API done: proceed to validate
    """
    next_action = state.get("next_action", "kundli_collect")
    if next_action == "validate":
        return "validate"
    return END   # still collecting — wait for user's next reply


def route_after_containment(state: AstroState) -> str:
    """After containing stress, wait for user to respond before continuing."""
    return END


def route_after_barnum(state: AstroState) -> str:
    """Barnum always feeds into validate (continues VPR flow)."""
    return "validate"


def route_after_validate(state: AstroState) -> str:
    """
    v2: validate can route to intent_feeling (love) or predict (all others).
    """
    next_action = state.get("next_action", "predict")
    if next_action == "intent_feeling":
        return "intent_feeling"
    return "predict"


def route_after_intent_feeling(state: AstroState) -> str:
    """Intent-vs-feeling always feeds into predict."""
    return "predict"


def route_after_predict(state: AstroState) -> str:
    """
    v2: predict can route to pattern (when 2+ themes) → remedy → hook,
    or skip pattern → remedy | hook.
    """
    next_action = state.get("next_action", "hook")
    if next_action in {"pattern", "remedy", "hook"}:
        return next_action
    return "hook"


def route_after_pattern(state: AstroState) -> str:
    """Pattern always feeds into remedy or hook depending on unlock status."""
    next_action = state.get("next_action", "hook")
    return next_action if next_action in {"remedy", "hook"} else "hook"


def route_after_remedy(state: AstroState) -> str:
    return "hook"


def route_after_hook(state: AstroState) -> str:
    return END


def route_after_recovery(state: AstroState) -> str:
    return END


def route_after_session_open(state: AstroState) -> str:
    return END


def route_after_vague(state: AstroState) -> str:
    return END


def route_after_intake(state: AstroState) -> str:
    return END


# ──────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────

def build_graph(checkpointer=None) -> StateGraph:
    """
    Build and compile the AstroLokal v2 conversation graph.

    Args:
        checkpointer: LangGraph checkpointer for persistent memory.
                      Pass MemorySaver() for in-memory (dev),
                      or a Redis/SQLite checkpointer for production.

    Returns:
        Compiled StateGraph.
    """
    builder = StateGraph(AstroState)

    # ── Register all nodes ──
    builder.add_node("profile_update",  profile_update_node)
    builder.add_node("classify",        classify_node)
    builder.add_node("intake",          intake_node)
    builder.add_node("containment",     containment_node)    # v2
    builder.add_node("barnum",          barnum_node)         # v2
    builder.add_node("validate",        validate_node)
    builder.add_node("intent_feeling",  intent_feeling_node) # v2
    builder.add_node("predict",         predict_node)
    builder.add_node("pattern",         pattern_node)        # v2
    builder.add_node("remedy",          remedy_node)
    builder.add_node("hook",            hook_node)
    builder.add_node("recover",         recover_node)
    builder.add_node("session_open",    session_open_node)
    builder.add_node("vague_input",     vague_input_node)
    builder.add_node("kundli_collect",  kundli_collect_node) # v3

    # ── Entry: every user message starts here ──
    builder.set_entry_point("profile_update")

    # ── profile_update → classify ──
    builder.add_edge("profile_update", "classify")

    # ── classify → conditional routing ──
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "containment":   "containment",
            "intake":        "intake",
            "validate":      "validate",
            "barnum":        "barnum",
            "recover":       "recover",
            "session_open":  "session_open",
            "vague_input":   "vague_input",
            "kundli_collect": "kundli_collect",
        }
    )

    # ── kundli_collect → validate (done) or END (still collecting) ──
    builder.add_conditional_edges(
        "kundli_collect",
        route_after_kundli,
        {
            "validate": "validate",
            END: END,
        }
    )

    # ── intake → END ──
    builder.add_conditional_edges("intake", route_after_intake, {END: END})

    # ── containment → END (wait for user to breathe and respond) ──
    builder.add_conditional_edges("containment", route_after_containment, {END: END})

    # ── barnum → validate ──
    builder.add_conditional_edges("barnum", route_after_barnum, {"validate": "validate"})

    # ── validate → intent_feeling | predict ──
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "intent_feeling": "intent_feeling",
            "predict":        "predict",
        }
    )

    # ── intent_feeling → predict ──
    builder.add_conditional_edges(
        "intent_feeling",
        route_after_intent_feeling,
        {"predict": "predict"}
    )

    # ── predict → pattern | remedy | hook ──
    builder.add_conditional_edges(
        "predict",
        route_after_predict,
        {
            "pattern": "pattern",
            "remedy":  "remedy",
            "hook":    "hook",
        }
    )

    # ── pattern → remedy | hook ──
    builder.add_conditional_edges(
        "pattern",
        route_after_pattern,
        {
            "remedy": "remedy",
            "hook":   "hook",
        }
    )

    # ── remedy → hook ──
    builder.add_conditional_edges("remedy", route_after_remedy, {"hook": "hook"})

    # ── hook → END ──
    builder.add_conditional_edges("hook", route_after_hook, {END: END})

    # ── recover → END ──
    builder.add_conditional_edges("recover", route_after_recovery, {END: END})

    # ── session_open → END ──
    builder.add_conditional_edges("session_open", route_after_session_open, {END: END})

    # ── vague_input → END ──
    builder.add_conditional_edges("vague_input", route_after_vague, {END: END})

    # ── Compile ──
    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


# ──────────────────────────────────────────────
# Default compiled graph (import this in your backend)
# ──────────────────────────────────────────────

astro_graph = build_graph()
