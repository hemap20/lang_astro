"""
chat.py — Main entry point for the AstroLokal chatbot.

This is the single function your backend calls for every incoming user message.
It handles:
  - First-time users (intake flow)
  - Return users (session continuity)
  - State persistence via LangGraph checkpointer

Usage:
    from astro_bot.chat import chat

    response = await chat(
        user_id="user_123",
        message="Mere love life ke baare mein batao",
        session_id="session_abc"   # optional, use same session_id to continue a thread
    )
    print(response["reply"])       # the bot's reply text
    print(response["state"])       # full state for debugging

Your backend (FastAPI + Redis) handles auth, rate limiting, and WebSocket delivery.
This module only handles the conversation logic.
"""

import os
from typing import Optional
from langchain_core.messages import HumanMessage

from .graph import build_graph
from .state import AstroState, EngagementTracker, ConversationPhase, RemedyState, KundliData
from langgraph.checkpoint.memory import MemorySaver


# ── Module-level graph (replace MemorySaver with your Redis checkpointer in prod) ──
_graph = build_graph(checkpointer=MemorySaver())


def get_initial_state(user_id: str, is_return_user: bool = False) -> dict:
    """
    Build the initial state for a brand-new conversation thread.
    For return users, is_return_user=True triggers session_open flow.
    """
    return {
        "messages": [],
        "user_profile": {},
        "engagement": EngagementTracker(
            consecutive_short_replies=0,
            repeated_question_count=0,
            last_question_topic="",
            engagement_level="high",
            positive_signals_count=0,
            session_count=1 if not is_return_user else 2,
            is_return_session=is_return_user,
        ),
        "phase": ConversationPhase(
            current_phase="intake",
            bot_turn_count=0,
            remedy_turn_unlocked=False,
            last_vpr_cycle=0,
            pending_drip=None,
            hooks_planted=[],
        ),
        "remedy_state": RemedyState(
            remedies_given=[],
            remedy_mechanisms_explained=[],
            active_remedy_day=None,
            remedy_progress_reported=False,
            pending_remedy=None,
        ),
        "kundli": KundliData(
            collection_complete=False,
            api_called=False,
            api_failed=False,
            dob_probes=0,
            tob_probes=0,
            pob_probes=0,
            dob_skipped=False,
            tob_skipped=False,
            pob_skipped=False,
        ),
        "next_action": "intake",
        "vpr_stage": "V",
        "current_user_message": "",
        "detected_topic": "general",
        "detected_emotion": "neutral",
        "is_vague_input": False,
        "user_question_intent": "unknown",
    }


def chat(
    user_id: str,
    message: str,
    thread_id: Optional[str] = None,
    is_return_user: bool = False,
) -> dict:
    """
    Process one user message and return the bot's reply.

    Args:
        user_id:        Unique user identifier (from your auth system)
        message:        The user's raw message text
        thread_id:      LangGraph thread ID for state persistence.
                        Use the same thread_id across a session to maintain memory.
                        Use a new thread_id for a genuinely new conversation.
        is_return_user: True if this user has chatted before (triggers session_open)

    Returns:
        {
            "reply": str,          # concatenated bot messages for this turn
            "thread_id": str,      # pass this back for the next turn
            "engagement_level": str,
            "phase": str,
            "debug": dict          # full state snapshot (remove in production)
        }
    """
    if thread_id is None:
        thread_id = f"{user_id}_session_1"

    config = {"configurable": {"thread_id": thread_id}}

    # ── Build input: inject user message into state ──
    input_state = {
        "messages": [HumanMessage(content=message)],
        "current_user_message": message,
    }

    # ── First turn: initialize full state ──
    # LangGraph checkpointer handles this automatically for subsequent turns
    # but we need to seed the state on the very first invocation
    try:
        snapshot = _graph.get_state(config)
        is_first_turn = len(snapshot.values.get("messages", [])) == 0
    except Exception:
        is_first_turn = True

    if is_first_turn:
        # Merge initial state with the user's first message
        initial = get_initial_state(user_id, is_return_user)
        initial["messages"] = [HumanMessage(content=message)]
        initial["current_user_message"] = message
        input_state = initial

    # ── Invoke the graph ──
    final_state = _graph.invoke(input_state, config=config)

    # ── Collect all AI messages generated this turn ──
    all_messages = final_state.get("messages", [])
    # Find new AI messages (those after the last HumanMessage)
    reply_parts = []
    for msg in reversed(all_messages):
        if hasattr(msg, "type") and msg.type == "human":
            break
        if hasattr(msg, "content"):
            reply_parts.insert(0, msg.content)

    reply = "\n\n".join(reply_parts) if reply_parts else "Ji bataiye, main sun raha hun."

    engagement = final_state.get("engagement", {})
    phase = final_state.get("phase", {})

    return {
        "reply": reply,
        "thread_id": thread_id,
        "engagement_level": engagement.get("engagement_level", "medium"),
        "phase": phase.get("current_phase", "intake"),
        "debug": {
            "vpr_stage": final_state.get("vpr_stage"),
            "bot_turn_count": phase.get("bot_turn_count"),
            "remedy_turn_unlocked": phase.get("remedy_turn_unlocked"),
            "user_profile": final_state.get("user_profile", {}),
            "next_action": final_state.get("next_action"),
            "kundli_status": {
                "collection_complete": final_state.get("kundli", {}).get("collection_complete"),
                "api_called": final_state.get("kundli", {}).get("api_called"),
                "api_failed": final_state.get("kundli", {}).get("api_failed"),
                "sun_sign": final_state.get("kundli", {}).get("sun_sign"),
                "moon_sign": final_state.get("kundli", {}).get("moon_sign"),
            },
        }
    }


# ──────────────────────────────────────────────
# Quick CLI test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("AstroLokal Chatbot — CLI Test Mode")
    print("Type 'quit' to exit.\n")

    user_id = "test_user_001"
    thread_id = f"{user_id}_dev_session"
    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if user_input.lower() in {"quit", "exit", "q"}:
            break

        if not user_input:
            continue

        turn += 1
        result = chat(
            user_id=user_id,
            message=user_input,
            thread_id=thread_id,
            is_return_user=(turn > 1),
        )

        print(f"\nAstrologer: {result['reply']}")
        print(f"  [Phase: {result['phase']} | Engagement: {result['engagement_level']} | Turn: {result['debug']['bot_turn_count']}]\n")
