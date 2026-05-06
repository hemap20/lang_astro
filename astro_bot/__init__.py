"""
astro_bot — AstroLokal LangGraph Conversation Engine

Entry points:
    from astro_bot.chat import chat          # single-message handler for your backend
    from astro_bot.graph import astro_graph  # raw compiled LangGraph (for custom use)
    from astro_bot.state import AstroState   # state TypedDict
"""

from .chat import chat
from .graph import astro_graph
from .state import AstroState

__all__ = ["chat", "astro_graph", "AstroState"]
