"""Data schemas for chat messages and conversational elements."""

from dataclasses import dataclass, field
import datetime
from typing import Any, Dict, List


@dataclass
class Message:
    """Represents a single chat message.

    Attributes:
        role: The role of the sender, typically 'user' or 'assistant'.
        content: The text message content.
        timestamp: Time string indicating when the message was sent.
        sources: List of retrieved legal citations or statutory snippets.
    """

    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%I:%M %p")
    )
    sources: List[Dict[str, Any]] = field(default_factory=list)
    verification_count: int = 0
    verification_status: str = "verified"
    query_variants: List[str] = field(default_factory=list)

