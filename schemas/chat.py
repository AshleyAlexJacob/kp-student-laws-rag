"""Data schemas for chat messages and conversational elements."""

from dataclasses import dataclass, field
import datetime


@dataclass
class Message:
    """Represents a single chat message.

    Attributes:
        role: The role of the sender, typically 'user' or 'assistant'.
        content: The text message content.
        timestamp: Time string indicating when the message was sent.
    """

    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%I:%M %p")
    )

