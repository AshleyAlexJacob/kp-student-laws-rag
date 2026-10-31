"""Session management and state storage for chat conversations."""

import datetime
from typing import Dict, List, Optional
import uuid
import streamlit as st

from kp_agent.logger import get_logger
from schemas.chat import Message

logger = get_logger(__name__)


class SessionStore:
    """Manages chat sessions stored in Streamlit session_state."""

    def __init__(self) -> None:
        """Initialize session storage if not already present in session_state."""
        if "sessions" not in st.session_state:
            # Create an initial default session for a smooth first-launch experience
            initial_id = str(uuid.uuid4())[:8]
            default_welcome_msg = Message(
                role="assistant",
                content=(
                    "👋 Welcome to the **KP Education Laws Assistant**.\n\n"
                    "You can ask questions regarding Khyber Pakhtunkhwa education laws "
                    "(1897–2020). Start a new chat or select a session from the sidebar."
                ),
            )
            st.session_state.sessions = {
                initial_id: {
                    "id": initial_id,
                    "title": "Welcome & Overview",
                    "created_at": datetime.datetime.now().strftime("%b %d, %I:%M %p"),
                    "messages": [
                        {
                            "role": default_welcome_msg.role,
                            "content": default_welcome_msg.content,
                            "timestamp": default_welcome_msg.timestamp,
                        }
                    ],
                }
            }
            st.session_state.active_session_id = initial_id
            logger.info("Initialized default session: %s", initial_id)

    def get_all_sessions(self) -> Dict[str, dict]:
        """Return all existing sessions.

        Returns:
            Dict[str, dict]: Dictionary mapping session IDs to session details.
        """
        return st.session_state.sessions

    def get_active_session_id(self) -> str:
        """Return the ID of the currently selected session.

        Returns:
            str: Active session ID.
        """
        return st.session_state.active_session_id

    def get_active_session(self) -> dict:
        """Return the full dictionary object for the active session.

        Returns:
            dict: Active session data.
        """
        active_id = self.get_active_session_id()
        if active_id not in st.session_state.sessions:
            # Fallback to first available session or create a fresh one
            if st.session_state.sessions:
                st.session_state.active_session_id = next(iter(st.session_state.sessions))
            else:
                self.create_session("New Session")
        return st.session_state.sessions[st.session_state.active_session_id]

    def create_session(self, title: str = "New Chat") -> str:
        """Create a new chat session and set it as active.

        Args:
            title: Display title for the new session.

        Returns:
            str: The newly created session ID.
        """
        new_id = str(uuid.uuid4())[:8]
        st.session_state.sessions[new_id] = {
            "id": new_id,
            "title": title,
            "created_at": datetime.datetime.now().strftime("%b %d, %I:%M %p"),
            "messages": [],
        }
        st.session_state.active_session_id = new_id
        logger.info("Created new session: %s (%s)", new_id, title)
        return new_id

    def switch_session(self, session_id: str) -> None:
        """Switch active conversation to the specified session ID.

        Args:
            session_id: ID of the session to switch to.
        """
        if session_id in st.session_state.sessions:
            st.session_state.active_session_id = session_id
            logger.info("Switched to session: %s", session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete a session by ID and adjust active session if needed.

        Args:
            session_id: ID of the session to remove.
        """
        if session_id in st.session_state.sessions:
            del st.session_state.sessions[session_id]
            logger.info("Deleted session: %s", session_id)
            # If the deleted session was active, reset active session
            if st.session_state.active_session_id == session_id:
                if st.session_state.sessions:
                    st.session_state.active_session_id = next(iter(st.session_state.sessions))
                else:
                    self.create_session("New Chat")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the currently active session.

        Args:
            role: Sender role ('user' or 'assistant').
            content: Text body of the message.
        """
        session = self.get_active_session()
        msg = Message(role=role, content=content)
        session["messages"].append(
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
            }
        )
        # Update session title automatically on first user prompt
        if role == "user" and (session["title"] in ["New Chat", "New Session"] or len(session["messages"]) <= 2):
            trimmed_title = content.strip().replace("\n", " ")[:32]
            session["title"] = f"{trimmed_title}..." if len(content) > 32 else trimmed_title

