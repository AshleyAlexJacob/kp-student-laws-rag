"""Streamlit Chat UI for KP Education Laws Assistant.

Provides a minimal, dark-mode chat interface with multi-session management,
session switching, and distinct sender-receiver message color styling.
Backend logic is deferred to subsequent project phases.
"""

from typing import List
import streamlit as st

from kp_agent.logger import get_logger
from schemas.chat import Message
from ui.css import inject_styles
from ui.session import SessionStore

logger = get_logger(__name__)


class ChatApp:
    """Main UI application class for the Streamlit chat interface."""

    def __init__(self) -> None:
        """Initialize the chat application and session store."""
        self.session_store = SessionStore()

    def render_sidebar(self) -> None:
        """Render the sidebar containing session management and previous sessions list."""
        with st.sidebar:
            st.markdown('<div class="sidebar-title">⚖️ KP Education Laws</div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-subtitle">RAG Assistant (1897–2020)</div>', unsafe_allow_html=True)

            # Button to create a new session
            if st.button("➕ New Chat", use_container_width=True, type="primary"):
                new_id = self.session_store.create_session("New Chat")
                st.rerun()

            st.markdown('<div class="session-label">Previous Sessions</div>', unsafe_allow_html=True)

            sessions = self.session_store.get_all_sessions()
            active_id = self.session_store.get_active_session_id()

            # Render sessions list in reverse order (newest first)
            for sess_id, sess_data in reversed(list(sessions.items())):
                col1, col2 = st.columns([0.82, 0.18])
                is_active = sess_id == active_id

                with col1:
                    # Highlight active session with indicator
                    btn_label = f"💬 {sess_data['title']}"
                    if is_active:
                        btn_label = f"▶ {sess_data['title']}"

                    if st.button(
                        btn_label,
                        key=f"sess_btn_{sess_id}",
                        use_container_width=True,
                        help=f"Created: {sess_data['created_at']}",
                    ):
                        self.session_store.switch_session(sess_id)
                        st.rerun()

                with col2:
                    # Quick delete button per session
                    if st.button("🗑️", key=f"del_sess_{sess_id}", help="Delete this session"):
                        self.session_store.delete_session(sess_id)
                        st.rerun()

    def render_chat_screen(self) -> None:
        """Render the chat header, conversation messages, and input prompt."""
        active_session = self.session_store.get_active_session()

        # Session Header
        st.markdown(
            f"""
            <div class="chat-header">
                <div>
                    <h2>{active_session['title']}</h2>
                </div>
                <span>Session: {active_session['id']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Message List Container
        messages: List[dict] = active_session["messages"]

        if not messages:
            st.info("💡 No messages in this session yet. Ask a question below to start chatting!")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")

            if role == "user":
                # Distinct teal container for user messages
                st.markdown(
                    f"""
                    <div class="msg-container-user">
                        <div class="msg-bubble-user">
                            <div>{content}</div>
                            <div class="msg-meta-user">👤 You • {timestamp}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Distinct slate container with teal accent for assistant messages
                st.markdown(
                    f"""
                    <div class="msg-container-bot">
                        <div class="msg-bubble-bot">
                            <div>{content}</div>
                            <div class="msg-meta-bot">🤖 Assistant • {timestamp}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Chat Input at page bottom
        user_input = st.chat_input("Ask a question about Khyber Pakhtunkhwa education laws...")
        if user_input:
            clean_input = user_input.strip()
            if clean_input:
                logger.info("Received user prompt in session %s", active_session["id"])
                # Record user question
                self.session_store.add_message("user", clean_input)

                # Simulated response for UI testing phase (no backend logic per requirement)
                mock_response = (
                    f"*(UI Phase Preview - Backend will be connected in future phases)*\n\n"
                    f"Received query: **\"{clean_input}\"**.\n\n"
                    f"In upcoming phases, this will search indexed KP education laws and cite "
                    f"the specific law title, year, and section."
                )
                self.session_store.add_message("assistant", mock_response)
                st.rerun()

    def run(self) -> None:
        """Run the complete chat UI."""
        inject_styles()
        self.render_sidebar()
        self.render_chat_screen()


def main() -> None:
    """Entrypoint function for Streamlit execution."""
    st.set_page_config(
        page_title="KP Education Laws Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
