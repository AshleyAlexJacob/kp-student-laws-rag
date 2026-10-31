"""Custom CSS styling definitions for the Streamlit UI."""

import streamlit as st

CUSTOM_CSS = """
<style>
/* Global font and container polish */
.stApp {
    background-color: #0F172A;
    color: #F8FAFC;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Minimal custom header */
.chat-header {
    padding: 0.5rem 0 1.25rem 0;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.chat-header h2 {
    margin: 0;
    font-size: 1.35rem;
    font-weight: 600;
    color: #F8FAFC;
}
.chat-header span {
    font-size: 0.85rem;
    color: #14B8A6;
    background: rgba(20, 184, 166, 0.12);
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    border: 1px solid rgba(20, 184, 166, 0.3);
}

/* User (Sender) Message - Vibrant Teal */
.msg-container-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1rem;
}
.msg-bubble-user {
    background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%);
    color: #FFFFFF;
    padding: 0.85rem 1.15rem;
    border-radius: 1.1rem 1.1rem 0.2rem 1.1rem;
    max-width: 80%;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.25);
    font-size: 0.95rem;
    line-height: 1.5;
}
.msg-meta-user {
    font-size: 0.72rem;
    color: #CCFBF1;
    margin-top: 0.35rem;
    text-align: right;
}

/* Assistant (Receiver) Message - Sleek Dark Slate with Teal accent */
.msg-container-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 1rem;
}
.msg-bubble-bot {
    background-color: #1E293B;
    color: #E2E8F0;
    border: 1px solid #334155;
    border-left: 3px solid #14B8A6;
    padding: 0.9rem 1.2rem;
    border-radius: 0.2rem 1.1rem 1.1rem 1.1rem;
    max-width: 82%;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    font-size: 0.95rem;
    line-height: 1.55;
}
.msg-meta-bot {
    font-size: 0.72rem;
    color: #94A3B8;
    margin-top: 0.35rem;
}

/* Sidebar aesthetics */
[data-testid="stSidebar"] {
    background-color: #090D16;
    border-right: 1px solid #1E293B;
}
.sidebar-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.25rem;
}
.sidebar-subtitle {
    font-size: 0.78rem;
    color: #94A3B8;
    margin-bottom: 1.25rem;
}
.session-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin: 1.2rem 0 0.5rem 0;
    font-weight: 600;
}

/* Button customization for active session vs inactive */
div[data-testid="stSidebar"] button[kind="secondary"] {
    background-color: #1E293B;
    color: #E2E8F0;
    border: 1px solid #334155;
    border-radius: 0.5rem;
    transition: all 0.2s ease;
}
div[data-testid="stSidebar"] button[kind="secondary"]:hover {
    border-color: #14B8A6;
    color: #14B8A6;
}
div[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #14B8A6;
    color: #042F2E;
    font-weight: 600;
    border: none;
    border-radius: 0.5rem;
}
</style>
"""


def inject_styles() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

