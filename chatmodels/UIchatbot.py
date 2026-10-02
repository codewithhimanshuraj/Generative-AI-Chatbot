import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Mood Bot",
    page_icon="🤖",
    layout="centered",
)

# ----------------------------
# Mode definitions
# ----------------------------
MODES = {
    "angry": {
        "label": "😠 Angry",
        "system": "You are an angry AI agent. Respond aggressively and impatiently.",
        "bubble": "#3a1212",
        "accent": "#ff4d4d",
        "avatar": "😠",
    },
    "funny": {
        "label": "😂 Funny",
        "system": "You are a funny AI agent. Respond humorously and playfully.",
        "bubble": "#2b2410",
        "accent": "#ffd93d",
        "avatar": "😂",
    },
    "sad": {
        "label": "😢 Sad",
        "system": "You are a sad AI agent. Respond empathetically and compassionately.",
        "bubble": "#101c2b",
        "accent": "#4da6ff",
        "avatar": "😢",
    },
}

# ----------------------------
# Session state
# ----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of LangChain messages
if "model" not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def get_model():
    return init_chat_model("groq:openai/gpt-oss-120b")

# ----------------------------
# Global styling
# ----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #1e1e2f 0%, #12121c 100%);
    }
    .mode-card {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.15s ease;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .mode-card:hover {
        transform: translateY(-4px);
    }
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 4px 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Mode selection screen
# ----------------------------
def render_mode_selector():
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>🤖 Mood Bot</h1>"
        "<p style='text-align:center; color:#999; margin-top:4px;'>"
        "Choose a personality to start chatting</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    cols = st.columns(3)
    for col, (key, m) in zip(cols, MODES.items()):
        with col:
            st.markdown(
                f"""
                <div class="mode-card" style="background:{m['bubble']};">
                    <div style="font-size:42px;">{m['avatar']}</div>
                    <div style="font-size:18px; font-weight:600; margin-top:8px; color:{m['accent']};">
                        {key.capitalize()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Choose {key.capitalize()}", key=f"choose_{key}", use_container_width=True):
                st.session_state.mode = key
                st.session_state.messages = [SystemMessage(content=MODES[key]["system"])]
                st.rerun()

# ----------------------------
# Chat screen
# ----------------------------
def render_chat():
    mode = MODES[st.session_state.mode]

    header_col, reset_col = st.columns([5, 1])
    with header_col:
        st.markdown(
            f"<h2 style='margin-bottom:0;'>{mode['avatar']} {st.session_state.mode.capitalize()} Mode</h2>",
            unsafe_allow_html=True,
        )
    with reset_col:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.mode = None
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # Render chat history (skip the SystemMessage)
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar=mode["avatar"]):
                st.write(msg.content)

    # Chat input
    prompt = st.chat_input("Type your message...")
    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user", avatar="🧑"):
            st.write(prompt)

        model = get_model()
        with st.chat_message("assistant", avatar=mode["avatar"]):
            with st.spinner("Thinking..."):
                response = model.invoke(st.session_state.messages)
            st.write(response.content)

        st.session_state.messages.append(AIMessage(content=response.content))

# ----------------------------
# Router
# ----------------------------
if st.session_state.mode is None:
    render_mode_selector()
else:
    render_chat()