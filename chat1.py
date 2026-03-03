
# chat1.py
# Streamlit chat app using Google Gen AI SDK (2026 models)
# Requirements:
#   - streamlit>=1.32.0
#   - google-genai>=1.65.0
# Secrets (Streamlit Cloud -> Settings -> Secrets):
#   GEMINI_API_KEY = "YOUR_NEW_API_KEY"

import os
import time
import streamlit as st

# New, recommended SDK (GA): google-genai
# ref: https://pypi.org/project/google-genai/  and SDK docs
from google import genai
from google.genai.types import GenerateContentConfig

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="Gemini Chat • chat1.py",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Gemini Chat (chat1.py)")
st.caption(
    "New Google Gen AI SDK • Models: gemini-3.1-pro (flagship) or gemini-3-flash-preview (fast)."
)

# -------------------------------
# API key (prefer Streamlit Secrets; fallback to env var)
# -------------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error(
        "Missing API key. Add `GEMINI_API_KEY` to Streamlit Secrets or your environment."
    )
    st.stop()

# -------------------------------
# Sidebar: model & generation settings
# -------------------------------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        options=[
            "gemini-3.1-pro",        # Latest flagship (Feb 19, 2026)
            "gemini-3-flash-preview" # Latest Flash-tier in 2026
        ],
        index=0,
        help="Use Pro for best reasoning; Flash for speed and cost."
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    top_k = st.slider("Top-k", 1, 100, 40, 1)
    max_output_tokens = st.number_input("Max output tokens", 64, 8192, 2048, 64)

    st.markdown("---")
    st.caption(
        "Tip: If you see NotFound/404, switch to a current model and ensure "
        "you're using the new SDK (google-genai)."
    )

# -------------------------------
# Initialize client (explicitly pass key)
# -------------------------------
client = genai.Client(api_key=API_KEY)

# -------------------------------
# Session-state chat history
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------------------
# Chat input & generation
# -------------------------------
prompt = st.chat_input("Type your message...")
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking…"):
            try:
                cfg = GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=max_output_tokens,
                )

                t0 = time.time()
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=cfg,
                )
                dt = time.time() - t0

                text = getattr(resp, "text", None) or "No text returned."
                placeholder.markdown(text)

                st.session_state.messages.append(
                    {"role": "assistant", "content": text}
                )
                st.caption(f"⏱ {dt:.2f}s • Model: {model}")

            except Exception as e:
                err = str(e)
                if "NotFound" in err or "404" in err:
                    st.error(
                        "Model not found. Use a current ID like "
                        "`gemini-3.1-pro` or `gemini-3-flash-preview`."
                    )
                elif "UNAUTHENTICATED" in err or "permission" in err.lower():
                    st.error(
                        "Authentication error. Confirm `GEMINI_API_KEY` is set in "
                        "Streamlit Secrets and is a valid Google AI Studio key."
                    )
                else:
                    st.error(f"Unexpected error: {err}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "Built with the Google Gen AI SDK (GA). For structured outputs, tool use, and streaming, "
    "see the SDK docs."
)
