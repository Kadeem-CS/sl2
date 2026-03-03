
import os
import time
import streamlit as st

# --- Google Gen AI SDK (new) ---
# pip install google-genai
from google import genai
from google.genai.types import GenerateContentConfig

# -------------------------------
# App config
# -------------------------------
st.set_page_config(
    page_title="Gemini Chat (Google Gen AI SDK)",
    page_icon="✨",
    layout="centered"
)

st.title("✨ Gemini Chat (2026)")
st.caption(
    "Powered by Google Gen AI SDK • Models: gemini-3.1-pro (flagship) or gemini-3-flash-preview (fast)"
)

# -------------------------------
# Secrets / API Key
# -------------------------------
# Prefer Streamlit Secrets; fallback to environment for local runs.
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error(
        "Missing API key. Add `GEMINI_API_KEY` to Streamlit Secrets or your environment."
    )
    st.stop()

# -------------------------------
# Sidebar controls
# -------------------------------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        options=[
            "gemini-3.1-pro",       # Latest flagship (Feb 2026)
            "gemini-3-flash-preview"  # Latest fast-tier (Flash) in 2026
        ],
        index=0,
        help="Choose Pro for best reasoning; Flash for speed/cost."
    )

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    top_k = st.slider("Top-k", 1, 100, 40, 1)
    max_output_tokens = st.number_input("Max output tokens", 64, 8192, 2048, 64)

    st.markdown("—")
    st.caption(
        "Tip: If you see `NotFound`, you're likely on an old model. "
        "Use one of the models above."
    )

# -------------------------------
# Client init (new SDK)
# -------------------------------
# You can rely on env/secret auto-pickup, but here we pass explicitly for clarity.
client = genai.Client(api_key=API_KEY)

# -------------------------------
# Chat history (Streamlit session state)
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------------------
# Chat input
# -------------------------------
prompt = st.chat_input("Type your message…")
if prompt:
    # Echo user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking…"):
            try:
                # Optional: basic generation config
                config = GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=max_output_tokens,
                )

                # Make the request
                t0 = time.time()
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                    # You can also set request-level timeouts via http_options
                )
                elapsed = time.time() - t0

                # Handle the response
                text = getattr(response, "text", None) or "No text returned."
                placeholder.markdown(text)

                # Save to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": text}
                )

                st.caption(f"⏱️ {elapsed:.2f}s · Model: {model}")

            except Exception as e:
