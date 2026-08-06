"""Streamlit interface for evidence-grounded Steam game recommendations."""

from __future__ import annotations

import os
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_backend import SteamReviewRAG


PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Steam Game Review Explorer", page_icon="🎮", layout="wide")

st.markdown(
    """
    <style>
      :root { --ink: #171b1e; --paper: #f4f0e7; --coral: #e95d3c; --blue: #155e75; }
      .stApp { background: var(--paper); }
      .hero { border-bottom: 3px solid var(--ink); padding: 0.8rem 0 1.4rem; margin-bottom: 1.2rem; }
      .eyebrow { color: var(--coral); font-size: 0.72rem; font-weight: 800; letter-spacing: 0.15em; text-transform: uppercase; }
      .hero h1 { font-family: Georgia, serif; font-size: clamp(2.2rem, 5vw, 4.4rem); letter-spacing: -0.06em; margin: 0.1rem 0; color: var(--ink); }
      .hero p { max-width: 50rem; color: #4d5557; font-size: 1.05rem; margin: 0; }
      .metric-strip { display: flex; gap: 1.5rem; margin: 0.8rem 0 1.5rem; flex-wrap: wrap; }
      .metric { border-left: 3px solid var(--coral); padding: 0.1rem 0 0.1rem 0.65rem; min-width: 9rem; }
      .metric strong { display: block; font-size: 1.25rem; color: var(--ink); }
      .metric span { color: #596164; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
      .evidence-label { color: var(--blue); font-size: 0.75rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the local retrieval index…")
def load_backend() -> SteamReviewRAG:
    return SteamReviewRAG.load(PROJECT_DIR)


def get_api_key() -> str | None:
    load_dotenv(PROJECT_DIR / ".env")
    value = os.getenv("GROQ_API_KEY")
    if value:
        return value.strip()
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def show_evidence(evidence) -> None:
    with st.expander(f"Evidence used ({len(evidence)} player reviews)", expanded=False):
        for position, row in evidence.iterrows():
            st.markdown(f"<div class='evidence-label'>[{position + 1}] {escape(str(row['game_name']))} · {escape(str(row['recommendation']))}</div>", unsafe_allow_html=True)
            st.caption(f"Semantic score {row['bi_score']:.3f} · reranker score {row['rerank_score']:.2f}")
            st.write(str(row["review"]))
            if position < len(evidence) - 1:
                st.divider()


try:
    backend = load_backend()
except Exception as exc:
    st.error("The local RAG artifacts could not be loaded.")
    st.exception(exc)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### Control room")
    model_labels = list(backend.config["groq_models"])
    default_value = backend.config["default_model"]
    default_index = next(
        (i for i, label in enumerate(model_labels) if backend.config["groq_models"][label] == default_value), 0
    )
    selected_label = st.selectbox("Answer model", model_labels, index=default_index)
    evidence_count = st.slider("Evidence reviews", min_value=3, max_value=8, value=5)
    key = get_api_key()
    if key:
        st.success("Generation is configured")
    else:
        st.info("Retrieval works locally. Add `GROQ_API_KEY` to `.env` to generate answers.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Retrieval device: {backend.device.upper()}")

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Steam review intelligence · evidence first</div>
      <h1>Steam Game Review Explorer</h1>
      <p>Ask for a kind of game, a player concern, or a recommendation. Every answer is grounded in retrieved Steam reviews and cites its evidence.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="metric-strip">
      <div class="metric"><strong>{len(backend.corpus):,}</strong><span>curated reviews</span></div>
      <div class="metric"><strong>{backend.corpus['game_name'].nunique():,}</strong><span>games represented</span></div>
      <div class="metric"><strong>2-stage</strong><span>retrieve + rerank</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("evidence") is not None:
            show_evidence(message["evidence"])

if question := st.chat_input("e.g. What is a calm game to play after work?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reranking player evidence…"):
            try:
                evidence = backend.retrieve_and_rerank(question, evidence_count)
                if key:
                    answer = backend.generate_answer(
                        question, evidence, key, backend.config["groq_models"][selected_label]
                    )
                else:
                    answer = (
                        "I found the evidence below. Add `GROQ_API_KEY` to `.env` to enable "
                        "a grounded written answer."
                    )
            except Exception as exc:
                st.error("The request could not be completed. Check the local model files or API key.")
                st.exception(exc)
                st.stop()
        st.markdown(answer)
        show_evidence(evidence)
    st.session_state.messages.append({"role": "assistant", "content": answer, "evidence": evidence})
