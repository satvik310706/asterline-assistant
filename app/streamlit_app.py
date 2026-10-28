"""Small Streamlit client; all policy and state live in the library."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from asterline_assistant.config import Settings
from asterline_assistant.corpus import load_corpus, validate_corpus_integrity
from asterline_assistant.graph import run_improved_research
from asterline_assistant.llm import make_provider
from asterline_assistant.memory import MemoryStore
from asterline_assistant.research import run_research
from asterline_assistant.retrieval import HybridRetriever


def active_memories(store: MemoryStore, user_id: str):
    return store.list_active(user_id)


def delete_memory(store: MemoryStore, user_id: str, memory_id: int) -> bool:
    return store.delete_memory(user_id, memory_id)


def forget_all(store: MemoryStore, user_id: str) -> int:
    return store.forget_all(user_id)


def export_memories(store: MemoryStore, user_id: str) -> str:
    return json.dumps([record.model_dump(mode="json") for record in active_memories(store, user_id)], indent=2)


def _store() -> MemoryStore:
    return MemoryStore(Settings.from_environment().sqlite_path)


def _render_memory_panel(store: MemoryStore, user_id: str) -> None:
    st.subheader("Memory")
    st.caption("Memory is user context, not company evidence.")
    records = active_memories(store, user_id)
    for record in records:
        left, right = st.columns([5, 1])
        left.write(f"`{record.type}`  {record.content}")
        if right.button("Delete", key=f"delete-{record.id}"):
            delete_memory(store, user_id, int(record.id))
            st.rerun()
    if not records:
        st.caption("No active memories.")
    st.download_button("Export JSON", export_memories(store, user_id), file_name=f"{user_id}-memories.json", mime="application/json")
    if st.button("Forget all", key="forget-all"):
        forget_all(store, user_id)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Asterline", layout="wide")
    st.markdown("<style>body{background:#0d0d0d}.stCode,code{font-family:monospace}.status{color:#8b8b8b}</style>", unsafe_allow_html=True)
    settings = Settings.from_environment()
    store = _store()
    if "user" not in st.session_state:
        st.session_state.user = "alice"
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    with st.sidebar:
        st.title("Asterline")
        user_id = st.selectbox("User", ["alice", "bob", "carol"], index=["alice", "bob", "carol"].index(st.session_state.user))
        if user_id != st.session_state.user:
            st.session_state.user = user_id
            st.session_state.session_id = uuid.uuid4().hex
            st.rerun()
        config = st.selectbox("Config", ["baseline", "improved"], index=1 if settings.config == "improved" else 0)
        provider_name = st.selectbox("Provider", ["fake", "gemini", "ollama"], index=0 if not os.getenv("GEMINI_API_KEY") else 1)
        if not os.getenv("GEMINI_API_KEY") and provider_name == "gemini":
            st.warning("No Gemini key configured — demo mode is available with FakeLLM.")
        if st.button("New session"):
            st.session_state.session_id = uuid.uuid4().hex
            st.rerun()
        with st.expander("Memory panel", expanded=True):
            _render_memory_panel(store, st.session_state.user)
    st.title("Research assistant")
    messages = store.history(st.session_state.session_id)
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask about the supplied corpus")
    if question:
        store.record_message(st.session_state.user, st.session_state.session_id, "user", question)
        with st.chat_message("user"):
            st.markdown(question)
        with st.status("Running agents", expanded=False):
            validate_corpus_integrity(settings.corpus_path)
            retriever = HybridRetriever(load_corpus(settings.corpus_path), settings.chroma_path, settings.fastembed_cache_path)
            provider = make_provider(provider_name, settings.gemini_model, settings.ollama_model)
            settings = settings.model_copy(update={"config": config})
            history = [*messages, {"role": "user", "content": question}]
            history_models = [type("History", (), item) for item in history]
            fn = run_improved_research if config == "improved" else run_research
            response = fn(question, st.session_state.user, retriever, provider, store, history=history_models)
        store.record_message(st.session_state.user, st.session_state.session_id, "assistant", response.answer)
        with st.chat_message("assistant"):
            st.markdown(response.answer)
            st.caption(f"{response.provider}/{response.model}" + (" · degraded" if response.degraded else ""))
            with st.expander("Evidence & citations"):
                for item in response.evidence:
                    st.write(f"`{item.chunk.chunk_id}` — {item.chunk.title} · {item.chunk.published} · v{item.chunk.version}")
                    st.caption(item.chunk.text)
            if response.memory_used:
                with st.expander("Memory used this turn"):
                    st.write("From what you told me earlier:", "; ".join(response.memory_used))


if __name__ == "__main__":
    main()
