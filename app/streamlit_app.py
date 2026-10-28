"""Small Streamlit client; all policy and state live in the library."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from asterline_assistant.config import Settings
from asterline_assistant.corpus import load_corpus, validate_corpus_integrity
from asterline_assistant.graph import is_chitchat_question, run_improved_research
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


def edit_memory(store: MemoryStore, user_id: str, memory_id: int, content: str) -> bool:
    return store.edit_memory(user_id, memory_id, content)


def export_memories(store: MemoryStore, user_id: str) -> str:
    return json.dumps([record.model_dump(mode="json") for record in active_memories(store, user_id)], indent=2)


def _store() -> MemoryStore:
    return MemoryStore(Settings.from_environment().sqlite_path)


@st.cache_resource(show_spinner=False)
def _retriever(corpus_file: str, chroma_dir: str, embedding_cache: str) -> HybridRetriever:
    corpus_path = Path(corpus_file)
    validate_corpus_integrity(corpus_path)
    return HybridRetriever(load_corpus(corpus_path), Path(chroma_dir), Path(embedding_cache))


def _safe_provider_diagnostics(response) -> str:
    """Expose classes and status only; never render raw model or SDK text."""
    primary = response.error_class or "provider failure"
    status = f"HTTP {response.error_status}" if response.error_status is not None else "status unavailable"
    quota = f", {response.quota_type} quota" if response.quota_type else ""
    details = [f"Gemini: {primary} ({status}{quota})"]
    if response.fallback_error:
        match = re.search(r"failed:\s*([A-Za-z][A-Za-z0-9_]*)", response.fallback_error)
        if match:
            details.append(f"Ollama: {match.group(1)}")
    return " · ".join(details)


def _new_session() -> None:
    session_id = uuid.uuid4().hex
    st.session_state.session_id = session_id
    # Set the widget's selection in its callback, before Streamlit constructs it.
    st.session_state.resume_session = session_id


def _render_memory_panel(store: MemoryStore, user_id: str) -> None:
    st.subheader("Memory panel")
    st.caption("Memory is user context, not company evidence.")
    records = active_memories(store, user_id)
    for record in records:
        left, middle, right = st.columns([4, 1, 1])
        with left:
            st.markdown(f"`{record.type}`  {record.content}")
            st.caption(f"saved {record.created_at[:19]} · confirmed {record.last_confirmed_at[:19]}")
        with middle:
            with st.popover("Edit", use_container_width=True):
                edited = st.text_area("Memory text", value=record.content, key=f"edit-text-{record.id}")
                if st.button("Save", key=f"save-{record.id}"):
                    if edit_memory(store, user_id, int(record.id), edited):
                        st.success("Memory updated")
                        st.rerun()
                    st.error("This memory is empty, sensitive, or contains a secret.")
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
    st.set_page_config(page_title="Asterline Research Assistant", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
      [data-testid="stAppViewContainer"] { background:#0d0d0d; color:#e8e8e8; }
      [data-testid="stSidebar"] { background:#111; border-right:1px solid #2a2a2a; }
      .asterline-kicker { color:#2fbf8f; font:600 0.72rem ui-monospace,monospace; letter-spacing:.12em; text-transform:uppercase; }
      .asterline-muted { color:#8b8b8b; }
      code, .stCode { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }
      div[data-testid="stChatMessage"] { border:1px solid #2a2a2a; border-radius:10px; margin:.65rem 0; padding:.3rem .45rem; }
      div[data-testid="stChatMessageContent"] { font-size:1rem; line-height:1.65; }
      [data-testid="stChatInput"] textarea { font-size:1rem !important; }
      [data-testid="stToolbar"] { visibility:hidden; }
      h1 { letter-spacing:-.025em; }
      @media (prefers-reduced-motion: no-preference) { div[data-testid="stChatMessage"] { animation:fade-in .18s ease-out; } @keyframes fade-in { from{opacity:.4;transform:translateY(2px)} to{opacity:1;transform:none} } }
    </style>
    """, unsafe_allow_html=True)
    settings = Settings.from_environment()
    store = _store()
    if "user" not in st.session_state:
        st.session_state.user = "alice"
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "resume_session" not in st.session_state:
        st.session_state.resume_session = st.session_state.session_id
    # The improved pipeline is the interactive default; baseline remains one click away.
    st.session_state.setdefault("ui_config", os.getenv("ASTERLINE_CONFIG", "improved").lower())
    with st.sidebar:
        st.markdown('<div class="asterline-kicker">Asterline Labs · research system</div>', unsafe_allow_html=True)
        st.title("Research controls")
        user_id = st.selectbox("User", ["alice", "bob", "carol"], index=["alice", "bob", "carol"].index(st.session_state.user))
        if user_id != st.session_state.user:
            st.session_state.user = user_id
            st.session_state.session_id = uuid.uuid4().hex
            st.session_state.resume_session = st.session_state.session_id
            st.rerun()
        config = st.selectbox("Config", ["baseline", "improved"], key="ui_config")
        provider_name = st.selectbox("Provider", ["fake", "gemini", "ollama"], index=0 if not os.getenv("GEMINI_API_KEY") else 1)
        if provider_name == "fake":
            st.info("DEMO MODE · deterministic FakeLLM")
        elif provider_name == "gemini" and not os.getenv("GEMINI_API_KEY"):
            st.warning("No Gemini key configured — switch to FakeLLM for offline demo mode.")
        else:
            st.caption(f"{provider_name} selected · availability checked on request")
        sessions = store.list_sessions(st.session_state.user)
        if st.session_state.session_id not in sessions:
            sessions.insert(0, st.session_state.session_id)
        selected_session = st.selectbox("Resume session", sessions, format_func=lambda value: value[:8], key="resume_session")
        if selected_session != st.session_state.session_id:
            st.session_state.session_id = selected_session
            st.rerun()
        st.button("New session", on_click=_new_session)
        with st.expander("Memory panel", expanded=True):
            _render_memory_panel(store, st.session_state.user)
    st.markdown('<div class="asterline-kicker">grounded answers · durable user context · explicit uncertainty</div>', unsafe_allow_html=True)
    st.title("Asterline Research Assistant")
    st.caption("Company facts come from retrieved corpus evidence. Saved context is never evidence.")
    messages = store.history(st.session_state.session_id)
    for message in messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🔎"):
            st.markdown(message["content"])
    if not messages:
        st.markdown("### Ask a grounded question")
        st.caption("Try: “What is the current Launch ingestion rate?” or “What is the Enterprise list price?”")
    question = st.chat_input("Ask about the supplied corpus")
    if question:
        store.record_message(st.session_state.user, st.session_state.session_id, "user", question)
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
        try:
            st.caption("Answer is shown after citation checks; provider and agent progress update while it runs.")
            with st.status("Running agents", expanded=True) as activity:
                agents = ["router", "memory_reader", "retriever", "evidence_analyst", "answer_writer", "verifier", "memory_writer"] if config == "improved" else ["router", "retriever", "answer_writer"]
                timeline = {agent: st.empty() for agent in agents} if config == "improved" else {}
                for index, agent in enumerate(agents):
                    if agent in timeline:
                        timeline[agent].write(f"{'◉' if index == 0 else '○'} {agent} · {'running' if index == 0 else 'pending'}")
                started = time.perf_counter()
                retriever = _retriever(str(settings.corpus_path), str(settings.chroma_path), str(settings.fastembed_cache_path))
                provider = make_provider(provider_name, settings.gemini_model, settings.ollama_model)
                provider_progress = st.empty()
                provider.status_callback = lambda message: provider_progress.caption(message)
                settings = settings.model_copy(update={"config": config})
                history = [*messages, {"role": "user", "content": question}]
                history_models = [type("History", (), item) for item in history]
                fn = run_improved_research if config == "improved" else run_research
                if config == "improved":
                    def on_agent_step(agent: str, elapsed: float) -> None:
                        chitchat = question.strip().lower() in {"hi", "hello", "thanks", "thank you"}
                        chitchat = chitchat or is_chitchat_question(question)
                        path = ["router", "answer_writer"] if chitchat else agents
                        agent_index = path.index(agent)
                        for index, name in enumerate(path):
                            if index < agent_index:
                                timeline[name].write(f"● {name} · done")
                            elif index == agent_index:
                                timeline[name].write(f"● {name} · done · {elapsed:.2f}s")
                            elif index == agent_index + 1:
                                timeline[name].write(f"◉ {name} · running")
                            else:
                                timeline[name].write(f"○ {name} · pending")
                        for name in set(agents) - set(path):
                            timeline[name].write(f"– {name} · skipped (chitchat) ")
                    response = fn(question, st.session_state.user, retriever, provider, store, history=history_models, on_step=on_agent_step)
                else:
                    st.write("Baseline path · hybrid retrieval → citation-checked answer")
                    response = fn(question, st.session_state.user, retriever, provider, store, history=history_models)
                elapsed = time.perf_counter() - started
                st.caption(f"agent graph completed in {elapsed:.2f}s")
                if response.provider_error:
                    activity.update(label="Sources only · answer not verified", state="complete", expanded=False)
                elif response.degraded:
                    activity.update(label="Completed · local fallback", state="complete", expanded=False)
                else:
                    activity.update(label="Agents complete", state="complete", expanded=False)
        except Exception as error:
            response = None
            st.error(f"The turn could not be completed cleanly ({error.__class__.__name__}). Try FakeLLM or retry later.")
        if response is None:
            st.stop()
        store.record_message(st.session_state.user, st.session_state.session_id, "assistant", response.answer)
        with st.chat_message("assistant", avatar="🔎"):
            st.markdown(response.answer)
            st.caption(f"{response.provider}/{response.model}" + (" · degraded" if response.degraded else ""))
            if response.provider_error:
                st.warning("Source-only result · no model-generated answer passed citation checks.")
                with st.expander("Provider diagnostics · sanitized"):
                    st.caption(_safe_provider_diagnostics(response))
            elif response.degraded:
                st.warning("Degraded mode · this answer came from the local fallback. Check the cited evidence below.")
                if response.primary_error:
                    with st.expander("Provider diagnostics · sanitized"):
                        st.caption(_safe_provider_diagnostics(response))
            with st.expander("Evidence & citations"):
                for item in response.evidence:
                    st.write(f"`{item.chunk.chunk_id}` — {item.chunk.title} · {item.chunk.published} · v{item.chunk.version}")
                    st.caption(item.chunk.text)
            if response.memory_used:
                with st.expander("Memory used this turn"):
                    st.write("From what you told me earlier:", "; ".join(response.memory_used))


if __name__ == "__main__":
    main()
