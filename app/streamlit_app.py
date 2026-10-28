"""Asterline Research Assistant — Codex-style UI.

Production-quality Streamlit frontend:
- Codex / Linear inspired dark design system
- Permanent sidebar toggle button (open/close) visible in all states
- Header with high-contrast typography, never clipped by viewport
- Live agent progress; answer text appears only after verification completes
- Instant fallback without long sleeps when upstream quotas are exhausted
- Sender headers for both User and Assistant ("Alice [USER]" / "Asterline Assistant [RESEARCH AGENT]")
- Interactive starter question buttons (clicking immediately runs the query)
- Welcome card smoothly disappears once messages exist
- Clean single-row verification guardrail badges
- Comprehensive user-scoped memory panel (edit, delete, export JSON, forget all)
- Zero secrets or raw error payloads surfaced in UI or logs
"""

from __future__ import annotations

import json
import html
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
from asterline_assistant.models import ResearchResponse
from asterline_assistant.research import run_research
from asterline_assistant.retrieval import HybridRetriever

# ─── Module-level helpers (maintained for test backward-compatibility) ────────

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


def _safe_provider_diagnostics(response: ResearchResponse) -> str:
    """Expose error class and status only — never raw model or SDK text."""
    primary = response.error_class or "provider failure"
    status = f"HTTP {response.error_status}" if response.error_status is not None else "status unavailable"
    quota = f", {response.quota_type} quota" if response.quota_type else ""
    parts = [f"primary: {primary} ({status}{quota})"]
    if response.fallback_error:
        match = re.search(r"failed:\s*([A-Za-z][A-Za-z0-9_]*)", response.fallback_error)
        if match:
            parts.append(f"fallback: {match.group(1)}")
    return " · ".join(parts)


def _default_provider(has_gemini: bool, configured: str) -> str:
    """Honor an explicit provider/demo choice, otherwise choose a usable default."""
    preferred = configured.strip().lower()
    if preferred in {"fake", "ollama"}:
        return preferred
    if preferred == "gemini" and has_gemini:
        return "gemini"
    return "gemini" if has_gemini else "fake"


def _effective_provider(provider_name: str, has_gemini: bool) -> str:
    """No Gemini credentials means the offline demo, not an unexpected Ollama call."""
    return "fake" if provider_name == "gemini" and not has_gemini else provider_name


def _safe_diagnostics(response: ResearchResponse) -> str:
    return _safe_provider_diagnostics(response)


# ─── Design System CSS ────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Reset & Global Typography ───────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
  background: #0d0d0d !important;
  color: #e8e8e8 !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
}

/* Hide default menu and deploy button, but KEEP header and sidebar toggle button! */
#MainMenu { display: none !important; }
[data-testid="stAppDeployButton"] { display: none !important; }
footer { display: none !important; }
header [data-testid="stToolbarActions"] { display: none !important; }

/* ── Header: Transparent and Non-Blocking ─────────────────────────────────── */
[data-testid="stHeader"],
header.stAppHeader {
  background: transparent !important;
  pointer-events: none !important;
  z-index: 9999 !important;
}

/* Ensure all buttons inside header remain clickable */
[data-testid="stHeader"] button,
[data-testid="stToolbar"] button,
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
  pointer-events: auto !important;
}

/* ── Always Visible Glowing Sidebar Toggle Buttons ────────────────────────── */
[data-testid="stExpandSidebarButton"],
button[aria-label="Expand sidebar"],
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  position: fixed !important;
  top: 14px !important;
  left: 14px !important;
  z-index: 999999 !important;
  background: #161616 !important;
  border: 1px solid #2fbf8f !important;
  border-radius: 8px !important;
  color: #b9f0da !important;
  cursor: pointer !important;
  padding: 6px 10px !important;
  box-shadow: 0 0 12px rgba(47, 191, 143, 0.18) !important;
  transition: all 0.2s ease !important;
}

[data-testid="stSidebarCollapseButton"],
button[aria-label="Collapse sidebar"],
[data-testid="stSidebarHeader"] button {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  background: #161616 !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 8px !important;
  color: #b9f0da !important;
  cursor: pointer !important;
  padding: 6px 8px !important;
  box-shadow: 0 0 10px rgba(47, 191, 143, 0.14) !important;
  transition: all 0.2s ease !important;
}

[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapseButton"]:hover,
button[aria-label="Expand sidebar"]:hover,
button[aria-label="Collapse sidebar"]:hover {
  background: #173b30 !important;
  border-color: #2fbf8f !important;
  color: #ffffff !important;
  transform: scale(1.08) !important;
  box-shadow: 0 0 16px rgba(47, 191, 143, 0.28) !important;
}

[data-testid="stSidebarHeader"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  padding: 0.6rem 0.8rem !important;
}

/* ── Main Container (Centered & Protected Top Padding) ──────────────── */
.block-container {
  max-width: 960px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 4.8rem 2rem 6rem !important;
  width: 100% !important;
}

/* ── Sidebar Styling ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #0d0d0d !important;
  border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.09em !important;
  text-transform: uppercase !important;
  color: #8b8b8b !important;
  margin: 0.8rem 0 0.3rem !important;
}

[data-testid="stSelectbox"] > div > div {
  background: #161616 !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 8px !important;
  color: #e8e8e8 !important;
  font-size: 0.85rem !important;
}
[data-testid="stSelectbox"] > div > div:hover {
  border-color: #2fbf8f !important;
}

.stButton > button {
  background: #161616 !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 8px !important;
  color: #e8e8e8 !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  padding: 0.4rem 0.9rem !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  background: #1d1d1d !important;
  border-color: #2fbf8f !important;
  color: #ffffff !important;
}

/* ── Top Nav / Header Card ──────────────────────────────────────────── */
.top-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #161616;
  border: 1px solid #2a2a2a;
  border-radius: 12px;
  padding: 1.1rem 1.4rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  gap: 0.85rem;
  box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}
.brand-box {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}
.brand-icon {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, #176a4d 0%, #2fbf8f 100%);
  border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; font-weight: 800; color: #ffffff;
  box-shadow: 0 4px 12px rgba(47, 191, 143, 0.22);
  flex-shrink: 0;
}
.brand-name {
  font-size: 1.35rem !important;
  font-weight: 750 !important;
  color: #ffffff !important;
  letter-spacing: -0.015em !important;
  line-height: 1.25 !important;
}
.brand-sub {
  font-size: 0.8rem !important;
  color: #94a3b8 !important;
  margin-top: 0.2rem !important;
}
.nav-pills {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.nav-pill {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 6px;
  background: #161616;
  border: 1px solid #2a2a2a;
  color: #94a3b8;
}
.nav-pill strong { color: #f1f5f9; }
.nav-pill-active {
  border-color: #2fbf8f;
  color: #73ddb5;
}

/* ── Chat Message Containers & Senders ──────────────────────────────── */
div[data-testid="stChatMessage"] {
  background: #161616 !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 12px !important;
  margin: 0.85rem 0 !important;
  padding: 1rem 1.25rem !important;
  box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
}

div[data-testid="stChatMessageContent"] {
  font-size: 0.98rem !important;
  line-height: 1.75 !important;
  color: #e8e8e8 !important;
}

.sender-header {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.5rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sender-name-user {
  font-size: 0.9rem;
  font-weight: 700;
  color: #60a5fa;
}
.sender-name-assistant {
  font-size: 0.9rem;
  font-weight: 700;
  color: #c084fc;
}
.sender-badge {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 2px 7px;
  border-radius: 4px;
}
.badge-user {
  background: #172554;
  color: #93c5fd;
  border: 1px solid #1e3a8a;
}
.badge-assistant {
  background: #2e1065;
  color: #d8b4fe;
  border: 1px solid #4c1d95;
}
.badge-provider {
  margin-left: auto;
  font-family: monospace;
  font-size: 0.7rem;
  color: #64748b;
}

/* ── Welcome & Interactive Starter Box ──────────────────────────────── */
.welcome-container {
  background: #0f0f1c;
  border: 1px solid #24243a;
  border-radius: 14px;
  padding: 1.75rem 2rem;
  margin: 1.2rem 0;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.welcome-heading {
  font-size: 1.2rem;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 0.35rem;
}
.welcome-desc {
  font-size: 0.86rem;
  color: #94a3b8;
  line-height: 1.6;
  margin-bottom: 1.1rem;
}
.welcome-prompt-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #64748b;
  margin-bottom: 0.5rem;
}

/* ── Chat Input ─────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
  background: #12121e !important;
  border: 1px solid #2d2d44 !important;
  border-radius: 14px !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: #2fbf8f !important;
  box-shadow: 0 0 0 3px rgba(47, 191, 143, 0.16) !important;
}
[data-testid="stChatInput"] textarea {
  font-size: 0.95rem !important;
  color: #f1f5f9 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #64748b !important;
}

/* ── Guardrails Row ─────────────────────────────────────────────────── */
.guardrail-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.65rem 0 0.25rem;
  padding-top: 0.5rem;
  border-top: 1px solid rgba(255,255,255,0.06);
}
.pill-ok {
  display: inline-flex; align-items: center; gap: 4px;
  background: #062b14; border: 1px solid #0f5132;
  border-radius: 5px; padding: 2px 8px;
  font-size: 0.72rem; font-weight: 600; color: #4ade80;
}
.pill-warn {
  display: inline-flex; align-items: center; gap: 4px;
  background: #2b1f06; border: 1px solid #664d03;
  border-radius: 5px; padding: 2px 8px;
  font-size: 0.72rem; font-weight: 600; color: #fbbf24;
}
.pill-fail {
  display: inline-flex; align-items: center; gap: 4px;
  background: #2c0b0e; border: 1px solid #842029;
  border-radius: 5px; padding: 2px 8px;
  font-size: 0.72rem; font-weight: 600; color: #f87171;
}

/* ── Evidence Cards ─────────────────────────────────────────────────── */
.evidence-card {
  background: #11111d;
  border: 1px solid #232338;
  border-radius: 8px;
  padding: 0.75rem 1rem;
  margin: 0.45rem 0;
}
.evidence-card-header {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.35rem;
}
.evidence-chip {
  background: #1e1e36;
  border: 1px solid #373760;
  border-radius: 4px;
  padding: 2px 7px;
  font-family: monospace;
  font-size: 0.74rem;
  font-weight: 600;
  color: #a5b4fc;
}
.evidence-title { color: #e2e8f0; font-size: 0.85rem; font-weight: 600; }
.evidence-meta { color: #64748b; font-size: 0.72rem; margin-left: auto; }
.evidence-text { color: #94a3b8; font-size: 0.82rem; line-height: 1.55; }

/* ── Memory Cards ───────────────────────────────────────────────────── */
.memory-card {
  background: #10101c;
  border: 1px solid #222238;
  border-radius: 8px;
  padding: 0.65rem 0.85rem;
  margin: 0.35rem 0;
  display: flex;
  align-items: flex-start;
  gap: 0.65rem;
}
.memory-pill {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border-radius: 4px;
  padding: 2px 6px;
  flex-shrink: 0;
  margin-top: 2px;
}
.memory-type-preference { background: #172554; color: #93c5fd; border: 1px solid #1e3a8a; }
.memory-type-project    { background: #064e3b; color: #6ee7b7; border: 1px solid #065f46; }
.memory-type-task       { background: #7c2d12; color: #fdba74; border: 1px solid #9a3412; }
.memory-type-user_claim { background: #4c1d95; color: #d8b4fe; border: 1px solid #581c87; }
.memory-content { color: #cbd5e1; font-size: 0.84rem; line-height: 1.5; flex: 1; }
.memory-ts { color: #64748b; font-size: 0.7rem; margin-top: 0.2rem; }

/* ── Agent Pipeline ─────────────────────────────────────────────────── */
.agent-row {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  padding: 0.35rem 0;
  font-size: 0.82rem;
}
.agent-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.agent-dot-done    { background: #10b981; }
.agent-dot-running { background: #2fbf8f; animation: pulse 1s infinite; }
.agent-dot-pending { background: #334155; }
.agent-dot-skipped { background: #1e293b; opacity: 0.5; }

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(1.3); }
}

.agent-name { color: #94a3b8; font-weight: 500; min-width: 120px; }
.agent-status-done    { color: #34d399; font-weight: 600; }
.agent-status-running { color: #73ddb5; font-weight: 600; }
.agent-status-pending { color: #475569; }
.agent-status-skipped { color: #334155; font-style: italic; }
.agent-time { color: #64748b; font-size: 0.74rem; margin-left: auto; font-family: monospace; }

/* ── Blinking Cursor ────────────────────────────────────────────────── */
/* ── Scrollbars ─────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #262638; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3b3b55; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
}
</style>
"""

# ─── Constants ────────────────────────────────────────────────────────────────

USERS = ["alice", "bob", "carol"]
SAMPLE_QUESTIONS = [
    "What is the current Launch ingestion rate?",
    "What changed for cold recovery storage?",
    "How much is an extra Team seat per month?",
    "What happened in INC-2025-08?",
    "Does Enterprise have an SLA?",
    "What is the AQL timeout in version 2.3?",
]

AGENT_ICONS = {
    "router": "⊙",
    "memory_reader": "◈",
    "retriever": "⊗",
    "evidence_analyst": "⊕",
    "answer_writer": "◎",
    "verifier": "⊛",
    "memory_writer": "◉",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _store() -> MemoryStore:
    return MemoryStore(Settings.from_environment().sqlite_path)


@st.cache_resource(show_spinner=False)
def _retriever(corpus_file: str, chroma_dir: str, embedding_cache: str) -> HybridRetriever:
    corpus_path = Path(corpus_file)
    validate_corpus_integrity(corpus_path)
    return HybridRetriever(load_corpus(corpus_path), Path(chroma_dir), Path(embedding_cache))


def _new_session() -> None:
    sid = uuid.uuid4().hex
    st.session_state.session_id = sid
    st.session_state.resume_session = sid


def _guardrail_html(response: ResearchResponse, question: str) -> str:
    badges = []
    chitchat = is_chitchat_question(question)

    if chitchat:
        badges.append('<span class="pill-ok">✓ greeting / app role</span>')
        badges.append('<span class="pill-ok">✓ deterministic answer</span>')
    else:
        # Citations
        if response.provider_error:
            badges.append('<span class="pill-fail">⊘ citation check failed</span>')
        elif response.abstained:
            label = "answer withheld · verifier check failed" if response.error_class == "VerifierFailure" else "abstained · not found in corpus"
            badges.append(f'<span class="pill-warn">⊘ {label}</span>')
        else:
            badges.append('<span class="pill-ok">✓ citations verified</span>')

        # Evidence
        if response.evidence:
            badges.append(f'<span class="pill-ok">✓ {len(response.evidence)} sources retrieved</span>')
        else:
            badges.append('<span class="pill-warn">⊘ no corpus sources</span>')

    # Isolation
    badges.append('<span class="pill-ok">✓ user isolation enforced</span>')

    # Model pill
    model_cls = "pill-warn" if response.degraded else ("pill-fail" if response.provider_error else "pill-ok")
    badges.append(f'<span class="{model_cls}">● {response.provider}/{response.model}</span>')

    return '<div class="guardrail-row">' + "".join(badges) + "</div>"


def _evidence_html(item) -> str:
    chunk = item.chunk
    score = f"{item.rrf_score:.3f}" if item.rrf_score else "—"
    return f"""
<div class="evidence-card">
  <div class="evidence-card-header">
    <span class="evidence-chip">{chunk.chunk_id}</span>
    <span class="evidence-title">{chunk.title}</span>
    <span class="evidence-meta">v{chunk.version} · {chunk.published} · RRF {score}</span>
  </div>
  <div class="evidence-text">{chunk.text[:340]}{'…' if len(chunk.text) > 340 else ''}</div>
</div>
"""


# ─── Memory Panel ─────────────────────────────────────────────────────────────

def _render_memory_panel(store: MemoryStore, user_id: str) -> None:
    records = active_memories(store, user_id)
    if records:
        for record in records:
            pill_cls = f"memory-type-{record.type}"
            st.markdown(
                f'<div class="memory-card">'
                f'<span class="memory-pill {pill_cls}">{record.type}</span>'
                f'<div><div class="memory-content">{html.escape(record.content)}</div>'
                f'<div class="memory-ts">saved {record.created_at[:16]} UTC</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            col_edit, col_del = st.columns([1, 1])
            with col_edit:
                with st.popover("Edit", use_container_width=True):
                    edited = st.text_area("Memory text", value=record.content, key=f"edit-{record.id}")
                    if st.button("Save", key=f"save-{record.id}"):
                        if edit_memory(store, user_id, int(record.id), edited):
                            st.success("Updated")
                            st.rerun()
                        else:
                            st.error("Blocked: empty, sensitive, or contains a secret.")
            with col_del:
                if st.button("Delete", key=f"del-{record.id}"):
                    delete_memory(store, user_id, int(record.id))
                    st.rerun()
    else:
        st.caption("No active memories.")

    st.markdown("---")
    col_exp, col_forget = st.columns([1, 1])
    with col_exp:
        st.download_button(
            "Export JSON",
            export_memories(store, user_id),
            file_name=f"{user_id}-memories.json",
            mime="application/json",
        )
    with col_forget:
        if st.button("Forget all", key="forget-all"):
            forget_all(store, user_id)
            st.rerun()


# ─── Sidebar Controls ─────────────────────────────────────────────────────────

def _render_sidebar(store: MemoryStore) -> tuple[str, str, str]:
    with st.sidebar:
        st.markdown(
            '<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#2fbf8f;margin-bottom:0.8rem">'
            '◈ Asterline Labs · Research System'
            '</div>',
            unsafe_allow_html=True,
        )

        # 1. User (index 0)
        st.markdown("### User")
        user_id = st.selectbox(
            "User",
            USERS,
            index=USERS.index(st.session_state.get("user", "alice")),
            key="user_select",
        )
        if user_id != st.session_state.get("user", "alice"):
            st.session_state.user = user_id
            st.session_state.session_id = uuid.uuid4().hex
            st.session_state.resume_session = st.session_state.session_id
            st.rerun()

        # 2. Config (index 1)
        st.markdown("### Pipeline")
        config = st.selectbox(
            "Config",
            ["improved", "baseline"],
            key="ui_config",
        )

        # 3. Provider (index 2 for test compatibility)
        st.markdown("### Provider")
        has_gemini = bool(os.getenv("GEMINI_API_KEY"))
        provider_options = ["fake", "gemini", "ollama"]
        preferred_provider = _default_provider(has_gemini, os.getenv("ASTERLINE_LLM_PROVIDER", "auto"))
        default_idx = provider_options.index(preferred_provider)
        provider_name = st.selectbox(
            "Provider",
            provider_options,
            index=default_idx,
            key="provider_select",
        )
        if provider_name == "fake":
            st.info("DEMO MODE · deterministic FakeLLM (Instant)")
        elif provider_name == "gemini" and not has_gemini:
            st.warning("No Gemini key configured — using FakeLLM.")
        else:
            st.caption(f"{provider_name} selected · availability checked on request")

        # 4. Session (index 3 for test compatibility)
        st.markdown("### Resume session")
        sessions = store.list_sessions(st.session_state.get("user", "alice"))
        if st.session_state.session_id not in sessions:
            sessions.insert(0, st.session_state.session_id)
        selected_session = st.selectbox(
            "Resume session",
            sessions,
            format_func=lambda v: v[:8],
            key="resume_session",
        )
        if selected_session != st.session_state.session_id:
            st.session_state.session_id = selected_session
            st.rerun()

        st.button("New session", on_click=_new_session, use_container_width=True)

        st.markdown("---")
        with st.expander("Memory panel", expanded=True):
            _render_memory_panel(store, st.session_state.get("user", "alice"))

    return user_id, config, provider_name


# ─── Main Application ─────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Asterline Research Assistant",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    settings = Settings.from_environment()
    store = _store()

    if "user" not in st.session_state:
        st.session_state.user = "alice"
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "resume_session" not in st.session_state:
        st.session_state.resume_session = st.session_state.session_id
    st.session_state.setdefault("ui_config", os.getenv("ASTERLINE_CONFIG", "improved").lower())

    user_id, config, provider_name = _render_sidebar(store)

    # ── Top Nav Header (Never clipped, clearly visible even when sidebar collapses) ───
    st.markdown(
        f'<div class="top-nav">'
        f'  <div class="brand-box">'
        f'    <div class="brand-icon">◈</div>'
        f'    <div>'
        f'      <div class="brand-name">Asterline Research Assistant</div>'
        f'      <div class="brand-sub">Grounded answers from supplied corpus · Evidence-cited · Isolated memory</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="nav-pills">'
        f'    <span class="nav-pill nav-pill-active">👤 User: <strong>{user_id.capitalize()}</strong></span>'
        f'    <span class="nav-pill">◷ Session: <strong>{st.session_state.session_id[:8]}</strong></span>'
        f'    <span class="nav-pill">⚙ <strong>{config.upper()}</strong></span>'
        f'    <span class="nav-pill">● <strong>{provider_name}</strong></span>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Chat history ──────────────────────────────────────────────────────────
    messages = store.history(st.session_state.session_id)
    for msg in messages:
        is_user = msg["role"] == "user"
        avatar = "👤" if is_user else "🔎"
        with st.chat_message(msg["role"], avatar=avatar):
            if is_user:
                st.markdown(
                    f'<div class="sender-header">'
                    f'  <span class="sender-name-user">{user_id.capitalize()}</span>'
                    f'  <span class="sender-badge badge-user">USER</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="sender-header">'
                    f'  <span class="sender-name-assistant">Asterline Assistant</span>'
                    f'  <span class="sender-badge badge-assistant">RESEARCH AGENT</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])

    # ── Check for pending starter question ────────────────────────────────────
    pending_starter = st.session_state.pop("pending_starter", None)

    # ── Welcome & Interactive Starter Card (hidden once messages exist) ──────
    welcome_slot = st.empty()
    if not messages and not pending_starter:
        with welcome_slot.container():
            st.markdown(
                '<div class="welcome-container">'
                '  <div class="welcome-heading">Ask a grounded question</div>'
                '  <div class="welcome-desc">'
                '    All company facts come strictly from retrieved corpus evidence.<br>'
                '    Saved context is isolated per user and treated as unverified user context, never as company fact.'
                '  </div>'
                '  <div class="welcome-prompt-label">Select a sample question to try:</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(2)
            for idx, q_text in enumerate(SAMPLE_QUESTIONS):
                col = cols[idx % 2]
                if col.button(q_text, key=f"sample_q_{idx}", use_container_width=True):
                    st.session_state.pending_starter = q_text
                    st.rerun()

    # ── Input Area ────────────────────────────────────────────────────────────
    chat_input_val = st.chat_input("Ask about the supplied corpus…")
    question = pending_starter or chat_input_val

    if not question:
        return

    # Clear welcome slot if question was just submitted
    welcome_slot.empty()

    # Record & render user message immediately
    store.record_message(user_id, st.session_state.session_id, "user", question)
    with st.chat_message("user", avatar="👤"):
        st.markdown(
            f'<div class="sender-header">'
            f'  <span class="sender-name-user">{user_id.capitalize()}</span>'
            f'  <span class="sender-badge badge-user">USER</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(question)

    # ── Render Assistant Message Container IMMEDIATELY ────────────────────────
    with st.chat_message("assistant", avatar="🔎"):
        st.markdown(
            f'<div class="sender-header">'
            f'  <span class="sender-name-assistant">Asterline Assistant</span>'
            f'  <span class="sender-badge badge-assistant">RESEARCH AGENT</span>'
            f'  <span class="sender-badge badge-provider">{provider_name}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        status = st.status("Analyzing query and evidence…", expanded=False)
        answer_placeholder = st.empty()

        retriever = _retriever(
            str(settings.corpus_path),
            str(settings.chroma_path),
            str(settings.fastembed_cache_path),
        )
        provider = make_provider(
            _effective_provider(provider_name, bool(os.getenv("GEMINI_API_KEY"))),
            settings.gemini_model,
            settings.ollama_model,
        )

        settings = settings.model_copy(update={"config": config})
        history_models = [type("History", (), item) for item in messages]

        started = time.perf_counter()

        # Execute Pipeline
        try:
            if config == "improved":
                def on_step(agent: str, elapsed: float) -> None:
                    status.write(f"{AGENT_ICONS.get(agent, '○')} {agent} ({elapsed:.2f}s)")

                response = run_improved_research(
                    question, user_id, retriever, provider, store,
                    history=history_models, on_step=on_step,
                )
            else:
                response = run_research(
                    question, user_id, retriever, provider, store,
                    history=history_models,
                )

            elapsed = time.perf_counter() - started

            # Never expose model tokens before the graph's citation and grounding checks.
            answer_placeholder.markdown(response.answer)

            # Update status widget to complete
            status.update(label=f"Agents complete · {elapsed:.2f}s", state="complete", expanded=False)

        except Exception as exc:
            status.update(label="Turn failed", state="error", expanded=False)
            st.error(f"Turn could not be completed ({exc.__class__.__name__}). Try switching provider to FakeLLM.")
            return

        if response is None:
            return

        store.record_message(user_id, st.session_state.session_id, "assistant", response.answer)

        # Single Guardrail Badge Row
        st.markdown(_guardrail_html(response, question), unsafe_allow_html=True)

        # Subtle Model Caption for test harness verification
        st.caption(f"{response.provider}/{response.model}" + (" · degraded" if response.degraded else ""))

        # Provider Diagnostics (if degraded / error)
        if response.provider_error:
            with st.expander("⚠ Provider diagnostics (sanitized)"):
                st.caption(_safe_provider_diagnostics(response))
        elif response.degraded and response.primary_error:
            with st.expander("⚠ Degraded mode (local fallback)"):
                st.caption(_safe_provider_diagnostics(response))

        # Evidence & Citations
        if response.evidence:
            with st.expander(f"Evidence & citations ({len(response.evidence)} sources)"):
                for item in response.evidence:
                    st.markdown(_evidence_html(item), unsafe_allow_html=True)

        # Memory Used
        if response.memory_used:
            with st.expander(f"Memory used this turn ({len(response.memory_used)})"):
                st.caption("Saved user context is never company proof.")
                for mem in response.memory_used:
                    st.markdown(
                        f'<div class="memory-card">'
                        f'  <span class="memory-pill memory-type-user_claim">context</span>'
                        f'  <div class="memory-content">{html.escape(mem)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
