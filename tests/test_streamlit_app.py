from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

from asterline_assistant.memory import MemoryStore
from asterline_assistant.models import ResearchResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_memory_panel_library_functions_are_user_scoped(tmp_path: Path) -> None:
    from app.streamlit_app import active_memories, delete_memory, edit_memory, export_memories, forget_all

    store = MemoryStore(tmp_path / "memory.sqlite3")
    alice = store.write_explicit("alice", "prefers concise answers")
    store.write_explicit("bob", "works on billing")
    assert len(active_memories(store, "alice")) == 1
    assert "works on billing" not in export_memories(store, "alice")
    assert edit_memory(store, "alice", int(alice.id), "prefers two sentence answers")
    assert "two sentence" in export_memories(store, "alice")
    assert delete_memory(store, "alice", int(alice.id))
    assert active_memories(store, "alice") == []
    assert forget_all(store, "bob") == 1


def test_streamlit_app_starts_in_fake_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ASTERLINE_LLM_PROVIDER", "fake")
    monkeypatch.setenv("ASTERLINE_SQLITE_PATH", str(tmp_path / "startup.sqlite3"))
    app = AppTest.from_file(ROOT / "app" / "streamlit_app.py").run(timeout=30)
    assert not app.exception
    assert any("User" in str(widget.label) for widget in app.selectbox)
    assert app.selectbox[2].value == "fake"
    assert any("No active memories" in str(widget.value) for widget in app.caption)


def test_memory_panel_escapes_user_supplied_html(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    db_path = tmp_path / "memory-html.sqlite3"
    monkeypatch.setenv("ASTERLINE_SQLITE_PATH", str(db_path))
    store = MemoryStore(db_path)
    store.write_explicit("alice", "<script>alert('x')</script>")

    app = AppTest.from_file(ROOT / "app" / "streamlit_app.py").run(timeout=30)
    assert not app.exception
    rendered = " ".join(str(widget.value) for widget in app.markdown)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_new_session_stays_selected_until_first_message(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ASTERLINE_SQLITE_PATH", str(tmp_path / "sessions.sqlite3"))
    app = AppTest.from_file(ROOT / "app" / "streamlit_app.py").run(timeout=30)
    previous = app.session_state["session_id"]
    next(button for button in app.button if button.label == "New session").click().run(timeout=30)
    assert app.session_state["session_id"] != previous
    assert app.selectbox[3].value == app.session_state["session_id"]


def test_provider_diagnostics_are_sanitized() -> None:
    from app.streamlit_app import _safe_provider_diagnostics

    response = ResearchResponse(
        answer="", evidence=[], provider="unavailable", model="llama3.2:3b",
        error_class="ClientError", error_status=429, quota_type="quota",
        primary_error="raw reply: secret primary body",
        fallback_error="fallback: ollama/model failed: MissingCitation raw reply: secret fallback body",
    )
    details = _safe_provider_diagnostics(response)
    assert "HTTP 429" in details and "quota" in details and "MissingCitation" in details
    assert "secret" not in details and "raw reply" not in details


def test_verifier_abstention_is_not_mislabeled_as_missing_corpus_evidence() -> None:
    from app.streamlit_app import _guardrail_html

    response = ResearchResponse(
        answer="I could not verify the answer after one repair.", evidence=[],
        provider="gemini", model="test", abstained=True, error_class="VerifierFailure",
    )
    rendered = _guardrail_html(response, "What is this supported fact?")

    assert "verifier check failed" in rendered
    assert "not found in corpus" not in rendered


def test_ui_provider_default_honors_fake_mode_and_missing_keys() -> None:
    from app.streamlit_app import _default_provider, _effective_provider

    assert _default_provider(True, "fake") == "fake"
    assert _default_provider(False, "auto") == "fake"
    assert _default_provider(True, "auto") == "gemini"
    assert _effective_provider("gemini", False) == "fake"


def test_streamlit_chat_remember_and_user_isolation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ASTERLINE_SQLITE_PATH", str(tmp_path / "ui.sqlite3"))
    app = AppTest.from_file(ROOT / "app" / "streamlit_app.py").run(timeout=30)
    # Provider is the third selectbox in the sidebar: user, config, provider.
    app.selectbox[2].set_value("fake")
    app.chat_input[0].set_value("remember I prefer concise answers")
    app.run(timeout=30)
    assert "concise answers" in " ".join(str(widget.value) for widget in app.markdown)
    app.selectbox[0].set_value("bob")
    app.run(timeout=30)
    assert "concise answers" not in " ".join(str(widget.value) for widget in app.markdown)


def test_streamlit_fake_question_shows_citation_evidence(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ASTERLINE_SQLITE_PATH", str(tmp_path / "question.sqlite3"))
    app = AppTest.from_file(ROOT / "app" / "streamlit_app.py").run(timeout=30)
    app.selectbox[2].set_value("fake")
    app.chat_input[0].set_value("What is the current Launch ingestion rate?")
    app.run(timeout=60)
    assert not app.exception
    rendered = " ".join(str(widget.value) for widget in [*app.markdown, *app.caption])
    assert "300" in rendered
    assert any("Evidence & citations" in str(widget.label) for widget in app.expander)


def test_streamlit_does_not_claim_unrelated_saved_context_was_used(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    db_path = tmp_path / "unrelated-memory.sqlite3"
    monkeypatch.setenv("ASTERLINE_SQLITE_PATH", str(db_path))
    store = MemoryStore(db_path)
    store.write_explicit("alice", "My primary project is Apollo on the Team plan")

    app = AppTest.from_file(ROOT / "app" / "streamlit_app.py").run(timeout=30)
    app.selectbox[2].set_value("fake")
    app.chat_input[0].set_value("What is the current Launch ingestion rate?")
    app.run(timeout=60)

    assert not app.exception
    rendered = " ".join(str(widget.value) for widget in [*app.markdown, *app.caption])
    assert "300" in rendered
    assert not any("Memory used this turn" in str(widget.label) for widget in app.expander)


def test_streamlit_improved_mode_runs_agent_graph(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ASTERLINE_SQLITE_PATH", str(tmp_path / "improved-ui.sqlite3"))
    monkeypatch.setenv("ASTERLINE_CONFIG", "improved")
    app = AppTest.from_file(ROOT / "app" / "streamlit_app.py").run(timeout=30)
    app.selectbox[2].set_value("fake")
    app.chat_input[0].set_value("What is the current Launch ingestion rate?")
    app.run(timeout=60)
    assert not app.exception
    assert "300" in " ".join(str(widget.value) for widget in app.markdown)
    assert any("Agents complete" in str(widget.label) for widget in app.status)


def test_streamlit_improved_greeting_uses_short_chitchat_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ASTERLINE_SQLITE_PATH", str(tmp_path / "greeting-ui.sqlite3"))
    monkeypatch.setenv("ASTERLINE_CONFIG", "improved")
    app = AppTest.from_file(ROOT / "app" / "streamlit_app.py").run(timeout=30)
    app.selectbox[2].set_value("fake")
    app.chat_input[0].set_value("hello")
    app.run(timeout=30)
    assert not app.exception
    rendered = " ".join(str(widget.value) for widget in [*app.markdown, *app.caption])
    assert "What would you like to know?" in rendered
    assert "rules/deterministic" in rendered
    assert "on-call runbook" not in rendered
    assert any("Agents complete" in str(widget.label) for widget in app.status)


def test_streamlit_role_and_chunk_count_is_fast_and_grounded_in_inventory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ASTERLINE_SQLITE_PATH", str(tmp_path / "meta-question.sqlite3"))
    monkeypatch.setenv("ASTERLINE_CONFIG", "improved")
    app = AppTest.from_file(ROOT / "app" / "streamlit_app.py").run(timeout=30)
    app.chat_input[0].set_value("hi, what is your role here? and how many chunks do you have")
    app.run(timeout=60)
    assert not app.exception
    rendered = " ".join(str(widget.value) for widget in [*app.markdown, *app.caption])
    assert "research assistant" in rendered.lower()
    assert "75 chunks across 25 documents" in rendered
    assert "rules/deterministic" in rendered
