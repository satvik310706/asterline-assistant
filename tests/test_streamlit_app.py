from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

from asterline_assistant.memory import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_memory_panel_library_functions_are_user_scoped(tmp_path: Path) -> None:
    from app.streamlit_app import active_memories, delete_memory, export_memories, forget_all

    store = MemoryStore(tmp_path / "memory.sqlite3")
    alice = store.write_explicit("alice", "prefers concise answers")
    store.write_explicit("bob", "works on billing")
    assert len(active_memories(store, "alice")) == 1
    assert "works on billing" not in export_memories(store, "alice")
    assert delete_memory(store, "alice", int(alice.id))
    assert active_memories(store, "alice") == []
    assert forget_all(store, "bob") == 1


def test_streamlit_app_starts_in_fake_mode(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(ROOT / "app" / "streamlit_app.py").run(timeout=30)
    assert not app.exception
    assert any("User" in str(widget.label) for widget in app.selectbox)
    assert any("No active memories" in str(widget.value) for widget in app.caption)


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
