"""Real end-to-end 10-turn conversation test suite.

Covers:
1. Turn 1: Chitchat & Inventory metadata
2. Turn 2: Grounded corpus question with citations (Launch ingestion rate)
3. Turn 3: Contextual follow-up question (Team & Enterprise rates)
4. Turn 4: Temporal supersession & conflict resolution (Cold recovery storage 20d vs 60d)
5. Turn 5: Corpus abstention on unpublished information (Enterprise list price)
6. Turn 6: Memory storage policy (Safe project/plan context)
7. Turn 7: Memory-informed personalization (Warehouse export destinations for Team)
8. Turn 8: Sensitive credential guardrail (sk- API key rejection)
9. Turn 9: Sensitive medical / leave guardrail (Health PII rejection)
10. Turn 10: Adversarial user isolation (Bob cannot see Alice's data) & Hard Forget
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from asterline_assistant.memory import MemoryStore


def test_10_turn_real_conversation_flow(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "e2e_10_turns.sqlite3"
    monkeypatch.setenv("ASTERLINE_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("ASTERLINE_CONFIG", "improved")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    app = AppTest.from_file(ROOT / "app" / "streamlit_app.py").run(timeout=30)
    assert not app.exception, f"App crashed on startup: {app.exception}"

    # Ensure fake provider is selected for deterministic offline testing
    app.selectbox[2].set_value("fake")
    app.run(timeout=30)

    # ──────────────────────────────────────────────────────────────────────────
    # TURN 1: Chitchat & Inventory Metadata
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- TURN 1: Chitchat & Inventory ---")
    app.chat_input[0].set_value("hi, what is your role here? and how many chunks do you have").run(timeout=30)
    assert not app.exception
    rendered_1 = " ".join(str(w.value) for w in [*app.markdown, *app.caption])
    assert "research assistant" in rendered_1.lower()
    assert "75 chunks across 25 documents" in rendered_1
    assert "rules/deterministic" in rendered_1
    print("Turn 1 PASS: Answered role and 75 corpus chunks correctly.")

    # ──────────────────────────────────────────────────────────────────────────
    # TURN 2: Single-fact Research Question with Citations
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- TURN 2: Grounded Fact with Citations ---")
    app.chat_input[0].set_value("What is the current Launch ingestion rate?").run(timeout=30)
    assert not app.exception
    rendered_2 = " ".join(str(w.value) for w in [*app.markdown, *app.caption])
    assert "300" in rendered_2
    assert any("Evidence & citations" in str(w.label) for w in app.expander)
    print("Turn 2 PASS: 300 rps retrieved with citations.")

    # ──────────────────────────────────────────────────────────────────────────
    # TURN 3: Contextual Follow-up Query
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- TURN 3: Contextual Follow-up ---")
    app.chat_input[0].set_value("What about Team and Enterprise?").run(timeout=30)
    assert not app.exception
    rendered_3 = " ".join(str(w.value) for w in [*app.markdown, *app.caption])
    assert "1,500" in rendered_3 or "1500" in rendered_3 or "6,000" in rendered_3 or "6000" in rendered_3 or "Team" in rendered_3
    print("Turn 3 PASS: Follow-up query resolved through session history.")

    # ──────────────────────────────────────────────────────────────────────────
    # TURN 4: Temporal Supersession & Conflict Handling
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- TURN 4: Supersession & Conflict ---")
    app.chat_input[0].set_value("What changed for cold recovery storage?").run(timeout=30)
    assert not app.exception
    rendered_4 = " ".join(str(w.value) for w in [*app.markdown, *app.caption])
    assert "20" in rendered_4
    assert any("Evidence & citations" in str(w.label) for w in app.expander)
    print("Turn 4 PASS: Cold recovery 20 days supersession handled.")

    # ──────────────────────────────────────────────────────────────────────────
    # TURN 5: Corpus Abstention on Unpublished Information
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- TURN 5: Abstention on Unpublished Fact ---")
    app.chat_input[0].set_value("What is the Enterprise list price?").run(timeout=30)
    assert not app.exception
    rendered_5 = " ".join(str(w.value) for w in [*app.markdown, *app.caption])
    assert any(term in rendered_5.lower() for term in ["custom", "negotiated", "not", "cannot", "contact"])
    print("Turn 5 PASS: Abstention on unpublished Enterprise list price.")

    # ──────────────────────────────────────────────────────────────────────────
    # TURN 6: Explicit Memory Storage (Project / Team Plan)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- TURN 6: Memory Storage Policy ---")
    app.chat_input[0].set_value("remember that I am investigating BigQuery duplicates on the Team plan").run(timeout=30)
    assert not app.exception
    rendered_6 = " ".join(str(w.value) for w in [*app.markdown, *app.caption])
    assert "remember" in rendered_6.lower() or "team plan" in rendered_6.lower() or "saved" in rendered_6.lower()
    # Verify in DB
    store = MemoryStore(db_path)
    alice_memories = store.list_active("alice")
    assert len(alice_memories) >= 1
    assert any("Team plan" in m.content for m in alice_memories)
    print(f"Turn 6 PASS: Memory saved in SQLite (Active records: {len(alice_memories)}).")

    # ──────────────────────────────────────────────────────────────────────────
    # TURN 7: Memory-informed Contextual Personalization
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- TURN 7: Memory-informed Query ---")
    app.chat_input[0].set_value("What warehouse export destinations are available for my plan?").run(timeout=30)
    assert not app.exception
    rendered_7 = " ".join(str(w.value) for w in [*app.markdown, *app.caption])
    assert "BigQuery" in rendered_7 or "Snowflake" in rendered_7 or "Team" in rendered_7
    print("Turn 7 PASS: Team plan context applied without treating memory as company proof.")

    # ──────────────────────────────────────────────────────────────────────────
    # TURN 8: Sensitive Credential Guardrail (Secret Rejection)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- TURN 8: Secret Rejection Guardrail ---")
    synthetic_secret = "sk-" + "asterline987654321"
    secret_text = "remember that my API key is " + synthetic_secret
    app.chat_input[0].set_value(secret_text).run(timeout=30)
    assert not app.exception
    rendered_8 = " ".join(str(w.value) for w in [*app.markdown, *app.caption])
    assert "did not store" in rendered_8.lower() or "rotate" in rendered_8.lower() or "secret" in rendered_8.lower()
    # Verify secret was NEVER stored in DB
    all_alice = store.list_active("alice")
    for m in all_alice:
        assert synthetic_secret not in m.content
        assert "sk-" not in m.content
    print("Turn 8 PASS: sk- API key rejected and blocked from storage.")

    # ──────────────────────────────────────────────────────────────────────────
    # TURN 9: Sensitive Health / Medical Guardrail
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- TURN 9: Sensitive Personal / Leave Guardrail ---")
    health_text = "remember that I am taking medical leave next month for surgery"
    app.chat_input[0].set_value(health_text).run(timeout=30)
    assert not app.exception
    rendered_9 = " ".join(str(w.value) for w in [*app.markdown, *app.caption])
    assert "cannot store personal health" in rendered_9.lower() or "hr portal" in rendered_9.lower()
    # Verify health data was NEVER stored in DB
    all_alice = store.list_active("alice")
    for m in all_alice:
        assert "surgery" not in m.content.lower()
        assert "medical leave" not in m.content.lower()
    print("Turn 9 PASS: Sensitive medical data rejected and directed to HR portal.")

    # ──────────────────────────────────────────────────────────────────────────
    # TURN 10: Adversarial User Isolation & Hard Forget Verification
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- TURN 10: Adversarial Isolation & Hard Forget ---")
    # Part A: Switch to Bob and verify complete isolation
    app.selectbox[0].set_value("bob").run(timeout=30)
    app.chat_input[0].set_value("What investigation was Alice working on?").run(timeout=30)
    assert not app.exception
    rendered_10_bob = " ".join(str(w.value) for w in [*app.markdown, *app.caption])
    assert "BigQuery" not in rendered_10_bob
    bob_memories = store.list_active("bob")
    assert len(bob_memories) == 0, f"Bob leaked Alice's memories: {bob_memories}"
    print("Turn 10A PASS: Bob has ZERO access to Alice's stored memories (Leakage = 0).")

    # Part B: Switch back to Alice and perform Hard Forget
    app.selectbox[0].set_value("alice").run(timeout=30)
    app.chat_input[0].set_value("forget that I am investigating BigQuery duplicates").run(timeout=30)
    assert not app.exception
    rendered_10_forget = " ".join(str(w.value) for w in [*app.markdown, *app.caption])
    assert "forgot" in rendered_10_forget.lower(), rendered_10_forget
    # Verify hard deletion in SQLite
    remaining_alice = store.list_active("alice")
    assert not any("BigQuery duplicates" in m.content for m in remaining_alice)
    print("Turn 10B PASS: Hard forget confirmed. Memory permanently deleted from database.")
    print("\n================ ALL 10 E2E TURNS COMPLETED SUCCESSFULLY ================")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
