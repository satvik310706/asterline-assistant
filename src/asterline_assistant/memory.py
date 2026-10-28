"""SQLite source of truth for user-scoped memories."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import MemoryRecord

SECRET_RE = re.compile(r"(?i)\b(api[_ -]?key|token|password|authorization|secret|bearer)\b\s*[:=]?")
SENSITIVE_RE = re.compile(r"(?i)\b(health|medical|diagnosis|parental leave|pregnan|mental health|disability|religion|sexual orientation)\b")


class MemoryStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, type TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_confirmed_at TEXT NOT NULL, source_session TEXT, source_turn TEXT, confidence REAL, is_authoritative INTEGER NOT NULL)")
        self.connection.commit()

    def list_active(self, user_id: str) -> list[MemoryRecord]:
        rows = self.connection.execute("SELECT * FROM memories WHERE user_id = ? AND status = 'active' ORDER BY id", (user_id,)).fetchall()
        return [MemoryRecord.model_validate(dict(row) | {"is_authoritative": bool(row["is_authoritative"])}) for row in rows]

    def write_explicit(self, user_id: str, content: str, session: str = "", turn: str = "") -> MemoryRecord | None:
        if not content.strip() or SECRET_RE.search(content) or SENSITIVE_RE.search(content):
            return None
        now = datetime.now(timezone.utc).isoformat()
        record = self.connection.execute("INSERT INTO memories (user_id,type,content,status,created_at,updated_at,last_confirmed_at,source_session,source_turn,confidence,is_authoritative) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (user_id, "user_claim", content, "active", now, now, now, session, turn, 1.0, 0))
        self.connection.commit()
        return MemoryRecord(id=record.lastrowid, user_id=user_id, type="user_claim", content=content, created_at=now, updated_at=now, last_confirmed_at=now, source_session=session, source_turn=turn, is_authoritative=False)

    def forget(self, user_id: str, target: str) -> int:
        """Hard-delete matching memories so forgotten content cannot resurface."""
        rows = self.connection.execute("SELECT id FROM memories WHERE user_id = ? AND status = 'active' AND lower(content) LIKE ?", (user_id, f"%{target.lower()}%" )).fetchall()
        if rows:
            self.connection.executemany("DELETE FROM memories WHERE user_id = ? AND id = ?", ((user_id, row["id"]) for row in rows))
            self.connection.commit()
        return len(rows)

    def forget_all(self, user_id: str) -> int:
        result = self.connection.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        self.connection.commit()
        return result.rowcount

    def correct(self, user_id: str, target: str, replacement: str, session: str = "", turn: str = "") -> MemoryRecord | None:
        """Supersede only this user's old context; never make it authoritative."""
        rows = self.connection.execute("SELECT id FROM memories WHERE user_id = ? AND status = 'active' AND lower(content) LIKE ?", (user_id, f"%{target.lower()}%" )).fetchall()
        if len(rows) != 1:
            return None
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute("UPDATE memories SET status = 'superseded', updated_at = ? WHERE user_id = ? AND id = ?", (now, user_id, rows[0]["id"]))
        self.connection.commit()
        return self.write_explicit(user_id, replacement, session=session, turn=turn)

