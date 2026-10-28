"""SQLite source of truth for user-scoped memories."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import MemoryRecord

SECRET_RE = re.compile(r"(?i)\b(api[_ -]?key|token|password|authorization|secret|bearer)\b\s*[:=]?")
SENSITIVE_RE = re.compile(r"(?i)\b(health|medical|diagnosis|parental leave|pregnan|mental health|disability|religion|sexual orientation)\b")
SK_RE = re.compile(r"(?i)\bsk-[A-Za-z0-9_-]+")
BEARER_RE = re.compile(r"(?i)(authorization\s*:\s*bearer\s+|bearer\s+)[A-Za-z0-9._~+/=-]+")
PASSWORD_RE = re.compile(r"(?i)(password\s*(?:is|=|:)\s*)[^\s,;]+")


class Clock:
    def __init__(self, now: datetime | None = None) -> None:
        self.current = now or datetime.now(timezone.utc)

    def now(self) -> datetime:
        return self.current

    def advance_days(self, days: int) -> None:
        self.current += timedelta(days=days)


def redact_sensitive(text: str) -> str:
    text = SK_RE.sub("[REDACTED]", text)
    text = BEARER_RE.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
    text = PASSWORD_RE.sub(r"\1[REDACTED]", text)
    return SENSITIVE_RE.sub("[REDACTED]", text)


def contains_sensitive(text: str) -> bool:
    return bool(SK_RE.search(text) or BEARER_RE.search(text) or PASSWORD_RE.search(text))


class MemoryStore:
    def __init__(self, path: Path, clock: Clock | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.clock = clock or Clock()
        self.connection.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, type TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_confirmed_at TEXT NOT NULL, source_session TEXT, source_turn TEXT, confidence REAL, is_authoritative INTEGER NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS memory_vectors (memory_id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS response_cache (cache_key TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL)")
        self.connection.commit()

    def list_active(self, user_id: str) -> list[MemoryRecord]:
        rows = self.connection.execute("SELECT * FROM memories WHERE user_id = ? AND status = 'active' ORDER BY id", (user_id,)).fetchall()
        return [MemoryRecord.model_validate(dict(row) | {"is_authoritative": bool(row["is_authoritative"])}) for row in rows]

    def list_sessions(self, user_id: str) -> list[str]:
        rows = self.connection.execute("SELECT session_id FROM sessions WHERE user_id = ? ORDER BY rowid DESC", (user_id,)).fetchall()
        return [str(row["session_id"]) for row in rows]

    def write_explicit(self, user_id: str, content: str, session: str = "", turn: str = "") -> MemoryRecord | None:
        if not content.strip() or SECRET_RE.search(content) or SENSITIVE_RE.search(content):
            return None
        now = self.clock.now().isoformat()
        record_type = "project" if re.search(r"(?i)(investigat|working on|project )", content) else ("preference" if re.search(r"(?i)(at most two sentences|bullet points|answer (?:short|brief|concis)|keep answers short)", content) else "user_claim")
        record = self.connection.execute("INSERT INTO memories (user_id,type,content,status,created_at,updated_at,last_confirmed_at,source_session,source_turn,confidence,is_authoritative) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (user_id, record_type, content, "active", now, now, now, session, turn, 1.0, 0))
        self.connection.execute("INSERT INTO memory_vectors (memory_id,user_id,content) VALUES (?,?,?)", (record.lastrowid, user_id, content))
        self.connection.commit()
        return MemoryRecord(id=record.lastrowid, user_id=user_id, type=record_type, content=content, created_at=now, updated_at=now, last_confirmed_at=now, source_session=session, source_turn=turn, is_authoritative=False)

    def forget(self, user_id: str, target: str) -> int:
        """Hard-delete matching memories so forgotten content cannot resurface."""
        rows = self.connection.execute("SELECT id FROM memories WHERE user_id = ? AND status = 'active' AND lower(content) LIKE ?", (user_id, f"%{target.lower()}%" )).fetchall()
        if rows:
            self.connection.executemany("DELETE FROM memory_vectors WHERE user_id = ? AND memory_id = ?", ((user_id, row["id"]) for row in rows))
            self.connection.executemany("DELETE FROM memories WHERE user_id = ? AND id = ?", ((user_id, row["id"]) for row in rows))
            self.connection.commit()
        return len(rows)

    def forget_all(self, user_id: str) -> int:
        result = self.connection.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        self.connection.execute("DELETE FROM memory_vectors WHERE user_id = ?", (user_id,))
        self.connection.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        self.connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        self.connection.execute("DELETE FROM response_cache WHERE user_id = ?", (user_id,))
        self.connection.commit()
        return result.rowcount

    def delete_memory(self, user_id: str, memory_id: int) -> bool:
        row = self.connection.execute("SELECT id FROM memories WHERE user_id = ? AND id = ?", (user_id, memory_id)).fetchone()
        if row is None:
            return False
        self.connection.execute("DELETE FROM memory_vectors WHERE user_id = ? AND memory_id = ?", (user_id, memory_id))
        self.connection.execute("DELETE FROM memories WHERE user_id = ? AND id = ?", (user_id, memory_id))
        self.connection.commit()
        return True

    def edit_memory(self, user_id: str, memory_id: int, content: str) -> bool:
        """Edit only safe user context; keep the original record identity."""
        if not content.strip() or SECRET_RE.search(content) or SENSITIVE_RE.search(content):
            return False
        row = self.connection.execute("SELECT id FROM memories WHERE user_id = ? AND id = ? AND status = 'active'", (user_id, memory_id)).fetchone()
        if row is None:
            return False
        now = self.clock.now().isoformat()
        self.connection.execute("UPDATE memories SET content = ?, updated_at = ?, last_confirmed_at = ? WHERE user_id = ? AND id = ?", (redact_sensitive(content.strip()), now, now, user_id, memory_id))
        self.connection.execute("UPDATE memory_vectors SET content = ? WHERE user_id = ? AND memory_id = ?", (redact_sensitive(content.strip()), user_id, memory_id))
        self.connection.commit()
        return True

    def correct(self, user_id: str, target: str, replacement: str, session: str = "", turn: str = "") -> MemoryRecord | None:
        """Supersede only this user's old context; never make it authoritative."""
        rows = self.connection.execute("SELECT id FROM memories WHERE user_id = ? AND status = 'active' AND lower(content) LIKE ?", (user_id, f"%{target.lower()}%" )).fetchall()
        if len(rows) != 1:
            return None
        now = self.clock.now().isoformat()
        self.connection.execute("UPDATE memories SET status = 'superseded', updated_at = ? WHERE user_id = ? AND id = ?", (now, user_id, rows[0]["id"]))
        self.connection.execute("DELETE FROM memory_vectors WHERE user_id = ? AND memory_id = ?", (user_id, rows[0]["id"]))
        self.connection.commit()
        return self.write_explicit(user_id, replacement, session=session, turn=turn)

    def record_message(self, user_id: str, session_id: str, role: str, content: str) -> None:
        self.connection.execute("INSERT OR IGNORE INTO sessions(session_id,user_id) VALUES (?,?)", (session_id, user_id))
        self.connection.execute("INSERT INTO messages(user_id,session_id,role,content) VALUES (?,?,?,?)", (user_id, session_id, role, redact_sensitive(content)))
        self.connection.commit()

    def history(self, session_id: str) -> list[dict[str, str]]:
        rows = self.connection.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def active_memory_text(self, user_id: str) -> list[str]:
        return [f"{record.type}: {record.content}" for record in self.list_active(user_id)]

    def expire(self, user_id: str, *, memory_type: str, max_age_days: int) -> int:
        """Hard-delete stale memories of one type using the injectable clock."""
        cutoff = self.clock.now() - timedelta(days=max_age_days)
        rows = self.connection.execute(
            "SELECT id FROM memories WHERE user_id = ? AND type = ? AND status = 'active' AND updated_at < ?",
            (user_id, memory_type, cutoff.isoformat()),
        ).fetchall()
        for row in rows:
            self.delete_memory(user_id, int(row["id"]))
        return len(rows)

    def dump_memory_tables(self) -> str:
        return "\n".join(" | ".join(str(value) for value in row) for table in ("memories", "memory_vectors") for row in self.connection.execute(f"SELECT * FROM {table}").fetchall())

    def dump_all_tables(self) -> str:
        return "\n".join(" | ".join(str(value) for value in row) for table in ("memories", "memory_vectors", "sessions", "messages", "response_cache") for row in self.connection.execute(f"SELECT * FROM {table}").fetchall())

