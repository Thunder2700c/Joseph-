"""Small explicit-write SQLite memory store.

This module never decides by itself that a conversation detail should be saved.
Call save_memory only when a future consent-aware conversation layer decides it
is appropriate to retain that detail.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversation_turns (
                turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS memories_category_idx ON memories(category);
            """
        )
        self.connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_memory(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        memory_id: str | None = None,
    ) -> str:
        content = content.strip()
        category = category.strip()
        if not content or not category:
            raise ValueError("memory content and category cannot be empty")
        if len(content) > 20_000:
            raise ValueError("memory content exceeds 20,000 characters")
        if not 0.0 <= importance <= 1.0:
            raise ValueError("importance must be between 0 and 1")

        memory_id = memory_id or str(uuid.uuid4())
        now = self._now()
        self.connection.execute(
            """INSERT INTO memories
               (memory_id, category, content, importance, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(memory_id) DO UPDATE SET
               category=excluded.category, content=excluded.content,
               importance=excluded.importance, updated_at=excluded.updated_at""",
            (memory_id, category, content, importance, now, now),
        )
        self.connection.commit()
        return memory_id

    def search_memories(self, query: str, limit: int = 20) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")
        pattern = f"%{query.strip()}%"
        rows = self.connection.execute(
            """SELECT * FROM memories
               WHERE content LIKE ? OR category LIKE ?
               ORDER BY importance DESC, updated_at DESC LIMIT ?""",
            (pattern, pattern, limit),
        )
        return [dict(row) for row in rows]

    def delete_memory(self, memory_id: str) -> bool:
        cursor = self.connection.execute(
            "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def add_turn(self, role: str, content: str) -> int:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be user, assistant, or system")
        if not content.strip():
            raise ValueError("turn content cannot be empty")
        cursor = self.connection.execute(
            "INSERT INTO conversation_turns(role, content, created_at) VALUES (?, ?, ?)",
            (role, content, self._now()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def recent_turns(self, limit: int = 20) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self.connection.execute(
            "SELECT * FROM conversation_turns ORDER BY turn_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
