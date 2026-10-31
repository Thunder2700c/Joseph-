"""Persistent one-thread conversation history plus searchable long-term memories."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STOPWORDS = {
    "the", "a", "an", "about", "tell", "me", "my", "to", "what", "is", "are",
    "of", "for", "can", "you", "please", "do", "does", "did", "and", "or",
    "in", "on", "at", "how", "why", "when", "where", "was", "were", "with",
    "this", "that", "it", "i", "we", "our", "your", "who",
}


def _search_terms(query: str) -> list[str]:
    terms = [
        term.lower() for term in re.findall(r"[\w]+", query, flags=re.UNICODE)
        if term.lower() not in _STOPWORDS and len(term) > 1
    ]
    return list(dict.fromkeys(terms))[:12]


class MemoryStore:
    """One continuous transcript. Full history stays local; context is retrieved."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._fts_enabled = False
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
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS memories_category_idx ON memories(category);
            CREATE INDEX IF NOT EXISTS conversation_turns_created_idx
                ON conversation_turns(created_at, turn_id);
            """
        )
        self._migrate_columns()
        self._setup_full_text_search()
        self.connection.commit()

    def _migrate_columns(self) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(conversation_turns)")
        }
        if "metadata_json" not in columns:
            self.connection.execute(
                "ALTER TABLE conversation_turns "
                "ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
            )

    def _setup_full_text_search(self) -> None:
        existed = self.connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='conversation_turns_fts'"
        ).fetchone() is not None
        try:
            self.connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS conversation_turns_fts
                   USING fts5(content, content='conversation_turns',
                              content_rowid='turn_id')"""
            )
            self.connection.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS conversation_turns_ai
                AFTER INSERT ON conversation_turns BEGIN
                    INSERT INTO conversation_turns_fts(rowid, content)
                    VALUES (new.turn_id, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS conversation_turns_ad
                AFTER DELETE ON conversation_turns BEGIN
                    INSERT INTO conversation_turns_fts(
                        conversation_turns_fts, rowid, content
                    ) VALUES ('delete', old.turn_id, old.content);
                END;
                CREATE TRIGGER IF NOT EXISTS conversation_turns_au
                AFTER UPDATE OF content ON conversation_turns BEGIN
                    INSERT INTO conversation_turns_fts(
                        conversation_turns_fts, rowid, content
                    ) VALUES ('delete', old.turn_id, old.content);
                    INSERT INTO conversation_turns_fts(rowid, content)
                    VALUES (new.turn_id, new.content);
                END;
                """
            )
            if not existed:
                self.connection.execute(
                    "INSERT INTO conversation_turns_fts(conversation_turns_fts) "
                    "VALUES ('rebuild')"
                )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_memory(
        self, content: str, category: str = "general",
        importance: float = 0.5, memory_id: str | None = None,
    ) -> str:
        content, category = content.strip(), category.strip()
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
        terms = _search_terms(query)
        clauses = " OR ".join(["content LIKE ? OR category LIKE ?"] * max(1, len(terms)))
        parameters: list[object] = []
        for term in terms or [query.strip()]:
            pattern = f"%{term}%"
            parameters.extend([pattern, pattern])
        rows = self.connection.execute(
            f"""SELECT * FROM memories WHERE {clauses}
                ORDER BY importance DESC, updated_at DESC LIMIT ?""",
            (*parameters, limit),
        )
        return [dict(row) for row in rows]

    def delete_memory(self, memory_id: str) -> bool:
        cursor = self.connection.execute(
            "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def add_turn(
        self, role: str, content: str, metadata: dict[str, Any] | None = None,
    ) -> int:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be user, assistant, or system")
        if not content.strip():
            raise ValueError("turn content cannot be empty")
        if len(content) > 1_000_000:
            raise ValueError("single turn exceeds the 1,000,000-character safety limit")
        cursor = self.connection.execute(
            """INSERT INTO conversation_turns
               (role, content, created_at, metadata_json)
               VALUES (?, ?, ?, ?)""",
            (role, content, self._now(), json.dumps(metadata or {}, ensure_ascii=False)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    @staticmethod
    def _turn_from_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json", "{}") or "{}")
        return result

    def recent_turns(self, limit: int = 20) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self.connection.execute(
            "SELECT * FROM conversation_turns ORDER BY turn_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._turn_from_row(row) for row in reversed(rows)]

    def get_turns(self, after_turn_id: int = 0, limit: int = 100) -> list[dict]:
        """Page through the complete continuous conversation in original order."""
        if after_turn_id < 0 or limit < 1:
            raise ValueError("after_turn_id must be non-negative and limit positive")
        rows = self.connection.execute(
            """SELECT * FROM conversation_turns
               WHERE turn_id > ? ORDER BY turn_id LIMIT ?""",
            (after_turn_id, limit),
        ).fetchall()
        return [self._turn_from_row(row) for row in rows]

    def search_turns(self, query: str, limit: int = 20) -> list[dict]:
        """Search the entire saved conversation, not just recent turns."""
        if limit < 1:
            raise ValueError("limit must be positive")
        terms = _search_terms(query)
        if not terms:
            return []
        if self._fts_enabled:
            match_query = " OR ".join(
                '"' + term.replace('"', '""') + '"' for term in terms
            )
            try:
                rows = self.connection.execute(
                    """SELECT t.* FROM conversation_turns_fts f
                       JOIN conversation_turns t ON t.turn_id = f.rowid
                       WHERE conversation_turns_fts MATCH ?
                       ORDER BY bm25(conversation_turns_fts), t.turn_id DESC LIMIT ?""",
                    (match_query, limit),
                ).fetchall()
                return [self._turn_from_row(row) for row in reversed(rows)]
            except sqlite3.OperationalError:
                pass
        clauses = " OR ".join(["content LIKE ?"] * len(terms))
        rows = self.connection.execute(
            f"""SELECT * FROM conversation_turns WHERE {clauses}
                ORDER BY turn_id DESC LIMIT ?""",
            (*[f"%{term}%" for term in terms], limit),
        ).fetchall()
        return [self._turn_from_row(row) for row in reversed(rows)]

    def conversation_context(
        self, query: str, recent_limit: int = 12, recalled_limit: int = 8,
    ) -> list[dict]:
        """Combine recent turns with older matching turns, preserving chronology."""
        recent = self.recent_turns(recent_limit)
        recalled = self.search_turns(query, recalled_limit)
        combined = {turn["turn_id"]: turn for turn in recalled}
        combined.update({turn["turn_id"]: turn for turn in recent})
        return [combined[key] for key in sorted(combined)]

    def turn_count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM conversation_turns"
        ).fetchone()
        return int(row["count"])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
