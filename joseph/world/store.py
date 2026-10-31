"""Persistent user-taught world facts, separate from Teyvat canon and chat logs."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

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


class WorldKnowledgeStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._fts_enabled = False
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS taught_facts (
                fact_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                category TEXT NOT NULL,
                statement TEXT NOT NULL,
                origin TEXT NOT NULL DEFAULT 'user_taught',
                tags_json TEXT NOT NULL DEFAULT '[]',
                importance REAL NOT NULL DEFAULT 0.7,
                include_in_training INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS taught_facts_subject_idx ON taught_facts(subject);
            CREATE INDEX IF NOT EXISTS taught_facts_category_idx ON taught_facts(category);
            """
        )
        existed = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='taught_facts_fts'"
        ).fetchone() is not None
        try:
            self.connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS taught_facts_fts
                   USING fts5(subject, category, statement, tags)"""
            )
            if not existed:
                self.connection.execute(
                    """INSERT INTO taught_facts_fts(subject,category,statement,tags)
                       SELECT subject,category,statement,tags_json FROM taught_facts"""
                )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False
        self.connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalise(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["tags"] = json.loads(item.pop("tags_json"))
        item["include_in_training"] = bool(item["include_in_training"])
        return item

    def teach(
        self, subject: str, statement: str, category: str = "general",
        tags: tuple[str, ...] | list[str] = (), importance: float = 0.7,
        include_in_training: bool = False, fact_id: str | None = None,
    ) -> str:
        subject, statement, category = subject.strip(), statement.strip(), category.strip()
        if not subject or not statement or not category:
            raise ValueError("subject, statement, and category are required")
        if len(statement) > 20_000:
            raise ValueError("a taught fact cannot exceed 20,000 characters")
        if not 0.0 <= importance <= 1.0:
            raise ValueError("importance must be between 0 and 1")
        clean_tags = tuple(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
        fact_id, now = fact_id or str(uuid.uuid4()), self._now()
        with self.connection:
            self.connection.execute(
                """INSERT INTO taught_facts
                   (fact_id, subject, category, statement, origin, tags_json,
                    importance, include_in_training, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'user_taught', ?, ?, ?, ?, ?)
                   ON CONFLICT(fact_id) DO UPDATE SET
                   subject=excluded.subject, category=excluded.category,
                   statement=excluded.statement, tags_json=excluded.tags_json,
                   importance=excluded.importance,
                   include_in_training=excluded.include_in_training,
                   updated_at=excluded.updated_at""",
                (
                    fact_id, subject, category, statement,
                    json.dumps(clean_tags, ensure_ascii=False), importance,
                    int(include_in_training), now, now,
                ),
            )
            if self._fts_enabled:
                row = self.connection.execute(
                    "SELECT rowid FROM taught_facts WHERE fact_id=?", (fact_id,)
                ).fetchone()
                self.connection.execute(
                    "DELETE FROM taught_facts_fts WHERE rowid=?", (row["rowid"],)
                )
                self.connection.execute(
                    """INSERT INTO taught_facts_fts(rowid,subject,category,statement,tags)
                       SELECT rowid,subject,category,statement,tags_json
                       FROM taught_facts WHERE fact_id=?""",
                    (fact_id,),
                )
        return fact_id

    def get(self, fact_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM taught_facts WHERE fact_id=?", (fact_id,)
        ).fetchone()
        return self._normalise(row) if row else None

    def search(self, query: str, limit: int = 12) -> list[dict]:
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
                    """SELECT f.* FROM taught_facts_fts x
                       JOIN taught_facts f ON f.rowid=x.rowid
                       WHERE taught_facts_fts MATCH ?
                       ORDER BY bm25(taught_facts_fts), f.importance DESC,
                                f.updated_at DESC LIMIT ?""",
                    (match_query, limit),
                ).fetchall()
                return [self._normalise(row) for row in rows]
            except sqlite3.OperationalError:
                pass
        clauses = " OR ".join(
            ["subject LIKE ? OR category LIKE ? OR statement LIKE ?"] * len(terms)
        )
        parameters: list[str] = []
        for term in terms:
            pattern = f"%{term}%"
            parameters.extend([pattern, pattern, pattern])
        rows = self.connection.execute(
            f"""SELECT * FROM taught_facts WHERE {clauses}
                ORDER BY importance DESC, updated_at DESC LIMIT ?""",
            (*parameters, limit),
        ).fetchall()
        return [self._normalise(row) for row in rows]

    def for_training(self, limit: int = 10000) -> list[dict]:
        rows = self.connection.execute(
            """SELECT * FROM taught_facts WHERE include_in_training=1
               ORDER BY created_at,fact_id LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._normalise(row) for row in rows]

    def set_training_approval(self, fact_id: str, approved: bool) -> bool:
        cursor = self.connection.execute(
            """UPDATE taught_facts SET include_in_training=?,updated_at=?
               WHERE fact_id=?""",
            (int(approved), self._now(), fact_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def context(self, query: str, limit: int = 8) -> str:
        facts = self.search(query, limit=limit)
        return "\n".join(
            f"- {item['subject']} ({item['category']}): {item['statement']}"
            for item in facts
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "WorldKnowledgeStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
