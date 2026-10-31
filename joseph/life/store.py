"""SQLite storage for atomic, ordered, source-linked life memories."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .schema import LifeMemory, LifeSource

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


class LifeMemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS life_sources (
                source_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS life_memories (
                memory_id TEXT PRIMARY KEY,
                sequence_no INTEGER NOT NULL UNIQUE,
                stage_id TEXT NOT NULL,
                memory_kind TEXT NOT NULL,
                canon_category TEXT NOT NULL,
                review_state TEXT NOT NULL,
                recalled_by_character INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS life_memories_stage_idx
                ON life_memories(stage_id, sequence_no);
            CREATE INDEX IF NOT EXISTS life_memories_review_idx
                ON life_memories(review_state);
            CREATE TABLE IF NOT EXISTS life_stages (
                stage_id TEXT PRIMARY KEY,
                stage_order INTEGER NOT NULL UNIQUE,
                title TEXT NOT NULL,
                required_before_awakening INTEGER NOT NULL,
                is_awakening_stage INTEGER NOT NULL,
                dependencies_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'collecting',
                review_note TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS life_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def add_source(self, source: LifeSource) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO life_sources(source_id,payload) VALUES (?,?)",
                (source.source_id, json.dumps(source.to_dict(), ensure_ascii=False)),
            )

    def has_source(self, source_id: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM life_sources WHERE source_id=?", (source_id,)
        ).fetchone() is not None

    def add_memory(self, memory: LifeMemory) -> None:
        missing = [
            source_id for source_id in memory.source_ids
            if not self.has_source(source_id)
        ]
        if missing:
            raise ValueError(f"unknown source IDs: {', '.join(missing)}")
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO life_memories
                   (memory_id,sequence_no,stage_id,memory_kind,canon_category,
                    review_state,recalled_by_character,payload)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    memory.memory_id,memory.sequence,memory.stage_id,
                    memory.memory_kind,memory.canon_category,memory.review_state,
                    int(memory.recalled_by_character),
                    json.dumps(memory.to_dict(),ensure_ascii=False),
                ),
            )
            self.connection.execute(
                """UPDATE life_stages SET status='collecting',review_note=''
                   WHERE stage_id=? AND status='completed'""",
                (memory.stage_id,),
            )

    def memories(
        self, stage_id: str | None = None, recalled_only: bool = False,
        limit: int | None = None,
    ) -> list[LifeMemory]:
        sql = "SELECT payload FROM life_memories WHERE 1=1"
        args: list[object] = []
        if stage_id is not None:
            sql += " AND stage_id=?"
            args.append(stage_id)
        if recalled_only:
            sql += " AND recalled_by_character=1"
        sql += " ORDER BY sequence_no"
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            sql += " LIMIT ?"
            args.append(limit)
        return [
            LifeMemory.from_dict(json.loads(row["payload"]))
            for row in self.connection.execute(sql,args)
        ]

    def search_memories(
        self, query: str, stage_id: str | None = None, limit: int = 20,
    ) -> list[LifeMemory]:
        if limit < 1:
            raise ValueError("limit must be positive")
        terms = _search_terms(query)
        if not terms:
            return []
        clauses = " OR ".join(["payload LIKE ?"] * len(terms))
        args: list[object] = [f"%{term}%" for term in terms]
        sql = f"SELECT payload FROM life_memories WHERE ({clauses})"
        if stage_id is not None:
            sql += " AND stage_id=?"
            args.append(stage_id)
        sql += " ORDER BY sequence_no LIMIT ?"
        args.append(limit)
        return [
            LifeMemory.from_dict(json.loads(row["payload"]))
            for row in self.connection.execute(sql,args)
        ]

    def verified_count(self, stage_id: str) -> int:
        row = self.connection.execute(
            """SELECT COUNT(*) AS count FROM life_memories
               WHERE stage_id=? AND review_state IN ('verified','user_authored')""",
            (stage_id,),
        ).fetchone()
        return int(row["count"])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LifeMemoryStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
