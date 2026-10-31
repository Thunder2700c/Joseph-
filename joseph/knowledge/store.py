"""SQLite-backed knowledge store with mandatory source provenance."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schema import EntityRecord, FactRecord, SourceRecord


class KnowledgeStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                part INTEGER,
                continuity TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL REFERENCES entities(entity_id),
                predicate TEXT NOT NULL,
                object_value TEXT NOT NULL,
                part INTEGER,
                continuity TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fact_sources (
                fact_id TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
                source_id TEXT NOT NULL REFERENCES sources(source_id),
                PRIMARY KEY (fact_id, source_id)
            );
            CREATE INDEX IF NOT EXISTS facts_part_idx ON facts(part);
            CREATE INDEX IF NOT EXISTS facts_subject_idx ON facts(subject_id);
            """
        )
        self.connection.commit()

    def add_source(self, source: SourceRecord) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO sources(source_id, payload) VALUES (?, ?)",
            (source.source_id, json.dumps(source.to_dict(), ensure_ascii=False)),
        )
        self.connection.commit()

    def add_entity(self, entity: EntityRecord) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO entities
               (entity_id, name, entity_type, part, continuity, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                entity.entity_id, entity.name, entity.entity_type, entity.part,
                entity.continuity, json.dumps(entity.to_dict(), ensure_ascii=False),
            ),
        )
        self.connection.commit()

    def add_fact(self, fact: FactRecord) -> None:
        if self.connection.execute(
            "SELECT 1 FROM entities WHERE entity_id = ?", (fact.subject_id,)
        ).fetchone() is None:
            raise ValueError(f"unknown subject entity: {fact.subject_id}")
        for source_id in fact.source_ids:
            if self.connection.execute(
                "SELECT 1 FROM sources WHERE source_id = ?", (source_id,)
            ).fetchone() is None:
                raise ValueError(f"unknown source: {source_id}")

        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO facts
                   (fact_id, subject_id, predicate, object_value, part,
                    continuity, status, confidence, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact.fact_id, fact.subject_id, fact.predicate, fact.object_value,
                    fact.part, fact.continuity, fact.status, fact.confidence,
                    json.dumps(fact.to_dict(), ensure_ascii=False),
                ),
            )
            self.connection.execute(
                "DELETE FROM fact_sources WHERE fact_id = ?", (fact.fact_id,)
            )
            self.connection.executemany(
                "INSERT INTO fact_sources(fact_id, source_id) VALUES (?, ?)",
                [(fact.fact_id, source_id) for source_id in fact.source_ids],
            )

    def search_facts(
        self,
        query: str,
        part: int | None = None,
        continuity: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")
        pattern = f"%{query.strip()}%"
        sql = """
            SELECT f.payload AS fact_payload, e.payload AS entity_payload
            FROM facts f JOIN entities e ON e.entity_id = f.subject_id
            WHERE (e.name LIKE ? OR f.predicate LIKE ? OR f.object_value LIKE ?)
        """
        args: list[object] = [pattern, pattern, pattern]
        if part is not None:
            sql += " AND f.part = ?"
            args.append(part)
        if continuity is not None:
            sql += " AND f.continuity IN (?, 'both', 'unknown')"
            args.append(continuity)
        sql += " ORDER BY f.confidence DESC, f.fact_id LIMIT ?"
        args.append(limit)

        results = []
        for row in self.connection.execute(sql, args):
            fact = json.loads(row["fact_payload"])
            entity = json.loads(row["entity_payload"])
            sources = []
            for source_id in fact["source_ids"]:
                source_row = self.connection.execute(
                    "SELECT payload FROM sources WHERE source_id = ?", (source_id,)
                ).fetchone()
                if source_row:
                    sources.append(json.loads(source_row["payload"]))
            results.append({"entity": entity, "fact": fact, "sources": sources})
        return results

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
