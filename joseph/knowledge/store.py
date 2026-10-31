"""SQLite knowledge store supporting legacy entity/fact records and common-world knowledge."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .schema import (
    EntityRecord, FactRecord, KnowledgeFact, KnowledgeSource, SourceRecord,
)

_STOPWORDS = {
    "the", "a", "an", "about", "tell", "me", "my", "to", "what", "is", "are",
    "of", "for", "can", "you", "please", "do", "does", "did", "and", "or",
    "in", "on", "at", "how", "why", "when", "where", "was", "were", "with",
    "this", "that", "it", "i", "we", "our", "your", "who", "explain",
}


def _terms(query: str) -> list[str]:
    return list(dict.fromkeys(
        term.lower() for term in re.findall(r"[\w]+", query, flags=re.UNICODE)
        if len(term) > 1 and term.lower() not in _STOPWORDS
    ))[:16]


class KnowledgeStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_sources (
                source_id TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS knowledge_facts (
                fact_id TEXT PRIMARY KEY, domain TEXT NOT NULL, topic TEXT NOT NULL,
                term TEXT NOT NULL, status TEXT NOT NULL, scope TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS knowledge_topic_idx
                ON knowledge_facts(domain, topic);
            CREATE INDEX IF NOT EXISTS knowledge_term_idx
                ON knowledge_facts(term);

            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY, name TEXT NOT NULL,
                entity_type TEXT NOT NULL, part INTEGER, continuity TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL REFERENCES entities(entity_id),
                predicate TEXT NOT NULL, object_value TEXT NOT NULL,
                part INTEGER, continuity TEXT NOT NULL, status TEXT NOT NULL,
                confidence REAL NOT NULL, payload TEXT NOT NULL
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
        self._fts_enabled = False
        existed = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_facts_fts'"
        ).fetchone() is not None
        try:
            self.connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_facts_fts
                   USING fts5(term, domain, topic, definition, tags, related_terms)"""
            )
            if not existed:
                self.connection.execute(
                    """INSERT INTO knowledge_facts_fts(
                           rowid,term,domain,topic,definition,tags,related_terms
                       )
                       SELECT rowid,term,domain,topic,
                              json_extract(payload,'$.definition'),
                              json_extract(payload,'$.tags'),
                              json_extract(payload,'$.related_terms')
                       FROM knowledge_facts"""
                )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False
        self.connection.commit()

    def add_source(self, source: KnowledgeSource | SourceRecord) -> None:
        if isinstance(source, SourceRecord):
            with self.connection:
                self.connection.execute(
                    "INSERT OR REPLACE INTO sources(source_id,payload) VALUES (?,?)",
                    (source.source_id, json.dumps(source.to_dict(), ensure_ascii=False)),
                )
            return
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO knowledge_sources(source_id,payload) VALUES (?,?)",
                (source.source_id, json.dumps(source.to_dict(), ensure_ascii=False)),
            )

    def add_entity(self, entity: EntityRecord) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO entities
                   (entity_id,name,entity_type,part,continuity,payload)
                   VALUES (?,?,?,?,?,?)""",
                (
                    entity.entity_id, entity.name, entity.entity_type, entity.part,
                    entity.continuity, json.dumps(entity.to_dict(), ensure_ascii=False),
                ),
            )

    def add_fact(self, fact: KnowledgeFact | FactRecord) -> None:
        if isinstance(fact, FactRecord):
            if self.connection.execute(
                "SELECT 1 FROM entities WHERE entity_id=?", (fact.subject_id,)
            ).fetchone() is None:
                raise ValueError(f"unknown subject entity: {fact.subject_id}")
            for source_id in fact.source_ids:
                if self.connection.execute(
                    "SELECT 1 FROM sources WHERE source_id=?", (source_id,)
                ).fetchone() is None:
                    raise ValueError(f"unknown source: {source_id}")
            with self.connection:
                self.connection.execute(
                    """INSERT OR REPLACE INTO facts
                       (fact_id,subject_id,predicate,object_value,part,continuity,
                        status,confidence,payload)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        fact.fact_id, fact.subject_id, fact.predicate, fact.object_value,
                        fact.part, fact.continuity, fact.status, fact.confidence,
                        json.dumps(fact.to_dict(), ensure_ascii=False),
                    ),
                )
                self.connection.execute(
                    "DELETE FROM fact_sources WHERE fact_id=?", (fact.fact_id,)
                )
                self.connection.executemany(
                    "INSERT INTO fact_sources(fact_id,source_id) VALUES (?,?)",
                    [(fact.fact_id, source_id) for source_id in fact.source_ids],
                )
            return

        missing = [
            source_id for source_id in fact.source_ids
            if self.connection.execute(
                "SELECT 1 FROM knowledge_sources WHERE source_id=?", (source_id,)
            ).fetchone() is None
        ]
        if missing:
            raise ValueError("unknown knowledge sources: " + ", ".join(missing))
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO knowledge_facts
                   (fact_id,domain,topic,term,status,scope,payload)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    fact.fact_id, fact.domain, fact.topic, fact.term, fact.status,
                    fact.scope, json.dumps(fact.to_dict(), ensure_ascii=False),
                ),
            )
            if self._fts_enabled:
                row = self.connection.execute(
                    "SELECT rowid FROM knowledge_facts WHERE fact_id=?", (fact.fact_id,)
                ).fetchone()
                self.connection.execute(
                    "DELETE FROM knowledge_facts_fts WHERE rowid=?", (row["rowid"],)
                )
                self.connection.execute(
                    """INSERT INTO knowledge_facts_fts
                       (rowid,term,domain,topic,definition,tags,related_terms)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        row["rowid"], fact.term, fact.domain, fact.topic,
                        fact.definition, " ".join(fact.tags), " ".join(fact.related_terms),
                    ),
                )

    def search(
        self, query: str, limit: int = 12,
        scope: str | None = None, domain: str | None = None,
    ) -> list[KnowledgeFact]:
        if limit < 1:
            raise ValueError("limit must be positive")
        terms = _terms(query)
        if not terms:
            return []
        if self._fts_enabled:
            match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
            sql = """SELECT f.payload FROM knowledge_facts_fts x
                     JOIN knowledge_facts f ON f.rowid=x.rowid
                     WHERE knowledge_facts_fts MATCH ?"""
            args: list[object] = [match]
            if scope:
                sql += " AND f.scope=?"
                args.append(scope)
            if domain:
                sql += " AND f.domain=?"
                args.append(domain)
            sql += " ORDER BY bm25(knowledge_facts_fts), f.term LIMIT ?"
            args.append(limit)
            try:
                rows = self.connection.execute(sql, args).fetchall()
                return [KnowledgeFact.from_dict(json.loads(row["payload"])) for row in rows]
            except sqlite3.OperationalError:
                pass

        clauses = " OR ".join(
            ["(term LIKE ? OR domain LIKE ? OR topic LIKE ? OR payload LIKE ?)"] * len(terms)
        )
        args = []
        for term in terms:
            p = f"%{term}%"
            args.extend([p, p, p, p])
        sql = f"SELECT payload FROM knowledge_facts WHERE ({clauses})"
        if scope:
            sql += " AND scope=?"
            args.append(scope)
        if domain:
            sql += " AND domain=?"
            args.append(domain)
        sql += " ORDER BY domain,topic,term LIMIT ?"
        args.append(limit)
        rows = self.connection.execute(sql, args).fetchall()
        return [KnowledgeFact.from_dict(json.loads(row["payload"])) for row in rows]

    def search_facts(
        self, query: str, part: int | None = None,
        continuity: str | None = None, limit: int = 20,
    ) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")
        pattern = f"%{query.strip()}%"
        sql = """SELECT f.payload AS fact_payload, e.payload AS entity_payload
                 FROM facts f JOIN entities e ON e.entity_id=f.subject_id
                 WHERE (e.name LIKE ? OR f.predicate LIKE ? OR f.object_value LIKE ?)"""
        args: list[object] = [pattern, pattern, pattern]
        if part is not None:
            sql += " AND f.part=?"
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
                    "SELECT payload FROM sources WHERE source_id=?", (source_id,)
                ).fetchone()
                if source_row:
                    sources.append(json.loads(source_row["payload"]))
            results.append({"entity": entity, "fact": fact, "sources": sources})
        return results

    def all_facts(self, scope: str | None = None) -> list[KnowledgeFact]:
        sql = "SELECT payload FROM knowledge_facts"
        args: tuple = ()
        if scope:
            sql += " WHERE scope=?"
            args = (scope,)
        sql += " ORDER BY domain,topic,term"
        return [
            KnowledgeFact.from_dict(json.loads(row["payload"]))
            for row in self.connection.execute(sql, args)
        ]

    def count(self) -> int:
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM knowledge_facts"
        ).fetchone()[0])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
