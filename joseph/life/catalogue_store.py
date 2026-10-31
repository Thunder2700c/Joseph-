"""SQLite store for Archon, Story, Hangout, World and Event quest indexes."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .catalogue_schema import QuestCatalogueEntry


_STOPWORDS = {
    "the", "a", "an", "about", "tell", "me", "my", "to", "what", "is",
    "are", "of", "for", "can", "you", "please", "do", "does", "did",
    "and", "or", "in", "on", "at", "how", "why", "when", "where",
    "was", "were", "with", "this", "that", "it", "i", "we", "our", "your",
}


def _terms(query: str) -> list[str]:
    return list(dict.fromkeys(
        term.lower() for term in re.findall(r"[\\w]+", query, flags=re.UNICODE)
        if len(term) > 1 and term.lower() not in _STOPWORDS
    ))[:12]


class QuestCatalogueStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS quest_sources (
                source_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS quest_catalogue (
                quest_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                region TEXT NOT NULL,
                quest_type TEXT NOT NULL,
                parent_quest_id TEXT NOT NULL,
                index_state TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS quest_catalogue_region_idx
                ON quest_catalogue(region, quest_type, title);
            CREATE INDEX IF NOT EXISTS quest_catalogue_parent_idx
                ON quest_catalogue(parent_quest_id);
            """
        )
        self.connection.commit()

    def add_source(self, source: dict) -> None:
        if not source.get("source_id") or not source.get("url"):
            raise ValueError("source_id and url are required")
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO quest_sources(source_id,payload) VALUES (?,?)",
                (source["source_id"], json.dumps(source, ensure_ascii=False)),
            )

    def add_quest(self, quest: QuestCatalogueEntry) -> None:
        missing = [
            source_id for source_id in quest.source_ids
            if self.connection.execute(
                "SELECT 1 FROM quest_sources WHERE source_id=?", (source_id,)
            ).fetchone() is None
        ]
        if missing:
            raise ValueError("unknown quest sources: " + ", ".join(missing))
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO quest_catalogue
                   (quest_id,title,region,quest_type,parent_quest_id,index_state,payload)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    quest.quest_id, quest.title, quest.region, quest.quest_type,
                    quest.parent_quest_id, quest.index_state,
                    json.dumps(quest.to_dict(), ensure_ascii=False),
                ),
            )

    def search(
        self,
        query: str = "",
        region: str | None = None,
        quest_type: str | None = None,
        limit: int = 100,
    ) -> list[QuestCatalogueEntry]:
        if limit < 1:
            raise ValueError("limit must be positive")
        sql = "SELECT payload FROM quest_catalogue WHERE 1=1"
        args: list[object] = []
        terms = _terms(query)
        if terms:
            sql += " AND (" + " OR ".join(
                ["(title LIKE ? OR payload LIKE ?)"] * len(terms)
            ) + ")"
            for term in terms:
                pattern = f"%{term}%"
                args.extend([pattern, pattern])
        if region:
            sql += " AND region=?"
            args.append(region)
        if quest_type:
            sql += " AND quest_type=?"
            args.append(quest_type)
        sql += " ORDER BY region, quest_type, title LIMIT ?"
        args.append(limit)
        return [
            QuestCatalogueEntry.from_dict(json.loads(row["payload"]))
            for row in self.connection.execute(sql, args)
        ]

    def counts_by_type(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT quest_type,COUNT(*) AS count FROM quest_catalogue GROUP BY quest_type"
        )
        return {row["quest_type"]: int(row["count"]) for row in rows}

    def count(self) -> int:
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM quest_catalogue"
        ).fetchone()[0])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "QuestCatalogueStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
