"""Store and search the publisher's real-world announcement feed."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .release_news_schema import OfficialBulletin


class OfficialNewsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS official_news_sources (
                source_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS official_news (
                news_id TEXT PRIMARY KEY,
                posted_at TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS official_news_date_idx
                ON official_news(posted_at DESC);
            """
        )
        self.connection.commit()

    def add_source(self, source: dict) -> None:
        if not source.get("source_id") or not source.get("url"):
            raise ValueError("source_id and url are required")
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO official_news_sources(source_id,payload) VALUES (?,?)",
                (source["source_id"], json.dumps(source, ensure_ascii=False)),
            )

    def add_bulletin(self, bulletin: OfficialBulletin) -> None:
        missing = [
            source_id for source_id in bulletin.source_ids
            if self.connection.execute(
                "SELECT 1 FROM official_news_sources WHERE source_id=?", (source_id,)
            ).fetchone() is None
        ]
        if missing:
            raise ValueError("unknown news sources: " + ", ".join(missing))
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO official_news
                   (news_id,posted_at,category,status,payload) VALUES (?,?,?,?,?)""",
                (
                    bulletin.news_id, bulletin.posted_at, bulletin.category,
                    bulletin.status, json.dumps(bulletin.to_dict(), ensure_ascii=False),
                ),
            )

    def search(
        self,
        query: str = "",
        category: str | None = None,
        limit: int = 50,
    ) -> list[OfficialBulletin]:
        if limit < 1:
            raise ValueError("limit must be positive")
        sql = "SELECT payload FROM official_news WHERE 1=1"
        args: list[object] = []
        if query.strip():
            sql += " AND payload LIKE ?"
            args.append(f"%{query.strip()}%")
        if category:
            sql += " AND category=?"
            args.append(category)
        sql += " ORDER BY posted_at DESC LIMIT ?"
        args.append(limit)
        return [
            OfficialBulletin.from_dict(json.loads(row["payload"]))
            for row in self.connection.execute(sql, args)
        ]

    def all_news(self, limit: int = 500) -> list[OfficialBulletin]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self.connection.execute(
            "SELECT payload FROM official_news ORDER BY posted_at DESC LIMIT ?",
            (limit,),
        )
        return [OfficialBulletin.from_dict(json.loads(row["payload"])) for row in rows]

    def count(self) -> int:
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM official_news"
        ).fetchone()[0])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "OfficialNewsStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
