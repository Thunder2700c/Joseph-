"""Versioned Teyvat story beats and the channels through which Hu Tao learns them."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


NEWS_CHANNELS = {
    "steambird", "merchant", "traveller", "adventurers_guild",
    "official_dispatch", "letter", "book", "performer", "rumour", "other",
}
RELIABILITY = {"official", "firsthand", "corroborated", "unverified", "rumour"}
KNOWLEDGE_BASIS = {"witnessed", "heard", "read", "inferred", "unknown"}
CANON_CATEGORIES = {"game_canon", "official_supplementary", "in_world_rumour", "user_au"}
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


@dataclass(frozen=True)
class StoryBeat:
    beat_id: str
    arc_id: str
    beat_order: int
    quest_id: str
    quest_title: str
    act_title: str
    region: str
    version_introduced: str
    beat_title: str
    summary: str
    source_ids: tuple[str, ...]
    canon_category: str = "game_canon"
    hu_tao_participates: bool = False
    chronology_note: str = ""
    spoiler_scope: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (
            self.beat_id,self.arc_id,self.quest_id,self.quest_title,self.act_title,
            self.region,self.beat_title,self.summary,
        )):
            raise ValueError("story-beat identifiers, titles, region and summary are required")
        if self.beat_order < 1:
            raise ValueError("beat_order must be positive")
        if self.canon_category not in CANON_CATEGORIES:
            raise ValueError("invalid canon category")
        if not self.source_ids:
            raise ValueError("every canon story beat requires at least one source")

    def to_dict(self) -> dict:
        data=asdict(self)
        data["source_ids"]=list(self.source_ids)
        data["spoiler_scope"]=list(self.spoiler_scope)
        return data


@dataclass(frozen=True)
class NewsTransmission:
    report_id: str
    beat_id: str
    channel: str
    reported_by: str
    report_text: str
    reliability: str
    knowledge_basis: str = "heard"
    source_ids: tuple[str, ...] = ()
    awareness_sequence: int = 0
    transmission_note: str = ""

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.report_id,self.beat_id,self.report_text)):
            raise ValueError("report_id, beat_id and report_text are required")
        if self.channel not in NEWS_CHANNELS:
            raise ValueError(f"unsupported news channel: {self.channel}")
        if self.reliability not in RELIABILITY:
            raise ValueError(f"unsupported reliability: {self.reliability}")
        if self.knowledge_basis not in KNOWLEDGE_BASIS:
            raise ValueError(f"unsupported knowledge basis: {self.knowledge_basis}")
        if self.awareness_sequence < 0:
            raise ValueError("awareness_sequence cannot be negative")

    def to_dict(self) -> dict:
        data=asdict(self)
        data["source_ids"]=list(self.source_ids)
        return data


_STOPWORDS = {
    "the", "a", "an", "about", "tell", "me", "my", "to", "what", "is",
    "are", "of", "for", "can", "you", "please", "do", "does", "did",
    "and", "or", "in", "on", "at", "how", "why", "when", "where",
    "was", "were", "with", "this", "that", "it", "i", "we", "our", "your",
}


def _search_terms(query: str) -> list[str]:
    return list(dict.fromkeys(
        term.lower() for term in re.findall(r"[\\w]+", query, flags=re.UNICODE)
        if len(term) > 1 and term.lower() not in _STOPWORDS
    ))[:12]


class TeyvatChronicleStore:
    def __init__(self, path: str | Path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.connection=sqlite3.connect(self.path)
        self.connection.row_factory=sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chronicle_sources (
                source_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS story_beats (
                beat_id TEXT PRIMARY KEY,
                arc_id TEXT NOT NULL,
                beat_order INTEGER NOT NULL,
                region TEXT NOT NULL,
                version_introduced TEXT NOT NULL,
                hu_tao_participates INTEGER NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(arc_id,beat_order)
            );
            CREATE INDEX IF NOT EXISTS story_beats_region_idx
                ON story_beats(region,arc_id,beat_order);
            CREATE TABLE IF NOT EXISTS news_transmissions (
                report_id TEXT PRIMARY KEY,
                beat_id TEXT NOT NULL REFERENCES story_beats(beat_id),
                channel TEXT NOT NULL,
                reliability TEXT NOT NULL,
                awareness_sequence INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS news_reports_beat_idx
                ON news_transmissions(beat_id);
            """
        )
        self.connection.commit()

    def add_source(self, source: dict) -> None:
        if not source.get("source_id") or not source.get("url"):
            raise ValueError("source_id and url are required")
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO chronicle_sources(source_id,payload) VALUES (?,?)",
                (source["source_id"],json.dumps(source,ensure_ascii=False)),
            )

    def _check_sources(self, source_ids: tuple[str, ...]) -> None:
        for source_id in source_ids:
            if self.connection.execute(
                "SELECT 1 FROM chronicle_sources WHERE source_id=?",(source_id,)
            ).fetchone() is None:
                raise ValueError(f"unknown chronicle source: {source_id}")

    def add_story_beat(self, beat: StoryBeat) -> None:
        self._check_sources(beat.source_ids)
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO story_beats
                   (beat_id,arc_id,beat_order,region,version_introduced,
                    hu_tao_participates,payload) VALUES (?,?,?,?,?,?,?)""",
                (
                    beat.beat_id,beat.arc_id,beat.beat_order,beat.region,
                    beat.version_introduced,int(beat.hu_tao_participates),
                    json.dumps(beat.to_dict(),ensure_ascii=False),
                ),
            )

    def add_report(self, report: NewsTransmission) -> None:
        self._check_sources(report.source_ids)
        if self.connection.execute(
            "SELECT 1 FROM story_beats WHERE beat_id=?",(report.beat_id,)
        ).fetchone() is None:
            raise ValueError(f"unknown story beat: {report.beat_id}")
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO news_transmissions
                   (report_id,beat_id,channel,reliability,awareness_sequence,payload)
                   VALUES (?,?,?,?,?,?)""",
                (
                    report.report_id,report.beat_id,report.channel,report.reliability,
                    report.awareness_sequence,json.dumps(report.to_dict(),ensure_ascii=False),
                ),
            )

    def search_beats(
        self,query: str="",region: str|None=None,arc_id: str|None=None,limit: int=50,
    ) -> list[dict]:
        if limit<1:
            raise ValueError("limit must be positive")
        sql="SELECT payload FROM story_beats WHERE 1=1"
        args:list[object]=[]
        terms=_search_terms(query)
        if terms:
            sql+=" AND ("+" OR ".join(["payload LIKE ?"]*len(terms))+")"
            args.extend(f"%{term}%" for term in terms)
        if region:
            sql+=" AND region=?"
            args.append(region)
        if arc_id:
            sql+=" AND arc_id=?"
            args.append(arc_id)
        sql+=" ORDER BY arc_id,beat_order LIMIT ?"
        args.append(limit)
        return [json.loads(row["payload"]) for row in self.connection.execute(sql,args)]

    def reports_for_beat(self,beat_id: str) -> list[dict]:
        rows=self.connection.execute(
            """SELECT payload FROM news_transmissions
               WHERE beat_id=? ORDER BY awareness_sequence,report_id""",(beat_id,)
        )
        return [json.loads(row["payload"]) for row in rows]

    def known_news(self,query: str="",limit: int=30) -> list[dict]:
        terms=_search_terms(query)
        sql="""
            SELECT b.payload AS beat_payload,n.payload AS report_payload
            FROM news_transmissions n JOIN story_beats b ON b.beat_id=n.beat_id
            WHERE 1=1
        """
        args:list[object]=[]
        if terms:
            sql+=" AND ("+" OR ".join(["(b.payload LIKE ? OR n.payload LIKE ?)"]*len(terms))+")"
            for term in terms:
                args.extend([f"%{term}%",f"%{term}%"])
        sql+=" ORDER BY n.awareness_sequence,b.arc_id,b.beat_order LIMIT ?"
        args.append(limit)
        return [
            {"beat":json.loads(row["beat_payload"]),"report":json.loads(row["report_payload"])}
            for row in self.connection.execute(sql,args)
        ]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "TeyvatChronicleStore":
        return self

    def __exit__(self,exc_type,exc,traceback) -> None:
        self.close()
