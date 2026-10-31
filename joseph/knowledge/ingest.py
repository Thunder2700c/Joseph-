"""Import JSONL source/entity/fact records into a local SQLite knowledge DB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .schema import EntityRecord, FactRecord, SourceRecord
from .store import KnowledgeStore


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
        records.append(record)
    return records


def ingest_bundle(
    sources_path: str | Path,
    entities_path: str | Path,
    facts_path: str | Path,
    database_path: str | Path,
) -> tuple[int, int, int]:
    sources = read_jsonl(sources_path)
    entities = read_jsonl(entities_path)
    facts = read_jsonl(facts_path)

    with KnowledgeStore(database_path) as store:
        for record in sources:
            store.add_source(SourceRecord(**record))
        for record in entities:
            record["aliases"] = tuple(record.get("aliases", ()))
            store.add_entity(EntityRecord(**record))
        for record in facts:
            record["source_ids"] = tuple(record.get("source_ids", ()))
            store.add_fact(FactRecord(**record))

    return len(sources), len(entities), len(facts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="datasets/jojo/sources.jsonl")
    parser.add_argument("--entities", default="datasets/jojo/parts.jsonl")
    parser.add_argument("--facts", default="datasets/jojo/facts.jsonl")
    parser.add_argument("--db", default="local/jojo_knowledge.sqlite")
    args = parser.parse_args()
    counts = ingest_bundle(args.sources, args.entities, args.facts, args.db)
    print(f"Imported sources={counts[0]}, entities={counts[1]}, facts={counts[2]}")
    print(f"Knowledge database: {args.db}")


if __name__ == "__main__":
    main()
