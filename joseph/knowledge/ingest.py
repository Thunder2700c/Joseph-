"""Import both legacy entity/fact bundles and source-aware common knowledge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import (
    EntityRecord, FactRecord, KnowledgeFact, KnowledgeSource, SourceRecord,
)
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
    sources_path: str | Path, entities_path: str | Path,
    facts_path: str | Path, database_path: str | Path,
) -> tuple[int, int, int]:
    sources, entities, facts = (
        read_jsonl(sources_path), read_jsonl(entities_path), read_jsonl(facts_path)
    )
    with KnowledgeStore(database_path) as store:
        for row in sources:
            store.add_source(SourceRecord(**row))
        for row in entities:
            row["aliases"] = tuple(row.get("aliases", ()))
            store.add_entity(EntityRecord(**row))
        for row in facts:
            row["source_ids"] = tuple(row.get("source_ids", ()))
            store.add_fact(FactRecord(**row))
    return len(sources), len(entities), len(facts)


def ingest(
    sources_path: str, facts_path: str, database_path: str,
) -> int:
    sources, facts = read_jsonl(sources_path), read_jsonl(facts_path)
    with KnowledgeStore(database_path) as store:
        for row in sources:
            store.add_source(KnowledgeSource(**row))
        for row in facts:
            # Accept the earlier misspelling in saved datasets for backward compatibility.
            if "tevyat_application" in row and "teyvat_application" not in row:
                row["teyvat_application"] = row.pop("tevyat_application")
            else:
                row.pop("tevyat_application", None)
            for key in ("source_ids", "tags", "related_terms"):
                row[key] = tuple(row.get(key, ()))
            store.add_fact(KnowledgeFact(**row))
        return store.count()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="datasets/common_world/sources.jsonl")
    parser.add_argument("--facts", default="datasets/common_world/facts.jsonl")
    parser.add_argument("--db", default="local/common_knowledge.sqlite")
    args = parser.parse_args()
    print(f"Loaded {ingest(args.sources,args.facts,args.db)} facts into {args.db}")


if __name__ == "__main__":
    main()
