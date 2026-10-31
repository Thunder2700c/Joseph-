"""Import source-aware common-world and Teyvat assumption records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import KnowledgeFact, KnowledgeSource
from .store import KnowledgeStore


def read_jsonl(path: str | Path) -> list[dict]:
    result = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        result.append(data)
    return result


def ingest(sources_path: str, facts_path: str, database: str) -> int:
    with KnowledgeStore(database) as store:
        for row in read_jsonl(sources_path):
            store.add_source(KnowledgeSource(**row))
        for row in read_jsonl(facts_path):
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
    count = ingest(args.sources, args.facts, args.db)
    print(f"Loaded {count} common-world and Teyvat-assumption facts into {args.db}")


if __name__ == "__main__":
    main()
