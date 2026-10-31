"""Load source-linked, line-sized memory records from JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import LifeMemory, LifeSource
from .store import LifeMemoryStore
from .builder import LifeBuilder


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(record)
    return records


def ingest(source_file: str, memory_file: str, plan_file: str, database: str) -> None:
    with LifeMemoryStore(database) as store:
        LifeBuilder(store, plan_file)
        for record in read_jsonl(source_file):
            store.add_source(LifeSource(**record))
        for record in read_jsonl(memory_file):
            store.add_memory(LifeMemory.from_dict(record))
        print(f"Loaded {len(store.memories())} ordered memories into {database}")
        print("Awakening ready:", LifeBuilder(store, plan_file).ready_to_awaken())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="character/life_sources.jsonl")
    parser.add_argument("--memories", default="character/life_memories.jsonl")
    parser.add_argument("--plan", default="character/life_stages.json")
    parser.add_argument("--db", default="local/hu_tao_life.sqlite")
    args = parser.parse_args()
    ingest(args.sources, args.memories, args.plan, args.db)


if __name__ == "__main__":
    main()
