"""Import the versioned quest index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalogue_schema import QuestCatalogueEntry
from .catalogue_store import QuestCatalogueStore


def read_jsonl(path: str | Path) -> list[dict]:
    items = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        items.append(row)
    return items


def ingest(sources_file: str, quests_file: str, database: str) -> int:
    with QuestCatalogueStore(database) as store:
        for source in read_jsonl(sources_file):
            store.add_source(source)
        for row in read_jsonl(quests_file):
            store.add_quest(QuestCatalogueEntry.from_dict(row))
        return store.count()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="datasets/teyvat/chronicle_sources.jsonl")
    parser.add_argument("--quests", default="datasets/teyvat/quest_catalogue.jsonl")
    parser.add_argument("--db", default="local/quest_catalogue.sqlite")
    args = parser.parse_args()
    print(f"Imported {ingest(args.sources, args.quests, args.db)} indexed quests.")


if __name__ == "__main__":
    main()
