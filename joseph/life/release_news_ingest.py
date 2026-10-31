"""Import official HoYoverse release announcements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .release_news_schema import OfficialBulletin
from .release_news_store import OfficialNewsStore


def read_jsonl(path: str | Path) -> list[dict]:
    result = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        result.append(row)
    return result


def ingest(sources_file: str, bulletins_file: str, database: str) -> int:
    with OfficialNewsStore(database) as store:
        for row in read_jsonl(sources_file):
            store.add_source(row)
        for row in read_jsonl(bulletins_file):
            store.add_bulletin(OfficialBulletin.from_dict(row))
        return store.count()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="datasets/teyvat/chronicle_sources.jsonl")
    parser.add_argument("--news", default="datasets/teyvat/release_news.jsonl")
    parser.add_argument("--db", default="local/official_news.sqlite")
    args = parser.parse_args()
    print(f"Imported {ingest(args.sources,args.news,args.db)} official publisher bulletins.")


if __name__ == "__main__":
    main()
