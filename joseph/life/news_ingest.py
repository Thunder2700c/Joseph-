"""Import sourced Teyvat story beats and Hu Tao's in-world news reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .news import NewsTransmission, StoryBeat, TeyvatChronicleStore


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        records.append(row)
    return records


def ingest(
    sources_file: str,
    beats_file: str,
    reports_file: str,
    database: str,
) -> tuple[int, int, int]:
    with TeyvatChronicleStore(database) as store:
        sources = read_jsonl(sources_file)
        beats = read_jsonl(beats_file)
        reports = read_jsonl(reports_file)
        for source in sources:
            store.add_source(source)
        for row in beats:
            row["source_ids"] = tuple(row.get("source_ids", ()))
            row["spoiler_scope"] = tuple(row.get("spoiler_scope", ()))
            store.add_story_beat(StoryBeat(**row))
        for row in reports:
            row["source_ids"] = tuple(row.get("source_ids", ()))
            store.add_report(NewsTransmission(**row))
        return len(sources), len(beats), len(reports)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="datasets/teyvat/chronicle_sources.jsonl")
    parser.add_argument("--beats", default="datasets/teyvat/story_beats.jsonl")
    parser.add_argument("--reports", default="datasets/teyvat/news_reports.jsonl")
    parser.add_argument("--db", default="local/teyvat_chronicle.sqlite")
    args = parser.parse_args()
    counts = ingest(args.sources, args.beats, args.reports, args.db)
    print(
        f"Imported {counts[0]} sources, {counts[1]} story beats, "
        f"and {counts[2]} news transmissions into {args.db}"
    )


if __name__ == "__main__":
    main()
