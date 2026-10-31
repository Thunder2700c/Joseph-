"""Export only user-approved taught facts to an optional training JSONL file.

Raw conversation turns are never exported by this command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from joseph.world import WorldKnowledgeStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="local/world_knowledge.sqlite")
    parser.add_argument("--output", default="artifacts/approved_world_facts.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with WorldKnowledgeStore(args.db) as store:
        approved = store.for_training()
        with output.open("w", encoding="utf-8") as handle:
            for item in approved:
                handle.write(json.dumps({
                    "subject": item["subject"],
                    "category": item["category"],
                    "text": item["statement"],
                    "origin": item["origin"],
                    "tags": item["tags"],
                }, ensure_ascii=False) + "\n")
    print(f"Exported {len(approved)} explicitly approved facts to {output}")


if __name__ == "__main__":
    main()
