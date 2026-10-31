"""Build deterministic train/validation text files from explicitly selected corpora.

Only paths passed with --input are read. The continuous conversation database is
not accessed by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _text_paragraphs(path: Path) -> Iterator[str]:
    buffer: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                buffer.append(line.rstrip())
            elif buffer:
                yield "\n".join(buffer).strip()
                buffer.clear()
    if buffer:
        yield "\n".join(buffer).strip()


def _jsonl_samples(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            if isinstance(row.get("messages"), list):
                parts = []
                for message in row["messages"]:
                    if isinstance(message, dict):
                        role = str(message.get("role", "user")).strip().lower()
                        text = str(message.get("content", "")).strip()
                        if text:
                            parts.append(f"<|{role}|>\n{text}")
                if parts:
                    yield "\n".join(parts)
            elif row.get("prompt") and row.get("completion"):
                yield (
                    "<|user|>\n" + str(row["prompt"]).strip()
                    + "\n<|assistant|>\n" + str(row["completion"]).strip()
                )
            else:
                value = row.get("text") or row.get("content")
                if isinstance(value, str) and value.strip():
                    yield value.strip()


def iter_samples(path: str | Path, mode: str = "auto") -> Iterator[str]:
    source = Path(path)
    selected = mode
    if selected == "auto":
        selected = "jsonl" if source.suffix.lower() == ".jsonl" else "text"
    if selected == "jsonl":
        yield from _jsonl_samples(source)
    elif selected == "text":
        yield from _text_paragraphs(source)
    else:
        raise ValueError(f"unsupported input mode: {mode}")


def build_corpus(
    inputs: list[str],
    output_dir: str,
    validation_percent: int = 5,
    minimum_characters: int = 24,
    input_mode: str = "auto",
) -> dict:
    if not inputs:
        raise ValueError("at least one explicit input path is required")
    if not 1 <= validation_percent <= 49:
        raise ValueError("validation_percent must be between 1 and 49")
    if minimum_characters < 1:
        raise ValueError("minimum_characters must be positive")

    seen: set[str] = set()
    train_samples: list[str] = []
    validation_samples: list[str] = []
    duplicates = 0
    too_short = 0

    for path in inputs:
        for raw in iter_samples(path, input_mode):
            sample = raw.strip()
            normalised = _normalise(sample).casefold()
            if len(normalised) < minimum_characters:
                too_short += 1
                continue
            digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            bucket = int(digest[:8], 16) % 100
            (validation_samples if bucket < validation_percent else train_samples).append(sample)

    if len(seen) < 2:
        raise ValueError("need at least two unique samples for a train/validation split")
    if not validation_samples:
        validation_samples.append(train_samples.pop())
    if not train_samples:
        train_samples.append(validation_samples.pop())

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    train_text = "\n\n".join(train_samples) + "\n"
    validation_text = "\n\n".join(validation_samples) + "\n"
    train_path = target / "train.txt"
    validation_path = target / "validation.txt"
    train_path.write_text(train_text, encoding="utf-8")
    validation_path.write_text(validation_text, encoding="utf-8")

    report = {
        "format": "joseph-corpus-report-v1",
        "inputs": [str(Path(path)) for path in inputs],
        "input_mode": input_mode,
        "validation_percent_target": validation_percent,
        "unique_samples": len(seen),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "duplicates_removed": duplicates,
        "too_short_removed": too_short,
        "train_sha256": hashlib.sha256(train_text.encode("utf-8")).hexdigest(),
        "validation_sha256": hashlib.sha256(validation_text.encode("utf-8")).hexdigest(),
        "private_chat_history_included": False,
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="explicit text or JSONL source; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--minimum-characters", type=int, default=24)
    parser.add_argument("--mode", choices=["auto", "text", "jsonl"], default="auto")
    args = parser.parse_args()
    report = build_corpus(
        args.input, args.output_dir, args.validation_percent,
        args.minimum_characters, args.mode,
    )
    print(
        f"Built corpus: train={report['train_samples']} "
        f"validation={report['validation_samples']} "
        f"duplicates_removed={report['duplicates_removed']}"
    )


if __name__ == "__main__":
    main()
