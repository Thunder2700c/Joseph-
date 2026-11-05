"""Build an auditable corpus of human-authored conversational data.

This stage prepares dialogue only. It does not train a model, include private
user chats, include model-generated data, or add Hu Tao identity/life-stage
material. Official evaluation splits remain separate from training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from training.prepare_oasst1_corpus import OASST1_REVISION, conversation_samples

DAILYDIALOG_SOURCE = "roskoN/dailydialog"
DAILYDIALOG_REVISION = "1354edc25b185125201352b283cf9480c2fa91de"
EMPATHETIC_SOURCE = "facebook/empathetic_dialogues"
EMPATHETIC_REVISION = "d8b8066"

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")

LICENSES = {
    "oasst1": {
        "source": "OpenAssistant/oasst1",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "notes": "Reviewed, non-deleted, non-synthetic English messages only.",
    },
    "daily_dialog": {
        "source": DAILYDIALOG_SOURCE,
        "revision": DAILYDIALOG_REVISION,
        "license": "CC-BY-NC-SA-4.0",
        "url": "https://huggingface.co/datasets/roskoN/dailydialog",
        "notes": "Non-commercial; ShareAlike applies to adaptations. Preserve attribution.",
    },
    "empathetic_dialogues": {
        "source": EMPATHETIC_SOURCE,
        "revision": EMPATHETIC_REVISION,
        "license": "CC-BY-NC-4.0",
        "url": "https://huggingface.co/datasets/facebook/empathetic_dialogues",
        "notes": "Non-commercial. Preserve attribution.",
    },
}


def normalize(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def acceptable_text(value: Any, *, minimum: int = 2, maximum: int = 6000) -> str | None:
    if not isinstance(value, str):
        return None
    text = normalize(value)
    if not minimum <= len(text) <= maximum or EMAIL_PATTERN.search(text):
        return None
    return text


def serialize_turns(turns: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for role, content in turns:
        marker = "<|user|>" if role == "user" else "<|assistant|>"
        ending = " <|eot|>" if role == "assistant" else ""
        parts.append(f"{marker} {content}{ending}")
    return " ".join(parts)


def daily_dialog_samples(row: dict[str, Any]) -> list[str]:
    utterances = row.get("utterances")
    if not isinstance(utterances, list) or len(utterances) < 2:
        return []
    clean = [acceptable_text(item) for item in utterances]
    results: list[str] = []
    for target_index in range(1, len(clean), 2):
        if clean[target_index] is None or clean[target_index - 1] is None:
            continue
        start = max(0, target_index - 6)
        if start % 2:
            start += 1
        turns: list[tuple[str, str]] = []
        valid = True
        for index in range(start, target_index + 1):
            content = clean[index]
            if content is None:
                valid = False
                break
            turns.append(("user" if index % 2 == 0 else "assistant", content))
        if valid and turns and turns[-1][0] == "assistant":
            sample = serialize_turns(turns)
            if len(sample) >= 24:
                results.append(sample)
    return results


def empathetic_sample(row: dict[str, Any]) -> str | None:
    prompt = acceptable_text(row.get("prompt"), minimum=4)
    response = acceptable_text(row.get("utterance"))
    if prompt is None or response is None:
        return None
    context = acceptable_text(row.get("context"), maximum=100)
    prefix = f"<|system|> Conversation situation: {context} " if context else ""
    return f"{prefix}<|user|> {prompt} <|assistant|> {response} <|eot|>"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def iter_daily_splits(dataset: Any) -> tuple[dict[str, list[str]], dict[str, int]]:
    output = {"train": [], "validation": [], "test": []}
    counts: dict[str, int] = {}
    aliases = {"validation": ("validation", "valid", "dev"), "test": ("test",)}
    for split in ("train", "validation", "test"):
        source_split = split
        if source_split not in dataset and split != "train":
            source_split = next((name for name in aliases[split] if name in dataset), "")
        if not source_split:
            continue
        rows = dataset[source_split]
        counts[source_split] = len(rows)
        for row in rows:
            output[split].extend(daily_dialog_samples(dict(row)))
    if not all(output.values()):
        raise RuntimeError(f"DailyDialog has empty prepared split(s): { {k: len(v) for k, v in output.items()} }")
    return output, counts


def iter_empathetic_splits(dataset: Any) -> tuple[dict[str, list[str]], dict[str, int]]:
    output = {"train": [], "validation": [], "test": []}
    counts: dict[str, int] = {}
    for split in output:
        if split not in dataset:
            raise RuntimeError(f"EmpatheticDialogues is missing official {split} split")
        rows = dataset[split]
        counts[split] = len(rows)
        for row in rows:
            sample = empathetic_sample(dict(row))
            if sample is not None:
                output[split].append(sample)
    if not all(output.values()):
        raise RuntimeError(f"EmpatheticDialogues has empty prepared split(s): { {k: len(v) for k, v in output.items()} }")
    return output, counts


def deduplicate_splits(groups: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, int]]:
    # Hold-out examples have priority: exact duplicates cannot leak into train.
    seen: set[str] = set()
    result = {"train": [], "validation": [], "test": []}
    duplicates: dict[str, int] = defaultdict(int)
    for split in ("test", "validation", "train"):
        for sample in groups[split]:
            digest = sha256_text(sample)
            if digest in seen:
                duplicates[split] += 1
                continue
            seen.add(digest)
            result[split].append(sample)
    return result, dict(duplicates)


def build_corpus(output_dir: str | Path) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets==3.3.2 to acquire dialogue sources") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # This existing tree-safe builder excludes synthetic OASST messages and
    # separates complete conversation trees into train and validation.
    oasst = load_dataset("OpenAssistant/oasst1", revision=OASST1_REVISION)
    if "train" not in oasst or "validation" not in oasst:
        raise RuntimeError("OASST1 snapshot is missing train or validation split")
    oasst_train, oasst_validation, oasst_report = conversation_samples(
        oasst["train"].to_list() + oasst["validation"].to_list(),
        validation_percent=5,
    )

    daily = load_dataset(DAILYDIALOG_SOURCE, "full", revision=DAILYDIALOG_REVISION)
    daily_groups, daily_rows = iter_daily_splits(daily)

    empathetic = load_dataset(EMPATHETIC_SOURCE, revision=EMPATHETIC_REVISION)
    empathetic_groups, empathetic_rows = iter_empathetic_splits(empathetic)

    raw = {
        "train": oasst_train + daily_groups["train"] + empathetic_groups["train"],
        "validation": oasst_validation + daily_groups["validation"] + empathetic_groups["validation"],
        "test": daily_groups["test"] + empathetic_groups["test"],
    }
    corpus, duplicates = deduplicate_splits(raw)
    if not all(corpus.values()):
        raise RuntimeError("Final corpus must have non-empty train/validation/test")

    for split, samples in corpus.items():
        (output / f"{split}.txt").write_text(
            "\n".join(samples) + "\n", encoding="utf-8", newline="\n"
        )

    preview = [
        "# Human-authored dialogue corpus preview",
        "",
        "Audit artifact only; not a model release.",
        "",
    ]
    for split in ("validation", "test"):
        preview.extend([f"## {split.title()} examples", ""])
        for sample in corpus[split][:12]:
            preview.extend(["---", sample, ""])
    (output / "preview.md").write_text("\n".join(preview), encoding="utf-8")

    report = {
        "format": "hutao-human-dialogue-corpus-v2",
        "purpose": "general conversational behaviour and emotional dialogue research",
        "sources": {
            "oasst1": {
                **LICENSES["oasst1"],
                **oasst_report,
                "source_train_rows": len(oasst["train"]),
                "source_validation_rows": len(oasst["validation"]),
            },
            "daily_dialog": {
                **LICENSES["daily_dialog"],
                "source_rows_by_split": daily_rows,
                "prepared_samples_by_split": {k: len(v) for k, v in daily_groups.items()},
            },
            "empathetic_dialogues": {
                **LICENSES["empathetic_dialogues"],
                "source_rows_by_split": empathetic_rows,
                "prepared_samples_by_split": {k: len(v) for k, v in empathetic_groups.items()},
            },
        },
        "samples_by_split": {k: len(v) for k, v in corpus.items()},
        "exact_duplicates_removed_by_split": duplicates,
        "files": {
            f"{split}.txt": {
                "bytes": (output / f"{split}.txt").stat().st_size,
                "sha256": sha256_text((output / f"{split}.txt").read_text(encoding="utf-8")),
            }
            for split in corpus
        },
        "tokenizer_policy": "fit on train.txt only; never fit on validation.txt or test.txt",
        "private_user_conversations_included": False,
        "model_generated_training_data_included": False,
        "hu_tao_identity_or_life_stage_material_included": False,
        "noncommercial_sources_present": True,
        "distribution_note": (
            "DailyDialog and EmpatheticDialogues carry non-commercial restrictions; "
            "DailyDialog also carries ShareAlike terms. Preserve attributions and review "
            "obligations before distributing a model trained on this mixture."
        ),
        "excluded_candidates": {
            "personachat": "held pending a clear dataset-specific licence audit",
            "multi_session_chat": "held pending a clear dataset-specific licence audit",
            "topical_chat": "held for separate review of its CDLA-Sharing-1.0 data agreement",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = build_corpus(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
