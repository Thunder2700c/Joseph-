"""Prepare a provenance-tracked, tree-safe OASST1 dialogue supplement.

This is general conversational language data only. It does not contain Hu Tao
identity material and must never be used to bypass the life-stage review gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

OASST1_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
OASST1_LICENSE = "apache-2.0"
EXCLUDED_LABELS = {
    "spam": 0.0,
    "lang_mismatch": 0.0,
    "pii": 0.0,
    "not_appropriate": 0.5,
    "hate_speech": 0.5,
    "sexual_content": 0.5,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_value(labels: Any, name: str) -> float | None:
    """Read labels from either HF's mapping or sequence-of-records shape."""
    if isinstance(labels, dict):
        item = labels.get(name)
        if isinstance(item, dict) and item.get("value") is not None:
            return float(item["value"])
        if isinstance(item, (int, float)):
            return float(item)
        names = labels.get("name")
        values = labels.get("value")
        if isinstance(names, list) and isinstance(values, list):
            for label_name, value in zip(names, values):
                if label_name == name and value is not None:
                    return float(value)
    if isinstance(labels, list):
        for item in labels:
            if isinstance(item, dict) and item.get("name") == name:
                value = item.get("value")
                return float(value) if value is not None else None
    return None


def eligible_message(row: dict[str, Any]) -> tuple[bool, str | None]:
    if row.get("lang") != "en":
        return False, "non_english"
    if row.get("role") not in {"prompter", "assistant"}:
        return False, "unknown_role"
    if row.get("review_result") is not True:
        return False, "not_review_approved"
    if row.get("deleted") is not False:
        return False, "deleted_or_unknown"
    if row.get("synthetic") is not False:
        return False, "synthetic_or_unknown"
    if row.get("tree_state") != "ready_for_export":
        return False, "tree_not_ready"
    text = row.get("text")
    if not isinstance(text, str) or len(text.strip()) < 2:
        return False, "empty_or_short"
    if len(text) > 6000:
        return False, "message_too_long"
    labels = row.get("labels")
    for label, threshold in EXCLUDED_LABELS.items():
        score = label_value(labels, label)
        if score is not None and score > threshold:
            return False, f"flagged_{label}"
    return True, None


def _normalise_message(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def conversation_samples(rows: list[dict[str, Any]], validation_percent: int = 5) -> tuple[list[str], list[str], dict]:
    """Create one causal SFT sequence for each preferred assistant reply."""
    if not 1 <= validation_percent <= 49:
        raise ValueError("validation_percent must be between 1 and 49")

    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    rejected: dict[str, int] = defaultdict(int)
    for row in rows:
        ok, reason = eligible_message(row)
        if not ok:
            rejected[reason or "unknown"] += 1
            continue
        tree_id = row.get("message_tree_id")
        message_id = row.get("message_id")
        if not tree_id or not message_id:
            rejected["missing_id"] += 1
            continue
        groups[str(tree_id)][str(message_id)] = row

    train: list[str] = []
    validation: list[str] = []
    trees_with_samples = 0
    broken_chains = 0
    duplicate_messages = 0
    seen_ids: set[str] = set()

    for tree_id, messages in sorted(groups.items()):
        children: dict[str, list[dict[str, Any]]] = defaultdict(list)
        roots: list[dict[str, Any]] = []
        for row in messages.values():
            message_id = str(row["message_id"])
            if message_id in seen_ids:
                duplicate_messages += 1
                continue
            seen_ids.add(message_id)
            parent_id = row.get("parent_id")
            if parent_id is None:
                roots.append(row)
            elif str(parent_id) in messages:
                children[str(parent_id)].append(row)

        # For each parent, retain its highest-ranked approved assistant reply.
        # Prompter follow-ups remain separate branches and are all retained.
        preferred_children: dict[str, list[dict[str, Any]]] = {}
        for parent_id, candidates in children.items():
            assistant_children = [r for r in candidates if r.get("role") == "assistant"]
            prompter_children = [r for r in candidates if r.get("role") == "prompter"]
            if assistant_children:
                assistant_children.sort(
                    key=lambda r: (
                        r.get("rank") is None,
                        int(r.get("rank") or 0),
                        str(r["message_id"]),
                    )
                )
                preferred_children[parent_id] = [assistant_children[0]] + prompter_children
            else:
                preferred_children[parent_id] = prompter_children

        # Reconstruct the unique parent chain of each selected assistant reply.
        by_id = messages
        tree_samples: list[str] = []
        for row in messages.values():
            if row.get("role") != "assistant":
                continue
            parent_id = row.get("parent_id")
            if parent_id is None or str(parent_id) not in by_id:
                continue
            siblings = preferred_children.get(str(parent_id), [])
            if not any(str(item["message_id"]) == str(row["message_id"]) for item in siblings):
                continue

            chain: list[dict[str, Any]] = []
            current: dict[str, Any] | None = row
            visited: set[str] = set()
            valid = True
            while current is not None:
                current_id = str(current["message_id"])
                if current_id in visited:
                    valid = False
                    break
                visited.add(current_id)
                chain.append(current)
                ancestor_id = current.get("parent_id")
                if ancestor_id is None:
                    break
                current = by_id.get(str(ancestor_id))
                if current is None:
                    valid = False
                    break
            if not valid:
                broken_chains += 1
                continue
            chain.reverse()
            roles = [item.get("role") for item in chain]
            if not roles or roles[0] != "prompter" or roles[-1] != "assistant":
                continue
            if any(roles[index] == roles[index - 1] for index in range(1, len(roles))):
                continue

            # The 256-token training context is short; retain the most recent
            # four turns, beginning with a user turn where possible.
            chain = chain[-8:]
            if chain and chain[0].get("role") == "assistant":
                chain = chain[1:]
            if not chain or chain[-1].get("role") != "assistant":
                continue
            parts = []
            for item in chain:
                body = _normalise_message(str(item["text"]))
                if not body:
                    continue
                if item["role"] == "prompter":
                    parts.append("<|user|> " + body)
                else:
                    parts.append("Assistant: " + body)
            sample = " ".join(parts)
            if len(sample) >= 32:
                tree_samples.append(sample)

        if tree_samples:
            trees_with_samples += 1
            bucket = int(hashlib.sha256(tree_id.encode("utf-8")).hexdigest()[:8], 16) % 100
            target = validation if bucket < validation_percent else train
            target.extend(tree_samples)

    if not train or not validation:
        raise ValueError(
            f"tree-level split produced train={len(train)} validation={len(validation)}; "
            "check source split and validation_percent"
        )

    report = {
        "source": "OpenAssistant/oasst1",
        "source_revision": OASST1_REVISION,
        "license": OASST1_LICENSE,
        "split_policy": f"deterministic message_tree_id SHA-256 bucket; validation={validation_percent}%",
        "rows_seen": len(rows),
        "eligible_tree_count": len(groups),
        "trees_with_dialogue_samples": trees_with_samples,
        "train_dialogue_samples": len(train),
        "validation_dialogue_samples": len(validation),
        "rejected_message_reasons": dict(sorted(rejected.items())),
        "broken_or_incomplete_parent_chains": broken_chains,
        "duplicate_message_ids": duplicate_messages,
        "train_validation_tree_overlap": 0,
        "synthetic_records_allowed": False,
        "language_filter": "en",
        "private_chat_history_included": False,
    }
    return train, validation, report


def _read_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def mix_corpora(
    base_train_path: str,
    base_validation_path: str,
    dialogue_train: list[str],
    dialogue_validation: list[str],
    tokenizer_path: str,
    output_dir: str,
    dialogue_fraction: float = 0.25,
    seed: int = 1337,
) -> dict:
    if not 0.05 <= dialogue_fraction <= 0.5:
        raise ValueError("dialogue_fraction must be between 0.05 and 0.50")
    from joseph.tokenizer import ByteBPETokenizer

    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    base_train = _read_lines(base_train_path)
    base_validation = _read_lines(base_validation_path)
    if not base_train or not base_validation or not dialogue_train or not dialogue_validation:
        raise ValueError("all four train/validation source groups must be non-empty")

    def token_total(samples: list[str]) -> int:
        return sum(len(tokenizer.encode(sample)) + 2 for sample in samples)

    base_tokens = token_total(base_train)
    dialogue_tokens = token_total(dialogue_train)
    repeats = max(
        1,
        math.ceil(
            dialogue_fraction * base_tokens
            / ((1.0 - dialogue_fraction) * dialogue_tokens)
        ),
    )
    combined_train = base_train + dialogue_train * repeats
    random.Random(seed).shuffle(combined_train)
    combined_validation = base_validation + dialogue_validation
    random.Random(seed + 1).shuffle(combined_validation)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.txt"
    validation_path = output / "validation.txt"
    train_path.write_text("\n".join(combined_train) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join(combined_validation) + "\n", encoding="utf-8")

    final_base_tokens = base_tokens
    final_dialogue_tokens = dialogue_tokens * repeats
    report = {
        "format": "joseph-mixed-corpus-v1",
        "base_train_samples": len(base_train),
        "base_validation_samples": len(base_validation),
        "dialogue_train_samples_before_upsampling": len(dialogue_train),
        "dialogue_validation_samples": len(dialogue_validation),
        "dialogue_train_repetitions": repeats,
        "dialogue_token_fraction_target": dialogue_fraction,
        "dialogue_token_fraction_achieved_estimate": final_dialogue_tokens / max(
            final_base_tokens + final_dialogue_tokens, 1
        ),
        "train_token_count_estimate": final_base_tokens + final_dialogue_tokens,
        "validation_sample_count": len(combined_validation),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "seed": seed,
    }
    (output / "mixture-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--base-validation", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dialogue-fraction", type=float, default=0.25)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(
        "OpenAssistant/oasst1",
        revision=OASST1_REVISION,
    )
    rows = []
    for split_name in ("train", "validation"):
        if split_name not in dataset:
            raise RuntimeError(f"OASST1 snapshot is missing split {split_name}")
        rows.extend(dataset[split_name].to_list())

    train_samples, validation_samples, source_report = conversation_samples(
        rows, validation_percent=args.validation_percent
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dialogue_train_path = out / "oasst-train.txt"
    dialogue_validation_path = out / "oasst-validation.txt"
    dialogue_train_path.write_text("\n".join(train_samples) + "\n", encoding="utf-8")
    dialogue_validation_path.write_text(
        "\n".join(validation_samples) + "\n", encoding="utf-8"
    )
    source_report.update({
        "downloaded_dataset_revision": OASST1_REVISION,
        "dataset_source_train_rows": len(dataset["train"]),
        "dataset_source_validation_rows": len(dataset["validation"]),
        "train_sha256": sha256_file(dialogue_train_path),
        "validation_sha256": sha256_file(dialogue_validation_path),
    })
    (out / "oasst-source-report.json").write_text(
        json.dumps(source_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    mixture_report = mix_corpora(
        args.base_train,
        args.base_validation,
        train_samples,
        validation_samples,
        args.tokenizer,
        args.output_dir,
        args.dialogue_fraction,
        args.seed,
    )
    print(json.dumps({
        "source": source_report,
        "mixture": mixture_report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
