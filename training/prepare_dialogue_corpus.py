"""Build a licensed, tree-safe general dialogue supplement for response SFT.

This is generic assistant training only. It does not contain Hu Tao identity
material and must not be used to bypass the character life-stage review gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from training.prepare_oasst1_corpus import (
    OASST1_LICENSE,
    OASST1_REVISION,
    conversation_samples,
    mix_corpora,
    sha256_file,
)

DOLLY_SOURCE = "databricks/databricks-dolly-15k"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DOLLY_LICENSE = "CC-BY-SA-3.0"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def dolly_row_to_sample(row: dict[str, Any]) -> str | None:
    """Convert one human-authored Dolly instruction/response row to chat format."""
    instruction = row.get("instruction")
    response = row.get("response")
    context = row.get("context", "")
    if not isinstance(instruction, str) or not isinstance(response, str):
        return None
    if context is None:
        context = ""
    if not isinstance(context, str):
        return None

    instruction = normalize_text(instruction)
    response = normalize_text(response)
    context = normalize_text(context)
    if len(instruction) < 4 or len(response) < 2:
        return None
    if len(instruction) > 6000 or len(context) > 6000 or len(response) > 6000:
        return None

    user_text = instruction
    if context:
        user_text += "\n\nContext: " + context
    return f"<|user|> {user_text} <|assistant|> {response} <|eot|>"


def prepare_dolly_samples(
    rows: list[dict[str, Any]],
    validation_percent: int = 5,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Deduplicate Dolly rows and split deterministically by content hash."""
    if not 1 <= validation_percent <= 99:
        raise ValueError("validation_percent must be between 1 and 99")

    train: list[str] = []
    validation: list[str] = []
    seen: set[str] = set()
    rejected = 0
    duplicates = 0

    for row in rows:
        sample = dolly_row_to_sample(row)
        if sample is None:
            rejected += 1
            continue
        fingerprint = hashlib.sha256(sample.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        bucket = int(fingerprint[:8], 16) % 100
        (validation if bucket < validation_percent else train).append(sample)

    if not train or not validation:
        raise ValueError(
            f"Dolly split produced train={len(train)} validation={len(validation)}"
        )

    report = {
        "source": DOLLY_SOURCE,
        "source_revision": DOLLY_REVISION,
        "license": DOLLY_LICENSE,
        "split_policy": (
            "deterministic SHA-256 bucket of normalized serialized instruction/response"
            f"; validation={validation_percent}%"
        ),
        "rows_seen": len(rows),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "duplicates_removed": duplicates,
        "rejected_rows": rejected,
        "private_chat_history_included": False,
        "synthetic_answers_included": False,
    }
    return train, validation, report


def prepare_dialogue_sources(
    oasst_rows: list[dict[str, Any]],
    dolly_rows: list[dict[str, Any]],
    validation_percent: int = 5,
) -> tuple[list[str], list[str], dict[str, Any]]:
    oasst_train, oasst_validation, oasst_report = conversation_samples(
        oasst_rows,
        validation_percent=validation_percent,
    )
    dolly_train, dolly_validation, dolly_report = prepare_dolly_samples(
        dolly_rows,
        validation_percent=validation_percent,
    )
    train = oasst_train + dolly_train
    validation = oasst_validation + dolly_validation
    report = {
        "format": "general-dialogue-sources-v1",
        "train_samples": len(train),
        "validation_samples": len(validation),
        "sources": {
            "oasst1": {
                **oasst_report,
                "license": OASST1_LICENSE,
                "train_samples": len(oasst_train),
                "validation_samples": len(oasst_validation),
            },
            "dolly_15k": dolly_report,
        },
        "serialization": (
            "<|user|> prompt <|assistant|> response <|eot|>; "
            "multi-turn OASST messages use the same alternating markers"
        ),
        "loss_policy": (
            "mask prompt/user tokens; train assistant marker, assistant response, "
            "end-of-turn and final EOS"
        ),
        "hu_tao_identity_data_included": False,
        "private_chat_history_included": False,
    }
    return train, validation, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--base-validation", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dialogue-fraction", type=float, default=0.35)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    from datasets import load_dataset

    oasst = load_dataset("OpenAssistant/oasst1", revision=OASST1_REVISION)
    for split_name in ("train", "validation"):
        if split_name not in oasst:
            raise RuntimeError(f"OASST1 snapshot is missing split {split_name}")
    oasst_rows = oasst["train"].to_list() + oasst["validation"].to_list()

    dolly = load_dataset(DOLLY_SOURCE, revision=DOLLY_REVISION, split="train")
    dolly_rows = dolly.to_list()

    dialogue_train, dialogue_validation, source_report = prepare_dialogue_sources(
        oasst_rows,
        dolly_rows,
        validation_percent=args.validation_percent,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_dialogue_path = output / "dialogue-train.txt"
    validation_dialogue_path = output / "dialogue-validation.txt"
    train_dialogue_path.write_text(
        "\n".join(dialogue_train) + "\n",
        encoding="utf-8",
    )
    validation_dialogue_path.write_text(
        "\n".join(dialogue_validation) + "\n",
        encoding="utf-8",
    )
    source_report["train_sha256"] = sha256_file(train_dialogue_path)
    source_report["validation_sha256"] = sha256_file(validation_dialogue_path)
    source_report["dataset_attribution"] = {
        "oasst1": (
            "OpenAssistant/oasst1, Apache-2.0; preserve upstream copyright and licence notices."
        ),
        "dolly_15k": (
            "Databricks databricks-dolly-15k, CC BY-SA 3.0; retain attribution to Databricks "
            "and the applicable ShareAlike terms for the source dataset."
        ),
    }
    (output / "dialogue-source-report.json").write_text(
        json.dumps(source_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    mixture_report = mix_corpora(
        args.base_train,
        args.base_validation,
        dialogue_train,
        dialogue_validation,
        args.tokenizer,
        args.output_dir,
        args.dialogue_fraction,
        args.seed,
    )
    (output / "dialogue-attribution.md").write_text(
        "# Dialogue training-data attribution\n\n"
        "## OpenAssistant OASST1\n"
        "- Source: https://huggingface.co/datasets/OpenAssistant/oasst1\n"
        "- Revision: " + OASST1_REVISION + "\n"
        "- Licence: Apache-2.0\n\n"
        "## Databricks Dolly 15K\n"
        "- Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k\n"
        "- Revision: " + DOLLY_REVISION + "\n"
        "- Licence: CC BY-SA 3.0; retain attribution and review ShareAlike obligations.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "sources": source_report,
        "mixture": mixture_report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
