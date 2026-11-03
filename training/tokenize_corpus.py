"""Encode line-oriented text into disk-backed token and response-loss-mask streams."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
from pathlib import Path

from joseph.tokenizer import ByteBPETokenizer


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_loss_mask(
    sample_ids: list[int],
    *,
    assistant_id: int,
    eot_id: int,
    eos_id: int,
    bos_id: int,
) -> list[int]:
    """Mask user/prompt tokens and train only assistant turns in SFT samples.

    Ordinary prose samples contain no assistant marker and retain full causal-LM
    loss. Dialogue samples must contain balanced assistant-marker / EOT pairs.
    The returned mask aligns one-to-one with [BOS] + sample + [EOS].
    """
    is_dialogue = assistant_id in sample_ids
    if not is_dialogue:
        return [0] + [1] * len(sample_ids) + [1]

    mask = [0]  # BOS is input context, never a prediction target.
    in_assistant_turn = False
    assistant_turns = 0
    for token_id in sample_ids:
        if token_id == assistant_id:
            if in_assistant_turn:
                raise ValueError("nested assistant marker before end-of-turn")
            in_assistant_turn = True
            assistant_turns += 1
            mask.append(1)  # Learn to begin the assistant turn.
        elif token_id == eot_id:
            if not in_assistant_turn:
                raise ValueError("end-of-turn outside an assistant response")
            mask.append(1)  # Learn when to stop the response.
            in_assistant_turn = False
        else:
            mask.append(int(in_assistant_turn))

    if in_assistant_turn:
        raise ValueError("dialogue sample has an unterminated assistant response")
    if assistant_turns == 0:
        raise ValueError("dialogue sample has no assistant response")
    mask.append(1)  # Learn EOS after the final EOT.
    if len(mask) != len(sample_ids) + 2:
        raise AssertionError("loss-mask/token alignment failure")
    return mask


def tokenize_text_file(
    input_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Tokenize samples and create an aligned uint8 assistant-only loss mask."""
    source = Path(input_path)
    tokenizer_file = Path(tokenizer_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    mask_target = target.with_suffix(target.suffix + ".loss-mask.bin")
    mask_temporary = mask_target.with_suffix(mask_target.suffix + ".tmp")

    if array.array("I").itemsize != 4:
        raise RuntimeError("this token cache format requires 32-bit unsigned integers")

    tokenizer = ByteBPETokenizer.load(tokenizer_file)
    bos_id = tokenizer.special_token_id("<bos>")
    eos_id = tokenizer.special_token_id("<eos>")
    assistant_id = tokenizer.special_token_id("<|assistant|>")
    eot_id = tokenizer.special_token_id("<|eot|>")

    token_buffer = array.array("I")
    mask_buffer = array.array("B")
    token_count = 0
    selected_target_count = 0
    sample_count = 0
    dialogue_sample_count = 0

    try:
        with source.open("r", encoding="utf-8") as input_file, \
                temporary.open("wb") as output_file, \
                mask_temporary.open("wb") as mask_file:
            token_buffer.append(bos_id)
            mask_buffer.append(0)
            token_count += 1

            for line_number, line in enumerate(input_file, start=1):
                sample = line.strip()
                if not sample:
                    continue
                ids = tokenizer.encode(sample)
                sample_mask = build_loss_mask(
                    ids,
                    assistant_id=assistant_id,
                    eot_id=eot_id,
                    eos_id=eos_id,
                    bos_id=bos_id,
                )
                full_ids = ids + [eos_id]
                token_buffer.extend(full_ids)
                mask_buffer.extend(sample_mask[1:])
                token_count += len(full_ids)
                selected_target_count += sum(sample_mask[1:])
                sample_count += 1
                if assistant_id in ids:
                    dialogue_sample_count += 1

                if len(token_buffer) >= 262_144:
                    token_buffer.tofile(output_file)
                    mask_buffer.tofile(mask_file)
                    token_buffer = array.array("I")
                    mask_buffer = array.array("B")

            if token_buffer:
                token_buffer.tofile(output_file)
            if mask_buffer:
                mask_buffer.tofile(mask_file)
            output_file.flush()
            mask_file.flush()
            os.fsync(output_file.fileno())
            os.fsync(mask_file.fileno())

        if sample_count == 0:
            raise ValueError(f"no non-empty text samples found in {source}")
        if selected_target_count == 0:
            raise ValueError("tokenized corpus contains no loss-bearing targets")
        os.replace(temporary, target)
        os.replace(mask_temporary, mask_target)
    finally:
        for path in (temporary, mask_temporary):
            if path.exists():
                path.unlink()

    manifest = {
        "format": "joseph-token-cache-v2",
        "input_file": str(source),
        "input_sha256": sha256_file(source),
        "tokenizer_file": str(tokenizer_file),
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "output_file": str(target),
        "output_sha256": sha256_file(target),
        "loss_mask_file": str(mask_target),
        "loss_mask_sha256": sha256_file(mask_target),
        "token_count": token_count,
        "sample_count": sample_count,
        "dialogue_sample_count": dialogue_sample_count,
        "selected_target_count": selected_target_count,
        "dtype": "int32",
        "mask_dtype": "uint8",
        "byte_order": "native",
        "bos_id": bos_id,
        "eos_id": eos_id,
        "assistant_id": assistant_id,
        "eot_id": eot_id,
        "loss_policy": "all ordinary-text targets; assistant marker, response, EOT and EOS for dialogue samples",
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create aligned disk-backed token and response-loss-mask caches."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = tokenize_text_file(args.input, args.tokenizer, args.output)
    print(
        f"Encoded {manifest['sample_count']:,} samples into "
        f"{manifest['token_count']:,} tokens; "
        f"{manifest['selected_target_count']:,} loss targets "
        f"({manifest['dialogue_sample_count']:,} dialogue samples): "
        f"{manifest['output_file']}"
    )


if __name__ == "__main__":
    main()
