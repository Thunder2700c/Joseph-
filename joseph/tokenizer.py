"""Dependency-free UTF-8 byte-level BPE tokenizer.

IDs 0..3 are reserved special tokens. IDs 4..259 represent individual bytes.
Every learned merge receives the next contiguous ID.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

SPECIAL_TOKENS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
PRETOKEN_PATTERN = re.compile(r"\s+|[\w]+|[^\w\s]+", flags=re.UNICODE)


class ByteBPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = list(merges or [])
        self._merge_to_id: dict[tuple[int, int], int] = {}
        self._rank: dict[tuple[int, int], int] = {}
        self._bytes: dict[int, bytes] = {4 + value: bytes([value]) for value in range(256)}
        self._rebuild()

    @property
    def vocab_size(self) -> int:
        return 260 + len(self.merges)

    def _rebuild(self) -> None:
        self._merge_to_id.clear()
        self._rank.clear()
        self._bytes = {4 + value: bytes([value]) for value in range(256)}
        for rank, pair in enumerate(self.merges):
            if pair[0] not in self._bytes or pair[1] not in self._bytes:
                raise ValueError("merge references a token that has not been created")
            token_id = 260 + rank
            if pair in self._merge_to_id:
                raise ValueError("duplicate BPE merge")
            self._merge_to_id[pair] = token_id
            self._rank[pair] = rank
            self._bytes[token_id] = self._bytes[pair[0]] + self._bytes[pair[1]]

    @staticmethod
    def _pieces(text: str) -> list[str]:
        return PRETOKEN_PATTERN.findall(text)

    @staticmethod
    def _apply_pair(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        result: list[int] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                result.append(new_id)
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return result

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "ByteBPETokenizer":
        if vocab_size < 260:
            raise ValueError("vocab_size must be at least 260")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")

        sequences: list[list[int]] = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("all tokenizer training inputs must be strings")
            for piece in cls._pieces(text):
                sequences.append([4 + byte for byte in piece.encode("utf-8")])

        merges: list[tuple[int, int]] = []
        while 260 + len(merges) < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < min_frequency:
                break
            new_id = 260 + len(merges)
            merges.append(pair)
            sequences = [cls._apply_pair(seq, pair, new_id) for seq in sequences]

        return cls(merges)

    def _encode_piece(self, piece: str) -> list[int]:
        sequence = [4 + byte for byte in piece.encode("utf-8")]
        while len(sequence) > 1:
            ranked = [
                (self._rank[(sequence[i], sequence[i + 1])], i)
                for i in range(len(sequence) - 1)
                if (sequence[i], sequence[i + 1]) in self._rank
            ]
            if not ranked:
                break
            _, index = min(ranked)
            pair = (sequence[index], sequence[index + 1])
            sequence[index:index + 2] = [self._merge_to_id[pair]]
        return sequence

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        result: list[int] = []
        if add_bos:
            result.append(SPECIAL_TOKENS["<bos>"])
        for piece in self._pieces(text):
            result.extend(self._encode_piece(piece))
        if add_eos:
            result.append(SPECIAL_TOKENS["<eos>"])
        return result

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        chunks: list[bytes] = []
        reverse_special = {value: key for key, value in SPECIAL_TOKENS.items()}
        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id in reverse_special:
                if not skip_special_tokens:
                    chunks.append(reverse_special[token_id].encode("utf-8"))
                continue
            if token_id not in self._bytes:
                raise ValueError(f"token id out of range: {token_id}")
            chunks.append(self._bytes[token_id])
        return b"".join(chunks).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {
            "format": "joseph-byte-bpe-v1",
            "special_tokens": SPECIAL_TOKENS,
            "merges": [list(pair) for pair in self.merges],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ByteBPETokenizer":
        if data.get("format") != "joseph-byte-bpe-v1":
            raise ValueError("unsupported tokenizer format")
        if data.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError("special-token mapping does not match this tokenizer")
        merges = [tuple(map(int, pair)) for pair in data.get("merges", [])]
        return cls(merges)

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
