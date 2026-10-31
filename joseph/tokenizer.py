"""Versioned byte-level BPE tokenizer backed by Hugging Face Tokenizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = (
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<|system|>", "<|user|>", "<|hutao|>", "<|memory|>", "<|report|>", "<|eot|>",
)


class ByteBPETokenizer:
    """Thin wrapper around the Rust-backed BPE engine with stable control-token IDs."""

    def __init__(self, backend: Tokenizer):
        self._backend = backend
        for expected_id, token in enumerate(SPECIAL_TOKENS):
            actual_id = backend.token_to_id(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"special-token ID drift: {token} expected {expected_id}, got {actual_id}"
                )

    @property
    def vocab_size(self) -> int:
        return self._backend.get_vocab_size(with_added_tokens=True)

    def special_token_id(self, token: str) -> int:
        token_id = self._backend.token_to_id(token)
        if token_id is None or token not in SPECIAL_TOKENS:
            raise ValueError(f"unknown special token: {token}")
        return token_id

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 8192,
        min_frequency: int = 2,
    ) -> "ByteBPETokenizer":
        minimum_vocab = len(SPECIAL_TOKENS) + 256
        if vocab_size < minimum_vocab:
            raise ValueError(f"vocab_size must be at least {minimum_vocab}")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")

        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        reserved = [
            AddedToken(token, normalized=False, special=True)
            for token in SPECIAL_TOKENS
        ]
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=reserved,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        backend.train_from_iterator(texts, trainer=trainer)
        return cls(backend)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = self._backend.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids.insert(0, self.special_token_id("<bos>"))
        if add_eos:
            ids.append(self.special_token_id("<eos>"))
        return ids

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self._backend.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=skip_special_tokens,
        )

    def to_dict(self) -> dict:
        return json.loads(self._backend.to_str(pretty=True))

    @classmethod
    def from_dict(cls, data: dict) -> "ByteBPETokenizer":
        if not isinstance(data, dict) or "model" not in data:
            raise ValueError("invalid Hugging Face tokenizer JSON")
        return cls(Tokenizer.from_str(json.dumps(data, ensure_ascii=False)))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._backend.save(str(target), pretty=True)

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        return cls(Tokenizer.from_file(str(path)))
