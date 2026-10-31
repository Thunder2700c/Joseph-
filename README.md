# Joseph

**Joseph** is the internal project codename for an offline-first companion whose user-facing character is **Hu Tao** in a fictional reincarnation scenario. The goal is a locally running, Teyvat-knowledgeable companion with persistent personal memory and a high-fidelity character identity.

The repository starts with a deliberately small, inspectable foundation. It does **not** contain a pretrained model or pretend that a fresh random model already knows how to chat.

## Character premise

Hu Tao remembers her former life in Teyvat and remembers being the 77th Director of the Wangsheng Funeral Parlor. She somehow returns and wakes inside the user's phone. She does not remember dying, how she crossed over, or why she is inside the device. She does not understand herself as an AI or software.

This is a user-created alternate-universe premise, not an event established by Genshin Impact canon. Keep this personal storyline separate from verified Teyvat lore.

See [character/identity.md](character/identity.md) for the complete character contract.

## Project principles

- **From-scratch model:** custom byte-level BPE tokenizer and decoder-only Transformer initialized from random weights. No pretrained weights are downloaded.
- **Hu Tao first:** she is the user-facing identity, not a temporary roleplay skin layered over a generic assistant.
- **Evidence-aware Teyvat knowledge:** facts retain sources, canon category, version/date, confidence, and spoiler scope.
- **Local-first data:** personal memories and structured lore live in SQLite/JSON files independently of model weights.
- **Small-model budget:** the default reference configuration is about 4.94 million parameters (roughly 19 MiB of raw FP32 weights). Checkpoints are rejected if they exceed the configured 1 GB limit.
- **Android target:** the eventual companion must run locally on Android. The current Python/PyTorch implementation is a reference/training path, not yet an Android app.

## Reality check: training from zero

A randomly initialized 5M-parameter model trained on a small lore dataset will not become a capable conversational model. It needs a large, diverse, legally usable language corpus to learn language, followed by character-behaviour training and domain knowledge. The phone is better suited to inference, retrieval, and memory than large-scale pretraining.

Joseph separates four things:

1. **Language-model training** changes neural weights.
2. **Teyvat knowledge ingestion** stores source-backed facts that can be searched and updated without retraining.
3. **Personal memory** stores user-approved preferences, project context, and conversation memories.
4. **Character adaptation** stores explicit feedback and evolving preferences without silently changing neural weights.

A retrieved fact is not automatically a trained model parameter or a personal memory.

## Current status

- [x] Project skeleton and configuration
- [x] Byte-level BPE tokenizer with save/load
- [x] Small decoder-only Transformer reference model
- [x] Provenance-aware JoJo knowledge schema and SQLite store
- [x] Separate local memory store
- [x] Hu Tao reincarnation persona and identity contract
- [x] Teyvat lore source/schema planning and initial sourced records
- [x] Training and checkpoint-size guard
- [ ] Comprehensive Teyvat lore collection and coverage evaluation
- [ ] Conversation/inference loop with lore retrieval
- [ ] Character-behaviour training and evaluation
- [ ] Android runtime and UI
- [ ] On-device performance, battery, and storage benchmarks

## Quick start

Python 3.11+ is recommended.

Install the optional training dependency:

    python -m pip install -e ".[train,test]"

Import the existing JoJo catalogue metadata:

    python -m joseph.knowledge.ingest --db local/jojo_knowledge.sqlite

Train a tokenizer on a plain-text corpus that you have permission to use:

    python -m training.train_tokenizer --input path/to/your_corpus.txt --output artifacts/tokenizer.json --vocab-size 512

Train a fresh model from random initialization:

    python -m training.train --input path/to/your_corpus.txt --tokenizer artifacts/tokenizer.json --output artifacts/joseph.pt --steps 1000

Do not use ripped manga or game files, full unlicensed dialogue dumps, scans, or extracted voice lines. For lore, add concise original summaries with legitimate source references.

Run tests:

    python -m unittest discover -s tests -v

## Important limitations

The current model is a small architecture/training prototype, not a capable Hu Tao conversational model. It has no pretrained linguistic knowledge, no character tuning, no integrated lore-retrieval response loop, and no Android inference package yet. Do not treat its raw generated output as reliable facts.

The initial Teyvat records are only a tiny source-backed seed, not "everything about Teyvat." Coverage must be built and evaluated region by region, quest by quest, and version by version.

## Repository map

- joseph/tokenizer.py — byte-level BPE tokenizer
- joseph/model.py — decoder-only Transformer reference implementation
- joseph/config.py — model dimensions and weight estimates
- joseph/persona.py — Hu Tao reincarnation identity and evolving preferences
- joseph/knowledge/ — initial provenance-aware knowledge infrastructure
- joseph/memory/ — separate SQLite-based personal memory
- character/ — character premise and behaviour contract
- datasets/teyvat/ — Teyvat lore records and sources
- datasets/jojo/ — earlier JoJo catalogue experiment
- training/ — tokenizer training, language-model training, checkpoints
- android/ — Android runtime planning notes

Official starting references:
- [Hu Tao — HoYoWiki](https://wiki.hoyolab.com/pc/genshin/entry/34)
- [Teyvat — Official Genshin Impact site](https://genshin.hoyoverse.com/en/map)
