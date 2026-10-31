# Joseph

**Joseph** is an offline-first AI companion research project. The long-term goal is a locally running friend with a distinct, evolving personality, explicit long-term memory, and a small language model whose architecture and weights are created by this project.

This repository starts with a deliberately small, inspectable foundation. It does **not** contain a pretrained model or pretend that a fresh random model already knows how to chat.

## Project principles

- **From-scratch model:** custom byte-level BPE tokenizer and a decoder-only Transformer initialized from random weights. No pretrained weights are downloaded.
- **Local-first data:** memories and structured knowledge live in SQLite/JSON files that can be backed up independently of model weights.
- **Evidence-aware knowledge:** JoJo facts retain source IDs, continuity (manga/anime), confidence, and verification status.
- **Small-model budget:** the default reference configuration is about 4.94 million parameters (roughly 19 MiB of raw FP32 weights). Checkpoints are rejected if they exceed the configured 1 GB limit.
- **Android is the target device:** training experiments can happen elsewhere, but the eventual companion must run locally on Android. The current Python/PyTorch implementation is a reference/training path, not yet an Android app.

## Reality check: training from zero

A randomly initialized 5M-parameter model trained only on a small JoJo dataset will not become a capable general conversational model. It needs a large, diverse, legally usable language corpus to learn language, followed by focused data for behaviour and domain knowledge. The phone is better suited to inference, retrieval, and memory than large-scale pretraining.

Joseph therefore separates three things:

1. **Language-model training** changes neural weights.
2. **Knowledge ingestion** stores sourced facts that can be searched and updated without retraining.
3. **Personal memory** stores user-approved preferences, project context, and conversation memories separately.

These systems will later be connected by a response pipeline. A retrieved fact is not automatically a trained model parameter.

## Current status

- [x] Project skeleton and configuration
- [x] Byte-level BPE tokenizer with save/load
- [x] Small decoder-only Transformer reference model
- [x] Provenance-aware JoJo knowledge schema and SQLite store
- [x] Separate local memory store
- [x] Training and checkpoint-size guard
- [x] Starter metadata for Parts 1–9 from the official JoJo portal
- [ ] General-language training corpus and data-license audit
- [ ] Conversation/inference loop
- [ ] Memory retrieval and knowledge-grounded response integration
- [ ] Android runtime and UI
- [ ] On-device performance, battery, and storage benchmarks

## Quick start

Python 3.11+ is recommended.

Install the optional training dependency:

    python -m pip install -e ".[train,test]"

Train a tokenizer on a plain-text corpus that you have permission to use:

    python -m training.train_tokenizer --input path/to/your_corpus.txt --output artifacts/tokenizer.json --vocab-size 512

Train a fresh model from random initialization:

    python -m training.train --input path/to/your_corpus.txt --tokenizer artifacts/tokenizer.json --output artifacts/joseph.pt --steps 1000

The corpus file should contain ordinary language text. Do not use ripped manga pages, scans, full manga scripts, or unlicensed anime subtitles. For JoJo-specific knowledge, add concise original summaries/facts with source records to datasets/jojo instead.

Run tests:

    python -m unittest discover -s tests -v

## Repository map

- joseph/tokenizer.py — byte-level BPE tokenizer
- joseph/model.py — decoder-only Transformer reference implementation
- joseph/config.py — model dimensions and parameter/weight estimates
- joseph/knowledge/ — sourced facts and continuity-aware retrieval
- joseph/memory/ — separate SQLite-based personal memory
- training/ — tokenizer training, language-model training, checkpoints
- datasets/jojo/ — small factual metadata starter, not copied story content
- android/ — Android runtime planning notes

## Important limitations

The current training loop is a prototype for small experiments. It loads the supplied text corpus into memory and is not a large-scale distributed trainer. The model has no pretrained linguistic knowledge, no chat tuning, and no Android inference package yet. Do not treat its generated output as reliable facts. Use the knowledge store and sources for factual answers until a grounded response layer exists.

## JoJo source policy

The starter catalogue identifies the nine numbered parts listed by the official JoJo portal. This is catalogue metadata only. The repository must not store copyrighted manga chapters, scans, anime episodes, or ripped subtitles. Add factual notes in your own words and preserve links to legitimate sources. Keep manga and anime continuity distinct when an adaptation changes, omits, or expands events.

Official catalogue: https://jojo-portal.com/comics/
