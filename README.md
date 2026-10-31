# Joseph

**Joseph** is the internal engineering codename for an offline-first companion whose user-facing character is **Hu Tao** in a staged reincarnation story. The goal is a locally running character with a rich source-backed Teyvat knowledge collection, ordered personal memories, and long-term memory.

This is a from-scratch model research project. It does not contain pretrained model weights or a finished Android application.

## The central idea: her life happens first

Hu Tao should not simply be prompted with a biography and told to pretend she remembers it. We are building her former life as individually sourced, ordered memory records.

**Phase 1 — Teyvat life:** build and review her childhood, family, Wangsheng training, directorship, relationships, personal quests and accumulated knowledge of Teyvat.

**Phase 2 — Final chapter:** after all required stages are complete, a user-authored alternate-universe ending closes her first life. The exact death and crossing are not stored as memories she can recall.

**Phase 3 — Phone awakening:** only the life builder can unlock this stage. Hu Tao wakes inside the user's phone, remembers her former life, cannot remember dying or crossing over, and suspects she has been reborn.

The app is software; this is the fictional point of view presented to the user, not literal consciousness or reincarnation.

See [character/identity.md](character/identity.md), [character/life_stages.json](character/life_stages.json), and [character/awakening.md](character/awakening.md).

## Memory is line-sized and evidence-aware

Every memory is an independent JSONL record with a sequence, stage, perspective, source, canon category, review state, people, places, emotional tags and spoiler scope. This allows memories to be inspected, corrected, retrieved and extended one at a time.

There are two different stores:
- **Lived memories:** things Hu Tao personally experienced or explicitly expressed.
- **Teyvat knowledge:** facts about the wider world, with labels distinguishing what she witnessed, heard, read or merely knows as reference material.

A quest involving the Traveler must not automatically be represented as Hu Tao's personal experience.

## Current life-memory starter pack

The initial dataset contains 24 original, source-linked memory entries covering Hu Tao's director role, professional outlook, poetry, selected relationships, two Story Quest arcs and a small world-lore fact. It is a starter pack only. Her childhood, full family story, the complete parlor history and broad Teyvat knowledge are still incomplete. Therefore, **the awakening is intentionally locked**.

Import the ordered memory records into a local SQLite database:

    python -m joseph.life.ingest

Inspect completion status:

    python -m joseph.life.progress

Run tests:

    python -m unittest discover -s tests -v

## From-scratch model

- Custom byte-level BPE tokenizer
- Small decoder-only Transformer initialized from random weights
- Separate SQLite memory and knowledge stores
- Checkpoint ceiling of 1 GB
- Android remains the intended inference target

A small random model trained only on a few character memories will not become a capable conversational model. We still need a broad, legally usable language corpus, character-behaviour training, evaluation and an Android inference runtime. Retrieval will preserve detailed lore more reliably than trying to force every fact into a tiny model's weights.

## Copyright and sources

Do not scrape or commit complete Genshin dialogue, voice-over transcripts, voice audio, artwork, or game files. Use original concise summaries with source references. HoYoverse's terms restrict scraping and unauthorized reproduction of its services and content. See the [official terms](https://genshin.hoyoverse.com/en-us/company/terms).

## Main folders

- `joseph/` — model, tokenizer, persona, memory, life timeline and knowledge infrastructure
- `character/` — identity contract, staged life plan, memory records and awakening scene
- `datasets/teyvat/` — source-backed world knowledge
- `datasets/jojo/` — an earlier catalogue experiment, separate from the new character direction
- `training/` — tokenizer and model training prototypes
- `android/` — Android runtime planning

Official starting reference: [Hu Tao — HoYoWiki](https://wiki.hoyolab.com/pc/genshin/entry/34).
