# Joseph

**Joseph** is the internal engineering codename for an offline-first companion whose user-facing character is **Hu Tao** in a staged reincarnation story. The goal is a locally running companion with a rich, source-aware Teyvat history, one continuous personal conversation, durable recall, and user-taught knowledge about our world.

This is a from-scratch model research project. It does not contain pretrained model weights or a finished Android application.

## The central idea: her life happens first

Hu Tao's former life is built as ordered, individually sourced memories. Her final phone awakening remains locked until the required life stages have been researched, reviewed and completed.

A Traveler-only event is never turned into Hu Tao's first-hand memory. For Fontaine, Natlan and other distant nations, the chronicle records the canon event separately from the news report through which Hu Tao learns its public outline.

See [character/identity.md](character/identity.md), [character/life_stages.json](character/life_stages.json), and [character/awakening.md](character/awakening.md).

## One continuous friend conversation

The intended app experience is a single persistent, WhatsApp-style thread with Hu Tao—not a list of disposable chat sessions.

- Every user and Hu Tao message is appended to one local chronological history.
- Older messages remain searchable; closing and reopening the app does not reset the thread.
- The raw transcript is never replaced by an automatic summary.
- Relevant old turns, personal memories, Teyvat memories, news and user-taught facts are retrieved when needed.
- The full transcript stays on the device; the model sees a relevant selection for each reply.

The backend supports this single timeline in `joseph/memory/store.py`. Android UI and inference integration are still to be built.

## Four independent memory stores

1. **Continuous conversation transcript:** the complete sequence of messages, saved locally and searchable.
2. **Personal memory:** durable user preferences and details, separate from the raw transcript.
3. **User-taught world knowledge:** facts the user teaches Hu Tao about our world. Each is attributed as user-taught, stored independently, and retrieved later.
4. **Teyvat life and news:** her own source-backed experiences, reference knowledge, and separate reports about events in other nations.

Teach a fact to the world database through `WorldKnowledgeStore.teach(subject, statement, ...)`. It is persisted immediately and can be retrieved in future conversations. It does not need model retraining to become available.

## News from other nations

The Teyvat chronicle stores a canon story beat and a separate news transmission. For example, a Fontaine Archon Quest act can be canon, while Hu Tao's awareness of it is labelled as something she read in the Steambird or heard from merchants.

Transmission channels include newspapers, merchants, travellers, Adventurers' Guild dispatches, letters, books and rumours. Every report retains a channel and reliability label. Hearing a story about Natlan is not the same as personally taking part in it.

The starter chronicle includes 10 act-level original summaries for Fontaine and Natlan, plus 9 separate reports that reach Hu Tao as news. The life-memory dataset now contains 33 source-linked entries, including 9 news-awareness memories. These are **not yet a scene-by-scene complete story database**. The format supports expanding each act into many ordered, independently retrievable story beats.

Import the chronicle:

    python -m joseph.life.news_ingest

## Model size is not memory size

The 1 GB limit is a **maximum neural-weight/checkpoint budget**, not a 1 GB context window.

- The current experimental Transformer is about 4.94 million parameters and roughly 19 MiB of raw FP32 weights.
- Its default active context is 256 tokens; that token limit is separate from the weight budget.
- The local SQLite transcript and knowledge databases can grow independently of the model weights, subject to phone storage.
- On each reply, retrieval selects useful old messages and facts rather than placing an entire lifetime of conversation into the active prompt.

The model is not yet capable of a convincing conversation. A small random model cannot become Hu Tao just by loading a few memories. It still needs broad language training, character-behaviour training, evaluation and an Android inference runtime.

## Model learning versus stored knowledge

User-taught world facts are saved immediately in the local knowledge database. By default, they are **not** silently baked into model weights.

A fact can be individually approved for future training. Export only those approved records with:

    python -m training.export_world_knowledge --db local/world_knowledge.sqlite

Raw chat transcripts are never included by that exporter. This keeps private conversations correctable and separate from neural training.

## Current Teyvat checkpoint

The current project knowledge checkpoint is **2026-10-31**. The official Version 7.2 update notes say Version 7.2 arrives on **November 4, 2026**, so its announced story is not treated as released playable history on the checkpoint date. Every story record must carry its version and release status.

## Run tests

    python -m unittest discover -s tests -v

## Story source handling

We can build detailed, original, scene-by-scene story summaries and fact records. Story beats must have original wording, a legitimate source reference, chronology, spoiler scope, and a label that says whether Hu Tao witnessed, heard or read about them.

## Main folders

- `joseph/` — model, tokenizer, persona, memory, Teyvat life, news and recall
- `character/` — staged life plan, source-linked memories and awakening
- `datasets/teyvat/` — source-backed world facts, story beats and news reports
- `datasets/world_user/` — user-taught facts schema and guidance
- `training/` — model training and explicitly approved world-fact export
- `android/` — single-thread UI and Android runtime planning

## Expanded knowledge checkpoint (2026-10-31)

- `datasets/common_world/` contains 262 source-aware everyday-life facts: anatomy, digestion, bowel and urinary functions, hygiene, sex and gender, reproduction, relationships, emotions, adult vocabulary and media literacy.
- Teyvat common-life assumptions are explicitly marked as assumptions, not canon. Non-human species are not automatically assigned human biology.
- `datasets/teyvat/story_beats.jsonl` contains 50 act-level Archon Quest and Interlude summaries, with 50 separate AU news transmissions in `news_reports.jsonl`.
- `quest_catalogue.jsonl` indexes 257 quest entries: 50 Archon/Interlude acts, 129 World Quest titles, 55 Story Quest chapters, 4 Hu Tao Story Quest subquests and 19 Hangout chapters.
- `release_news.jsonl` contains 14 out-of-world HoYoverse publisher bulletins, kept separate from Hu Tao's Teyvat awareness.
- The World Quest, Story Quest and Hangout entries are title-level catalogue records; they are not all fully summarised. Scene-level Archon Quest expansion and complete side-story coverage remain unfinished.

The database import commands are:

    python -m joseph.knowledge.ingest --db local/common_knowledge.sqlite
    python -m joseph.life.news_ingest --db local/teyvat_chronicle.sqlite
    python -m joseph.life.catalogue_ingest --db local/quest_catalogue.sqlite
    python -m joseph.life.release_news_ingest --db local/official_news.sqlite

The knowledge corpus is retrieved into the model context when relevant. This is not neural training; neural training begins only after dataset review and a separate training plan.
