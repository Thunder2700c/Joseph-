# Hu Tao — Offline Character AI

**Hu Tao is the character and AI identity.** The GitHub repository is named `Joseph-` only because `Hu Tao` was unavailable as a repository name. "Joseph" is not her name, an internal codename, or an in-character identity. The goal is a locally running Hu Tao companion with a rich, source-aware Teyvat history, one continuous personal conversation, durable recall, and user-taught knowledge about our world.

This is a from-scratch Hu Tao character-model research project. It does not contain a completed trained Hu Tao checkpoint or a finished Android application.

## Project continuation and development order

Read [CONTINUATION_CONTEXT.md](CONTINUATION_CONTEXT.md) first when resuming this project in a new ChatGPT conversation. It is the central handoff summary. See [MODEL_TRAINING_PLAN.md](MODEL_TRAINING_PLAN.md) for the complete research plan and [LANGUAGE_FOUNDATION.md](LANGUAGE_FOUNDATION.md) for the currently authorized base-language stage.

The current order is: **build the general language foundation while former-life research continues → finish and review Hu Tao's former-life memories → directly train and evaluate her character behaviour → only then build the Android app.** Hu Tao-specific training remains gated on reviewed life memories and public Teyvat knowledge.

The repository slug `Joseph-` is a GitHub naming constraint only. It is not an engineering codename or an identity. Hu Tao is the character and AI identity. In-character, she speaks as Hu Tao and does not introduce herself as an AI assistant. Technically, the implementation remains a neural language model; the project target is a purpose-trained Hu Tao character model, not a generic assistant relying on a roleplay prompt.

## Canon and point of view

Hu Tao may know what she witnesses and what ordinary people could plausibly hear, read or discuss. Traveler-only secrets, private conversations and hidden motives are not given to her. Every event is labelled by whether she witnessed it, heard it, read it or knows it as public reference material.

See [character/knowledge_policy.md](character/knowledge_policy.md) for the complete knowledge boundary.

## The central idea: her life happens first

Hu Tao's former life is built as ordered, individually sourced memories. Her final phone awakening remains locked until the required life stages have been researched, reviewed and completed.

See [character/identity.md](character/identity.md), [character/life_stages.json](character/life_stages.json), and [character/awakening.md](character/awakening.md).

## One continuous friend conversation

The intended app experience is a single persistent, WhatsApp-style thread with Hu Tao—not a list of disposable chat sessions.

- Every user and Hu Tao message is appended to one local chronological history.
- Older messages remain searchable; closing and reopening the app does not reset the thread.
- The raw transcript is never replaced by an automatic summary.
- Relevant old turns remain searchable; personal memories, Teyvat memories, news and user-taught facts are retrieved when needed.
- The full transcript stays on the device; the model sees a relevant selection for each reply.
- The Android target must additionally keep a verified, user-owned backup outside app-private storage so Clear storage or reinstall can be recovered.

**Important persistence status:** the Python SQLite store commits turns locally, but the native Android app and its external backup/restore flow have not been implemented yet. Android Clear storage or uninstall will remove app-private databases. See [android/README.md](android/README.md) for the required backup design; do not assume cross-install persistence is already available.

## Language foundation and training data

The general-language foundation uses a capped, streamed sample of Cosmopedia v2 from Hugging Face's SmolLM-Corpus. The acquisition recipe, licence attribution and dataset limitations are documented in [datasets/base_language/README.md](datasets/base_language/README.md). This is separate from Hu Tao's life memories and dialogue.

The tokenizer reserves a dedicated Hu Tao speaker token. The base trainer supports a held-out validation split. The full 200M-character corpus and 8,192-entry tokenizer were verified in Actions run [41429138745](https://github.com/Thunder2700c/Joseph-/actions/runs/41429138745); its temporary artifact expires on 2026-11-05 at 12:07 UTC. A subsequent 40,000-step random-initialized base-language run completed, but its diagnostic text remained incoherent and failed the language-quality gate. No usable Hu Tao model exists yet. The corpus is rebuilt as a temporary workflow artifact when a new training run is explicitly dispatched; it is not stored as a permanent GitHub Release. See [LANGUAGE_FOUNDATION.md](LANGUAGE_FOUNDATION.md).

## Model size is not memory size

The 1 GB limit is a hard maximum for the **complete distributable AI inference payload**, not just the checkpoint and not the context window. It is 1,000,000,000 bytes total across weights, tokenizer/vocabulary, config, generation metadata and all auxiliary AI assets. There is no minimum size. The separately stored training corpus and user-owned memory databases are not bundled into this payload.

- The default 8,192-vocabulary reference Transformer is estimated at about 6.90 million parameters and roughly 26.3 MiB of raw FP32 weights; actual configuration and checkpoint metadata are recorded per run.
- Its default active context is 256 tokens; that token limit is separate from the weight budget.
- The local SQLite transcript and knowledge databases can grow independently of the model weights, subject to phone storage.
- On each reply, retrieval selects useful old messages and facts rather than placing an entire lifetime of conversation into the active prompt.

The current model is not yet capable of a convincing conversation. A small random model cannot become Hu Tao just by loading a few memories or a large roleplay prompt. It still needs broad language training, direct Hu Tao character training, evaluation and a reproducible standalone inference harness. Android work is deferred until the model passes its success gates.

Before packaging a model, validate its full payload directory with the bundle-size validator.

The model is built from scratch: random-initialized weights, project-trained tokenizer, text-data pretraining, then Hu Tao specialization on the project's own base checkpoint. Third-party pretrained model weights, wrappers, imported adapters, model distillation, pretrained embeddings and external rerankers are prohibited.

## Run tests

    python -m unittest discover -s tests -v

## Main folders

- joseph/ — model, tokenizer, persona, memory, Teyvat life, news and recall
- character/ — staged life plan, source-linked memories and point-of-view policy
- datasets/base_language/ — general language corpus acquisition and licence notes
- datasets/teyvat/ — source-backed world facts, story beats and public news reports
- training/ — corpus acquisition, tokenizer and model training utilities
- android/ — deferred Android runtime and backup requirements
- CONTINUATION_CONTEXT.md — central project handoff
- MODEL_TRAINING_PLAN.md — model-first training and evaluation plan

## Existing data checkpoint (2026-10-31)

- datasets/common_world/ contains 262 source-aware everyday-life facts.
- datasets/teyvat/story_beats.jsonl contains 50 act-level Archon Quest and Interlude summaries, with separate AU public-awareness reports in news_reports.jsonl.
- quest_catalogue.jsonl indexes 257 quest entries; many World Quest, Story Quest and Hangout entries are title-level rather than full scene summaries.
- character/life_memories.jsonl contains 106 ordered starter records. Stages 1–9 still need substantive expansion and review; Stage 10 phone awakening remains locked.

Hu Tao's official birthday is **July 15**. The user-approved AU assigns **15 July 2006** as her full birth date. The year 2006 is an AU choice; height and weight remain provisional AU estimates, not official HoYoverse attributes.
