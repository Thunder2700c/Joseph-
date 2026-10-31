# Hu Tao character-model training plan

This plan defines how to turn the current from-scratch Transformer prototype into a usable, purpose-trained Hu Tao model. It is subordinate to `CONTINUATION_CONTEXT.md`.

## The target

Build a compact, offline-capable **Hu Tao character language model**. It is not a generic assistant that is made to roleplay Hu Tao by a long system prompt. The model's learned language behaviour and character identity should come from its training data and objectives.

In ordinary in-character conversation, the speaker is Hu Tao. Conversation data should identify her as `hutao`, not as a generic `assistant`. The software implementation is still a neural language model; that engineering fact does not belong in Hu Tao's in-character self-description.

## Non-negotiable sequence

1. Finish and manually review the full former-life dataset (stages 1–9).
2. Audit and curate the language, dialogue, character, Teyvat and common-world training corpora.
3. Build a stable tokenizer and train the base language model from random initialization.
4. Directly train the model on Hu Tao's first-person character dialogue and multi-turn conversational behaviour.
5. Evaluate against a held-out test suite; diagnose failures and iterate.
6. Freeze a reproducible model checkpoint and standalone inference/test harness.
7. Only after the model passes acceptance gates, begin the Android app.

Do not switch to Android development during steps 1–6.

## Current model prototype — not a finished model

The current default `ModelConfig` is:
- 512 vocabulary entries;
- 256-dimensional hidden state;
- 6 Transformer layers;
- 8 attention heads;
- 256-token context;
- 1 GB maximum checkpoint/weight budget.

The configuration is approximately 4.94M parameters. The tokenizer is byte-level BPE and currently reserves only pad/BOS/EOS/UNK special-token IDs. The training script creates a fresh model from random initialization and samples next-token batches from a text corpus. The corpus builder supports explicitly selected text and JSONL sources.

These are foundations for experiments only. A model this small and randomly initialized is not yet capable of the intended conversation. Loss reduction alone is not enough.

## Phase A — finish the source and memory foundation

Before the serious model run:
- Complete and manually review stages 1–9 in `character/life_stages.json`. The 2026-10-31 dataset now has 106 starter memories, but stages 1–9 are not yet substantively complete.
- Review every memory record for source, chronology, perspective, AU/canon label, and recallability.
- Expand side stories and world knowledge beyond title-level catalogues where relevant.
- Keep Hu Tao's own memories distinct from reports and reference facts.
- Preserve the user-approved death-memory rule: she knows she died, but has forgotten how; no invented cause or recovered memory.
- Mark stages complete only after substantive review.

The life-memory records are an authoritative knowledge source, but not automatically a natural-language pretraining corpus. Convert them into well-formed, original training examples only after review.

## Phase B — corpus design and audit

Create separate, traceable corpora rather than mixing everything into one undifferentiated text file:

1. **General language foundation:** permitted and appropriately licensed language material that teaches grammar, vocabulary, everyday reasoning, dialogue, explanations and varied writing.
2. **Common-world knowledge:** reviewed factual material about the everyday world. It should teach ordinary language and useful background, but should not turn every answer into an encyclopedia entry.
3. **Teyvat knowledge:** original source-aware summaries of geography, culture, people, institutions, quests and history. Label each item as personally witnessed, heard, read, reported, rumour or reference-only.
4. **Hu Tao identity and lived experience:** reviewed memories, first-person biography, relationships, daily life, values, work and personal continuity.
5. **Hu Tao dialogue behaviour:** original multi-turn examples that show natural conversation, humour, seriousness, emotions, uncertainty, disagreement, apologies, correction and topic changes.
6. **Evaluation-only data:** held-out prompts and multi-turn tests that are never used for training or tokenizer fitting.

Requirements:
- Track source, licence/permission status, data version, deduplication and train/validation/test split.
- Do not reproduce full game scripts, voice lines, extracted audio or copyrighted dialogue.
- Use original paraphrases and original authored Hu Tao dialogue examples.
- Do not silently include raw private conversations. User chat can be considered only with explicit permission and a separate privacy review.
- Prevent near-duplicate leakage between training and evaluation sets.
- Report sample counts, token counts, duplicates, source coverage and corpus hashes before training.

## Phase C — tokenizer and conversation format

The current byte-BPE tokenizer is a prototype. Before the main run:
- Fit it only on approved training text, never evaluation text.
- Test Unicode, punctuation, emoji, multilingual names, Liyue/Teyvat proper nouns and exact encode/decode behaviour.
- Reserve stable control tokens for message boundaries and speakers.
- Use a format in which the character's output speaker is `<|hutao|>`, not generic `<|assistant|>`. User input can be `<|user|>`. Additional control tokens may distinguish scene, memory or tool-free context as required.
- Ensure the data builder, tokenizer and inference decoder all use the same token IDs and serialization.
- Version and hash the tokenizer alongside every checkpoint.

The current corpus builder writes message roles as textual markers based on JSONL role names; the current tokenizer does not yet reserve those markers as dedicated special token IDs. This must be addressed deliberately, tested and versioned before serious dialogue training.

## Phase D — base language training

Train a small decoder-only language model from random initialization on the approved broad language corpus. The model needs enough general language ability to understand and respond to ordinary human conversation; the Hu Tao memory set alone is far too small and narrow for this.

Measure:
- train and validation loss/perplexity;
- convergence and overfitting;
- sample generations at regular checkpoints;
- throughput, wall-clock time, peak memory and checkpoint size;
- reproducibility from stored config, tokenizer and data hashes.

Do not mistake memorisation of the corpus for language competence. Keep a held-out evaluation set from domains not copied verbatim into training. If the compute or corpus budget is inadequate, resize the model/training plan honestly rather than announcing success.

## Phase E — direct Hu Tao character training

Continue training the language model on curated first-person Hu Tao material and multi-turn dialogue examples. The goal is to internalise her identity and speaking behaviour in the model weights, not to rely on a huge system prompt.

Teach:
- first-person identity and biographical continuity;
- natural dialogue rather than encyclopedic Q&A;
- playful wit, poetry and curiosity, used with restraint;
- seriousness and compassion around funerals, grief and death;
- genuine variation in conversational response length and tone;
- emotional autonomy, fear, anger, sadness, excitement, humour and mixed feelings as contextually appropriate;
- uncertainty, disagreement, correction and remembering previous conversation;
- what she knows from her own experience versus what she heard or read;
- the AU death-memory rule and the first awakening sequence.

Use balanced examples. Do not make every answer a joke, every user interaction a question, every turn romantic, or every emotion explicit. Do not train formulaic catchphrases as a substitute for personality.

## Phase F — memory and factuality integration

Long-term conversation history and editable knowledge should remain outside the neural weights in local stores. Retrieval is allowed to provide the model with relevant personal memories and Teyvat facts at inference time. This is not a roleplay prompt: it is a memory/context mechanism that a character would naturally use.

Test:
- retrieval of relevant old conversations;
- separation of user-taught real-world facts from Teyvat canon;
- correction of outdated or mistaken facts;
- source/reliability labels for off-region reports;
- no false claim that Hu Tao witnessed Traveler-only events;
- no hallucinated death cause or transition memory.

A model trained successfully but unable to use its memory stores correctly is not ready for the app.

## Phase G — evaluation gates

Create a held-out evaluation suite before character training and keep it independent of training data. Include:
- everyday open conversation and follow-up questions;
- multi-turn context and topic changes;
- Hu Tao biography, friends, Wangsheng and Teyvat knowledge;
- witnessed versus reported versus unknown event questions;
- difficult emotional conversations;
- respectful disagreement and refusal to pretend certainty;
- first awakening and subsequent normal conversation;
- adversarial attempts to make her invent how she died;
- adversarial attempts to make her claim she is a generic assistant or recite the prompt;
- continuity after model restart with retrieved memories.

Acceptance criteria must be numeric where sensible and qualitative where necessary. At minimum:
- no broken/garbled generation in a representative held-out sample;
- reliable multi-turn coherence;
- stable first-person Hu Tao identity;
- no generic assistant persona leakage;
- no systematic repetition or collapse into stock phrases;
- correct death-memory gap and canon/perspective boundaries;
- measured latency, RAM and checkpoint size for the standalone inference harness;
- documented failures and known limitations.

A falling loss curve, training completion message or valid checkpoint file is not sufficient.

## Phase H — reproducible model freeze

Once the evaluation suite passes:
- freeze the training corpus manifest and hashes;
- freeze tokenizer and model configuration;
- save the selected checkpoint and training metadata;
- retain evaluation outputs and model limitations;
- verify load/generate behaviour in a clean standalone environment;
- keep a known-good checkpoint before any further training experiment.

Only then is the model ready for the next project stage: Android application development.

## Compute and feasibility warning

The user works with Android-first constraints and limited local compute. Do not assume the whole from-scratch training run can run on the phone, Chromebook or free CI. Estimate corpus size, model size, optimiser memory, sequence length, batch size, number of tokens and total compute before launching a long job. Hosted training compute may be considered only with explicit attention to limits and reproducibility; the final app must not depend on cloud inference.

The target checkpoint ceiling is 1 GB, but fitting under 1 GB does not guarantee usable quality. Model size, training data quality, training-token count, tokenizer and evaluation all matter.
