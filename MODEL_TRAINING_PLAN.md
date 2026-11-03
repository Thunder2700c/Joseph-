# Hu Tao character-model training plan

This plan defines how to turn the current from-scratch Transformer prototype into a usable, purpose-trained Hu Tao model. It is subordinate to `CONTINUATION_CONTEXT.md`.

## The target

**Hard size ceiling:** the complete distributable AI inference payload must be at most **1,000,000,000 bytes** including model weights, tokenizer, vocabulary, configuration, generation metadata and any auxiliary AI models/assets. There is no minimum size and no requirement to fill the cap. Smaller is acceptable; select the smallest model that passes evaluation. Run the bundle validator before release.

**From-scratch only:** initialize all neural weights randomly and train the tokenizer from approved text. Do not download, wrap, rebuild, distil from, or fine-tune third-party pretrained LLMs or use third-party model weights, adapters, embedding models or rerankers. The later Hu Tao character-specialization stage continues training only the base model created by this project. Licensed text datasets are allowed; pretrained neural weights are not.

Build a compact, offline-capable **Hu Tao character language model**. It is not a generic assistant that is made to roleplay Hu Tao by a long system prompt. The model's learned language behaviour and character identity should come from its training data and objectives.

In ordinary in-character conversation, the speaker is Hu Tao. Conversation data should identify her as `hutao`, not as a generic `assistant`. The software implementation is still a neural language model; that engineering fact does not belong in Hu Tao's in-character self-description.

## Non-negotiable sequence

1. Build the general language foundation in parallel with former-life research: curate permitted base text, train the tokenizer and run controlled base-language experiments from random initialization.
2. Finish and manually review Hu Tao's former-life dataset and public Teyvat knowledge (stages 1–9). Her knowledge includes what she witnesses and what ordinary people can plausibly hear, read or discuss; Traveler-only secrets and private conversations are excluded.
3. Only after the life-memory review gate passes, create and directly train on Hu Tao's first-person character dialogue and multi-turn conversational behaviour.
4. Evaluate against held-out language and character test suites; diagnose failures and iterate.
5. Freeze a reproducible checkpoint, tokenizer, corpus manifest and standalone inference/test harness.
6. Only after the model passes acceptance gates, begin the Android app.

Do not switch to Android development during steps 1–6.

## Current model prototype — not a finished model

The reference/default `ModelConfig` remains a random-initialized baseline with 8,192 vocabulary entries, 256-dimensional hidden state, 6 Transformer layers, 8 attention heads and 256-token context (approximately 6.90M parameters). The first completed CPU calibration experiment used a separate smaller configuration: 8,192 vocabulary entries, 128-dimensional hidden state, 4 layers, 4 heads and 128-token context (1,858,304 parameters).

The project-trained Rust-backed ByteLevel BPE tokenizer has 8,192 vocabulary entries and ten reserved control IDs, including `<|user|>` and `<|hutao|>`. The Cosmopedia v2 general-language corpus contains 199,947,904 characters from 20 verified partitions; its canonical build is Actions run `41429138745`. The prior temporary artifact expires on 2026-11-05, and the workflow now rebuilds the corpus when training is manually dispatched. Permanent GitHub Releases are not authorized. The disk-backed int32 token-cache pipeline and random-initialized training loop are operational.

**First base-language calibration pilot: completed successfully** in GitHub Actions run [41541564443](https://github.com/Thunder2700c/Joseph-/actions/runs/41541564443). The measured-results record is [reports/BASE_LANGUAGE_CALIBRATION_2026-11-02.md](reports/BASE_LANGUAGE_CALIBRATION_2026-11-02.md). At step 1,000, evaluated training loss was 7.7335, validation loss 7.8389, training perplexity 2,283.62, validation perplexity 2,537.51, average loop throughput 6,219.78 tokens/sec, elapsed training-loop time 164.64 seconds, and checkpoint size 7,454,309 bytes. The run processed 1,024,000 optimizer token exposures.

The first pilot was followed by a warm-start continuation in Actions run [41594999963](https://github.com/Thunder2700c/Joseph-/actions/runs/41594999963), adding 10,240,000 optimizer-token exposures and reaching global checkpoint step 11,000. Final sampled training/validation losses were 7.2779/7.2709, validation perplexity was 1,437.87, average CPU throughput was 5,772.17 tokens/sec, training-loop time was 1,774.03 seconds, and checkpoint size was 7,454,541 bytes. See [reports/BASE_LANGUAGE_CONTINUATION_2026-11-03.md](reports/BASE_LANGUAGE_CONTINUATION_2026-11-03.md).

The continuation's three diagnostic generations were still incoherent and repetitive. Therefore the general-language foundation has **not** passed its qualitative competence gate. Both experiments validate a working training/checkpoint pipeline and show sampled loss reduction; neither proves useful conversation. The next phase must improve data diversity and generation quality, run held-out evaluations and further controlled training. The 11,264,000 optimizer-token exposures across both runs are not a full epoch over the 41.75M-token training split. Loss reduction or checkpoint creation alone is not success.

## Phase A — finish the source and memory foundation

Before Hu Tao-specific identity/dialogue training:
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

The tokenizer pipeline now uses Hugging Face Tokenizers ByteLevel BPE. Before a reproducible base training run:
- Fit it only on approved training text, never evaluation text.
- Test Unicode, punctuation, emoji, multilingual names, Liyue/Teyvat proper nouns and exact encode/decode behaviour.
- Reserve stable control tokens for message boundaries and speakers.
- Use a format in which the character's output speaker is `<|hutao|>`, not generic `<|assistant|>`. User input can be `<|user|>`. Additional control tokens may distinguish scene, memory or tool-free context as required.
- Ensure the data builder, tokenizer and inference decoder all use the same token IDs and serialization.
- Version and hash the tokenizer alongside every checkpoint.

The tokenizer now reserves stable IDs for system, user, Hu Tao, memory, report and end-of-turn markers. The complete conversation serializer, role-aware packing, and inference stopping behaviour must still be implemented and tested before character dialogue training.

## Phase D — base language training (may proceed alongside Phase A)

Train this project's decoder-only language model from random initialization on the approved broad language corpus. No third-party checkpoint, pretrained model or neural weight may be used. This general-language phase may proceed while Hu Tao's former-life stages are being researched and reviewed; it does not authorise Hu Tao-specific identity/dialogue training. The model needs enough general language ability to understand and respond to ordinary human conversation; the Hu Tao memory set alone is far too small and narrow for this.

The current `training/train.py` uses disk-backed int32 token caches and samples mini-batches from the memory-mapped files, avoiding a full token-stream copy into RAM/VRAM. Measure:
- train and validation loss/perplexity;
- convergence and overfitting;
- sample generations at regular checkpoints;
- throughput, wall-clock time, peak memory and checkpoint size;
- reproducibility from stored config, tokenizer and data hashes.

The first calibration result is recorded in [reports/BASE_LANGUAGE_CALIBRATION_2026-11-02.md](reports/BASE_LANGUAGE_CALIBRATION_2026-11-02.md), and its 10.24M-token-exposure warm-start continuation is recorded in [reports/BASE_LANGUAGE_CONTINUATION_2026-11-03.md](reports/BASE_LANGUAGE_CONTINUATION_2026-11-03.md). Both establish a measured CPU baseline, but qualitative generations remain incoherent; the base-language phase is not complete.

Do not mistake memorisation of the corpus for language competence. Keep a held-out evaluation set from domains not copied verbatim into training. If the compute or corpus budget is inadequate, resize the model/training plan honestly rather than announcing success.

## Phase E — direct Hu Tao character training

Continue training the same base checkpoint created by this project on curated first-person Hu Tao material and multi-turn dialogue examples. This is staged training of our own from-scratch model, not fine-tuning or rebuilding a third-party model. The goal is to internalise her identity and speaking behaviour in the model weights, not to rely on a huge system prompt.

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

The first selected source is the Cosmopedia v2 configuration from Hugging Face SmolLM-Corpus; only its generated `text` field is used. The dataset card declares ODC-By 1.0. Its full configuration is roughly 122 GB compressed, so the acquisition script streams a capped sample and writes a source/hash manifest rather than downloading the complete corpus. This educational/story text is not a natural conversation corpus and does not teach Hu Tao's persona; separate dialogue and direct character data are still required.

The user works with Android-first constraints and limited local compute. Do not assume the whole from-scratch training run can run on the phone, Chromebook or free CI. Estimate corpus size, model size, optimiser memory, sequence length, batch size, number of tokens and total compute before launching a long job. Hosted training compute may be considered only with explicit attention to limits and reproducibility; the final app must not depend on cloud inference.

The complete distributable AI payload has an absolute maximum of 1,000,000,000 bytes. Count the checkpoint, tokenizer/vocabulary, configuration, metadata and every auxiliary AI asset together. No additional embedding model, reranker, adapter or second model may be included if the total would exceed the cap. The cap is not a size target; any smaller package is acceptable. The training corpus is not part of the inference bundle and must not be shipped with it. Model size under the cap does not guarantee usable quality; data, training-token count and evaluation still matter.
