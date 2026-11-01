# Language foundation — current execution stage

Decision date: 2026-10-31  
Character: Hu Tao  
Repository codename: Joseph

## Permanent model constraints

- The entire distributable AI inference payload must remain at or below **1,000,000,000 bytes**. This includes checkpoint weights, tokenizer/vocabulary, config, generation metadata and any other neural/AI assets. There is no minimum size; smaller is fine. Do not add model components that take the combined bundle over the cap.
- Every model weight is initialized randomly. The tokenizer is trained on approved text. No third-party pretrained LLM, model weights, adapters, model distillation, pretrained embeddings or external reranking model may be used. Later character training continues only from this project's own base checkpoint.
- Text-only datasets are allowed after licence review and stay outside the final inference bundle.

## What changed

The user confirmed that Hu Tao's Teyvat knowledge must be limited to what she personally experiences or what ordinary people could plausibly hear, read or discuss. Traveler-only secrets and private conversations are not part of her knowledge. See character/knowledge_policy.md.

This resolves the perspective rule enough to begin the general language foundation. Broad language-model preparation and base-language experiments can proceed in parallel with continued former-life research. Hu Tao-specific character training remains gated on substantive review of stages 1–9.

## Selected base corpus

The first source is Cosmopedia v2 from Hugging Face's SmolLM-Corpus, using only the generated text field. The dataset card declares ODC-By 1.0. Its full configuration is approximately 122 GB compressed, so the repository contains a capped streaming acquisition script rather than a copy of the dataset.

The script streams rows, deduplicates normalized text, generates deterministic train/validation partitions and records source revision, settings and file hashes. The generated prompt and seed_data fields are excluded. The corpus is general educational/story text, not Hu Tao dialogue or a full conversational dataset.

## Implemented in the repository

- Rust-backed Hugging Face Tokenizers ByteLevel BPE.
- Ten reserved tokens with fixed IDs: pad, bos, eos, unk, system, user, hutao, memory, report and end-of-turn.
- Tokenizer training uses train.txt only and writes a sidecar SHA-256 manifest.
- Corpus acquisition is streamed and capped; train/validation are split without document overlap.
- Text is converted to disk-backed int32 token caches, one sample per line, adding BOS once and EOS after each sample. The trainer uses configurable model width, layer count, head count and dropout, and CI exercises a tiny full training/checkpoint smoke test.
- The training loop samples batches from memory-mapped token files rather than loading the entire token stream into RAM/VRAM.
- The trainer supports held-out validation loss and perplexity.

## Current execution state

GitHub Actions run #69 passed for commit 9f9005a1f92b955e69d85dc97328042aa97bd814: 47 unit tests passed, Python modules compiled, and a tiny two-step CPU training smoke test measured train/validation loss and saved a checkpoint. This verifies the execution path and checkpoint creation, not the actual external dataset download or useful language-model quality.

No completed Cosmopedia tokenizer artifact has been verified yet, no base model has been trained, and no Hu Tao character model exists. The repository now has a bounded 10M pilot and a 20-partition full-corpus workflow; completion must be verified from GitHub Actions artifacts. The current chat runtime has CPU-only PyTorch, no cached corpus and no outbound access to the dataset. A meaningful base-language training run therefore still needs an external machine with network and suitable training compute.

The current approximately 4.94M-parameter default Transformer remains a small research baseline (the 8,192-token tokenizer setting increases embedding parameters). It is not a plausible finished conversational model merely because it fits a checkpoint or has lower validation loss. The current 200-million-character corpus cap is only a pipeline pilot, roughly tens of millions of tokens, not a serious pretraining budget. For scale context, Hugging Face reports that SmolLM's 135M/360M models were trained on 600B tokens and the 1.7B model on 1T tokens; those runs also used a broader data mixture and are not a direct recipe for this project. The final architecture and token budget must be selected from actual corpus size, measured training throughput, available compute and offline inference constraints.

## Commands

Install dependencies:

    python -m pip install -e ".[data,train,test]"

Acquire a capped corpus sample:

    python -m training.acquire_base_corpus --output-dir artifacts/base_language/cosmopedia-v2 --max-samples 5000 --max-characters 10000000 --shuffle-buffer 1000

Fit the tokenizer on the training split only:

    python -m training.train_tokenizer --input artifacts/base_language/cosmopedia-v2/train.txt --output artifacts/base_language/tokenizer.json --vocab-size 8192

Run an experimental from-scratch baseline with validation:

    python -m training.train --input artifacts/base_language/cosmopedia-v2/train.txt --validation artifacts/base_language/cosmopedia-v2/validation.txt --tokenizer artifacts/base_language/tokenizer.json --output artifacts/base_language/base-pilot.pt --steps 10000 --batch-size 8 --seq-len 256 --d-model 256 --n-layers 6 --n-heads 8 --device cuda

Token caches are generated automatically under artifacts/base_language/token_cache by default. They use disk-backed int32 files so the full corpus need not be copied into RAM/VRAM. The sample command is a pilot, not the final compute recipe. Do not run a long job on a phone or 4 GB Chromebook.

## Hosted corpus and tokenizer build

The repository contains two hosted workflows:

- `.github/workflows/build-base-tokenizer.yml` is the small one-stream tokenizer pilot.
- `.github/workflows/build-base-corpus.yml` builds the full capped corpus from **20 disjoint source partitions** and fits the final tokenizer once after merging.

The full workflow uses the pinned Cosmopedia v2 revision `3ba9d60`. The acquisition script now supports `--num-shards` and `--shard-index`; it partitions the Hugging Face streaming source before shuffling, so each matrix job reads a different deterministic subset of source shards. Each of the 20 jobs has a 10-million-character and 5,000-document cap. The matrix runs at most two jobs concurrently, with a 45-minute limit per shard. The theoretical upper bound is 900 Linux runner-minutes for shard jobs plus a 60-minute merge/tokenizer job; actual use should be lower.

After the shard jobs finish, the merge job:
- checks that all 20 source partition indexes exist exactly once;
- checks matching source revision, dataset licence, split settings and per-file SHA-256 hashes;
- globally deduplicates by normalized, case-folded text;
- rechecks the hash-based train/validation assignment to reject leakage;
- caps the merged corpus at 200,000,000 characters;
- fits one 8,192-entry ByteLevel BPE tokenizer on the combined `train.txt` only;
- verifies the tokenizer manifest, reserved IDs and Unicode encode/decode round trips.

Intermediate shard artifacts are retained for one day and the merged corpus/tokenizer artifact for three days. The raw training corpus is stored as a workflow artifact, not committed to the repository and never included in the final inference bundle.

The first non-sharded three-hour attempt was cancelled before producing a verified tokenizer. The smaller 10-million-character pilot and new 20-shard pipeline are intended to isolate the slow acquisition work and keep each individual job bounded. An unauthenticated Hugging Face warning is informational unless the run logs show actual rate-limit or download errors; no token is required for this public dataset.

**Execution is not considered successful merely because workflow files were committed.** Confirm a completed GitHub Actions run and download the `joseph-cosmopedia-v2-merged-200m-tokenizer` artifact before claiming the full corpus or final tokenizer exists. The 20 × 10M settings are upper bounds: the final manifest reports the actual unique character and sample counts, which may be lower after global deduplication. This stage acquires text and trains the tokenizer; it does not train neural weights.

## Live acquisition status (2026-11-01)

The user made `Thunder2700c/Joseph-` public. This was independently verified from GitHub repository metadata (`private: false`). Standard GitHub-hosted runner usage for public repositories is free; artifact storage remains subject to the GitHub plan's shared storage allowance, so intermediate shard artifacts are retained for one day and the final artifact for three days. See GitHub Actions billing documentation.

The 10-million-character tokenizer pilot:
- Run: [41293340375](https://github.com/Thunder2700c/Joseph-/actions/runs/41293340375)
- At the latest check: `in_progress`, still at the “Acquire capped Cosmopedia v2 corpus” step.
- A second trigger was created by commit `a2f1ea0ccb3d0ed921f599a21628a287a78bd2c0`; run [41296346001](https://github.com/Thunder2700c/Joseph-/actions/runs/41296346001) was `pending` behind the same workflow concurrency group. Do not start another duplicate pilot unless the state changes.

The 20-partition 200-million-character acquisition workflow:
- Run: [41295607839](https://github.com/Thunder2700c/Joseph-/actions/runs/41295607839)
- At the latest check: 20 matrix jobs created; shard 0 completed successfully, shards 1 and 2 were acquiring data, and 17 shard jobs were queued.
- The source reports 104 underlying data shards.
- Shard 0 downloaded and validated 9,999,831 unique characters: 2,539 train samples, 147 validation samples, zero duplicates within the shard, and an approximate 2,499,958 tokens. Its uploaded artifact was 3,714,469 bytes.
- Its logs showed the unauthenticated Hugging Face warning, but the shard still completed, passed provenance/size checks and uploaded successfully.

These are live snapshots, not final results. The overall 200M workflow still needs all 20 partitions, global deduplication, merged manifest verification and final 8,192-vocabulary tokenizer fitting. No neural-model training has started. Refresh the run links before reporting updated progress.

## Conversation-data follow-up

Cosmopedia v2 supplies educational prose and stories, not authentic conversational turn structure. A candidate supplementary source is OpenAssistant OASST1: its dataset card declares Apache-2.0, provides 84,437 training and 4,401 validation messages in its ready export, and preserves parent-linked conversation trees. It also contains multilingual and synthetic/model-generated rows, so a later adapter must preserve train/validation separation, reconstruct coherent trees, filter to the selected language and reviewed human-authored records, and keep it as a small dialogue supplement rather than allowing generic assistant responses to define Hu Tao. It has not been downloaded or added to training yet.

## Remaining steps

1. Run corpus acquisition and inspect its manifest, token count and licence attribution.
2. Fit and test the tokenizer on the actual training split, then archive its manifest.
3. Choose model dimensions and training-token budget from measured hardware throughput; the 4.94M model is only a baseline.
4. Run controlled base training, record loss/perplexity, throughput, peak memory, generation samples and checkpoint size.
5. Continue expanding/reviewing Hu Tao's former-life memories and public Teyvat knowledge.
6. Once stages 1–9 pass review, build original first-person Hu Tao dialogue data and continue training for her character.
7. Complete held-out character evaluation before any Android implementation.
