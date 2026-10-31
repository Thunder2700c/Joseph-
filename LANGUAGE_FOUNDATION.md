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

No tokenizer has been fitted on Cosmopedia v2 yet, no base model has been trained, and no Hu Tao character model exists. The current chat runtime has CPU-only PyTorch, no cached corpus and no outbound access to the dataset. A meaningful base-language training run therefore still needs an external machine with network and suitable training compute.

The current approximately 4.94M-parameter default Transformer remains a small research baseline (the 8,192-token tokenizer setting increases embedding parameters). It is not a plausible finished conversational model merely because it fits a checkpoint or has lower validation loss. The current 200-million-character corpus cap is only a pipeline pilot, roughly tens of millions of tokens, not a serious pretraining budget. For scale context, Hugging Face reports that SmolLM's 135M/360M models were trained on 600B tokens and the 1.7B model on 1T tokens; those runs also used a broader data mixture and are not a direct recipe for this project. The final architecture and token budget must be selected from actual corpus size, measured training throughput, available compute and offline inference constraints.

## Commands

Install dependencies:

    python -m pip install -e ".[data,train,test]"

Acquire a capped corpus sample:

    python -m training.acquire_base_corpus --output-dir artifacts/base_language/cosmopedia-v2 --max-samples 100000 --max-characters 200000000

Fit the tokenizer on the training split only:

    python -m training.train_tokenizer --input artifacts/base_language/cosmopedia-v2/train.txt --output artifacts/base_language/tokenizer.json --vocab-size 8192

Run an experimental from-scratch baseline with validation:

    python -m training.train --input artifacts/base_language/cosmopedia-v2/train.txt --validation artifacts/base_language/cosmopedia-v2/validation.txt --tokenizer artifacts/base_language/tokenizer.json --output artifacts/base_language/base-pilot.pt --steps 10000 --batch-size 8 --seq-len 256 --d-model 256 --n-layers 6 --n-heads 8 --device cuda

Token caches are generated automatically under artifacts/base_language/token_cache by default. They use disk-backed int32 files so the full corpus need not be copied into RAM/VRAM. The sample command is a pilot, not the final compute recipe. Do not run a long job on a phone or 4 GB Chromebook.

## Hosted tokenizer build

The repository now contains `.github/workflows/build-base-tokenizer.yml`. It runs on matching pushes to `main` and can also be started from the GitHub Actions page with **Run workflow**. The job:

- streams the pinned Cosmopedia v2 revision `3ba9d60`;
- caps acquisition at 200,000,000 characters and 100,000 samples;
- fits the 8,192-entry ByteLevel BPE tokenizer on `train.txt` only;
- verifies the tokenizer manifest hashes, all reserved token IDs, and Unicode encode/decode round trips;
- uploads the tokenizer, corpus splits, manifests and licence notes as the `joseph-base-tokenizer-cosmopedia-v2` Actions artifact for 30 days.

The workflow definition is committed, but do not treat the tokenizer as built until the hosted job reports success and the artifact is available. This step only acquires text and fits the tokenizer; it does not train any neural weights.

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
