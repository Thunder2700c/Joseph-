# Language foundation — current execution stage

Decision date: 2026-10-31  
Character: Hu Tao  
Repository codename: Joseph

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

GitHub Actions passed unit tests and Python compilation for commit a99883fd762d387a06f87d360f7ff85b25aaf3f0. This verifies the tokenizer, data-preparation helpers and existing project tests, not the actual external dataset download or a trained model.

No tokenizer has been fitted on Cosmopedia v2 yet, no base model has been trained, and no Hu Tao character model exists. The current chat runtime has CPU-only PyTorch, no cached corpus and no outbound access to the dataset. A meaningful base-language training run therefore still needs an external machine with network and suitable training compute.

The current approximately 4.94M-parameter Transformer remains a small research baseline. It is not a plausible finished conversational model merely because it fits a checkpoint or has lower validation loss. The final architecture and token budget must be selected from actual corpus size, measured training throughput, available compute and offline inference constraints.

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

## Remaining steps

1. Run corpus acquisition and inspect its manifest, token count and licence attribution.
2. Fit and test the tokenizer on the actual training split, then archive its manifest.
3. Choose model dimensions and training-token budget from measured hardware throughput; the 4.94M model is only a baseline.
4. Run controlled base training, record loss/perplexity, throughput, peak memory, generation samples and checkpoint size.
5. Continue expanding/reviewing Hu Tao's former-life memories and public Teyvat knowledge.
6. Once stages 1–9 pass review, build original first-person Hu Tao dialogue data and continue training for her character.
7. Complete held-out character evaluation before any Android implementation.
