# Language foundation — current execution stage

Decision date: 2026-10-31  
Character: Hu Tao  
Repository codename: Joseph

## What changed

The user confirmed that Hu Tao's Teyvat knowledge must be limited to what she personally experiences or what ordinary people could plausibly hear, read or discuss. Traveler-only secrets and private conversations are not part of her knowledge. See character/knowledge_policy.md.

This resolves the perspective rule enough to begin the general language foundation. Work on broad language ability can proceed while the detailed former-life/Teyvat memory collection continues. Hu Tao-specific character training must still wait until the former-life material is substantively researched and reviewed.

## Training sequence

1. Prepare and audit an appropriately licensed general-language corpus.
2. Train a byte-level BPE tokenizer with dedicated control tokens.
3. Train a small decoder-only language model from random initialization; use a held-out validation set and record data/tokenizer hashes.
4. Continue expanding and reviewing Hu Tao's former-life memories and public Teyvat knowledge under the point-of-view policy.
5. Only after that material is ready, train the model directly on original Hu Tao biography, first-person dialogue and multi-turn examples.
6. Evaluate character identity, conversational quality, factual perspective boundaries and the death-memory rule.
7. Begin Android work only after the standalone model passes its acceptance gates.

## Selected base corpus

The first source is Cosmopedia v2 from Hugging Face's SmolLM-Corpus, using only the generated text field. The dataset card declares ODC-By 1.0. Its full configuration is approximately 122 GB compressed, so the repository contains a capped streaming acquisition script rather than a copy of the dataset. The script saves train/validation files and a reproducibility manifest locally.

This synthetic educational/story corpus is useful for a first language-model foundation, but it is not a complete conversational corpus. It does not teach Hu Tao's persona. Later stages need original multi-turn dialogue material and separate reviewed character/Teyvat training data.

## Repository changes in this stage

- The tokenizer uses the Rust-backed Hugging Face Tokenizers implementation and ByteLevel BPE.
- Ten reserved IDs are allocated in a fixed order: pad, bos, eos, unk, system, user, hutao, memory, report and eot markers.
- Tokenizer training writes a sidecar manifest with source hashes and the reserved-token list.
- The corpus acquisition tool streams a capped Cosmopedia v2 sample, deduplicates it, makes a deterministic hash-based validation split and records hashes.
- The model trainer supports a held-out validation file and reports validation loss/perplexity.

## What has not happened

No tokenizer has been fitted on Cosmopedia v2 yet, no base model has been trained, and no Hu Tao character model exists. The current chat environment has CPU-only PyTorch, no cached corpus and no working outbound network access, so it cannot honestly perform the real corpus download or a meaningful pretraining run. The code is prepared for a machine with network access and suitable training compute.

The existing roughly 4.94M-parameter Transformer remains a prototype. Its ability to save a checkpoint is not evidence of language competence. The next real training run must log its token count, model parameter count, validation metrics, runtime, peak memory, checkpoint size and generation samples.

## Commands

Install dependencies:

    python -m pip install -e ".[data,train,test]"

Acquire a pilot corpus:

    python -m training.acquire_base_corpus --output-dir artifacts/base_language/cosmopedia-v2 --max-samples 100000 --max-characters 200000000

Fit tokenizer on the training split only:

    python -m training.train_tokenizer --input artifacts/base_language/cosmopedia-v2/train.txt --output artifacts/base_language/tokenizer.json --vocab-size 8192

Run an experimental from-scratch baseline with validation:

    python -m training.train --input artifacts/base_language/cosmopedia-v2/train.txt --validation artifacts/base_language/cosmopedia-v2/validation.txt --tokenizer artifacts/base_language/tokenizer.json --output artifacts/base_language/base-pilot.pt --steps 10000 --batch-size 8 --seq-len 256 --device cuda

That sample command is a pipeline pilot, not the final compute recipe or model-size recommendation. Do not run a long job on a phone or 4 GB Chromebook. The final architecture and token budget must be chosen from measured training and inference constraints.
