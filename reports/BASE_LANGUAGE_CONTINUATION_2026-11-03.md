# General-language foundation continuation — measured results

**Run date:** 2026-11-03 UTC / India time  
**GitHub Actions run:** [41594999963](https://github.com/Thunder2700c/Joseph-/actions/runs/41594999963)  
**Run commit:** `d96a2724c8cd64a812966647a6f4cac8b15c6b4b`  
**Artifact:** [joseph-base-language-continuation-10m.zip](https://github.com/Thunder2700c/Joseph-/actions/runs/41594999963/artifacts/13213295962)  
**Artifact ID:** `13213295962`  
**Artifact ZIP size:** 6,880,948 bytes  
**Artifact SHA-256:** `91dd140d8444ec78e2d3dc974a0f1d8706c2cc533e4e9dec74a4579ed5fff6b8`

## Scope and continuation semantics

This is general-language foundation work only. It is not Hu Tao identity, biography, personality, relationship, emotional-autonomy or Teyvat character training. No Hu Tao-specific records were used.

The run loaded the exact 1,858,304-parameter checkpoint from the first calibration pilot, verified its SHA-256 (`7700fdf27b701133d13bdf6d564dbb0f88097f7230e232b8f962ffcf8de389fc`) and checked that its architecture matched the requested configuration. It then performed a **warm-start continuation** for 10,000 more optimizer steps.

Important: the prior checkpoint did not contain AdamW optimizer state. The model weights were restored, but a new AdamW optimizer state was initialized. This is not an exact interrupted-run resume. The model checkpoint records global step 11,000.

## Final measured results

Metrics below are the mean loss over 32 sampled mini-batches for each split at the final checkpoint. Each optimizer step processes 8 × 128 = 1,024 token positions.

| Measurement | Result |
|---|---:|
| Additional optimizer steps | 10,000 |
| Global checkpoint step | 11,000 |
| Additional optimizer-token exposures | 10,240,000 |
| Total exposures across first pilot + continuation | 11,264,000 |
| Final evaluated training loss | 7.2779244 |
| Final evaluated validation loss | 7.2709180 |
| Training perplexity | 1,447.9794 |
| Validation perplexity | 1,437.8699 |
| Average training throughput | 5,772.17 tokens/sec |
| Final 2,000-step interval throughput | 5,787.49 tokens/sec |
| Training-loop elapsed time | 1,774.03 sec (29 min 34 sec) |
| Checkpoint size | 7,454,541 bytes |
| Combined experimental inference bundle | 8,014,978 bytes |
| Remaining under 1,000,000,000-byte bundle ceiling | 991,985,022 bytes |

At the start-of-run evaluation (after the first optimizer update, using 32 evaluation batches), training loss was 7.9373904 and validation loss was 7.8811633. At the end, they were 7.2779244 and 7.2709180 respectively. The validation perplexity fell from 2,646.95 to 1,437.87 within this run. These are sampled estimates, not a full sweep over every token in the validation file.

## Evaluation progression

| Global step | Training loss | Validation loss | Validation perplexity |
|---:|---:|---:|---:|
| 1,001 | 7.9374 | 7.8812 | 2,646.95 |
| 3,000 | 7.5250 | 7.5315 | 1,865.97 |
| 5,000 | 7.4169 | 7.4200 | 1,669.03 |
| 7,000 | 7.3355 | 7.3507 | 1,557.22 |
| 9,000 | 7.3077 | 7.2939 | 1,471.36 |
| 11,000 | 7.2779 | 7.2709 | 1,437.87 |

## Qualitative generation result — NOT PASSED

The workflow generated three reproducible diagnostic continuations from generic prompts. Manual inspection of the artifact shows that all three are still largely incoherent: fragmented/repeated words, broken syntax and weak semantic continuity. These are not usable conversational responses.

Therefore:
- The training pipeline and checkpoint continuation are operational.
- Sampled validation loss/perplexity improved during the continuation.
- **Language competence is not demonstrated.** Do not call this a conversational model or a completed general-language foundation.
- The next work must address data quality/domain mix, token budget, architecture/sequence length and evaluation. Add suitable conversation data only after licence, human-authorship, language and split review. Avoid tuning solely to Cosmopedia validation loss.

## Hardware and configuration

- CPU-only GitHub-hosted runner, Ubuntu 24.04.5, Python 3.12.15, PyTorch 2.15.0+cpu, 2 CPU threads.
- Random-origin model; warm-started from the project's own previous checkpoint.
- 1,858,304 parameters; 8,192-token vocabulary; width 128; 4 Transformer layers; 4 heads; context 128; dropout 0.1.
- Batch size 8; sequence length 128; AdamW learning rate 0.0001; weight decay 0.1; seed 1337.
- Evaluation every 2,000 local steps, 32 mini-batches per split.
- Training split: 41,751,252 tokens; validation split: 2,204,121 tokens.
- Training file SHA-256: `5b2a3f8f1ac98c4fa5a4be8b08f425839988763662da0cceb521b10d66a967df`
- Validation file SHA-256: `9338a643c0fd46f0f093065ed5e2d74f1a7a0d09ec267e9f65ccd362d2f04f5d`
- Tokenizer SHA-256: `57911f2d25e2733f36b172160e07dc6649c1501afa2e76b440b71829adc995ab`
- Final checkpoint SHA-256: `25062181726b1251f59fa48fffa9058b347df40728f24a53a3df43ed6c80eea0`

The validator confirmed a complete experimental inference bundle of 8,014,978 bytes, comprising config (149 bytes), checkpoint (7,454,541 bytes) and tokenizer (560,288 bytes), below the project's 1,000,000,000-byte hard ceiling.

## CI and artifact

The separate repository test run [41595104438](https://github.com/Thunder2700c/Joseph-/actions/runs/41595104438) also passed: **56 tests passed**, followed by successful Python-module compilation. This includes the updated trainer smoke test that exercises warm-start checkpoint continuation.

The Actions artifact contains the checkpoint, JSONL metrics, run metadata, three diagnostic text samples and the inference-pilot config. It is retained by GitHub Actions for 30 days. It is an experimental checkpoint, not a release model.

## Gate status

- General-language training pipeline: operational.
- First calibration pilot: complete.
- 10.24M-token-exposure warm-start continuation: complete.
- Qualitative language competence: **not passed**.
- General-language foundation acceptance: **not passed**.
- Hu Tao life-history review stages 1–9: still required.
- Hu Tao-specific identity/dialogue training: **still blocked**.
- Android implementation: still deferred.
