# Base-language calibration pilot — measured results

**Run date:** 2026-11-02 UTC / 2026-11-03 India time  
**GitHub Actions run:** [41541564443](https://github.com/Thunder2700c/Joseph-/actions/runs/41541564443)  
**Run commit:** `97f4dc597281c0141bda61521c4b9e3dc808f2eb`  
**Artifact:** [joseph-base-language-calibration-pilot.zip](https://github.com/Thunder2700c/Joseph-/actions/runs/41541564443/artifacts/13186499154)  
**Artifact ID:** `13186499154`  
**Artifact SHA-256:** `fb99a441b671ac2c45b1d0f0744b8ccb0b9de82bf360d9c582e1616fde1d1df4`

## Scope

This is the **general-language foundation calibration pilot**. It is not Hu Tao identity, biography, personality, Teyvat-knowledge or character-dialogue training. The network was initialized randomly and trained on the general Cosmopedia v2 text split. No third-party pretrained model weights were used.

Hu Tao-specific training remains blocked until former-life stages 1–9 and the public Teyvat knowledge dataset pass substantive review.

## Measured results

The final metrics below are from the saved JSONL telemetry at step 1,000. Train and validation loss are evaluation means over 10 sampled mini-batches, not the loss of only the final optimizer mini-batch.

| Measurement | Result |
|---|---:|
| Completed optimizer steps | 1,000 / 1,000 |
| Final evaluated training loss | 7.7335175 |
| Final evaluated validation loss | 7.8389405 |
| Training perplexity | 2,283.6208 |
| Validation perplexity | 2,537.5149 |
| Average training throughput over the measured loop | 6,219.78 tokens/sec |
| Final 100-step interval throughput | 6,464.36 tokens/sec |
| Training-loop elapsed time | 164.6359 sec (2 min 44.64 sec) |
| Optimizer-seen token exposures | 1,024,000 |
| Saved checkpoint size | 7,454,309 bytes (7.45 MB / 7.11 MiB) |

The last optimizer mini-batch loss was 8.4032249; it is not the same statistic as the averaged evaluated training loss of 7.7335175.

## Experiment configuration

- Random initialization; 1,858,304 parameters.
- Vocabulary: 8,192 entries.
- Transformer: 4 layers, hidden width 128, 4 attention heads, 128-token context, dropout 0.1.
- Batch size 8; sequence length 128; 1,024 tokens per optimizer step.
- AdamW; learning rate 0.0003; weight decay 0.1; seed 1337.
- CPU-only hosted runner; Ubuntu 24.04.5; Python 3.12.15; PyTorch 2.15.0+cpu; 2 CPU threads.
- Training split available to the sampler: 41,751,252 tokens.
- Held-out validation split: 2,204,121 tokens.
- Only 1,024,000 optimizer token exposures were processed. This was not a full pass/epoch over the training split; batches were sampled from the disk-backed token stream.
- The checkpoint is a pilot artifact, not a frozen release model. Its size does not include the full distributable inference package.

The training loop took 164.64 seconds. The full Actions job, from runner initialization through artifact upload and cleanup, took approximately 264.66 seconds (4 min 24.66 sec), including dependency installation, archive retrieval/extraction, cache preparation, validation and upload.

## Interpretation and limitations

- Both losses decreased, but the final validation perplexity of about 2,538 remains very high. This is not a usable conversational model.
- The throughput measurement is **training** throughput, not autoregressive generation speed.
- No GPU memory measurement applies because the run used CPU.
- No peak system RAM measurement was recorded.
- This run validates that the random-initialized training path, held-out metric calculation, checkpoint saving and artifact upload work. It does not demonstrate general conversational competence, Hu Tao identity, personality, factual reliability, or Android readiness.
- The small number of optimizer-seen tokens means the result is only a calibration point. The next base-language experiments must increase training exposure and use held-out qualitative generation/evaluation before any success claim.

## Reproducibility

The run metadata records:

- Training file SHA-256: `5b2a3f8f1ac98c4fa5a4be8b08f425839988763662da0cceb521b10d66a967df`
- Validation file SHA-256: `9338a643c0fd46f0f093065ed5e2d74f1a7a0d09ec267e9f65ccd362d2f04f5d`
- Tokenizer SHA-256: `57911f2d25e2733f36b172160e07dc6649c1501afa2e76b440b71829adc995ab`

The Actions artifact contains the checkpoint, JSONL metric history, run metadata and inference-pilot config. The compressed ZIP is 6,874,845 bytes; the checkpoint file inside it is 7,454,309 bytes.
