# Hu Tao realism-focused model rebuild (v2)

Status: new direction approved by the user; model redesign and full training are not complete.

## Intended result

Build one genuinely offline Hu Tao character model that can sustain ordinary conversation rather than merely answer prompts. The target behaviours include casual short replies, natural turn-taking, follow-up questions, humour, teasing, curiosity, serious discussion, disagreement, embarrassment, sadness, anger, affection, consensual flirting, scene-based role-play and continuity across long conversations.

The model should portray emotions coherently. That is a behavioural target; a neural model producing emotional language is not evidence that it has subjective feelings or is literally a human woman.

## Final download budget

- Target final download: approximately 1.0–1.5 GB.
- Hard complete inference bundle limit: 1,500,000,000 decimal bytes.
- Count quantized weights, tokenizer, configuration, generation settings and every other required inference asset.
- Training corpus, optimizer state, training checkpoints, metrics and evaluation sets do not count toward the phone download and must not be bundled.
- A Q4-quantized model around 1.5–2.0B parameters is a planning estimate, not a guaranteed fit or quality result. Measure the exact exported artifact and validate the full bundle before calling it complete.

## Rules carried forward

- Initialize all neural weights randomly. Train the tokenizer from approved text.
- No third-party pretrained LLM weights, adapters, pretrained embeddings, external rerankers or model distillation. The user has not revoked this existing constraint.
- Do not use private user conversation history for training without separate explicit permission.
- Do not use copied Genshin game scripts, voice lines, extracted audio or verbatim character dialogue.
- Hu Tao remains the character identity. The repository name is only a GitHub slug.
- General-language and non-character dialogue research can proceed now. Hu Tao-specific life and identity training remains gated until life stages 1–9 are substantively reviewed and approved.
- Keep original authored character examples separate from public dialogue corpora and trace their sources.

## Why the 13.9M pilot is rejected

The previous completed pilot trained a 13,891,584-parameter model with an 8,192-token vocabulary, 256-token context, 10,000 steps and 10.24M token exposures. It used a small FineWeb-Edu sample mixed with OASST1 and Dolly. The loss decreased, but the generated answers were not competent conversation.

Do not keep extending that checkpoint or treat its loss curve as proof of language ability. Model capacity, context length, data volume and quality, tokenizer, training objective, optimization and generation all need a controlled redesign.

## Architecture direction

The current hand-written GPT prototype uses learned absolute position embeddings, LayerNorm, GELU and a small context. It is not a suitable foundation for a modern quantized Android model.

Build a decoder-only architecture that can be exported through a tested GGUF-compatible route:
- RoPE positional embeddings.
- RMSNorm.
- SwiGLU MLP.
- Grouped-query attention to reduce KV-cache cost.
- A project-trained tokenizer, likely in the 32K-vocabulary range after corpus statistics are measured.
- Start with a 4K context; move to 8K only after memory and latency testing.
- Random initialization, reproducible configuration, checkpoint-resume tests and recorded data/tokenizer hashes.
- Test conversion and quantization from a tiny smoke checkpoint before beginning long pretraining.

Do not hard-code a final parameter count based only on the package cap. First implement a parameter estimator for the exact architecture, test export, quantized size, context memory and Android inference. The intended range is roughly 1.5–2.0B parameters if the measured Q4 bundle fits; select the largest size that passes device constraints and training feasibility.

## Corpus design

Keep separate corpora for distinct capabilities. Do not mix everything into one opaque text file.

### 1. General language pretraining

Use a large, diverse, deduplicated text mixture appropriate to a billion-parameter model. FineWeb-Edu can remain one source, but it must not be the whole foundation. Add curated reference and encyclopedic text, public-domain books where rights are clear, explanatory prose and additional properly licensed text. Filter boilerplate, low-quality pages, duplicates, obvious personal data and malformed documents. Record source revisions, licenses, filters, document counts, token counts, hashes and train/validation/test splits.

The prior FineWeb workflow capped each of four shards at five million characters. That was a pipeline test, not a serious pretraining corpus for a 1.5–2.0B model.

### 2. Natural conversation and emotional language

The first v2 human-dialogue audit uses:
- OASST1: English, reviewed, non-deleted and explicitly non-synthetic human messages; preserve conversation-tree split isolation.
- DailyDialog: multi-turn everyday conversations and human emotion/communication labels.
- EmpatheticDialogues: crowdsourced emotional-situation conversations with the official train/validation/test split.

The initial audit has a non-commercial licensing constraint: DailyDialog is CC BY-NC-SA 4.0 and EmpatheticDialogues is CC BY-NC 4.0. Preserve attribution, and do not assume a commercial model release is permitted. OASST1 is Apache-2.0.

These sources teach different things. DailyDialog provides everyday turn-taking; EmpatheticDialogues provides emotion-grounded responses; OASST1 contributes broader human-written conversational responses. None alone is sufficient for role-play.

### 3. Persona and long-running relationships

Investigate before inclusion:
- PersonaChat / ConvAI2 for conversations between people given consistent profiles.
- Multi-Session Chat for human conversations across sessions, including learning and recalling partner details.
- Topical-Chat for longer knowledge-grounded human-to-human dialogue.

Do not include a mirror simply because it is downloadable. Verify the original dataset-specific rights, licence and allowed uses first. Topical-Chat's source release uses CDLA-Sharing-1.0 and requires a separate agreement review. PersonaChat and Multi-Session Chat are held until their exact data rights are established.

### 4. Hu Tao identity, role-play and relationships

Only after the life-stage review gate passes, build original multi-turn first-person Hu Tao material grounded in approved biography and knowledge. Include:
- her identity, daily life, values, work, friendships and relationships;
- natural conversational rhythm, short replies, topic changes, questions and context-sensitive humour;
- emotional variation without announcing every emotion or using the same catchphrase;
- disagreement, uncertainty, correction, apology and clear personal boundaries;
- role-play scene continuity and actions;
- playful teasing, affection and flirting only when appropriate to the established relationship and boundaries;
- her knowledge limits, including the distinction between personally witnessed events, hearsay, public information and unknown private events.

Do not copy game dialogue or create fake canon quotations. Original examples should convey her character without reproducing protected text.

### 5. Human preference evidence

Use preference labels only when the dataset actually contains human ratings or rankings and the task semantics justify them. Do not invent preference pairs, automatically rank examples by length, or silently substitute model-generated judgments.

## Training stages

1. Finish and inspect the current human-dialogue corpus audit. Review random samples from every source and every split before promoting the corpus.
2. Build the large general-language corpus with source-level provenance and held-out splits.
3. Implement the new architecture and verify random initialization, tokenization, loss masks, checkpoint resume, export and quantization using a small smoke run.
4. Benchmark the exact architecture on the intended accelerator and derive achievable tokens per second, peak memory, checkpoint size and a realistic total training budget.
5. Pretrain from random initialization on the broad general-language corpus.
6. Continue training on human conversational, persona and emotion-grounded dialogue.
7. After life stages 1–9 have been reviewed and approved, train Hu Tao's character identity and role-play behaviour on original source-grounded examples.
8. Quantize, export and evaluate the exact phone bundle.
9. Freeze a checkpoint only after the language, role-play, factuality and on-device gates pass.

## Evaluation gates

Never judge success by validation loss alone. Maintain fixed test sets that do not enter tokenizer fitting or training. Evaluate:
- casual one-line conversation and appropriate response length;
- multi-turn turn-taking, topic shifts and follow-up questions;
- consistency over 20–50 turn conversations;
- role-play scenes over multiple exchanges;
- emotional appropriateness for happy, sad, angry, embarrassed, affectionate, neutral and mixed situations;
- humour and flirting without repetitive stock phrases;
- disagreement, boundaries, uncertainty and correction;
- ordinary factual questions and knowledge limits;
- repetition, garbling, format leakage and unwanted assistant-like meta talk;
- Android load time, generation speed, RAM and KV-cache use at 4K/8K context;
- exact final bundle size.

The model passes only after representative samples are manually reviewed and measurable gates pass. A successful workflow, a decreasing loss or a valid checkpoint does not mean Hu Tao is ready.

## Compute and workflow reality

Unlimited corpus size is not the same as unlimited compute. The current GitHub Actions standard hosted runner is a CPU runner; it is useful for data preparation, tests, corpus audits, packaging and orchestration. It is not an appropriate environment for serious pretraining of a 1.5–2.0B model from random initialization. Hosted job limits also prevent one uninterrupted multi-week training run.

Long training needs an actual accelerator environment with enough GPU memory and reliable resumable storage. The training system must save checkpoints and metadata so runs can resume across jobs. Do not launch a multi-billion-parameter CPU job on the standard runner and call that a serious attempt. A GPU runner or cloud accelerator would need to be made available and any paid compute must be explicitly budgeted and authorized. The final inference runtime remains fully offline.

## Current active work

- training/audit_human_dialogue_corpus.py builds an initial corpus from OASST1, DailyDialog and EmpatheticDialogues.
- .github/workflows/audit-human-dialogue.yml acquires, filters and audits the corpus, then uploads a temporary artifact.
- The first successful corpus preparation reported 118,567 train examples, 16,030 validation examples and 14,581 test examples. The artifact upload failed on an invalid workflow variable escape and is being corrected.
- This is a data milestone only. It is not a trained Hu Tao model.
