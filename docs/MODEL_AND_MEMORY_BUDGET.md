# Model, context and memory budgets

The project has three distinct storage concepts. The complete AI payload has a strict one-billion-byte ceiling. Active context is counted in tokens. Persistent user memory is stored separately.

## 1. Complete distributable AI payload: hard maximum 1,000,000,000 bytes

This is a hard maximum of **1,000,000,000 decimal bytes** for the complete AI inference bundle. It is not a target that must be filled; any smaller size is acceptable and there is no minimum model size.

The bundle total includes:
- all neural checkpoint/model weights;
- tokenizer and vocabulary files;
- model configuration and generation metadata;
- every additional model, adapter, embedding/reranking weight or other AI asset required at inference time.

Do not include a second model or AI asset if the combined package would exceed the limit. Do not rely on compressed archive size alone: validate the actual extracted payload directory.

Run the validator before release:

    python -m training.validate_ai_bundle --root <complete-ai-bundle-directory>

The validator recursively counts all regular files and rejects packages over the cap. The checkpoint configuration also prevents setting its own maximum above the project limit.

The cap applies to the distributable AI inference payload. It does not include the separate training corpus, source datasets, Android/application code or runtime libraries, nor user-owned conversation/memory databases. Training corpora must never be copied into the shipped AI bundle. If the eventual product packaging changes, recalculate the full AI payload and enforce the same ceiling.

## 2. From-scratch model construction

This project creates its own model rather than wrapping or rebuilding an existing AI:
- all neural weights begin from random initialization;
- the tokenizer and vocabulary are trained from approved text;
- base language training uses permitted text data and next-token prediction;
- Hu Tao specialization continues from the base checkpoint produced by this project itself.

Prohibited: importing/fine-tuning a third-party pretrained LLM, downloading third-party model weights, external pretrained embeddings or rerankers, LoRA/adapters from other models, and model distillation. Licensed text corpora are allowed because they are training data, not pretrained neural weights.

## 3. Active context: token count

The active context window counts tokens, not bytes and not stored history. The current reference config uses a 256-token context window; the training script uses a separately selected sequence length. These values must be benchmarked before building the conversational runtime.

A full WhatsApp-like history cannot be loaded into every prompt. Instead, the recall layer assembles recent turns plus older messages and facts relevant to the current topic.

## 4. Persistent local memory: device storage

- conversation.sqlite: the complete single-thread transcript, in chronological order.
- world_knowledge.sqlite: user-taught facts about the user's world.
- hu_tao_life.sqlite: ordered personal memories and Teyvat life-stage progress.
- teyvat_chronicle.sqlite: canon story beats and reports through which Hu Tao learns of distant events.

These databases are independent of AI weights and the distributable AI bundle. Phone storage remains finite, so the Android app must show storage use and provide local backup/export controls.

## User-taught world facts

Every fact is written to the local store immediately. Facts are labelled user_taught; they should not be presented as independently verified public facts unless a separate source supports them.

Facts default to not being included in training. The user must explicitly approve a fact before it can be exported to a future training dataset. Raw conversation messages are not exported by this process.

## No forgetting by summarization

The complete raw transcript is retained. Derived long-term memories can summarize stable personal details, but they supplement the original transcript and never replace it. The recall layer searches both recent conversation and older relevant turns using meaningful query terms rather than requiring exact wording.

## Offline and privacy constraints

All runtime memory is intended to stay on-device. The current Python database prototype is not yet encrypted at rest; before a public Android release, add database encryption, export/import validation, and an explicit delete/backup screen.
