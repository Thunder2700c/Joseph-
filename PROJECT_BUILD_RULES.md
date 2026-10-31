# Mandatory model build rules

These rules apply to every future ChatGPT conversation, implementation, training experiment and release.

## Hard distributable AI size ceiling

The complete distributable AI inference payload must never exceed **1,000,000,000 bytes** (one billion decimal bytes). This is a hard maximum, not a target to fill. The model may be any smaller size; there is no minimum model size. Choose the smallest architecture that passes the measured quality tests within the cap.

The size calculation includes, together:
- all model/checkpoint weights;
- the tokenizer and vocabulary files;
- model configuration and generation metadata;
- any additional neural weights, adapters, embedding models, rerankers or other AI assets needed at inference time.

Do not add a second model or auxiliary AI component if the combined package would exceed the cap. Every candidate release must pass:

    python -m training.validate_ai_bundle --root <complete-ai-bundle-directory>

The validator counts all regular files recursively and rejects symlinks and over-limit packages. Keep the actual bundle below the ceiling with reasonable room for metadata rather than aiming at exactly one billion bytes.

This is a cap on the distributable AI model payload, not on the separate training corpus, the app/runtime code or user-owned conversation and memory databases. Training datasets must not be copied into the final AI bundle. Any future change to what is included in the deployable payload must be counted before release.

## From-scratch requirement — no rebuild of an existing AI

Build the language model itself from scratch:
- initialize every neural model weight randomly;
- train the tokenizer/vocabulary from approved training text;
- train the base language model with next-token prediction on permitted text;
- after the base model is trained, continue training that same project-created model on reviewed Hu Tao memories and original Hu Tao dialogue.

Do not download, wrap, import, merge or fine-tune an existing pretrained language model. Do not use pretrained model weights, LoRA/adapters from another model, external embedding models, external rerankers, model distillation or a third-party model as the base. Do not describe a wrapper around an existing model as the requested project.

Using properly licensed text datasets as training material is allowed; using an existing neural model's learned weights is not. The second character-training phase must continue from the base checkpoint created by this project, not from a third-party checkpoint.

## Correct order

1. Continue researching/reviewing former-life and public Teyvat knowledge under the established point-of-view boundary.
2. Acquire permitted text-only language data; record provenance and licence. Do not download pretrained weights.
3. Fit this project's tokenizer on the approved training split only.
4. Train a small random-initialized base Transformer and measure actual quality, token budget, RAM, speed and bundle size.
5. Choose model dimensions by measured quality and feasibility; never increase size merely to consume the 1 GB cap.
6. Complete/review Hu Tao's stages 1–9.
7. Continue training the project-created base model on original Hu Tao identity, memories and multi-turn dialogue.
8. Validate the complete AI payload against the 1,000,000,000-byte ceiling and evaluate the standalone companion.
9. Only after it passes quality and size gates, begin Android integration.

The final model is offline. Hosted compute may be used to train the project's own randomly initialized weights, but runtime inference must not depend on cloud services.

## Instructions for future ChatGPT chats

Always read PROJECT_BUILD_RULES.md, CONTINUATION_CONTEXT.md, MODEL_TRAINING_PLAN.md and docs/MODEL_AND_MEMORY_BUDGET.md before proposing architecture changes or model training.

Never suggest a pretrained LLM, rebuilding an existing model, external model weights, or a plan that could make the combined AI payload exceed 1 GB. When unsure of a component's installed size, measure it and include it in the bundle budget before adopting it.
