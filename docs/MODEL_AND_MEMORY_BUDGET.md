# Model, context and memory budgets

These are three different limits and must not be conflated.

## 1. Neural weights: maximum 1 GB

The project checkpoint validator enforces `max_weight_bytes = 1,000,000,000`. The current reference network is much smaller than that ceiling (about 4.94M parameters, roughly 19 MiB raw FP32 weights). Do not enlarge the network merely to consume the entire budget. Increase capacity only after there is a valid language corpus, meaningful training progress, and measured Android RAM/latency.

## 2. Active context: token count

The active context window counts tokens, not bytes and not stored history. The current reference config uses a 256-token context window; the training script uses a separately selected sequence length. These values will need to grow and be benchmarked before building a natural conversational runtime.

A full WhatsApp-like history cannot be loaded into every prompt. Instead, the recall layer assembles recent turns plus older messages and facts relevant to the current topic.

## 3. Persistent local memory: device storage

- `conversation.sqlite`: the complete single-thread transcript, in chronological order.
- `world_knowledge.sqlite`: user-taught facts about the user's world.
- `hu_tao_life.sqlite`: ordered personal memories and Teyvat life-stage progress.
- `teyvat_chronicle.sqlite`: canon story beats and reports through which Hu Tao learns of distant events.

These databases are independent of model weights and are not truncated to fit the active context. Phone storage remains finite, so the Android app must show storage use and provide local backup/export controls.

## User-taught world facts

Every fact is written to the local store immediately. Facts are labelled `user_taught`; they should not be presented as independently verified public facts unless a separate source supports them.

Facts default to not being included in training. The user must explicitly approve a fact before it can be exported to a future training dataset. Raw conversation messages are not exported by this process.

## No forgetting by summarization

The complete raw transcript is retained. Derived long-term memories can summarize stable personal details, but they supplement the original transcript and never replace it. The recall layer searches both recent conversation and older relevant turns using meaningful query terms rather than requiring exact wording.

## Offline and privacy constraints

All runtime memory is intended to stay on-device. The current Python database prototype is not yet encrypted at rest; before a public Android release, add database encryption, export/import validation, and an explicit delete/backup screen.
