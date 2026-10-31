# Android target notes

The target is an offline Android companion running on-device. This repository does not yet contain an Android application or a proven Android inference runtime.

Current model code is a PyTorch reference implementation for architecture experiments and training. Before choosing an Android runtime, benchmark and compare export paths (for example ONNX Runtime Mobile or a native runtime) against the exact operations and tensor shapes used by Joseph. Do not assume that a desktop checkpoint can simply be copied into an Android app.

Android acceptance goals:

- no network required for chat, memory lookup, or saved JoJo facts;
- model weights capped at 1 GB, with the initial small configuration far below that;
- memory/knowledge databases stored separately and independently backupable;
- no dependency on cloud inference;
- measure peak RAM, cold-start time, tokens/second, battery use, and device temperature;
- gracefully fall back to retrieval-based answers when the language model is uncertain.

## Project sequence: Android work is deferred

Do not begin the Android application while the former-life memory stages are unfinished or while the standalone Hu Tao model is still being trained/evaluated.

The approved sequence is:

1. Complete and manually review the former-life stages 1–9.
2. Train the purpose-built Hu Tao character language model and make it pass the standalone evaluation gates in [MODEL_TRAINING_PLAN.md](../MODEL_TRAINING_PLAN.md).
3. Freeze and verify the trained checkpoint and standalone inference harness.
4. Only then start the Android application, integrating that already-working model.

The first Android milestone, once authorized by the sequence above, is an inference-only native shell. Large-scale model training is not an Android-app task and must not be used as a reason to start app development early.

## Durable memory across clear-data and reinstall

**Current limitation:** The repository has a Python SQLite conversation store, but it does not yet contain a native Android application or a tested Android backup/restore implementation. SQLite commits protect conversation turns while the database file remains present; Android's Clear storage and uninstall remove app-private files. Therefore the current project cannot yet promise that memories survive either operation.

The Android implementation must use two layers:

1. **Primary local state:** SQLite databases in app-private storage for the continuous transcript, durable personal memories, learned user facts and other runtime state. Commit each completed user/character turn and each approved memory immediately. Keep persona/relationship state versioned and saved with the same durable state. The first-awakening flag is one-shot: after the first Hu Tao response has been committed to the transcript, call `HuTaoPersona.mark_opening_scene_seen()` and save the persona so the opening scene is not replayed on later launches.
2. **Portable user-owned backup:** maintain a versioned, integrity-checked backup outside the app's private directory, in a folder the user chooses through Android's Storage Access Framework (for example, a `Joseph` folder in Documents). Refresh it after every completed exchange and important memory/persona change, and before database migrations. This must work with a local on-device folder while offline; cloud storage may be optional, never required.

The backup bundle must include the entire conversation database, personal-memory database, user-taught knowledge, persona/relationship state, life/awakening state and a manifest with schema version, creation time, file sizes and cryptographic checksums. Use SQLite's online backup API or an equivalent consistent snapshot; do not copy a live database file blindly while SQLite may be writing to its WAL. Write a new snapshot to a temporary document and replace the previous snapshot only after verification succeeds. Keep at least one previous known-good snapshot or a recoverable journal so a failed write does not destroy the only backup.

On first run, ask the user to choose a persistent backup location. After reinstall or Clear storage, ask them to select the existing backup and restore it before opening a new conversation. The file survives app removal only while the user does not delete it; Android access grants may need to be selected again after reinstall. Never claim the state is protected until the external backup has been written and verified. Show the last successful backup time and a clear warning when backup is unavailable or stale.

Because chat history is private, production backups should be encrypted. A recovery design must survive app reinstall: Android Keystore-only keys are insufficient if uninstall removes the key, so provide a user-controlled recovery passphrase or another tested recovery mechanism. Do not rely exclusively on automatic Android/Google backup, because restore behaviour depends on device, account and settings.

The portable bundle must not be stored only in GitHub or only in an app-specific external directory; neither is a suitable user conversation backup. Do not upload private chat history without explicit user consent.