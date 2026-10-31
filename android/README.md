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

The first Android milestone should be inference-only. Large-scale training on the phone is not part of the initial plan.
