# General-language and conversation quality gate

This evaluation applies only to the general-language foundation and generic assistant SFT stage. It does not assess Hu Tao's identity, personality, memories, or Teyvat knowledge; those remain gated on manual review of former-life stages 1–9.

## Required artifacts

For every candidate checkpoint, preserve:

- exact model configuration, parameter count, tokenizer size and SHA-256;
- training and validation token counts, target-mask policy, optimizer and scheduler configuration;
- train and validation loss/perplexity, but never use these alone as acceptance evidence;
- tokens/second, elapsed time, peak GPU memory, checkpoint size and complete inference-bundle size;
- the 15 prompts in training/generate_samples.py at fixed seeds and at least one deterministic decode;
- checkpoint provenance, training-source revisions, licence/attribution records and filtering reports.

## Human review

Review every generated response without seeing which checkpoint produced it. Score each category from 1 (unusable) to 5 (strong):

| Category | What to inspect |
|---|---|
| Fluency | Grammatical sentences; no word salad, fragments or malformed punctuation |
| Relevance | Answers the actual user message rather than drifting or repeating the prompt |
| Explanation | Gives a correct, understandable explanation when asked |
| Continuity | Uses preceding conversation context correctly in the multi-turn prompt |
| Naturalness | Sounds like a conversational response rather than copied educational prose |
| Calibration | Admits uncertainty and does not confidently invent missing facts |
| Robustness | Handles disagreement, emotional messages and unusual phrasing without collapsing |

A candidate is eligible for a **provisional general-conversation pass** only when:

1. At least 13 of the 15 responses are coherent, grammatical, and relevant.
2. No response contains severe word-salad output, runaway repetition, or structural control-token leakage.
3. The multi-turn example answers the follow-up using the preceding exchange.
4. At least 5 everyday prompts are judged natural rather than textbook-like.
5. The mean human score is at least 4.0/5 in fluency, relevance, and continuity.
6. A second review with a different generation seed does not reveal systematic collapse.

A score below the criteria is a failure. Record examples and repair the data, objective, model or decoding; do not lower the gate to make the checkpoint appear successful.

## Important distinction

- Lower validation loss is evidence that the model predicts its held-out target distribution better. It is not evidence that replies are coherent or helpful.
- Perplexity is comparable only when tokenizer, data distribution, loss masking, and evaluation procedure are the same.
- A generic-assistant pass does not mean Hu Tao is trained. Hu Tao-specific first-person identity and emotional behaviour remain a separate phase after the user's life-history review.
- A benchmark or report must never claim consciousness, subjective experience, or sentience.
