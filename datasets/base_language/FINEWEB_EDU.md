# FineWeb-Edu base-language source

## Source and exact revision

- Dataset: `HuggingFaceFW/fineweb-edu`
- Configuration: `sample-10BT` (the publisher describes it as a random sample of approximately 10 billion GPT-2 tokens)
- Pinned dataset repository revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- Dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
- Dataset-card licence: Open Data Commons Attribution (ODC-By 1.0)
- Content: English educational web text filtered from Common Crawl, with per-row language confidence, educational score, and source token-count metadata.

The publisher's card documents the `sample-10BT` configuration as an approximately 10-billion-GPT-2-token sample. The project does **not** ingest all of that subset. The current acquisition recipe selects a deterministic fraction of its source shards, filters rows, and caps each shard at 100 million characters; 20 shards therefore have a maximum of 2 billion characters. The final exact tokenizer token count is measured after fitting the project tokenizer. Characters divided by four is only a planning estimate.

## Filtering and split policy

Only the `text` field becomes model input. Metadata fields are used solely for filtering and provenance:

- `language == "en"`
- `language_score >= 0.90`
- `int_score >= 4`
- source `token_count` between 100 and 30,000
- normalized text length between 200 and 100,000 characters
- duplicate suppression by SHA-256 of case-folded normalized text
- deterministic train/validation split by the same content hash (5% validation by default)
- basic exclusion of documents containing obvious email, US Social Security number, or 10-or-more-digit patterns

The PII pattern filter is only a lightweight precaution. It is **not** a comprehensive PII detector and does not guarantee that all personal information is removed. The dataset itself is public web text, not private user conversation history.

The 20 disjoint source shards are partitioned before per-shard shuffling. The merge step checks source identity, selected/filter fields, source revision, shard coverage, per-file SHA-256 values, split ownership, and global duplicate removal before writing the merged manifest. The manifest records the precise row filter thresholds and rejection counts.

## Licensing and attribution limitation

The FineWeb-Edu dataset card declares ODC-By 1.0. The card also states that the underlying material is filtered from Common Crawl and that Common Crawl terms of use apply. This project records the dataset-level licence and preserves this caveat; it does **not** claim to have independently cleared every underlying webpage's copyright, licence, or reuse restrictions.

Suggested dataset attribution:

> Hugging Face FineData, FineWeb-Edu, configuration sample-10BT, https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Review the current dataset card and applicable Common Crawl terms before redistributing derived data or model weights. The training corpus is not included in the final inference bundle.

## Role in Hu Tao's project

FineWeb-Edu is used only for general language pretraining. It is not a source of Hu Tao's memories, biography, personality, or Teyvat facts. Generic conversation supervision is prepared separately from reviewed OASST1 and Databricks Dolly 15K examples. Hu Tao-specific identity training remains blocked until the user manually reviews former-life stages 1–9.
