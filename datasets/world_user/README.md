# User-taught knowledge about our world

This is a separate durable store for things the user teaches Hu Tao about the real world outside Teyvat: local places, geography, technology, culture, slang, everyday routines, people the user introduces, and facts about how the user's world works.

A new fact is saved immediately to `local/world_knowledge.sqlite`. The original teaching message remains in the continuous conversation transcript too.

Every record is attributed as `user_taught`. Hu Tao may use it naturally in conversation, but the database must not claim that the fact is independently verified by an official source.

Use `WorldKnowledgeStore.teach(subject, statement, category, tags, importance)` to add knowledge. Search it using `WorldKnowledgeStore.search(query)`.

Records default to `include_in_training=false`. The user can explicitly approve stable, appropriate facts for future training. Export approved facts only:

    python -m training.export_world_knowledge --db local/world_knowledge.sqlite

The exporter does not read or export the chat transcript.
