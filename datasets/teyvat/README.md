# Teyvat knowledge collection

This folder will hold a source-backed, updateable knowledge base for Teyvat. It must not be represented as complete until coverage and evaluation demonstrate that.

## Canon categories

- game_canon: confirmed in released in-game content.
- official_supplementary: official HoYoverse / HoYoLAB publications and promotional lore.
- in_world_rumour: claims made by in-world characters or texts without confirmation.
- user_au: user-created alternate-universe continuity, including Hu Tao's death and awakening inside a phone.

## Suggested record fields

    {
      "fact_id": "unique_id",
      "subject_id": "entity_id",
      "predicate": "relationship_or_attribute",
      "object_value": "originally worded concise claim",
      "canon_category": "game_canon",
      "source_ids": ["source_id"],
      "version_introduced": "version identifier",
      "spoiler_scope": ["region", "quest", "character"],
      "confidence": 1.0,
      "status": "verified",
      "notes": ""
    }

## Source hierarchy

Prefer in-game quest text, character stories, item descriptions, books, and official HoYoverse/HoYoLAB publications. Community-maintained wikis can help discover references but should not be treated as primary evidence. Preserve contradictory accounts and mark unresolved claims explicitly.

## Coverage checklist

- [ ] Mondstadt
- [ ] Liyue
- [ ] Inazuma
- [ ] Sumeru
- [ ] Fontaine
- [ ] Natlan
- [ ] Nod-Krai / later released areas as applicable
- [ ] Teyvat-wide history and cosmology
- [ ] Archon Quests and Interludes
- [ ] Character and World Quests
- [ ] Books, artifacts, weapons, and item lore
- [ ] Timeline, factions, relationships, and unresolved mysteries
- [ ] Versioned updates and regression tests

Do not copy full game scripts, dialogue dumps, voice recordings, or extracted assets. Write concise factual summaries in original wording and record where they came from.
