# Teyvat story, world knowledge, quests and news

**Knowledge checkpoint: 2026-10-31.** Version 7.1 is treated as the latest released version at this checkpoint. Version 7.2 is announced for November 4, 2026; until release, its future story must not be represented as completed history.

## Current dataset contents

| Layer | Records | Meaning |
|---|---:|---|
| Archon Quest and Interlude act summaries | 50 | Act-level canon synopses from the Prologue through the current Chapter VII index |
| In-world news transmissions | 50 | One AU information report for each indexed Archon Quest/Interlude act |
| Quest catalogue | 257 | 50 Archon/Interlude, 129 World Quest titles, 55 Story Quest chapters, 4 Hu Tao Story Quest subquest titles and 19 Hangout chapters |
| Official publisher bulletins | 14 | Real-world HoYoverse news/patch announcements, not Hu Tao's in-world awareness |

The existing life-memory collection has 33 records and is still only a starter set. The common knowledge corpus is stored separately in `datasets/common_world/`.

## Three separate knowledge layers

1. **Canon story events:** what happened in released quests. A story beat does not imply Hu Tao was present.
2. **Hu Tao's in-world news:** the user-authored alternate-universe information route by which public information reaches Liyue through The Steambird, merchants, travellers, letters, books, performers and Adventurers' Guild dispatches. Each report carries a channel, reliability and heard/read basis. This mechanism is AU, not an in-game canon scene.
3. **Official publisher bulletins:** HoYoverse patch notes, trailers, banner schedules, security notices and web-event announcements. These are out-of-world records and must not be injected into Hu Tao's Teyvat memories.

The Traveler's private conversations, hidden discoveries and choices remain unknown to Hu Tao unless released canon establishes that the information became public. A public report may convey the broad outcome without revealing private details.

## Quest catalogue and granularity

- `story_beats.jsonl`: 50 act-level Archon Quest/Interlude summaries.
- `news_reports.jsonl`: 50 corresponding AU in-world public-report records.
- `quest_catalogue.jsonl`: 257 searchable title/index entries across Archon Quests, Interludes, World Quests, Story Quests, Hu Tao's Story Quest subquests and Hangouts.
- `release_news.jsonl`: 14 official publisher bulletins.

The 129 World Quest entries are title-level records. The 55 Story Quest and 19 Hangout entries are also title-level. They do not contain every plot event, branch, ending or conversation. Scene-by-scene expansion is not yet complete.

## World Quest title index checkpoint

| Region | Entries |
|---|---:|
| Mondstadt | 17 |
| Liyue | 18 |
| Inazuma | 15 |
| Sumeru | 15 |
| Fontaine | 15 |
| Natlan | 20 |
| Nod-Krai | 24 |
| Snezhnaya | 5 |
| **Total** | **129** |

These are a discovery/index layer, not proof that every World Quest is present or fully summarised. Each title should be checked against released in-game content before training.

## Remaining lore coverage before "complete"

1. Expand every Archon Quest act into chronological scene/story beats and mark what is public versus private.
2. Verify and expand all World Quest chains, subquests and prerequisite chronology; create original summaries and plausible news routes only where justified.
3. Add the remaining Story Quest subquests and Hangout branch routes/endings.
4. Index limited-time event stories, event-only appearances and interludes.
5. Cover books, letters, notes, artifacts, weapons, item descriptions, character stories and voice-over lore.
6. Expand places, settlements, cultures, factions, species, institutions, trade, technology, customs and regional histories.
7. Add post-7.1 content only after official release and review.
8. Expand Hu Tao's early life and Wangsheng history from reliable source material without inventing undocumented childhood events.

## Import

    python -m joseph.life.news_ingest --db local/teyvat_chronicle.sqlite
    python -m joseph.life.catalogue_ingest --db local/quest_catalogue.sqlite
    python -m joseph.life.release_news_ingest --db local/official_news.sqlite

These are three separate stores. Common-world knowledge is imported with `python -m joseph.knowledge.ingest`.
