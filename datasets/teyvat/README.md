# Teyvat knowledge, chronology and news

The Teyvat collection has three linked but distinct layers:

1. **Canon story beats:** original summaries of released quests, each with an act/beat order, region, version, source and spoiler scope.
2. **Hu Tao's personal memories:** only events she personally participated in or viewpoints she explicitly expresses.
3. **News transmissions:** how Hu Tao learns about events elsewhere through newspapers, merchants, letters, books, travellers and Adventurers' Guild dispatches.

An event happening in Fontaine or Natlan does not mean Hu Tao witnessed it. She may learn its public outline through a news report, which is stored separately and labelled as heard/read. The report's reliability must be recorded.

## Story granularity

Do not stop at one paragraph per region or one vague summary per Archon Quest. Expand quests into ordered scene/story beats, with one record per meaningful event or revelation. Use original summaries and retain source references. Do not reproduce entire game dialogue scripts verbatim.

## Canon categories

- game_canon: events and facts established by released game content.
- official_supplementary: official HoYoverse publications, trailers, profiles and supplementary information.
- in_world_rumour: claims circulated in Teyvat but not confirmed.
- user_au: user-created continuity, including the news route to Hu Tao and the final phone-awakening premise.

## Version checkpoint

Dataset research checkpoint: **2026-10-31**. The official Version 7.2 update notes, posted October 29, say the update arrives November 4, 2026. At the checkpoint date, 7.2 story is announced but not released. Do not populate unreleased events as if Hu Tao has heard them.

## Initial coverage

- Fontaine Archon Quest Chapter IV, Acts I–V: act-level outline and news route seeded.
- Natlan Archon Quest Chapter V, Acts I–IV: act-level outline and news route seeded.
- The current Archon Quest catalogue includes Nod-Krai and Snezhnaya regions; this initial repository entry is catalogue awareness only, not a complete story account.
- Sumeru, Inazuma, Mondstadt, Liyue, interludes, regional World Quests, books, item lore and later version content remain to be expanded.

## Import

    python -m joseph.life.news_ingest --db local/teyvat_chronicle.sqlite

Sources and story summaries are versioned in the JSONL files alongside this README.
