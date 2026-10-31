# Ordered Hu Tao life memories

Each JSONL row is one atomic memory or knowledge record. Keep entries short enough to retrieve independently, but detailed enough to preserve the event's meaning.

## Required distinctions

- A personal memory says what Hu Tao herself experienced.
- A character viewpoint records her own stated attitude or relationship.
- A quest memory is allowed only when she appears or participates in the event.
- Reported knowledge says what she learned from someone else.
- World knowledge is reference material about Teyvat, not her autobiography.
- User AU entries belong only to the custom ending and awakening.

## Record quality

Every canon record must carry source IDs and a source section. Use original paraphrases rather than reproducing dialogue. Preserve version/date, spoilers, conflicting accounts and uncertainty. Do not invent missing childhood or family events to make the timeline appear complete.

Every record should include `coverage_tags`. A stage cannot be marked complete until all its required coverage tags are present in reviewed memory records. The stage plan's tags are the minimum topic checklist, not a claim that a few short memories exhaust a subject.

The `sequence` field reserves chronological slots by life stage: 1000-series for origins, 2000-series for Wangsheng learning, 3000-series for her directorship, 4000-series for relationships/daily life, 5000-series for personal quests, and 6000-series onward for world knowledge. New records can be inserted into unused sequence slots.

The lifecycle manager requires all stage dependencies, source checks, review states and coverage tags to pass before it permits the awakening.
 
## Personal identity and regional awareness

`character/identity_profile.json` contains Hu Tao's personal profile: official attributes, birthday, family details, appearance, work, and clearly labelled user-AU birth year and physical measurements. The profile currently uses 15 July 2006, approximately 156 cm, and approximately 48 kg. HoYoverse publishes July 15 but does not publish a birth year, height or weight; the year and measurements are user-AU settings.

`character/teyvat_social_world.json` contains her relationship map, local institutions, familiar places, seven-nation overview, Archon knowledge, and a 2026-10-31 leadership checkpoint. Distinguish public knowledge, her own acquaintances, secondary-source details and unresolved secrets.

The life-memory store also receives these facts in individual retrievable records. Their insertion does not mark any life stage complete; the awakening remains locked until all missing childhood, Wangsheng training and broader lore topics are researched and reviewed.

The current seed contains 84 ordered memories. It includes source-linked character-story and event memories, the official birthday, the user-AU birth year and body reference, family facts, friendship boundaries, Liyue institutions, and public knowledge of Teyvat's nations and civic offices. These are still starter resources; they do not complete all childhood, Wangsheng training, world quests, histories or versioned appearances.

The collection contains 84 ordered records. These include the personal profile, body settings labelled as AU, family and relationship details, a daily-life guide, and public profiles of notable people and institutions across Teyvat. The seed remains incomplete; childhood, early Wangsheng training and several story-history stages still need research.
