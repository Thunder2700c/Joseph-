"""Versioned index entries for all quest families, separate from story summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass

QUEST_TYPES = {
    "archon_quest", "interlude_quest", "story_quest", "hangout_event",
    "world_quest", "tribal_chronicle", "event_quest", "subquest",
    "commission_chain", "other",
}
INDEX_STATES = {"released", "announced", "catalogued", "needs_verification"}


@dataclass(frozen=True)
class QuestCatalogueEntry:
    quest_id: str
    title: str
    region: str
    quest_type: str
    source_ids: tuple[str, ...]
    version_introduced: str = ""
    parent_quest_id: str = ""
    subregion: str = ""
    index_state: str = "catalogued"
    summary_state: str = "not_summarised"
    chronology_note: str = ""
    notes: str = ""
    public_summary: str = ""
    public_knowledge_level: str = "not_publicly_known"
    public_knowledge_note: str = ""
    knowledge_basis: str = "unknown"

    def __post_init__(self) -> None:
        if not self.quest_id.strip() or not self.title.strip() or not self.region.strip():
            raise ValueError("quest_id, title and region are required")
        if self.quest_type not in QUEST_TYPES:
            raise ValueError(f"invalid quest_type: {self.quest_type}")
        if self.index_state not in INDEX_STATES:
            raise ValueError(f"invalid index_state: {self.index_state}")
        if self.public_knowledge_level not in {
            "public", "regional", "reported_uncertain", "not_publicly_known"
        }:
            raise ValueError("invalid public knowledge level")
        if self.public_knowledge_level == "not_publicly_known" and self.public_summary:
            raise ValueError("non-public quest entries must not have a public summary")
        if self.public_summary and not self.public_knowledge_note:
            raise ValueError("public quest summaries require a knowledge-boundary note")
        if self.knowledge_basis not in {
            "unknown", "heard_or_read", "publicly_observed", "personal_memory"
        }:
            raise ValueError("invalid quest knowledge basis")
        if not self.source_ids:
            raise ValueError("quest index entries require at least one source")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["source_ids"] = list(self.source_ids)
        return result

    @classmethod
    def from_dict(cls, row: dict) -> "QuestCatalogueEntry":
        item = dict(row)
        item["source_ids"] = tuple(item.get("source_ids", ()))
        return cls(**item)
