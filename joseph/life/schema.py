"""Schema for ordered character memories and sourced life experiences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar

MEMORY_KINDS = {
    "autobiographical", "character_viewpoint", "quest_experience",
    "reported_knowledge", "world_knowledge", "user_au_event",
}
PERSPECTIVES = {"first_person", "third_person", "reference_only", "hidden_transition"}
CANON_CATEGORIES = {
    "game_canon", "official_supplementary", "in_world_rumour", "user_au",
}
REVIEW_STATES = {"needs_review", "verified", "user_authored"}


@dataclass(frozen=True)
class LifeSource:
    source_id: str
    title: str
    url: str
    publisher: str = ""
    source_type: str = "official"
    accessed_on: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.title.strip():
            raise ValueError("source_id and title are required")
        if not self.url.strip():
            raise ValueError("source URL or local source reference is required")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LifeMemory:
    memory_id: str
    sequence: int
    stage_id: str
    memory_kind: str
    perspective: str
    memory_text: str
    source_ids: tuple[str, ...] = ()
    canon_category: str = "game_canon"
    review_state: str = "needs_review"
    source_section: str = ""
    people: tuple[str, ...] = ()
    places: tuple[str, ...] = ()
    emotional_tags: tuple[str, ...] = ()
    spoiler_scope: tuple[str, ...] = ()
    coverage_tags: tuple[str, ...] = ()
    chronology_note: str = ""
    recalled_by_character: bool = True

    def __post_init__(self) -> None:
        if not self.memory_id.strip() or not self.stage_id.strip():
            raise ValueError("memory_id and stage_id are required")
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        if self.memory_kind not in MEMORY_KINDS:
            raise ValueError(f"invalid memory_kind: {self.memory_kind}")
        if self.perspective not in PERSPECTIVES:
            raise ValueError(f"invalid perspective: {self.perspective}")
        if self.canon_category not in CANON_CATEGORIES:
            raise ValueError(f"invalid canon_category: {self.canon_category}")
        if self.review_state not in REVIEW_STATES:
            raise ValueError(f"invalid review_state: {self.review_state}")
        if not self.memory_text.strip():
            raise ValueError("memory_text cannot be empty")
        if len(self.memory_text) > 4_000:
            raise ValueError("memory_text cannot exceed 4,000 characters")
        if self.canon_category == "user_au":
            if self.review_state == "verified":
                raise ValueError("user AU entries must not be labelled canon-verified")
        elif not self.source_ids:
            raise ValueError("canon and supplementary memories require source_ids")
        if self.perspective == "hidden_transition" and self.recalled_by_character:
            raise ValueError("hidden transition memories cannot be recalled by Hu Tao")
        if any(not tag.strip() for tag in self.coverage_tags):
            raise ValueError("coverage tags cannot be empty")

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in (
            "source_ids", "people", "places", "emotional_tags",
            "spoiler_scope", "coverage_tags",
        ):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "LifeMemory":
        values = dict(data)
        for key in (
            "source_ids", "people", "places", "emotional_tags",
            "spoiler_scope", "coverage_tags",
        ):
            values[key] = tuple(values.get(key, ()))
        return cls(**values)
