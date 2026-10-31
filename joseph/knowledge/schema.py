"""Schemas shared by the Teyvat lore and everyday knowledge stores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

CONTINUITIES = {"manga", "anime", "both", "supplementary", "unknown"}
STATUSES = {"verified", "unverified", "disputed", "fan_interpretation"}
FACT_STATUSES = {
    "established", "reasonable_assumption", "social_convention",
    "context_dependent", "fictional_canon", "in_world_rumour",
}
FACT_SCOPES = {
    "earth_human_baseline", "teyvat_assumption", "language_and_slang",
    "social_knowledge", "teyvat_canon",
}


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    url: str
    publisher: str = ""
    source_type: str = "official"
    language: str = "en"
    retrieved_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.title.strip() or not self.url.strip():
            raise ValueError("source_id, title, and url are required")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    name: str
    entity_type: str
    part: Optional[int] = None
    continuity: str = "manga"
    aliases: tuple[str, ...] = ()
    region: str = ""
    canon_category: str = ""

    def __post_init__(self) -> None:
        if not self.entity_id.strip() or not self.name.strip() or not self.entity_type.strip():
            raise ValueError("entity_id, name, and entity_type are required")
        if self.part is not None and not 1 <= self.part <= 9:
            raise ValueError("part must be between 1 and 9")
        if self.continuity not in CONTINUITIES:
            raise ValueError(f"invalid continuity: {self.continuity}")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["aliases"] = list(self.aliases)
        return data


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    subject_id: str
    predicate: str
    object_value: str
    source_ids: tuple[str, ...] = ()
    part: Optional[int] = None
    continuity: str = "unknown"
    status: str = "unverified"
    confidence: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (
            self.fact_id, self.subject_id, self.predicate, self.object_value
        )):
            raise ValueError("fact_id, subject_id, predicate, and object_value are required")
        if self.part is not None and not 1 <= self.part <= 9:
            raise ValueError("part must be between 1 and 9")
        if self.continuity not in CONTINUITIES:
            raise ValueError(f"invalid continuity: {self.continuity}")
        if self.status not in STATUSES:
            raise ValueError(f"invalid status: {self.status}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.status == "verified" and not self.source_ids:
            raise ValueError("verified facts must have at least one source_id")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["source_ids"] = list(self.source_ids)
        return data


@dataclass(frozen=True)
class KnowledgeSource:
    source_id: str
    title: str
    url: str
    publisher: str = ""
    source_type: str = "reference"
    accessed_on: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.title.strip() or not self.url.strip():
            raise ValueError("source_id, title and url are required")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeFact:
    fact_id: str
    domain: str
    topic: str
    term: str
    definition: str
    status: str
    scope: str
    source_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    related_terms: tuple[str, ...] = ()
    teyvat_application: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not all(x.strip() for x in (
            self.fact_id, self.domain, self.topic, self.term, self.definition
        )):
            raise ValueError("fact_id, domain, topic, term and definition are required")
        if self.status not in FACT_STATUSES:
            raise ValueError(f"invalid knowledge status: {self.status}")
        if self.scope not in FACT_SCOPES:
            raise ValueError(f"invalid knowledge scope: {self.scope}")
        if self.status == "established" and not self.source_ids:
            raise ValueError("established facts require source_ids")
        if self.scope == "teyvat_assumption" and self.status != "reasonable_assumption":
            raise ValueError("Teyvat assumptions must use reasonable_assumption status")
        if len(self.definition) > 5000:
            raise ValueError("definition cannot exceed 5,000 characters")

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("source_ids", "tags", "related_terms"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeFact":
        values = dict(data)
        for key in ("source_ids", "tags", "related_terms"):
            values[key] = tuple(values.get(key, ()))
        return cls(**values)
