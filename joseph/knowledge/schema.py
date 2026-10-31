"""Validated records for source-aware fictional-universe knowledge."""

from dataclasses import asdict, dataclass
from typing import Optional

CONTINUITIES = {"manga", "anime", "both", "supplementary", "unknown"}
STATUSES = {"verified", "unverified", "disputed", "fan_interpretation"}


@dataclass
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


@dataclass
class EntityRecord:
    entity_id: str
    name: str
    entity_type: str
    part: Optional[int] = None
    continuity: str = "manga"
    aliases: tuple[str, ...] = ()

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


@dataclass
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
