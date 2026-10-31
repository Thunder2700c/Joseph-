"""Typed, source-aware facts for everyday life and cross-world assumptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar

FACT_STATUSES = {
    "established", "reasonable_assumption", "social_convention",
    "context_dependent", "fictional_canon", "in_world_rumour",
}
FACT_SCOPES = {
    "earth_human_baseline", "teyvat_assumption", "language_and_slang",
    "social_knowledge", "teyvat_canon",
}


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
    tevyat_application: str = ""
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
