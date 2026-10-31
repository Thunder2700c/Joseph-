"""Schema for real-world HoYoverse bulletins, separate from in-world Teyvat news."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OfficialBulletin:
    news_id: str
    posted_at: str
    headline: str
    category: str
    status: str
    summary: str
    source_ids: tuple[str, ...]
    is_inworld_knowledge: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not all(x.strip() for x in (
            self.news_id, self.posted_at, self.headline, self.category,
            self.status, self.summary,
        )):
            raise ValueError("official bulletin fields are required")
        if not self.source_ids:
            raise ValueError("official bulletins require a source")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["source_ids"] = list(self.source_ids)
        return data

    @classmethod
    def from_dict(cls, row: dict) -> "OfficialBulletin":
        item = dict(row)
        item["source_ids"] = tuple(item.get("source_ids", ()))
        return cls(**item)
