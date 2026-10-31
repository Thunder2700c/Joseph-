"""Sourced, continuity-aware fictional-universe knowledge store."""

from .schema import EntityRecord, FactRecord, SourceRecord
from .store import KnowledgeStore

__all__ = ["EntityRecord", "FactRecord", "SourceRecord", "KnowledgeStore"]
