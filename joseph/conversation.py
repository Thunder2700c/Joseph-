"""One continuous, locally persisted conversation with Hu Tao."""

from __future__ import annotations

from typing import Any

from joseph.knowledge import KnowledgeStore
from joseph.life import (
    LifeMemoryStore,
    QuestCatalogueStore,
    TeyvatChronicleStore,
)
from joseph.memory import MemoryStore
from joseph.persona import HuTaoPersona
from joseph.recall import RecallContextBuilder
from joseph.world import WorldKnowledgeStore


class ContinuousConversation:
    """A single timeline; no session IDs or separate chat histories."""

    def __init__(
        self,
        transcript: MemoryStore,
        world_knowledge: WorldKnowledgeStore,
        persona: HuTaoPersona,
        common_knowledge: KnowledgeStore | None = None,
        life_memories: LifeMemoryStore | None = None,
        chronicle: TeyvatChronicleStore | None = None,
        quest_catalogue: QuestCatalogueStore | None = None,
    ):
        self.transcript = transcript
        self.world_knowledge = world_knowledge
        self.persona = persona
        self.recall = RecallContextBuilder(
            conversation=transcript,
            world_knowledge=world_knowledge,
            common_knowledge=common_knowledge,
            life_memories=life_memories,
            chronicle=chronicle,
            quest_catalogue=quest_catalogue,
        )

    def append_user_message(
        self, text: str, metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persist the user's message before starting inference."""
        return self.transcript.add_turn("user", text, metadata=metadata)

    def prepare_context(self, latest_user_message: str) -> str:
        """Build persona plus relevant recalled context for model inference."""
        persona = self.persona.render_context()
        recall = self.recall.build(latest_user_message)
        if recall:
            return persona + "\n\n" + recall
        return persona

    def append_hutao_reply(
        self, text: str, metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persist Hu Tao's reply into the same continuous timeline."""
        return self.transcript.add_turn("assistant", text, metadata=metadata)

    def teach_world_fact(
        self,
        subject: str,
        statement: str,
        category: str = "general",
        tags: tuple[str, ...] = (),
        include_in_training: bool = False,
    ) -> str:
        """Save a user-taught fact without silently changing model weights."""
        return self.world_knowledge.teach(
            subject=subject,
            statement=statement,
            category=category,
            tags=tags,
            include_in_training=include_in_training,
        )
