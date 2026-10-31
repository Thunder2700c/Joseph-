"""Assemble a small relevant context from the complete local memory stores.

The complete history remains on disk. This builder selects recent and relevant
pieces for a model turn so the active prompt does not grow without bound.
"""

from __future__ import annotations

from dataclasses import dataclass

from joseph.life import LifeMemoryStore, TeyvatChronicleStore
from joseph.memory import MemoryStore
from joseph.world import WorldKnowledgeStore
from joseph.knowledge import KnowledgeStore
from joseph.life import QuestCatalogueStore


@dataclass
class RecallContextBuilder:
    conversation: MemoryStore
    world_knowledge: WorldKnowledgeStore
    life_memories: LifeMemoryStore | None = None
    chronicle: TeyvatChronicleStore | None = None
    common_knowledge: KnowledgeStore | None = None
    quest_catalogue: QuestCatalogueStore | None = None

    def build(
        self,
        user_message: str,
        recent_turns: int = 10,
        recalled_turns: int = 8,
        world_facts: int = 8,
        life_memories: int = 8,
        news_reports: int = 8,
    ) -> str:
        if not user_message.strip():
            raise ValueError("user_message cannot be empty")

        sections: list[str] = []
        history = self.conversation.conversation_context(
            user_message,
            recent_limit=recent_turns,
            recalled_limit=recalled_turns,
        )
        if history:
            lines = [
                f"{turn['role'].upper()}: {turn['content']}"
                for turn in history
            ]
            sections.append("RELEVANT CONTINUOUS CONVERSATION HISTORY\n" + "\n".join(lines))

        world = self.world_knowledge.search(user_message, limit=world_facts)
        if world:
            lines = [
                f"- {item['subject']} [{item['category']}; taught by user]: "
                f"{item['statement']}"
                for item in world
            ]
            sections.append("FACTS LEARNED ABOUT THE USER'S WORLD\n" + "\n".join(lines))

        if self.life_memories is not None:
            found = self.life_memories.search_memories(
                user_message, limit=life_memories
            )
            if found:
                lines = [
                    f"- {item.memory_text} "
                    f"(basis={item.perspective}; type={item.memory_kind}; "
                    f"canon={item.canon_category})"
                    for item in found
                ]
                sections.append("RELEVANT TEYVAT LIFE MEMORIES\n" + "\n".join(lines))

        if self.chronicle is not None:
            beats = self.chronicle.search_beats(user_message, limit=news_reports)
            public_beats = [item for item in beats if item.get("public_summary")]
            if public_beats:
                lines = [
                    f"- {item['region']} — {item['beat_title']}: "
                    f"{item['public_summary']} "
                    f"(public knowledge level={item['public_knowledge_level']}; "
                    "broad public outcome only)"
                    for item in public_beats
                ]
                sections.append("PUBLICLY KNOWN TEYVAT STORY OUTCOMES\n" + "\n".join(lines))

            reports = [
                item for item in self.chronicle.known_news(user_message, limit=news_reports)
                if item["beat"].get("public_summary")
            ]
            if reports:
                lines = [
                    f"- Hu Tao learned via {item['report']['channel']} "
                    f"({item['report']['reliability']}; {item['report']['knowledge_basis']}): "
                    f"{item['report']['report_text']}"
                    for item in reports
                ]
                sections.append("NEWS HEARD ABOUT OTHER REGIONS\n" + "\n".join(lines))

        if self.quest_catalogue is not None:
            quests = self.quest_catalogue.search(user_message, limit=8)
            public_quests = [item for item in quests if item.public_summary]
            if public_quests:
                lines = [
                    f"- {item.parent_quest_id} — {item.title}: {item.public_summary} "
                    f"(public knowledge level={item.public_knowledge_level}; "
                    "broad public outcome only)"
                    for item in public_quests
                ]
                sections.append("PUBLICLY KNOWN CHARACTER QUEST OUTCOMES\n" + "\n".join(lines))

        if self.common_knowledge is not None:
            common = self.common_knowledge.search(user_message, limit=10)
            if common:
                lines = [
                    f"- {item.term}: {item.definition} "
                    f"(status={item.status}; scope={item.scope})"
                    for item in common
                ]
                sections.append("EVERYDAY LIFE AND GENERAL WORLD KNOWLEDGE\n" + "\n".join(lines))

        return "\n\n".join(sections)
