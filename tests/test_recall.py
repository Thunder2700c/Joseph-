import tempfile
import unittest
from pathlib import Path

from joseph.knowledge import KnowledgeFact, KnowledgeStore, KnowledgeSource
from joseph.life import QuestCatalogueEntry, QuestCatalogueStore
from joseph.memory import MemoryStore
from joseph.recall import RecallContextBuilder
from joseph.world import WorldKnowledgeStore


class RecallContextTests(unittest.TestCase):
    def test_context_uses_recent_history_and_user_taught_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with MemoryStore(root / "conversation.sqlite") as chat, WorldKnowledgeStore(
                root / "world.sqlite"
            ) as world:
                chat.add_turn("user", "I live in a place with a famous clock tower.")
                chat.add_turn("assistant", "Tell me more about it.")
                world.teach("Hometown", "Your city is known for a clock tower.", category="place")
                builder = RecallContextBuilder(chat, world)
                context = builder.build("Tell me about the clock tower.")
                self.assertIn("RELEVANT CONTINUOUS CONVERSATION HISTORY", context)
                self.assertIn("FACTS LEARNED ABOUT THE USER'S WORLD", context)
                self.assertIn("known for a clock tower", context)

    def test_context_retrieves_earth_knowledge_and_quest_titles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with MemoryStore(root / "conversation.sqlite") as chat, WorldKnowledgeStore(
                root / "world.sqlite"
            ) as world, KnowledgeStore(root / "common.sqlite") as common, QuestCatalogueStore(
                root / "quests.sqlite"
            ) as quests:
                common.add_source(KnowledgeSource(
                    source_id="source", title="Everyday knowledge",
                    url="https://example.invalid/everyday",
                ))
                common.add_fact(KnowledgeFact(
                    fact_id="digestion", domain="human_body", topic="digestion",
                    term="digestion", definition="Food is broken down and nutrients absorbed.",
                    status="established", scope="earth_human_baseline",
                    source_ids=("source",),
                ))
                quests.add_source({
                    "source_id":"quest_source","title":"Quest index",
                    "url":"https://example.invalid/quests"
                })
                quests.add_quest(QuestCatalogueEntry(
                    quest_id="hu_tao_story", title="Yet the Butterfly Flutters Away",
                    region="Liyue", quest_type="story_quest",
                    source_ids=("quest_source",), parent_quest_id="Hu Tao",
                ))
                builder = RecallContextBuilder(
                    chat, world, common_knowledge=common, quest_catalogue=quests
                )
                context = builder.build("Explain digestion and Hu Tao's quest")
                self.assertIn("EVERYDAY LIFE AND GENERAL WORLD KNOWLEDGE", context)
                self.assertIn("QUEST CATALOGUE MATCHES", context)
                self.assertIn("Yet the Butterfly Flutters Away", context)



    def test_recall_exposes_public_outcome_but_never_private_canon_summary(self):
        from joseph.life import StoryBeat, TeyvatChronicleStore

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with MemoryStore(root / "conversation.sqlite") as chat, WorldKnowledgeStore(
                root / "world.sqlite"
            ) as world, TeyvatChronicleStore(root / "chronicle.sqlite") as chronicle:
                chronicle.add_source({
                    "source_id": "canon",
                    "title": "Quest reference",
                    "url": "https://example.invalid/quest",
                })
                chronicle.add_story_beat(StoryBeat(
                    beat_id="hidden_test",
                    arc_id="test_arc",
                    beat_order=1,
                    quest_id="test_quest",
                    quest_title="A hidden chapter",
                    act_title="A hidden chapter",
                    region="Fontaine",
                    version_introduced="4.7",
                    beat_title="A hidden chapter",
                    summary="PRIVATE_SECRET_DETAILS_DO_NOT_SURFACE",
                    source_ids=("canon",),
                    public_summary="Fontaine's public crisis was resolved and the nation survived.",
                    public_knowledge_level="public",
                    public_knowledge_note="Broad public outcome only.",
                ))
                builder = RecallContextBuilder(chat, world, chronicle=chronicle)
                context = builder.build("Tell me what happened in Fontaine")
                self.assertIn("Fontaine's public crisis was resolved", context)
                self.assertNotIn("PRIVATE_SECRET_DETAILS_DO_NOT_SURFACE", context)


if __name__ == "__main__":
    unittest.main()
