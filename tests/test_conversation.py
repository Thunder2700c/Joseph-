import tempfile
import unittest
from pathlib import Path

from joseph.conversation import ContinuousConversation
from joseph.memory import MemoryStore
from joseph.persona import HuTaoPersona
from joseph.world import WorldKnowledgeStore


class ContinuousConversationTests(unittest.TestCase):
    def test_single_timeline_persists_user_and_hutao_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with MemoryStore(root / "conversation.sqlite") as transcript, WorldKnowledgeStore(
                root / "world.sqlite"
            ) as world:
                conversation = ContinuousConversation(
                    transcript, world, HuTaoPersona(phase="reborn_in_device")
                )
                user_id = conversation.append_user_message("My hometown has a river.")
                prompt = conversation.prepare_context("Tell me about my hometown.")
                reply_id = conversation.append_hutao_reply("A river, hm? Tell me more!")
                self.assertEqual((user_id, reply_id), (1, 2))
                self.assertEqual(transcript.turn_count(), 2)
                self.assertIn("You are Hu Tao", prompt)
                self.assertIn("river", transcript.get_turns()[0]["content"])

    def test_teaching_world_fact_does_not_add_a_chat_turn_or_train_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with MemoryStore(root / "conversation.sqlite") as transcript, WorldKnowledgeStore(
                root / "world.sqlite"
            ) as world:
                conversation = ContinuousConversation(transcript, world, HuTaoPersona())
                fact_id = conversation.teach_world_fact(
                    "Local custom", "People greet neighbours with a specific phrase.",
                )
                fact = world.get(fact_id)
                self.assertEqual(fact["origin"], "user_taught")
                self.assertFalse(fact["include_in_training"])
                self.assertEqual(transcript.turn_count(), 0)


if __name__ == "__main__":
    unittest.main()
