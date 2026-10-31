import tempfile
import unittest
from pathlib import Path

from joseph.memory import MemoryStore
from joseph.world import WorldKnowledgeStore
from joseph.recall import RecallContextBuilder


class RecallContextTests(unittest.TestCase):
    def test_context_uses_recent_history_and_user_taught_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore(Path(directory) / "memory.sqlite") as chat, WorldKnowledgeStore(
                Path(directory) / "world.sqlite"
            ) as world:
                chat.add_turn("user", "I live in a place with a famous clock tower.")
                chat.add_turn("assistant", "Tell me more about it.")
                world.teach("Hometown", "Your city is known for a clock tower.", category="place")
                builder = RecallContextBuilder(chat, world)
                context = builder.build("Tell me about the clock tower.")
                self.assertIn("RELEVANT CONTINUOUS CONVERSATION HISTORY", context)
                self.assertIn("FACTS LEARNED ABOUT THE USER'S WORLD", context)
                self.assertIn("known for a clock tower", context)


if __name__ == "__main__":
    unittest.main()
