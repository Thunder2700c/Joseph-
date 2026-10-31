import tempfile
import unittest
from pathlib import Path

from joseph.memory import MemoryStore


class PersistentConversationTests(unittest.TestCase):
    def test_all_turns_persist_in_one_ordered_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore(Path(directory) / "memory.sqlite") as store:
                first = store.add_turn("user", "My hometown has a clock tower.")
                store.add_turn("assistant", "I'll remember that detail.")
                store.add_turn("user", "I also like old poetry.")
                self.assertEqual(first, 1)
                self.assertEqual(store.turn_count(), 3)
                self.assertEqual(
                    [turn["role"] for turn in store.get_turns()],
                    ["user", "assistant", "user"],
                )
                self.assertEqual(
                    [turn["turn_id"] for turn in store.get_turns(after_turn_id=1)],
                    [2, 3],
                )

    def test_old_turns_are_searchable_and_recalled_with_recent_context(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore(Path(directory) / "memory.sqlite") as store:
                store.add_turn("user", "My hometown has an old clock tower.")
                store.add_turn("assistant", "That sounds like a lovely landmark.")
                for index in range(15):
                    store.add_turn("user", f"Unrelated everyday message {index}.")
                matches = store.search_turns("clock tower")
                self.assertEqual(len(matches), 1)
                context = store.conversation_context("clock tower", recent_limit=2, recalled_limit=4)
                self.assertTrue(any("clock tower" in turn["content"] for turn in context))
                self.assertEqual(context, sorted(context, key=lambda x: x["turn_id"]))

    def test_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore(Path(directory) / "memory.sqlite") as store:
                store.add_turn("user", "Remember this place.", {"source":"user_input"})
                self.assertEqual(
                    store.recent_turns()[0]["metadata"]["source"],
                    "user_input",
                )


if __name__ == "__main__":
    unittest.main()
