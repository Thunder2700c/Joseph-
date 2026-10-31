import tempfile
import unittest
from pathlib import Path

from joseph.world import WorldKnowledgeStore


class WorldKnowledgeTests(unittest.TestCase):
    def test_user_taught_fact_persists_and_is_searchable(self):
        with tempfile.TemporaryDirectory() as directory:
            with WorldKnowledgeStore(Path(directory) / "world.sqlite") as store:
                fact_id = store.teach(
                    "Ludhiana",
                    "The user explains that Ludhiana is a city in Punjab, India.",
                    category="place",
                    tags=("home", "Punjab"),
                )
                fact = store.get(fact_id)
                self.assertEqual(fact["origin"], "user_taught")
                self.assertFalse(fact["include_in_training"])
                matches = store.search("Ludhiana")
                self.assertEqual(matches[0]["fact_id"], fact_id)
                self.assertIn("Ludhiana", store.context("Ludhiana"))

    def test_training_requires_explicit_approval_and_exports_no_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            with WorldKnowledgeStore(Path(directory) / "world.sqlite") as store:
                first = store.teach("Custom word", "A local slang word.")
                second = store.teach(
                    "Custom place", "A made-up place name.",
                    include_in_training=True,
                )
                self.assertEqual([f["fact_id"] for f in store.for_training()], [second])
                self.assertTrue(store.set_training_approval(first, True))
                self.assertEqual(len(store.for_training()), 2)


if __name__ == "__main__":
    unittest.main()
