import tempfile
import unittest
from pathlib import Path

from joseph.life.catalogue_ingest import ingest
from joseph.life.catalogue_store import QuestCatalogueStore


class QuestCatalogueTests(unittest.TestCase):
    def test_catalogue_ingests_archon_world_story_and_hangout_indexes(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "catalogue.sqlite"
            count = ingest(
                str(root / "datasets/teyvat/chronicle_sources.jsonl"),
                str(root / "datasets/teyvat/quest_catalogue.jsonl"),
                str(db),
            )
            self.assertEqual(count, 257)
            with QuestCatalogueStore(db) as store:
                counts = store.counts_by_type()
                self.assertEqual(counts["archon_quest"], 45)
                self.assertEqual(counts["interlude_quest"], 5)
                self.assertEqual(counts["world_quest"], 129)
                self.assertEqual(counts["story_quest"], 55)
                self.assertEqual(counts["subquest"], 4)
                self.assertEqual(counts["hangout_event"], 19)
                public_story_summaries = [
                    item for item in store.search("", quest_type="story_quest", limit=100)
                    if item.public_summary
                ]
                self.assertEqual(len(public_story_summaries), 13)
                self.assertTrue(all(
                    item.public_knowledge_level in {"public", "regional", "reported_uncertain"}
                    for item in public_story_summaries
                ))
                hutao = store.search("Yet the Butterfly Flutters Away")
                self.assertEqual(len(hutao), 1)
                self.assertEqual(hutao[0].parent_quest_id, "Hu Tao")
                self.assertEqual(
                    len(store.search("The Joyous Floating Island", region="Nod-Krai")),
                    1,
                )
                self.assertEqual(
                    len(store.search("Shenanigans and Sweet Wine", quest_type="hangout_event")),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
