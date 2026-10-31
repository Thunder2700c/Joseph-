import tempfile
import unittest
from pathlib import Path

from joseph.knowledge import KnowledgeFact, KnowledgeStore, KnowledgeSource
from joseph.knowledge.ingest import ingest


class CommonKnowledgeTests(unittest.TestCase):
    def test_everyday_terms_and_bodily_functions_are_searchable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with KnowledgeStore(root / "knowledge.sqlite") as store:
                store.add_source(KnowledgeSource(
                    source_id="digestive", title="Digestive system",
                    url="https://example.invalid/digestive",
                ))
                store.add_fact(KnowledgeFact(
                    fact_id="stool", domain="human_body", topic="digestion",
                    term="stool", definition="Waste passed from the bowel.",
                    status="established", scope="earth_human_baseline",
                    source_ids=("digestive",), related_terms=("poop", "faeces"),
                ))
                self.assertEqual(store.search("poop")[0].term, "stool")

    def test_teyvat_assumptions_are_not_returned_as_canon(self):
        with tempfile.TemporaryDirectory() as directory:
            with KnowledgeStore(Path(directory) / "knowledge.sqlite") as store:
                store.add_fact(KnowledgeFact(
                    fact_id="teyvat_toilet", domain="teyvat_assumptions",
                    topic="daily life", term="sanitation",
                    definition="Assume practical sanitation exists for ordinary humans.",
                    status="reasonable_assumption", scope="teyvat_assumption",
                ))
                assumptions = store.search("sanitation", scope="teyvat_assumption")
                self.assertEqual(len(assumptions), 1)
                self.assertEqual(assumptions[0].status, "reasonable_assumption")

    def test_full_dataset_ingests_and_includes_adult_and_social_vocabulary(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "common.sqlite"
            count = ingest(
                str(root / "datasets/common_world/sources.jsonl"),
                str(root / "datasets/common_world/facts.jsonl"),
                str(db),
            )
            self.assertEqual(count, 262)
            with KnowledgeStore(db) as store:
                self.assertTrue(store.search("urine"))
                self.assertTrue(store.search("digestion"))
                self.assertTrue(store.search("gender"))
                self.assertTrue(store.search("romantic love"))
                self.assertTrue(store.search("porn"))
                self.assertTrue(store.search("fuck"))
                self.assertTrue(store.search("Liyue"))
                self.assertTrue(store.search("Teyvat sanitation", scope="teyvat_assumption"))
                self.assertTrue(store.search("heart lungs brain", scope="earth_human_baseline"))
                self.assertTrue(store.search("puberty menstrual cycle", scope="earth_human_baseline"))
                self.assertTrue(store.search("consensual relationship", scope="social_knowledge"))


if __name__ == "__main__":
    unittest.main()
