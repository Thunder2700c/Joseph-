import tempfile
import unittest
from pathlib import Path

from joseph.life import NewsTransmission, StoryBeat, TeyvatChronicleStore
from joseph.life.news_ingest import ingest


class TeyvatChronicleTests(unittest.TestCase):
    def test_hu_tao_learns_about_faraway_story_through_news_not_witnessing(self):
        with tempfile.TemporaryDirectory() as directory:
            with TeyvatChronicleStore(Path(directory) / "chronicle.sqlite") as store:
                store.add_source({
                    "source_id":"canon", "title":"Official quest update",
                    "url":"https://example.invalid/quest"
                })
                store.add_source({
                    "source_id":"au", "title":"News reaches Liyue",
                    "url":"local://character/identity.md"
                })
                store.add_story_beat(StoryBeat(
                    beat_id="fontaine_act", arc_id="fontaine_archon",
                    beat_order=1, quest_id="fontaine_quest",
                    quest_title="Fontaine Archon Quest",
                    act_title="Act I", region="Fontaine",
                    version_introduced="4.0", beat_title="A trial in Fontaine",
                    summary="An original summary of the Traveler's public involvement.",
                    source_ids=("canon",), hu_tao_participates=False,
                ))
                store.add_report(NewsTransmission(
                    report_id="fontaine_news", beat_id="fontaine_act",
                    channel="steambird", reported_by="The Steambird",
                    report_text="Hu Tao hears a report about the Fontaine proceedings.",
                    reliability="corroborated", knowledge_basis="heard",
                    source_ids=("au",), awareness_sequence=5001,
                ))
                result = store.known_news("Tell me about the Fontaine trial")
                self.assertEqual(len(result), 1)
                self.assertFalse(result[0]["beat"]["hu_tao_participates"])
                self.assertEqual(result[0]["report"]["knowledge_basis"], "heard")
                self.assertEqual(store.reports_for_beat("fontaine_act")[0]["channel"], "steambird")

    def test_reports_require_an_existing_story_beat(self):
        with tempfile.TemporaryDirectory() as directory:
            with TeyvatChronicleStore(Path(directory) / "chronicle.sqlite") as store:
                with self.assertRaises(ValueError):
                    store.add_report(NewsTransmission(
                        report_id="missing", beat_id="not_here", channel="letter",
                        reported_by="A friend", report_text="Some news.",
                        reliability="unverified",
                    ))

    def test_archon_quest_catalogue_and_reports_ingest(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "chronicle.sqlite"
            counts = ingest(
                str(root / "datasets/teyvat/chronicle_sources.jsonl"),
                str(root / "datasets/teyvat/story_beats.jsonl"),
                str(root / "datasets/teyvat/news_reports.jsonl"),
                str(db),
            )
            self.assertEqual(counts, (23, 50, 50))
            with TeyvatChronicleStore(db) as store:
                self.assertEqual(len(store.search_beats(region="Mondstadt")), 3)
                self.assertEqual(len(store.search_beats(region="Liyue")), 4)
                self.assertEqual(len(store.search_beats(region="Liyue Sea")), 1)
                self.assertEqual(len(store.search_beats(region="Inazuma")), 3)
                self.assertEqual(len(store.search_beats(region="The Chasm (Liyue)")), 2)
                self.assertEqual(len(store.search_beats(region="Sumeru")), 10)
                self.assertEqual(len(store.search_beats(region="Fontaine")), 6)
                self.assertEqual(len(store.search_beats(region="Natlan")), 7)
                self.assertEqual(len(store.search_beats(region="Nod-Krai")), 9)
                self.assertEqual(len(store.search_beats(region="Snezhnaya")), 4)
                self.assertEqual(len(store.known_news("Fontaine")), 6)
                self.assertEqual(len(store.known_news("Natlan")), 7)
                all_beats = store.search_beats(limit=100)
                self.assertEqual(len(all_beats), 50)
                self.assertTrue(all(not item["hu_tao_participates"] for item in all_beats))
                act_ten = store.search_beats("Truth Amongst the Pages of Purana")
                self.assertEqual(len(act_ten), 1)
                self.assertEqual(act_ten[0]["version_introduced"], "Luna VII")


if __name__ == "__main__":
    unittest.main()
