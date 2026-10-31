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
                result = store.known_news("Fontaine")
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

    def test_official_seed_is_act_indexed_and_hu_tao_only_hears_reports(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "chronicle.sqlite"
            counts = ingest(
                str(root / "datasets/teyvat/chronicle_sources.jsonl"),
                str(root / "datasets/teyvat/story_beats.jsonl"),
                str(root / "datasets/teyvat/news_reports.jsonl"),
                str(db),
            )
            self.assertEqual(counts, (8, 10, 9))
            with TeyvatChronicleStore(db) as store:
                beats = store.search_beats(region="Fontaine")
                natlan = store.search_beats(region="Natlan")
                self.assertEqual(len(beats), 5)
                self.assertEqual(len(natlan), 4)
                self.assertTrue(all(not item["hu_tao_participates"] for item in beats + natlan))
                self.assertEqual(len(store.known_news("Fontaine")), 5)
                self.assertEqual(len(store.known_news("Natlan")), 4)


if __name__ == "__main__":
    unittest.main()
