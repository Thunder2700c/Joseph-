import tempfile
import unittest
from pathlib import Path

from joseph.life.release_news_ingest import ingest
from joseph.life.release_news_store import OfficialNewsStore


class OfficialNewsTests(unittest.TestCase):
    def test_current_official_bulletins_import_and_keep_out_of_world_knowledge(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "official_news.sqlite"
            count = ingest(
                str(root / "datasets/teyvat/chronicle_sources.jsonl"),
                str(root / "datasets/teyvat/release_news.jsonl"),
                str(db),
            )
            self.assertEqual(count, 14)
            with OfficialNewsStore(db) as store:
                latest = store.search("Version 7.2")
                self.assertTrue(latest)
                self.assertTrue(all(not item.is_inworld_knowledge for item in store.all_news()))
                scheduled = store.search("November 4")
                self.assertTrue(any(item.status == "announced_not_released" for item in scheduled))
                furina = store.search("Furina Event Wish")
                self.assertEqual(furina[0].source_ids, ("furina_72_event_wish",))


if __name__ == "__main__":
    unittest.main()
