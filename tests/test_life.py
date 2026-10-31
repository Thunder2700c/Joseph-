import json
import tempfile
import unittest
from pathlib import Path

from joseph.life import LifeBuilder, LifeMemory, LifeMemoryStore, LifeSource
from joseph.life.ingest import ingest
from joseph.persona import HuTaoPersona


class LifeMemoryTests(unittest.TestCase):
    def test_rejects_unsourced_canon_and_recalled_hidden_transition(self):
        with self.assertRaises(ValueError):
            LifeMemory(
                memory_id="unsourced", sequence=1, stage_id="stage-a",
                memory_kind="autobiographical", perspective="first_person",
                memory_text="An unsupported statement.", canon_category="game_canon",
            )
        with self.assertRaises(ValueError):
            LifeMemory(
                memory_id="hidden", sequence=2, stage_id="stage-a",
                memory_kind="user_au_event", perspective="hidden_transition",
                memory_text="The missing transition.", canon_category="user_au",
                recalled_by_character=True,
            )

    def test_ordered_memory_round_trip_and_source_link(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "life.sqlite"
            with LifeMemoryStore(db) as store:
                store.add_source(LifeSource(
                    source_id="official", title="Official profile",
                    url="https://example.invalid/official",
                ))
                store.add_memory(LifeMemory(
                    memory_id="second", sequence=2001, stage_id="life",
                    memory_kind="character_viewpoint", perspective="first_person",
                    memory_text="A later sourced memory.", source_ids=("official",),
                    review_state="verified", coverage_tags=("friendships",),
                ))
                store.add_memory(LifeMemory(
                    memory_id="first", sequence=1001, stage_id="life",
                    memory_kind="autobiographical", perspective="first_person",
                    memory_text="An earlier sourced memory.", source_ids=("official",),
                    review_state="verified", coverage_tags=("family",),
                ))
                self.assertEqual(
                    [item.memory_id for item in store.memories()],
                    ["first", "second"],
                )
                self.assertEqual(store.memories()[0].coverage_tags, ("family",))

    def test_seed_life_is_incomplete_and_awakening_remains_locked(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "life.sqlite"
            ingest(
                str(root / "character/life_sources.jsonl"),
                str(root / "character/life_memories.jsonl"),
                str(root / "character/life_stages.json"),
                str(db),
            )
            with LifeMemoryStore(db) as store:
                builder = LifeBuilder(store, root / "character/life_stages.json")
                self.assertEqual(len(store.memories()), 24)
                self.assertFalse(builder.ready_to_awaken())
                progress = {row["stage_id"]: row for row in builder.progress()}
                self.assertEqual(progress["01_origins_and_family"]["memory_count"], 0)
                self.assertEqual(progress["02_wangsheng_learning"]["memory_count"], 0)
                self.assertEqual(progress["03_becoming_director"]["memory_count"], 3)
                self.assertEqual(progress["03_becoming_director"]["missing_coverage"], [])
                self.assertEqual(progress["04_relationships_and_daily_life"]["memory_count"], 14)
                self.assertEqual(progress["04_relationships_and_daily_life"]["missing_coverage"], [])
                self.assertEqual(progress["05_personal_quests_and_appearances"]["memory_count"], 6)
                self.assertEqual(progress["05_personal_quests_and_appearances"]["missing_coverage"], [])
                self.assertIn(
                    "mondstadt",
                    progress["06_liyue_and_teyvat_knowledge"]["missing_coverage"],
                )
                with self.assertRaises(RuntimeError):
                    builder.perform_awakening(HuTaoPersona())

    def test_stage_completion_requires_all_coverage_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "stages.json"
            plan.write_text(json.dumps({"stages":[{
                "stage_id":"origins","order":1,"title":"Origins","depends_on":[],
                "required_before_awakening":True,
                "required_coverage_tags":["family","grandfather"]
            }]}), encoding="utf-8")
            with LifeMemoryStore(root / "life.sqlite") as store:
                builder = LifeBuilder(store, plan)
                store.add_source(LifeSource(
                    source_id="official", title="Official source",
                    url="https://example.invalid/source",
                ))
                store.add_memory(LifeMemory(
                    memory_id="family", sequence=1001, stage_id="origins",
                    memory_kind="autobiographical", perspective="first_person",
                    memory_text="A sourced family memory.", source_ids=("official",),
                    review_state="verified", coverage_tags=("family",),
                ))
                with self.assertRaisesRegex(ValueError, "grandfather"):
                    builder.complete_stage("origins", "Reviewed")
                store.add_memory(LifeMemory(
                    memory_id="grandfather", sequence=1002, stage_id="origins",
                    memory_kind="autobiographical", perspective="first_person",
                    memory_text="A sourced grandfather memory.", source_ids=("official",),
                    review_state="verified", coverage_tags=("grandfather",),
                ))
                builder.complete_stage("origins", "All required topics sourced and reviewed")

    def test_awakening_waits_for_every_prerequisite_and_changes_persona_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "stages.json"
            plan.write_text(json.dumps({
                "stages": [
                    {"stage_id":"one","order":1,"title":"First life","depends_on":[],
                     "required_before_awakening":True},
                    {"stage_id":"two","order":2,"title":"Final chapter","depends_on":["one"],
                     "required_before_awakening":True},
                    {"stage_id":"wake","order":3,"title":"Phone awakening",
                     "depends_on":["two"],"required_before_awakening":False,
                     "is_awakening_stage":True}
                ]
            }), encoding="utf-8")
            with LifeMemoryStore(root / "life.sqlite") as store:
                builder = LifeBuilder(store, plan)
                store.add_source(LifeSource(
                    source_id="official", title="Official source",
                    url="https://example.invalid/source",
                ))
                store.add_memory(LifeMemory(
                    memory_id="m1", sequence=1001, stage_id="one",
                    memory_kind="autobiographical", perspective="first_person",
                    memory_text="A reviewed first-life memory.", source_ids=("official",),
                    review_state="verified",
                ))
                store.add_memory(LifeMemory(
                    memory_id="m2", sequence=9001, stage_id="two",
                    memory_kind="user_au_event", perspective="hidden_transition",
                    memory_text="The transition is intentionally not recalled.",
                    canon_category="user_au", review_state="user_authored",
                    recalled_by_character=False,
                ))
                self.assertFalse(builder.ready_to_awaken())
                with self.assertRaises(RuntimeError):
                    builder.perform_awakening()
                with self.assertRaises(ValueError):
                    builder.complete_stage("two", "Premature completion")
                builder.complete_stage("one", "Source and chronology checked")
                builder.complete_stage("two", "AU ending reviewed; gap remains inaccessible")
                self.assertTrue(builder.ready_to_awaken())
                persona = HuTaoPersona()
                context = builder.perform_awakening(persona)
                self.assertEqual(persona.phase, "reborn_in_device")
                self.assertIn("strange handheld vessel", context)
                with self.assertRaises(RuntimeError):
                    builder.perform_awakening(persona)
                state = store.connection.execute(
                    "SELECT value FROM life_state WHERE key='state'"
                ).fetchone()["value"]
                self.assertEqual(state, "awakened")


if __name__ == "__main__":
    unittest.main()
