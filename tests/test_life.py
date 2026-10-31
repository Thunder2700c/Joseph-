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

    def test_rejects_user_authored_state_for_canon_memory(self):
        with self.assertRaisesRegex(ValueError, "cannot be user-authored"):
            LifeMemory(
                memory_id="mislabelled-canon", sequence=3, stage_id="stage-a",
                memory_kind="autobiographical", perspective="first_person",
                memory_text="A sourced canon memory with the wrong review state.",
                source_ids=("official",), canon_category="game_canon",
                review_state="user_authored",
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
                self.assertEqual(len(store.memories()), 95)
                self.assertFalse(builder.ready_to_awaken())
                progress = {row["stage_id"]: row for row in builder.progress()}
                self.assertEqual(progress["01_origins_and_family"]["memory_count"], 5)
                self.assertEqual(progress["01_origins_and_family"]["missing_coverage"], [])
                self.assertEqual(progress["02_wangsheng_learning"]["memory_count"], 1)
                self.assertEqual(
                    progress["02_wangsheng_learning"]["missing_coverage"],
                    ["early_work", "living_at_parlor", "parlor_history"],
                )
                self.assertEqual(progress["03_becoming_director"]["memory_count"], 14)
                self.assertEqual(progress["03_becoming_director"]["missing_coverage"], [])
                self.assertEqual(progress["04_relationships_and_daily_life"]["memory_count"], 32)
                self.assertEqual(progress["04_relationships_and_daily_life"]["missing_coverage"], [])
                self.assertEqual(progress["05_personal_quests_and_appearances"]["memory_count"], 9)
                self.assertEqual(progress["06_liyue_and_teyvat_knowledge"]["memory_count"], 23)
                self.assertEqual(progress["06_liyue_and_teyvat_knowledge"]["missing_coverage"], [])
                self.assertEqual(progress["07_world_history_and_archon_events"]["memory_count"], 11)
                self.assertEqual(
                    progress["07_world_history_and_archon_events"]["missing_coverage"],
                    [],
                )
                with self.assertRaises(RuntimeError):
                    builder.perform_awakening(HuTaoPersona())


    def test_personal_profile_and_social_world_are_structured(self):
        root = Path(__file__).resolve().parents[1]
        profile = json.loads((root / "character/identity_profile.json").read_text(encoding="utf-8"))
        world = json.loads((root / "character/teyvat_social_world.json").read_text(encoding="utf-8"))
        self.assertEqual(profile["identity"]["date_of_birth_user_au"], "2006-07-15")
        self.assertEqual(profile["official_profile"]["birthday"], "July 15")
        self.assertEqual(profile["physical_profile"]["weight_kg"], 48)
        self.assertEqual(profile["physical_profile"]["height_cm"], 156)
        self.assertIsNone(profile["official_profile"]["height_cm"])
        self.assertIsNone(profile["official_profile"]["weight_kg"])
        self.assertEqual(len(world["nations"]), 8)
        self.assertEqual(len(world["notable_people_by_region"]), 8)
        self.assertIn("political_literacy", world)
        self.assertTrue(any(person["name"] == "Xingqiu" for person in profile["relationships"]))

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

    def test_ready_to_awaken_rechecks_coverage_after_plan_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "stages.json"
            initial_plan = {
                "stages": [
                    {"stage_id": "first", "order": 1, "title": "Origins",
                     "depends_on": [], "required_before_awakening": True,
                     "required_coverage_tags": ["family"]},
                    {"stage_id": "wake", "order": 2, "title": "Awakening",
                     "depends_on": ["first"], "required_before_awakening": False,
                     "is_awakening_stage": True}
                ]
            }
            plan.write_text(json.dumps(initial_plan), encoding="utf-8")
            with LifeMemoryStore(root / "life.sqlite") as store:
                builder = LifeBuilder(store, plan)
                store.add_source(LifeSource(
                    source_id="official", title="Official source",
                    url="https://example.invalid/source",
                ))
                store.add_memory(LifeMemory(
                    memory_id="family-memory", sequence=1001, stage_id="first",
                    memory_kind="autobiographical", perspective="first_person",
                    memory_text="A sourced family memory.", source_ids=("official",),
                    review_state="verified", coverage_tags=("family",),
                ))
                builder.complete_stage("first", "Initial requirements reviewed")
                self.assertTrue(builder.ready_to_awaken())

                changed_plan = {
                    "stages": [
                        {"stage_id": "first", "order": 1, "title": "Origins",
                         "depends_on": [], "required_before_awakening": True,
                         "required_coverage_tags": ["family", "childhood"]},
                        {"stage_id": "wake", "order": 2, "title": "Awakening",
                         "depends_on": ["first"], "required_before_awakening": False,
                         "is_awakening_stage": True}
                    ]
                }
                plan.write_text(json.dumps(changed_plan), encoding="utf-8")
                refreshed_builder = LifeBuilder(store, plan)
                self.assertFalse(refreshed_builder.ready_to_awaken())
                progress = {row["stage_id"]: row for row in refreshed_builder.progress()}
                self.assertEqual(progress["first"]["missing_coverage"], ["childhood"])
                with self.assertRaises(RuntimeError):
                    refreshed_builder.perform_awakening(HuTaoPersona())

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
                self.assertIn("You know that you died", context)
                self.assertIn("cannot remember how it happened", context)
                self.assertIn("Where am I? Where is my body? What is this box?", context)
                self.assertTrue(persona.opening_scene_pending)
                with self.assertRaises(RuntimeError):
                    builder.perform_awakening(persona)
                state = store.connection.execute(
                    "SELECT value FROM life_state WHERE key='state'"
                ).fetchone()["value"]
                self.assertEqual(state, "awakened")


if __name__ == "__main__":
    unittest.main()
