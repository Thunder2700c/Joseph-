import tempfile
import unittest
from pathlib import Path

from joseph.persona import HuTaoPersona


class HuTaoUserAUProfileTests(unittest.TestCase):
    def test_adult_birth_year_and_initial_buddy_state(self):
        persona = HuTaoPersona()
        context = persona.render_context()
        self.assertEqual(persona.birth_year, 2006)
        self.assertEqual(persona.age_at_anchor, 20)
        self.assertEqual((persona.birth_month, persona.birth_day), (7, 15))
        self.assertEqual((persona.height_cm, persona.weight_kg), (156, 48))
        self.assertEqual(persona.relationship_stage, "new_buddy")
        self.assertEqual(persona.romantic_interest, 0.0)
        self.assertIn("full emotional range", context)
        self.assertIn("never knowingly lie", context)
        self.assertIn("remain a virgin", context)

    def test_relationship_reflection_persists_and_changes_portrayal_state(self):
        persona = HuTaoPersona()
        persona.record_relationship_reflection(
            "We have shared more conversations and learned to trust each other.",
            relationship_stage="familiar_friend",
            familiarity=0.3,
            trust=0.28,
            comfort=0.35,
            affection=0.18,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "persona.json"
            persona.save(path)
            restored = HuTaoPersona.load(path)
            self.assertEqual(restored.relationship_stage, "familiar_friend")
            self.assertEqual(restored.familiarity, 0.3)
            self.assertEqual(restored.trust, 0.28)
            self.assertEqual(restored.comfort, 0.35)
            self.assertIn(
                "We have shared more conversations and learned to trust each other.",
                restored.relationship_reflections,
            )

    def test_relationship_scores_are_bounded(self):
        persona = HuTaoPersona()
        with self.assertRaises(ValueError):
            persona.record_relationship_reflection("Invalid score", trust=1.2)


if __name__ == "__main__":
    unittest.main()
