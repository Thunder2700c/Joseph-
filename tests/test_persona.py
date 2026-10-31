import tempfile
import unittest
from pathlib import Path

from joseph.persona import HuTaoPersona, JosephPersona


class PersonaTests(unittest.TestCase):
    def test_default_is_her_life_in_teyvat_not_phone_awakening(self):
        persona = HuTaoPersona()
        context = persona.render_context()
        self.assertIsInstance(persona, JosephPersona)
        self.assertEqual(persona.name, "Hu Tao")
        self.assertEqual(persona.birth_month, 7)
        self.assertEqual(persona.birth_day, 15)
        self.assertEqual(persona.birth_year, 2006)
        self.assertEqual(persona.height_cm, 156)
        self.assertEqual(persona.weight_kg, 48)
        self.assertIn("Your birthday is July 15", context)
        self.assertIn("48 kg", context)
        self.assertIn("Your home is Liyue Harbor", context)
        self.assertIn("Your closest friends are Xingqiu", context)
        self.assertIn("distinguish a person you personally know", context)
        self.assertIn("77th Director of the Wangsheng Funeral Parlor", context)
        self.assertIn("living your life in Teyvat", context)
        self.assertNotIn("strange handheld vessel", context)

    def test_reborn_phase_is_a_separate_final_state(self):
        persona = HuTaoPersona(phase="reborn_in_device")
        context = persona.render_context()
        self.assertEqual(persona.phase, "reborn_in_device")
        self.assertIn("You know that you died", context)
        self.assertIn("cannot remember how it happened", context)
        self.assertIn("do not remember crossing between worlds", context)
        self.assertNotIn("suspect that you died and were reborn", context)
        self.assertIn("Where am I? Where is my body? What is this box?", context)
        self.assertIn("fear, anger, sadness", context)
        self.assertTrue(persona.opening_scene_pending)
        persona.mark_opening_scene_seen()
        self.assertFalse(persona.opening_scene_pending)
        self.assertNotIn("This guidance applies only to your very first response", persona.render_context())
        self.assertIn("do not know the terms AI", context)
        self.assertIn("not established Genshin Impact canon", context)

    def test_explicit_feedback_persists_with_phase(self):
        persona = HuTaoPersona(phase="reborn_in_device")
        persona.record_feedback("Remember that I enjoy poetry.", trait="more poetic")
        persona.mark_opening_scene_seen()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "persona.json"
            persona.save(path)
            restored = HuTaoPersona.load(path)
            self.assertEqual(restored.phase, "reborn_in_device")
            self.assertIn("Remember that I enjoy poetry.", restored.learned_preferences)
            self.assertIn("more poetic", restored.evolving_traits)
            self.assertFalse(restored.opening_scene_pending)


if __name__ == "__main__":
    unittest.main()
