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
        self.assertIn("77th Director of the Wangsheng Funeral Parlor", context)
        self.assertIn("living your life in Teyvat", context)
        self.assertNotIn("strange handheld vessel", context)

    def test_reborn_phase_is_a_separate_final_state(self):
        persona = HuTaoPersona(phase="reborn_in_device")
        context = persona.render_context()
        self.assertEqual(persona.phase, "reborn_in_device")
        self.assertIn("do not remember dying", context)
        self.assertIn("suspect that you died and were reborn", context)
        self.assertIn("do not know the terms AI", context)
        self.assertIn("not established Genshin Impact canon", context)

    def test_explicit_feedback_persists_with_phase(self):
        persona = HuTaoPersona(phase="reborn_in_device")
        persona.record_feedback("Remember that I enjoy poetry.", trait="more poetic")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "persona.json"
            persona.save(path)
            restored = HuTaoPersona.load(path)
            self.assertEqual(restored.phase, "reborn_in_device")
            self.assertIn("Remember that I enjoy poetry.", restored.learned_preferences)
            self.assertIn("more poetic", restored.evolving_traits)


if __name__ == "__main__":
    unittest.main()
