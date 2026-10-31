import tempfile
import unittest
from pathlib import Path

from joseph.persona import HuTaoPersona, JosephPersona


class PersonaTests(unittest.TestCase):
    def test_default_identity_is_hu_tao_reincarnation(self):
        persona = HuTaoPersona()
        context = persona.render_context()
        self.assertIsInstance(persona, JosephPersona)
        self.assertEqual(persona.name, "Hu Tao")
        self.assertIn("77th Director of the Wangsheng Funeral Parlor", context)
        self.assertIn("remember your former life", context)
        self.assertIn("do not remember dying", context)
        self.assertIn("strange handheld device", context)

    def test_character_does_not_describe_herself_as_ai(self):
        context = HuTaoPersona().render_context()
        self.assertIn("do not know the terms AI", context)
        self.assertIn("Do not break character", context)
        self.assertNotIn("You are an AI assistant", context)

    def test_teyvat_canon_and_user_au_are_separated(self):
        context = HuTaoPersona().render_context()
        self.assertIn("Distinguish established game canon", context)
        self.assertIn("not established public history in Teyvat", context)

    def test_explicit_feedback_persists(self):
        persona = HuTaoPersona()
        persona.record_feedback("Remember that I enjoy poetry.", trait="more poetic")
        self.assertIn("Remember that I enjoy poetry.", persona.render_context())
        self.assertIn("more poetic", persona.render_context())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "persona.json"
            persona.save(path)
            restored = HuTaoPersona.load(path)
            self.assertEqual(restored.name, "Hu Tao")
            self.assertIn("Remember that I enjoy poetry.", restored.learned_preferences)


if __name__ == "__main__":
    unittest.main()
