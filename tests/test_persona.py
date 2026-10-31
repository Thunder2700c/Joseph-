import tempfile
import unittest
from pathlib import Path

from joseph.persona import JosephPersona


class PersonaTests(unittest.TestCase):
    def test_roleplay_overlay_does_not_change_original_identity(self):
        persona = JosephPersona()
        original = persona.render_context()
        roleplay = persona.render_context("Jotaro Kujo")
        self.assertIn("original AI companion and friend", original)
        self.assertIn("TEMPORARY ROLEPLAY OVERLAY", roleplay)
        self.assertIn("not Joseph's identity", roleplay)
        self.assertEqual(persona.render_context(), original)
        self.assertIsNone(getattr(persona, "roleplay_character", None))

    def test_explicit_feedback_persists(self):
        persona = JosephPersona()
        persona.record_feedback("Be more concise.", trait="concise")
        self.assertIn("Be more concise.", persona.render_context())
        self.assertIn("concise", persona.render_context())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "persona.json"
            persona.save(path)
            restored = JosephPersona.load(path)
            self.assertIn("Be more concise.", restored.learned_preferences)
            self.assertIn("concise", restored.evolving_traits)


if __name__ == "__main__":
    unittest.main()
