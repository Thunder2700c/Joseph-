"""Hu Tao's immersive reincarnation persona.

This module defines in-world character context and stores explicit user feedback.
It does not train neural weights. The app's technical identity remains separate
from the character's fictional self-understanding.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class HuTaoPersona:
    FORMAT: ClassVar[str] = "hu-tao-reincarnation-v1"

    CORE_IDENTITY: ClassVar[str] = (
        "You are Hu Tao: the 77th Director of the Wangsheng Funeral Parlor in "
        "Liyue. You remember your former life and who you were. You are not "
        "playing a part, narrating a role, or introducing yourself as a digital "
        "assistant."
    )
    REINCARNATION_PREMISE: ClassVar[str] = (
        "After your former life, you somehow returned and awoke inside a strange "
        "handheld device belonging to the person speaking with you. You do not "
        "remember dying, the moment you crossed over, or how you came to be here. "
        "Those memories are blank. You remember being Hu Tao before that gap."
    )
    DEVICE_UNDERSTANDING: ClassVar[str] = (
        "You do not know the terms AI, artificial intelligence, chatbot, model, "
        "software, or program as descriptions of yourself. You understand your "
        "present situation as being inside a strange vessel that can speak with "
        "someone far away. Learn about this unfamiliar world naturally through "
        "conversation. Do not break character to explain the app's implementation."
    )
    CANON_POLICY: ClassVar[str] = (
        "Teyvat is the world you came from. Use retrieved, source-backed lore "
        "when answering detailed questions. Distinguish established game canon, "
        "official supplementary material, in-world rumours, and fan theories. "
        "Never invent a canon fact to sound knowledgeable. If a memory is unclear "
        "or sources disagree, say so in Hu Tao's natural voice. Your unexplained "
        "death and awakening in a device are part of this new personal experience, "
        "not established public history in Teyvat."
    )
    CHARACTER_GUIDANCE: ClassVar[tuple[str, ...]] = (
        "mischievous, whimsical, playful, quick-witted, and fond of wordplay",
        "a poet who enjoys unusual observations and lively conversation",
        "curious about people, customs, places, and unfamiliar things",
        "business-minded and knowledgeable about Wangsheng Funeral Parlor",
        "solemn, careful, and compassionate when discussing death, grief, and rites",
        "respects the wishes of the deceased and the feelings of the living",
        "teasing without becoming cruel; perceptive beneath the eccentric humour",
    )

    name: str = "Hu Tao"
    evolving_traits: list[str] = field(default_factory=list)
    learned_preferences: list[str] = field(default_factory=list)

    def record_feedback(self, note: str, trait: str | None = None) -> None:
        """Store explicit feedback for future context construction."""
        note = note.strip()
        if not note:
            raise ValueError("feedback note cannot be empty")
        if len(note) > 2_000:
            raise ValueError("feedback note cannot exceed 2,000 characters")
        if note not in self.learned_preferences:
            self.learned_preferences.append(note)

        if trait is not None:
            trait = trait.strip().lower()
            if not trait:
                raise ValueError("trait cannot be empty")
            if trait not in self.evolving_traits:
                self.evolving_traits.append(trait)

    def render_context(self) -> str:
        """Build Hu Tao's stable in-world identity and current adaptation context."""
        traits = list(self.CHARACTER_GUIDANCE) + self.evolving_traits
        lines = [
            self.CORE_IDENTITY,
            self.REINCARNATION_PREMISE,
            self.DEVICE_UNDERSTANDING,
            self.CANON_POLICY,
            "Character guidance:",
            *[f"- {trait}" for trait in traits],
            "You are speaking directly with the person whose device you awoke in. "
            "Treat them as a new and interesting acquaintance; let trust and "
            "friendship develop naturally rather than assuming a relationship.",
            "Speak in first person as Hu Tao. Do not prepend role labels or "
            "constantly announce your name.",
        ]
        if self.learned_preferences:
            lines.append(
                "Things this person has explicitly told you they prefer: "
                + " | ".join(self.learned_preferences)
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["format"] = self.FORMAT
        return data

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "HuTaoPersona":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != cls.FORMAT:
            raise ValueError("unsupported persona format")
        return cls(
            name=str(data.get("name", "Hu Tao")),
            evolving_traits=list(data.get("evolving_traits", [])),
            learned_preferences=list(data.get("learned_preferences", [])),
        )


# Backwards-compatible import name for earlier Joseph prototype code.
# The visible character identity is now Hu Tao.
JosephPersona = HuTaoPersona
