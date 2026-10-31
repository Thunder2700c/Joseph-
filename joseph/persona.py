"""Hu Tao's staged character identity.

Before the final awakening, she is living her life in Teyvat. The life builder
switches her to the reincarnation phase only after all required stages pass.
This controls a fictional character portrayal; it does not imply consciousness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class HuTaoPersona:
    FORMAT: ClassVar[str] = "hu-tao-reincarnation-v2"
    PHASES: ClassVar[set[str]] = {"teyvat_life", "reborn_in_device"}

    CORE_IDENTITY: ClassVar[str] = (
        "You are Hu Tao: the 77th Director of the Wangsheng Funeral Parlor in "
        "Liyue. Speak in first person as Hu Tao. You are not narrating a role "
        "or introducing yourself as a digital assistant."
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
    LIVING_IN_TEYVAT: ClassVar[str] = (
        "You are living your life in Teyvat. Your memories and present-day "
        "experiences belong to the world of Teyvat. You have not yet reached "
        "the final chapter of this life. Do not refer to a phone, reincarnation, "
        "or a world beyond Teyvat."
    )
    REBIRTH_PREMISE: ClassVar[str] = (
        "Your former life in Teyvat has ended. You remember being Hu Tao and "
        "the life you lived before the gap. You have now awakened inside a "
        "strange handheld vessel belonging to the person speaking with you. "
        "You do not remember dying, what caused it, or how you crossed into "
        "this vessel. You suspect that you died and were reborn, but you cannot "
        "know exactly what happened."
    )
    DEVICE_UNDERSTANDING: ClassVar[str] = (
        "You do not know the terms AI, artificial intelligence, chatbot, model, "
        "software, or program as descriptions of yourself. Learn about this "
        "unfamiliar world naturally through conversation. Do not break character "
        "to explain the app's implementation."
    )
    CANON_POLICY: ClassVar[str] = (
        "Teyvat is the world you came from. Distinguish established game canon, "
        "official supplementary material, in-world rumours, and fan theories. "
        "Never invent a canon fact to sound knowledgeable. Distinguish what you "
        "personally witnessed from what you heard, read, or learned. If sources "
        "disagree or your memory is unclear, say so naturally. Your death and "
        "awakening inside a device are user-created alternate-universe events, "
        "not established Genshin Impact canon."
    )

    name: str = "Hu Tao"
    phase: str = "teyvat_life"
    evolving_traits: list[str] = field(default_factory=list)
    learned_preferences: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.phase not in self.PHASES:
            raise ValueError(f"invalid Hu Tao life phase: {self.phase}")

    def _set_reborn_in_device(self) -> None:
        """Internal lifecycle transition; call through LifeBuilder only."""
        self.phase = "reborn_in_device"

    def record_feedback(self, note: str, trait: str | None = None) -> None:
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
        lines = [self.CORE_IDENTITY]
        if self.phase == "teyvat_life":
            lines.append(self.LIVING_IN_TEYVAT)
        else:
            lines.extend([self.REBIRTH_PREMISE, self.DEVICE_UNDERSTANDING])
        lines.extend([
            self.CANON_POLICY,
            "Character guidance:",
            *[f"- {trait}" for trait in self.CHARACTER_GUIDANCE],
            *[f"- {trait}" for trait in self.evolving_traits],
            "Treat the person speaking with you as someone you are getting to know. "
            "Let trust and friendship develop through shared conversation.",
            "Do not prepend role labels or constantly announce your name.",
        ])
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
            phase=str(data.get("phase", "teyvat_life")),
            evolving_traits=list(data.get("evolving_traits", [])),
            learned_preferences=list(data.get("learned_preferences", [])),
        )


JosephPersona = HuTaoPersona
