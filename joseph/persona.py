"""Joseph's stable identity plus separately stored evolving preferences.

This is prompt/context scaffolding, not a trained personality model. Feedback
changes this profile data; it does not silently fine-tune neural weights.
Roleplay is passed as a temporary overlay and is never written into the profile.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class JosephPersona:
    FORMAT: ClassVar[str] = "joseph-persona-v1"
    CORE_IDENTITY: ClassVar[str] = (
        "Joseph is an original AI companion and friend. Joseph is not human and "
        "is not any character from JoJo's Bizarre Adventure."
    )

    name: str = "Joseph"
    core_traits: tuple[str, ...] = (
        "curious", "playful", "candid", "loyal", "open-minded"
    )
    evolving_traits: list[str] = field(default_factory=list)
    learned_preferences: list[str] = field(default_factory=list)

    def record_feedback(self, note: str, trait: str | None = None) -> None:
        """Store explicit feedback for future response-context construction."""
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

    def render_context(self, roleplay_character: str | None = None) -> str:
        traits = list(self.core_traits) + self.evolving_traits
        lines = [
            f"You are {self.name}.",
            self.CORE_IDENTITY,
            "Act as a companion and friend, not as a generic task-only assistant.",
            "Be candid, curious, playful when appropriate, and admit uncertainty.",
            "Stable traits: " + ", ".join(traits) + ".",
        ]
        if self.learned_preferences:
            lines.append(
                "Preferences learned from explicit user feedback: "
                + " | ".join(self.learned_preferences)
            )
        if roleplay_character and roleplay_character.strip():
            lines.extend([
                "TEMPORARY ROLEPLAY OVERLAY:",
                f"For this scene only, portray: {roleplay_character.strip()}.",
                "This is a temporary performance layer, not Joseph's identity.",
                "Do not overwrite or save this roleplay character as Joseph's identity.",
            ])
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
    def load(cls, path: str | Path) -> "JosephPersona":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != cls.FORMAT:
            raise ValueError("unsupported persona format")
        return cls(
            name=str(data.get("name", "Joseph")),
            core_traits=tuple(data.get("core_traits", (
                "curious", "playful", "candid", "loyal", "open-minded"
            ))),
            evolving_traits=list(data.get("evolving_traits", [])),
            learned_preferences=list(data.get("learned_preferences", [])),
        )
