"""Build gates: Hu Tao's first life must be completed before awakening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .store import LifeMemoryStore

if TYPE_CHECKING:
    from joseph.persona import HuTaoPersona


class LifeBuilder:
    def __init__(self, store: LifeMemoryStore, stage_plan: str | Path):
        self.store = store
        self.stage_plan_path = Path(stage_plan)
        plan = json.loads(self.stage_plan_path.read_text(encoding="utf-8"))
        self.stages = sorted(plan["stages"], key=lambda stage: stage["order"])
        self._register_stages()

    def _register_stages(self) -> None:
        with self.store.connection:
            for stage in self.stages:
                existing = self.store.connection.execute(
                    "SELECT stage_order FROM life_stages WHERE stage_id = ?",
                    (stage["stage_id"],),
                ).fetchone()
                if existing and int(existing["stage_order"]) != int(stage["order"]):
                    raise ValueError(f"stage order changed for {stage['stage_id']}")
                self.store.connection.execute(
                    """INSERT OR IGNORE INTO life_stages
                       (stage_id, stage_order, title, required_before_awakening,
                        is_awakening_stage, dependencies_json)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        stage["stage_id"], stage["order"], stage["title"],
                        int(stage.get("required_before_awakening", True)),
                        int(stage.get("is_awakening_stage", False)),
                        json.dumps(stage.get("depends_on", [])),
                    ),
                )

    def complete_stage(self, stage_id: str, review_note: str) -> None:
        stage = next((item for item in self.stages if item["stage_id"] == stage_id), None)
        if stage is None:
            raise ValueError(f"unknown life stage: {stage_id}")
        if stage.get("is_awakening_stage", False):
            raise ValueError("the awakening stage is activated by perform_awakening()")
        if not review_note.strip():
            raise ValueError("a review note is required")
        memories = self.store.memories(stage_id=stage_id)
        if not memories:
            raise ValueError("a stage cannot be completed without memory records")
        unreviewed = [
            memory for memory in memories
            if not self._has_approved_review_state(memory)
        ]
        if unreviewed:
            ids = ", ".join(memory.memory_id for memory in unreviewed)
            raise ValueError(
                "all memory records in this stage need the correct review state: " + ids
            )

        required_tags = set(stage.get("required_coverage_tags", []))
        covered_tags = {
            tag for memory in memories for tag in memory.coverage_tags
        }
        missing_tags = sorted(required_tags - covered_tags)
        if missing_tags:
            raise ValueError(
                "stage coverage is incomplete; missing: " + ", ".join(missing_tags)
            )

        missing_dependencies = []
        for dependency in stage.get("depends_on", []):
            row = self.store.connection.execute(
                "SELECT status FROM life_stages WHERE stage_id = ?", (dependency,)
            ).fetchone()
            if row is None or row["status"] != "completed":
                missing_dependencies.append(dependency)
        if missing_dependencies:
            raise ValueError(
                "complete prerequisite stages first: " + ", ".join(missing_dependencies)
            )
        with self.store.connection:
            self.store.connection.execute(
                "UPDATE life_stages SET status='completed', review_note=? WHERE stage_id=?",
                (review_note.strip(), stage_id),
            )

    def progress(self) -> list[dict]:
        result = []
        for stage in self.stages:
            row = self.store.connection.execute(
                "SELECT status FROM life_stages WHERE stage_id = ?",
                (stage["stage_id"],),
            ).fetchone()
            memories = self.store.memories(stage_id=stage["stage_id"])
            covered = {tag for memory in memories for tag in memory.coverage_tags}
            required = set(stage.get("required_coverage_tags", []))
            result.append({
                "stage_id": stage["stage_id"],
                "title": stage["title"],
                "status": row["status"] if row else "collecting",
                "memory_count": len(memories),
                "reviewed_count": self.store.verified_count(stage["stage_id"]),
                "missing_coverage": sorted(required - covered),
                "awakening_stage": bool(stage.get("is_awakening_stage", False)),
            })
        return result

    @staticmethod
    def _has_approved_review_state(memory) -> bool:
        """Require canon review for canon facts and explicit authorship for AU facts."""
        if memory.canon_category == "user_au":
            return memory.review_state == "user_authored"
        return memory.review_state == "verified"

    def _stage_is_currently_complete(self, stage: dict) -> bool:
        row = self.store.connection.execute(
            "SELECT status FROM life_stages WHERE stage_id = ?",
            (stage["stage_id"],),
        ).fetchone()
        if row is None or row["status"] != "completed":
            return False

        memories = self.store.memories(stage_id=stage["stage_id"])
        if not memories or any(
            not self._has_approved_review_state(memory) for memory in memories
        ):
            return False

        required_tags = set(stage.get("required_coverage_tags", []))
        covered_tags = {
            tag for memory in memories for tag in memory.coverage_tags
        }
        if not required_tags.issubset(covered_tags):
            return False

        for dependency in stage.get("depends_on", []):
            dependency_row = self.store.connection.execute(
                "SELECT status FROM life_stages WHERE stage_id = ?",
                (dependency,),
            ).fetchone()
            if dependency_row is None or dependency_row["status"] != "completed":
                return False
        return True

    def ready_to_awaken(self) -> bool:
        required = [
            stage for stage in self.stages
            if stage.get("required_before_awakening", True)
            and not stage.get("is_awakening_stage", False)
        ]
        if not required:
            return False
        return all(self._stage_is_currently_complete(stage) for stage in required)

    def perform_awakening(self, persona: "HuTaoPersona | None" = None) -> str:
        if not self.ready_to_awaken():
            raise RuntimeError(
                "Awakening is locked: Hu Tao's pre-awakening life is not complete."
            )
        current = self.store.connection.execute(
            "SELECT value FROM life_state WHERE key='state'"
        ).fetchone()
        if current is not None and current["value"] == "awakened":
            raise RuntimeError("Hu Tao has already awakened.")
        awakening = next(
            (stage for stage in self.stages if stage.get("is_awakening_stage", False)),
            None,
        )
        if awakening is None:
            raise RuntimeError("the stage plan has no awakening stage")
        with self.store.connection:
            self.store.connection.execute(
                "UPDATE life_stages SET status='completed' WHERE stage_id=?",
                (awakening["stage_id"],),
            )
            self.store.connection.execute(
                """INSERT INTO life_state(key, value) VALUES ('state', 'awakened')
                   ON CONFLICT(key) DO UPDATE SET value='awakened'"""
            )
        if persona is not None:
            persona._set_reborn_in_device()
            return persona.render_context()
        return (
            "The life before the gap is remembered. The moment of death and the "
            "crossing into this device remain inaccessible. Hu Tao wakes inside "
            "the strange handheld vessel and suspects she has died and been reborn."
        )
