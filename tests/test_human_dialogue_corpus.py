import unittest

from training.audit_human_dialogue_corpus import (
    iter_taskmaster_splits,
    taskmaster_context_turns,
    taskmaster_row_sample,
)


class TaskmasterCorpusTests(unittest.TestCase):
    def test_context_parser_keeps_utterances_and_discards_action_markup(self):
        context = (
            "<PR>find_theaters<PRAN>location<PAV>Spring Hills<C>"
            "<U>Could you find a funny movie?<PN>find_movies<PAN>genre<PAV>comedy"
            "<A>I found two comedies nearby.<U>Which one is newer?"
        )
        self.assertEqual(
            taskmaster_context_turns(context),
            [
                ("user", "Could you find a funny movie?"),
                ("assistant", "I found two comedies nearby."),
                ("user", "Which one is newer?"),
            ],
        )

    def test_row_serializer_appends_target_as_final_assistant_turn(self):
        row = {
            "conversation_id": "dlg-123",
            "context": "<C><U>Can you help me?<A>Sure, what do you need?<U>Find a comedy.",
            "target": "I can help you look for one.",
        }
        conversation_id, sample = taskmaster_row_sample(row)
        self.assertEqual(conversation_id, "dlg-123")
        self.assertEqual(
            sample,
            "<|user|> Can you help me? <|assistant|> Sure, what do you need? <|eot|> "
            "<|user|> Find a comedy. <|assistant|> I can help you look for one. <|eot|>",
        )

    def test_mojibake_and_malformed_turn_sequences_are_rejected(self):
        row = {
            "conversation_id": "dlg-bad",
            "context": "<U>Hello there<A>ThatÃ¢ÂÂs broken.<U>What?",
            "target": "A final reply.",
        }
        self.assertIsNone(taskmaster_row_sample(row)[1])
        self.assertEqual(
            taskmaster_context_turns("<U>Hi<A>Hello<A>Again"),
            [],
        )

    def test_taskmaster_is_resplit_by_conversation_without_leakage(self):
        import hashlib

        chosen = {"train": None, "validation": None, "test": None}
        candidate = 0
        while any(value is None for value in chosen.values()):
            conversation_id = f"dialog-{candidate}"
            digest = hashlib.sha256(
                f"hutao-taskmaster-group-split-v1:{conversation_id}".encode("utf-8")
            ).digest()
            bucket = int.from_bytes(digest[:4], "big") % 100
            split = "test" if bucket < 5 else "validation" if bucket < 10 else "train"
            chosen[split] = chosen[split] or conversation_id
            candidate += 1

        def row(conversation_id, question, answer):
            return {
                "conversation_id": conversation_id,
                "context": f"<U>{question}",
                "target": answer,
            }

        repeated_id = chosen["train"]
        groups, report = iter_taskmaster_splits({
            "train": [
                row(repeated_id, "How are you?", "I am well."),
                row(chosen["validation"], "What time is it?", "It is noon."),
            ],
            "dev": [
                row(repeated_id, "Would you like tea?", "Yes, please."),
            ],
            "test": [
                row(chosen["test"], "Can you help me?", "Certainly."),
            ],
        })
        self.assertEqual({key: len(value) for key, value in groups.items()}, {
            "train": 2, "validation": 1, "test": 1
        })
        self.assertEqual(report["unique_conversations"], 3)
        self.assertEqual(report["source_rows_total"], 4)
        self.assertEqual(report["conversation_id_overlap"], {
            "train__validation": 0,
            "train__test": 0,
            "validation__test": 0,
        })
        self.assertEqual(
            report["split_policy"],
            "deterministic SHA-256 by conversation_id; 90% train, 5% validation, 5% test",
        )


if __name__ == "__main__":
    unittest.main()
