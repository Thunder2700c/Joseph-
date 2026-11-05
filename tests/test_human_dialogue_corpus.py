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

    def test_official_splits_are_kept_separate_by_conversation(self):
        row_train = {
            "conversation_id": "train-1",
            "context": "<U>Hello there?",
            "target": "Yes, hello.",
        }
        row_validation = {
            "conversation_id": "valid-1",
            "context": "<U>Are you there?",
            "target": "I am here.",
        }
        row_test = {
            "conversation_id": "test-1",
            "context": "<U>Can you help?",
            "target": "Certainly.",
        }
        groups, report = iter_taskmaster_splits({
            "train": [row_train],
            "dev": [row_validation],
            "test": [row_test],
        })
        self.assertEqual({key: len(value) for key, value in groups.items()}, {
            "train": 1, "validation": 1, "test": 1
        })
        self.assertEqual(set(report["conversation_id_overlap"].values()), {0})

    def test_cross_split_conversation_reuse_is_rejected(self):
        row = {
            "conversation_id": "same-conversation",
            "context": "<U>Hello?",
            "target": "Hi.",
        }
        with self.assertRaisesRegex(RuntimeError, "cross train/validation splits"):
            iter_taskmaster_splits({
                "train": [row],
                "validation": [row],
                "test": [{"conversation_id": "test", "context": "<U>Hey?", "target": "Hello."}],
            })


if __name__ == "__main__":
    unittest.main()
