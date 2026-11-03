import unittest

from training.prepare_oasst1_corpus import conversation_samples


def make_row(tree_id, message_id, parent_id, role, text, rank=None, synthetic=False):
    return {
        "message_tree_id": tree_id,
        "message_id": message_id,
        "parent_id": parent_id,
        "role": role,
        "text": text,
        "lang": "en",
        "review_result": True,
        "deleted": False,
        "synthetic": synthetic,
        "tree_state": "ready_for_export",
        "rank": rank,
        "labels": {
            "spam": {"value": 0.0},
            "lang_mismatch": {"value": 0.0},
            "pii": {"value": 0.0},
            "not_appropriate": {"value": 0.0},
            "hate_speech": {"value": 0.0},
            "sexual_content": {"value": 0.0},
        },
    }


class OASSTPreparationTests(unittest.TestCase):
    def test_tree_split_is_disjoint_and_preferred_assistant_reply_is_selected(self):
        rows = []
        for index in range(40):
            tree = f"tree-{index}"
            root = f"{tree}-root"
            rows.append(make_row(tree, root, None, "prompter", "Can you explain this idea?"))
            rows.append(make_row(tree, f"{tree}-best", root, "assistant", "Here is a clear explanation.", rank=0))
            rows.append(make_row(tree, f"{tree}-alternate", root, "assistant", "A lower-ranked alternative.", rank=1))
        train, validation, report = conversation_samples(rows, validation_percent=50)
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertEqual(len(train) + len(validation), 40)
        self.assertEqual(report["train_validation_tree_overlap"], 0)
        self.assertTrue(all("Here is a clear explanation." in sample for sample in train + validation))
        self.assertFalse(any("lower-ranked alternative" in sample for sample in train + validation))

    def test_synthetic_messages_are_excluded(self):
        rows = [
            make_row("tree-a", "root-a", None, "prompter", "Please explain this clearly."),
            make_row("tree-a", "reply-a", "root-a", "assistant", "This is an approved human reply.", rank=0),
            make_row("tree-b", "root-b", None, "prompter", "Please explain this synthetic test."),
            make_row("tree-b", "reply-b", "root-b", "assistant", "Synthetic reply must not enter.", rank=0, synthetic=True),
        ]
        train, validation, report = conversation_samples(rows, validation_percent=50)
        all_samples = train + validation
        self.assertTrue(all("Synthetic reply must not enter." not in sample for sample in all_samples))
        self.assertGreater(report["rejected_message_reasons"].get("synthetic_or_unknown", 0), 0)


if __name__ == "__main__":
    unittest.main()
