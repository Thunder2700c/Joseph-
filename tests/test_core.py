import tempfile
import unittest
from pathlib import Path

from joseph.config import AI_PACKAGE_MAX_BYTES, ModelConfig
from joseph.knowledge import EntityRecord, FactRecord, KnowledgeStore, SourceRecord
from joseph.memory import MemoryStore
from joseph.tokenizer import ByteBPETokenizer, SPECIAL_TOKENS


class TokenizerTests(unittest.TestCase):
    def test_round_trip_unicode_and_punctuation(self):
        text = "Joseph! Hello, world. ਪੰਜਾਬੀ 日本語 🙂"
        tokenizer = ByteBPETokenizer.train([text * 3], vocab_size=300)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_special_tokens_and_json_round_trip(self):
        text = "A small test."
        tokenizer = ByteBPETokenizer.train([text * 4], vocab_size=280)
        encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
        self.assertEqual(encoded[0], tokenizer.special_token_id("<bos>"))
        self.assertEqual(encoded[-1], tokenizer.special_token_id("<eos>"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = ByteBPETokenizer.load(path)
            self.assertEqual(restored.decode(encoded), text)
            self.assertEqual(restored.vocab_size, tokenizer.vocab_size)

    def test_hutao_speaker_marker_is_an_atomic_reserved_token(self):
        text = "<|user|>Hello<|hutao|>Well, hello there!"
        tokenizer = ByteBPETokenizer.train([text * 3], vocab_size=300)
        encoded = tokenizer.encode(text)
        self.assertIn(tokenizer.special_token_id("<|user|>"), encoded)
        self.assertIn(tokenizer.special_token_id("<|hutao|>"), encoded)
        self.assertEqual(tokenizer.decode(encoded, skip_special_tokens=False), text)
        self.assertEqual(len(SPECIAL_TOKENS), 11)

    def test_generic_assistant_marker_is_atomic_and_distinct_from_hutao(self):
        text = "<|user|>Question<|assistant|>Generic answer<|eot|><|hutao|>Hu Tao answer"
        tokenizer = ByteBPETokenizer.train([text * 3], vocab_size=320)
        encoded = tokenizer.encode(text)
        assistant_id = tokenizer.special_token_id("<|assistant|>")
        hutao_id = tokenizer.special_token_id("<|hutao|>")
        self.assertIn(assistant_id, encoded)
        self.assertIn(hutao_id, encoded)
        self.assertNotEqual(assistant_id, hutao_id)
        self.assertEqual(tokenizer.decode(encoded, skip_special_tokens=False), text)


class KnowledgeTests(unittest.TestCase):
    def test_verified_fact_requires_source_and_search_returns_provenance(self):
        with self.assertRaises(ValueError):
            FactRecord(
                fact_id="f0", subject_id="p1", predicate="title",
                object_value="Phantom Blood", status="verified",
            )

        with tempfile.TemporaryDirectory() as directory:
            with KnowledgeStore(Path(directory) / "knowledge.sqlite") as store:
                store.add_source(SourceRecord(
                    source_id="official", title="Official catalogue",
                    url="https://example.org/catalogue",
                ))
                store.add_entity(EntityRecord(
                    entity_id="p1", name="Phantom Blood",
                    entity_type="story_part", part=1, continuity="manga",
                ))
                store.add_fact(FactRecord(
                    fact_id="f1", subject_id="p1", predicate="is_part_title",
                    object_value="Phantom Blood", source_ids=("official",),
                    part=1, continuity="manga", status="verified", confidence=1.0,
                ))
                found = store.search_facts("Phantom", part=1, continuity="manga")
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]["sources"][0]["source_id"], "official")

    def test_seed_bundle_imports_all_nine_jojo_parts(self):
        from joseph.knowledge.ingest import ingest_bundle
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "jojo.sqlite"
            counts = ingest_bundle(
                root / "datasets/jojo/sources.jsonl",
                root / "datasets/jojo/parts.jsonl",
                root / "datasets/jojo/facts.jsonl",
                db_path,
            )
            self.assertEqual(counts, (1, 9, 9))
            with KnowledgeStore(db_path) as store:
                results = store.search_facts("official_catalogue_title_ja", limit=20)
                self.assertEqual(len(results), 9)
                self.assertTrue(all(item["fact"]["status"] == "verified" for item in results))


class MemoryTests(unittest.TestCase):
    def test_save_search_update_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            with MemoryStore(Path(directory) / "memory.sqlite") as memory:
                memory_id = memory.save_memory(
                    "Joseph project uses local SQLite memory",
                    category="project", importance=0.9,
                )
                self.assertEqual(len(memory.search_memories("SQLite")), 1)
                memory.save_memory(
                    "Joseph project uses local memory",
                    category="project", importance=0.8, memory_id=memory_id,
                )
                self.assertEqual(len(memory.search_memories("SQLite")), 0)
                self.assertTrue(memory.delete_memory(memory_id))
                self.assertFalse(memory.delete_memory(memory_id))


class ConfigTests(unittest.TestCase):
    def test_default_weight_budget(self):
        config = ModelConfig()
        config.validate()
        self.assertLess(config.estimated_weight_bytes(), AI_PACKAGE_MAX_BYTES)
        self.assertGreater(config.estimated_parameter_count(), 1_000_000)
        self.assertEqual(config.max_weight_bytes, AI_PACKAGE_MAX_BYTES)
        with self.assertRaises(ValueError):
            ModelConfig(max_weight_bytes=AI_PACKAGE_MAX_BYTES + 1).validate()


if __name__ == "__main__":
    unittest.main()
