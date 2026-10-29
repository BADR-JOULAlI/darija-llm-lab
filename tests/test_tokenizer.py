"""Tests sur des exemples inventes, sans telechargement de donnees AtlasIA."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/atelier_tokenizer.py"
spec = importlib.util.spec_from_file_location("atelier_tokenizer", SCRIPT)
atelier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atelier)


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_fixture = [
            "salam, labas?", "ana kanqra b darija.",
            "ghadi nji demain.", "salam khouya, labas 3lik?",
        ]
        cls.tokenizer = atelier.train_bpe(cls.training_fixture, vocab_size=300)

    def test_unicode_and_literal_special_tokens_roundtrip(self):
        samples = [
            "سلام، لاباس؟", "cafe\u0301", "🙂", "salam  labas\n",
            "literal [UNK] [PAD] [BOS] [EOS]", "",
        ]
        atelier.check_roundtrip(self.tokenizer, samples)

    def test_save_reload_preserves_encoding(self):
        text = "salam [EOS] 🙂"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(str(path))
            loaded = atelier.Tokenizer.from_file(str(path))
            loaded.encode_special_tokens = True
            self.assertEqual(self.tokenizer.encode(text).ids, loaded.encode(text).ids)
            atelier.check_roundtrip(loaded, [text])

    def test_windows_have_correct_shift_and_last_target(self):
        result = list(atelier.windows([10, 20, 30, 40, 50, 60], length=3, stride=2))
        self.assertEqual(result, [
            ([10, 20, 30], [20, 30, 40]),
            ([30, 40, 50], [40, 50, 60]),
        ])
        self.assertEqual(list(atelier.windows([10, 20, 30], length=3)), [])
        with self.assertRaises(ValueError):
            list(atelier.windows([10, 20], length=1, stride=0))

    def test_windows_stay_inside_document(self):
        texts = ["salam [EOS]", "labas [UNK]"]
        expected = []
        eos = self.tokenizer.token_to_id("[EOS]")
        for text in texts:
            ids = self.tokenizer.encode(text).ids
            self.assertNotIn(eos, ids)
            expected.extend(atelier.windows(ids + [eos], length=3, stride=1))
        actual = list(atelier.document_windows(
            self.tokenizer, texts, length=3, stride=1
        ))
        self.assertEqual(actual, expected)

    def test_duplicate_normalization_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invented_fixture.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["darija"])
                writer.writeheader()
                for text in ["café", " cafe\u0301 ", "salam", "labas", ""]:
                    writer.writerow({"darija": text})
            documents = atelier.read_corpus(path)
            self.assertEqual(set(documents), {"café", "salam", "labas"})
            train, valid = atelier.split_corpus(documents)
            self.assertFalse(set(train) & set(valid))
            self.assertEqual(len(train) + len(valid), 3)


if __name__ == "__main__":
    unittest.main()
