"""Atelier du chapitre 2 : vocabulaire et fusions appris sur les donnees AtlasIA.

Installation : python -m pip install tokenizers==0.22.1
Exemple : python atelier_tokenizer.py --csv atlasia_stories.csv --limit 1000
Sans --csv, le fichier public est telecharge dans le dossier de travail.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from urllib.request import urlopen

import tokenizers
from tokenizers import Tokenizer, decoders, models, normalizers
from tokenizers import pre_tokenizers, trainers

DATASET = "atlasia/darija_english"
SOURCE_URL = (
    "https://huggingface.co/datasets/atlasia/darija_english/"
    "resolve/main/atlasia_stories.csv"
)
SPECIALS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]


def normalize(text):
    return unicodedata.normalize("NFC", text)


class WordTokenizer:
    """Version pedagogique : le decodage ne preserve pas les espaces."""

    def __init__(self, texts):
        pieces = {
            piece for text in texts
            for piece in re.findall(r"\w+|[^\w\s]", normalize(text))
        }
        self.vocab = {"[UNK]": 0}
        self.vocab.update({s: i + 1 for i, s in enumerate(sorted(pieces))})
        self.inverse = {i: s for s, i in self.vocab.items()}

    def encode(self, text):
        pieces = re.findall(r"\w+|[^\w\s]", normalize(text))
        return [self.vocab.get(s, 0) for s in pieces]

    def decode(self, ids):
        return " ".join(self.inverse[i] for i in ids)


def read_corpus(csv_path, limit=1000):
    if limit < 2:
        raise ValueError("Il faut au moins deux documents.")
    documents, seen = [], set()
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if "darija" not in (rows.fieldnames or []):
            raise ValueError("Colonne 'darija' absente : verifier le CSV AtlasIA.")
        for row in rows:
            text = normalize(row.get("darija") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            documents.append(text)
            if len(documents) >= limit:
                break
    if len(documents) < 2:
        raise ValueError("Le CSV contient moins de deux textes distincts.")
    return documents


def split_corpus(documents, seed=42):
    documents = list(documents)
    random.Random(seed).shuffle(documents)
    cut = max(1, min(len(documents) - 1, int(0.9 * len(documents))))
    return documents[:cut], documents[cut:]


def train_bpe(train_texts, vocab_size=2000):
    if vocab_size < 256 + len(SPECIALS):
        raise ValueError("La taille doit couvrir 256 octets et les tokens speciaux.")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIALS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    tokenizer.encode_special_tokens = True
    return tokenizer


def windows(ids, length=8, stride=4):
    if length < 1 or stride < 1:
        raise ValueError("length et stride doivent etre positifs.")
    for start in range(0, len(ids) - length, stride):
        x = ids[start:start + length]
        y = ids[start + 1:start + length + 1]
        yield x, y


def document_windows(tokenizer, texts, length=8, stride=4):
    eos = tokenizer.token_to_id("[EOS]")
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids + [eos]
        yield from windows(ids, length=length, stride=stride)


def check_roundtrip(tokenizer, samples):
    for text in samples:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        restored = tokenizer.decode(ids, skip_special_tokens=False)
        assert restored == normalize(text), (text, restored)
        assert tokenizer.token_to_id("[UNK]") not in ids


def validation_stats(tokenizer, texts):
    encodings = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    n_tokens = sum(len(e.ids) for e in encodings)
    n_chars = sum(len(normalize(t)) for t in texts)
    unk = tokenizer.token_to_id("[UNK]")
    n_unk = sum(e.ids.count(unk) for e in encodings)
    return {
        "validation_documents": len(texts),
        "validation_tokens": n_tokens,
        "tokens_per_100_characters": round(100 * n_tokens / max(1, n_chars), 2),
        "unknown_tokens": n_unk,
    }


def download_csv(destination):
    """Telecharge les seules donnees du dataset public, sans modele."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(SOURCE_URL, timeout=45) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="CSV AtlasIA deja telecharge")
    parser.add_argument("--out", type=Path, default=Path("atelier_darija"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--vocab-size", type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv or args.out / "atlasia_stories.csv"
    if args.csv is None and not csv_path.exists():
        print("Telechargement du CSV public AtlasIA...")
        download_csv(csv_path)
    documents = read_corpus(csv_path, limit=args.limit)
    train_texts, valid_texts = split_corpus(documents)
    tokenizer = train_bpe(train_texts, vocab_size=args.vocab_size)
    check_roundtrip(tokenizer, valid_texts)
    target = args.out / "tokenizer_darija.json"
    tokenizer.save(str(target))
    reloaded = Tokenizer.from_file(str(target))
    reloaded.encode_special_tokens = True
    assert reloaded.encode(valid_texts[0]).ids == tokenizer.encode(valid_texts[0]).ids
    manifest = {
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "seed": 42,
        "normalization": "NFC; strip document edges before split",
        "literal_special_tokens_as_text": True,
        "train_documents": len(train_texts),
        "requested_vocabulary": args.vocab_size,
        "actual_vocabulary": tokenizer.get_vocab_size(),
        "tokenizers_version": tokenizers.__version__,
        **validation_stats(tokenizer, valid_texts),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    first = next(document_windows(tokenizer, train_texts), None)
    if first:
        print("Entree :", first[0])
        print("Cible  :", first[1])
    print("Tokenizer sauvegarde :", target)


if __name__ == "__main__":
    main()
