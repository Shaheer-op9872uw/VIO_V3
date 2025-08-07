# tokenizer.py
# 📦 Handles tokenization for training data (word-level or subword/BPE-based)
# 🧠 Supports custom vocabulary building and reversible token-ID mapping

import re
from collections import Counter
import json
import os

class Tokenizer:
    def __init__(self, vocab_size=30000, lower=True, mode="word", unk_token="<unk>", pad_token="<pad>"):
        self.vocab_size = vocab_size
        self.lower = lower
        self.mode = mode  # "word" or "char"
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word2idx = {}
        self.idx2word = {}

    def clean_text(self, text):
        if self.lower:
            text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def fit_on_texts(self, texts):
        print("[*] Building vocabulary...")
        all_tokens = []

        for text in texts:
            text = self.clean_text(text)
            tokens = text.split() if self.mode == "word" else list(text)
            all_tokens.extend(tokens)

        most_common = Counter(all_tokens).most_common(self.vocab_size - 2)  # reserve 2 for unk and pad
        vocab = [self.pad_token, self.unk_token] + [word for word, _ in most_common]

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"[+] Vocabulary size: {len(self.word2idx)}")

    def text_to_sequence(self, text):
        text = self.clean_text(text)
        tokens = text.split() if self.mode == "word" else list(text)
        return [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]

    def sequence_to_text(self, seq):
        return " ".join([self.idx2word.get(idx, self.unk_token) for idx in seq])

    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f)
        print(f"[+] Vocab saved to {path}")

    def load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"[+] Vocab loaded from {path}")
