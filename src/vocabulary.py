# ============================================================
# Image Captioning using ResNet + LSTM
# File: src/vocabulary.py
# Purpose: Vocabulary and token conversion utilities
# ============================================================

import json


# ============================================================
# Vocabulary Class
# ============================================================

class Vocabulary:
    """
    Handles the vocabulary used by the trained captioning model.

    Special tokens:

        <PAD>   = 0
        <START> = 1
        <END>   = 2
        <UNK>   = 3
    """

    def __init__(self, vocabulary_path):
        """
        Load the saved vocabulary.

        Args:
            vocabulary_path:
                Path to vocabulary.json
        """

        with open(
            vocabulary_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        # ----------------------------------------------------
        # Load the exact format used by the trained project.
        # ----------------------------------------------------

        self.word_to_idx = {
            str(word): int(index)
            for word, index in data["word_to_id"].items()
        }

        # ----------------------------------------------------
        # Load reverse mapping.
        # ----------------------------------------------------

        self.idx_to_word = {
            int(index): str(word)
            for index, word in data["id_to_word"].items()
        }

        # ----------------------------------------------------
        # Vocabulary metadata
        # ----------------------------------------------------

        self.min_freq = data.get("min_freq", None)

        self.size = int(
            data.get(
                "vocab_size",
                len(self.word_to_idx)
            )
        )

    # ========================================================
    # Word → Token ID
    # ========================================================

    def word_to_index(self, word):
        """
        Convert a word into its token ID.

        Unknown words become <UNK>.
        """

        return self.word_to_idx.get(
            word,
            self.word_to_idx["<UNK>"]
        )

    # ========================================================
    # Token ID → Word
    # ========================================================

    def index_to_word(self, index):
        """
        Convert a token ID into a word.

        Unknown IDs become <UNK>.
        """

        return self.idx_to_word.get(
            int(index),
            "<UNK>"
        )

    # ========================================================
    # Token IDs → Words
    # ========================================================

    def decode(self, token_ids):
        """
        Convert token IDs into a list of words.

        Special tokens are handled as follows:

            <START> → ignored
            <END>   → stop decoding
            <PAD>   → ignored
        """

        words = []

        for token_id in token_ids:

            word = self.index_to_word(
                token_id
            )

            if word == "<START>":
                continue

            if word == "<END>":
                break

            if word == "<PAD>":
                continue

            words.append(word)

        return words

    # ========================================================
    # Token IDs → Sentence
    # ========================================================

    def decode_sentence(self, token_ids):
        """
        Convert token IDs into a readable sentence.
        """

        return " ".join(
            self.decode(token_ids)
        )

    # ========================================================
    # Words → Token IDs
    # ========================================================

    def encode(self, words):
        """
        Convert a list of words into token IDs.
        """

        return [
            self.word_to_index(word)
            for word in words
        ]

    # ========================================================
    # Special Token IDs
    # ========================================================

    @property
    def pad_id(self):
        return self.word_to_idx["<PAD>"]

    @property
    def start_id(self):
        return self.word_to_idx["<START>"]

    @property
    def end_id(self):
        return self.word_to_idx["<END>"]

    @property
    def unk_id(self):
        return self.word_to_idx["<UNK>"]


# ============================================================
# Quick Vocabulary Test
# ============================================================

if __name__ == "__main__":

    vocabulary_path = "../config/vocabulary.json"

    vocab = Vocabulary(
        vocabulary_path
    )

    print(
        "Vocabulary size:",
        vocab.size
    )

    print(
        "<PAD>   :",
        vocab.pad_id
    )

    print(
        "<START> :",
        vocab.start_id
    )

    print(
        "<END>   :",
        vocab.end_id
    )

    print(
        "<UNK>   :",
        vocab.unk_id
    )

    # Test a few known words
    test_words = [
        "a",
        "man",
        "dog",
        "running"
    ]

    print("\nWord → ID:")

    for word in test_words:

        print(
            f"{word:10s} → "
            f"{vocab.word_to_index(word)}"
        )