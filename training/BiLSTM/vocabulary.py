"""
Vocabulary for BiLSTM Text Classification

This file defines the vocabulary used to convert AG News text into
integer token IDs that can be processed by the BiLSTM model.

The vocabulary is built from the training data using a minimum
frequency threshold. It includes special tokens for padding and
unknown words. Text is tokenized using lowercase whitespace
tokenization, and unseen words are mapped to the <UNK> token.

The vocabulary is used by the AGNewsDataset to convert text into
sequences of indices before they are passed to the BiLSTM.
"""

from collections import Counter


class Vocabulary:

    def __init__(self, min_freq=2):

        self.min_freq = min_freq

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        self.word_to_idx = {
            self.pad_token: 0,
            self.unk_token: 1,
        }

        self.idx_to_word = {
            0: self.pad_token,
            1: self.unk_token,
        }

    def tokenize(self, text):

        return text.lower().split()

    def build(self, texts):

        counter = Counter()

        for text in texts:

            tokens = self.tokenize(text)

            counter.update(tokens)

        for word, frequency in counter.items():

            if frequency >= self.min_freq:

                idx = len(self.word_to_idx)

                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word

    def encode(self, text):

        tokens = self.tokenize(text)

        return [
            self.word_to_idx.get(
                token,
                self.word_to_idx[self.unk_token]
            )
            for token in tokens
        ]

    def decode(self, indices):

        return [
            self.idx_to_word.get(
                idx,
                self.unk_token
            )
            for idx in indices
        ]

    def __len__(self):

        return len(self.word_to_idx)