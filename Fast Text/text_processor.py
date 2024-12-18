# text_processor.py
import os
import pandas as pd
import torch
from collections import Counter
import numpy as np
import re

class TextPreprocessor:
    """
    Class untuk memproses teks dan membuat embedding matrix dari GloVe.
    """
    def __init__(self, config):
        self.config = config
        self.word_to_vec = self.load_glove_embeddings(config.GLOVE_PATH)

    @staticmethod
    def load_glove_embeddings(glove_path):
        """
        Memuat GloVe embeddings dari file.
        """
        embeddings = {}
        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.array(values[1:], dtype='float32')
                    embeddings[word] = vector
        except Exception as e:
            print(f"Error loading GloVe: {e}")
        return embeddings

    @staticmethod
    def clean_text(text):
        """
        Membersihkan teks dengan menghapus karakter non-alfabet dan mengubah menjadi huruf kecil.
        """
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        return text

    def create_embedding_matrix(self, word_to_idx):
        """
        Membuat embedding matrix dari GloVe.
        """
        embedding_matrix = np.zeros((len(word_to_idx), self.config.EMBEDDING_DIM))
        for word, idx in word_to_idx.items():
            if word in self.word_to_vec:
                embedding_matrix[idx] = self.word_to_vec[word]
            else:
                embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, self.config.EMBEDDING_DIM)
        return embedding_matrix

    @staticmethod
    def create_vocab(texts, max_size=20000):
        """
        Membuat vocabulary dari teks dengan ukuran maksimum tertentu.
        """
        all_words = ' '.join(texts).split()
        word_counts = Counter(all_words)
        vocab = ['<pad>', '<unk>'] + [word for word, _ in word_counts.most_common(max_size-2)]
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        return word_to_idx, vocab