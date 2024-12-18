import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import re
import numpy as np
from collections import Counter

class TextPreprocessor:
    def __init__(self, config):
        self.config = config
        self.embeddings_index = self.load_glove_embeddings(config.GLOVE_PATH)

    def load_glove_embeddings(self, glove_path):
        """
        Memuat GloVe embeddings dari file .txt
        """
        embeddings_index = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    @staticmethod
    def clean_text(text):
        """
        Membersihkan teks dengan menghapus karakter non-alfabet dan mengubahnya menjadi huruf kecil.
        """
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        return text

    def create_embedding_matrix(self, word_to_idx):
        """
        Membuat embedding matrix dari word embeddings GloVe.
        """
        embedding_matrix = np.zeros((len(word_to_idx), self.config.EMBEDDING_DIM))
        for word, idx in word_to_idx.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
            else:
                embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, self.config.EMBEDDING_DIM)
        return embedding_matrix

    @staticmethod
    def create_vocab(texts, max_size=20000):
        """
        Membuat vocabulary dari teks.
        """
        all_words = ' '.join(texts).split()
        word_counts = Counter(all_words)
        vocab = ['<pad>', '<unk>'] + [word for word, _ in word_counts.most_common(max_size - 2)]
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        return word_to_idx, vocab


class AGNewsDataset(Dataset):
    def __init__(self, dataframe, word_to_idx, max_length=100):
        self.texts = dataframe[1].apply(TextPreprocessor.clean_text)
        self.labels = dataframe[0].astype(int) - 1
        self.word_to_idx = word_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        # Tokenize and pad/truncate text
        tokens = text.split()[:self.max_length]
        indexed_tokens = [self.word_to_idx.get(token, self.word_to_idx['<unk>']) for token in tokens]
        if len(indexed_tokens) < self.max_length:
            indexed_tokens += [self.word_to_idx['<pad>']] * (self.max_length - len(indexed_tokens))

        return {
            'text': torch.tensor(indexed_tokens, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_dataset(config):
    """
    Memuat dataset dan membuat vocabulary serta embedding matrix.
    """
    train_path = os.path.join(config.DATASET_PATH, config.TRAIN_FILE)
    test_path = os.path.join(config.DATASET_PATH, config.TEST_FILE)

    # Membaca dataset
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # Membuat vocabulary dan embedding matrix
    preprocessor = TextPreprocessor(config)
    word_to_idx, vocab = preprocessor.create_vocab(train_df[1], max_size=config.VOCAB_SIZE)
    embedding_matrix = preprocessor.create_embedding_matrix(word_to_idx)

    # Membuat dataset
    train_dataset = AGNewsDataset(train_df, word_to_idx, config.MAX_SEQ_LEN)
    test_dataset = AGNewsDataset(test_df, word_to_idx, config.MAX_SEQ_LEN)

    return train_dataset, test_dataset, word_to_idx, embedding_matrix
