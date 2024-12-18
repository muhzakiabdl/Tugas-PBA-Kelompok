# data_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import re
from collections import Counter
import gensim
import numpy as np
from text_processor import TextPreprocessor


class AGNewsDataset(Dataset):
    """
    Dataset untuk AG News dengan preprocessing teks dan tokenisasi.
    """
    def __init__(self, dataframe, word_to_idx, config, max_length=100):
        self.preprocessor = TextPreprocessor(config)
        texts = dataframe[1].apply(self.preprocessor.clean_text)
        self.data = []
        for text, label in zip(texts, dataframe[0]):
            tokens = text.split()[:max_length]
            indexed_tokens = [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens]
            if len(indexed_tokens) < max_length:
                indexed_tokens += [word_to_idx['<pad>']] * (max_length - len(indexed_tokens))
            self.data.append({
                'text': torch.tensor(indexed_tokens, dtype=torch.long),
                'label': torch.tensor(int(label) - 1, dtype=torch.long)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_dataset(config, train_file='C:/Users/andre/OneDrive/Desktop/UJI_Tugas PBA Fast Text_121450004 - Copy/data/train.csv', test_file='C:/Users/andre/OneDrive/Desktop/UJI_Tugas PBA Fast Text_121450004 - Copy/data/test.csv'):
    """
    Memuat dataset dari file CSV, membuat vocabulary, dan dataset AGNews.
    """
    train_path = os.path.join(config.DATASET_PATH, train_file)
    test_path = os.path.join(config.DATASET_PATH, test_file)
    try:
        # Membaca dataset
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)

        # Membuat preprocessing dan vocabulary
        preprocessor = TextPreprocessor(config)
        word_to_idx, vocab = TextPreprocessor.create_vocab(train_df[1], max_size=config.VOCAB_SIZE)
        embedding_matrix = preprocessor.create_embedding_matrix(word_to_idx)

        # Membuat dataset AGNews
        train_dataset = AGNewsDataset(train_df, word_to_idx, config)
        test_dataset = AGNewsDataset(test_df, word_to_idx, config)

        return train_dataset, test_dataset, word_to_idx, embedding_matrix

    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None, None, None, None