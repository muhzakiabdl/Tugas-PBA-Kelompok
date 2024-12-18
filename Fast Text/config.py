# config.py
import torch

class Config:
    # Konfigurasi Dataset
    DATASET_PATH = 'C:/Users/andre/OneDrive/Desktop/UJI_Tugas PBA Fast Text_121450004 - Copy/data'
    NUM_CLASSES = 4
    CLASSES = ['World', 'Sports', 'Business', 'Sci/Tech']

    # Word Embedding Configuration
    EMBEDDING_TYPE = 'glove'  # Gunakan GloVe sebagai embedding
    EMBEDDING_DIM = 100  # Dimensi embedding GloVe
    GLOVE_PATH = 'C:/Users/andre/OneDrive/Desktop/UJI_Tugas PBA Fast Text_121450004 - Copy/data/glove.6B.100d.txt'  # Path ke file GloVe

    # Hyperparameter Model
    VOCAB_SIZE = 20000  # Ukuran maksimum vocabulary

    # Konfigurasi Pelatihan
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
