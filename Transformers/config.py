import torch

class Config:
    # Konfigurasi dataset
    DATASET_PATH = 'C:/Users/acer/Downloads/UJI_Tugas_PBA_Transformer_121450023'
    TRAIN_FILE = 'C:/Users/acer/Downloads/UJI_Tugas PBA Transformer_121450023/UJI_Tugas PBA Transformer_121450023/data/train.csv'
    TEST_FILE = 'C:/Users/acer/Downloads/UJI_Tugas PBA Transformer_121450023/UJI_Tugas PBA Transformer_121450023/data/test.csv'
    NUM_CLASSES = 10
    CLASSES = ['World', 'Sports', 'Business', 'Sci/Tech']

    # Konfigurasi embedding
    EMBEDDING_TYPE = 'glove'  # Menggunakan GloVe
    EMBEDDING_DIM = 100  # Ukuran embedding GloVe (100, 200, atau 300 tergantung file)
    GLOVE_PATH = 'C:/Users/acer/Downloads/UJI_Tugas PBA Transformer_121450023/UJI_Tugas PBA Transformer_121450023/data/glove.6B.100d.txt'
    NHEAD = 5
    # Hyperparameter model
    VOCAB_SIZE = 20000
    MAX_SEQ_LEN = 100

    # Konfigurasi pelatihan
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
