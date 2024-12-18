import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, max_seq_len, embedding_matrix):
        super(TransformerModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        nhead = 5  # Jumlah kepala perhatian, pastikan EMBEDDING_DIM % nhead == 0
        assert self.embedding_dim % nhead == 0, "embedding_dim must be divisible by nhead"

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=512, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Fully connected layer for classification
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Average pooling over sequence length
        x = self.fc(x)
        return x
