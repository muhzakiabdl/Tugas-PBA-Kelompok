# model.py
import torch
import torch.nn as nn

class FastTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, embedding_matrix=None, padding_idx=0):
        super().__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_matrix), 
                freeze=True,
                padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        output = self.fc(embedded)
        return output