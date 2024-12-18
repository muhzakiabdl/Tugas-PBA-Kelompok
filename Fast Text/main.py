# main.py
from torch.utils.data import DataLoader
from config import Config
from data_loader import load_dataset
from model import FastTextModel
from trainer import train_model

def main():
    config = Config()
    train_dataset, test_dataset, word_to_idx, embedding_matrix = load_dataset(config)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    model = FastTextModel(
        vocab_size=len(word_to_idx),
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=config.NUM_CLASSES,
        embedding_matrix=embedding_matrix
    ).to(config.DEVICE)

    results = train_model(model, train_loader, test_loader, config)

    print("\nHasil Akhir:")
    print(f"Akurasi Terakhir - Train: {results['train_accuracies'][-1]:.2f}%")
    print(f"Akurasi Terakhir - Test: {results['test_accuracies'][-1]:.2f}%")
    print(f"Waktu Pelatihan Total: {results['total_time']:.2f} detik")

if __name__ == "__main__":
    main()