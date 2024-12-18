import torch
from torch.utils.data import DataLoader
from config import Config
from data_loader import load_dataset
from modeltransformer import TransformerModel
from trainer import train_model

def main():
    config = Config()

    # Load dataset
    print("Loading dataset...")
    train_dataset, test_dataset, word_to_idx, embedding_matrix = load_dataset(config)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Vocabulary size: {len(word_to_idx)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    print("Data loaders created.")

    # Initialize model
    print("Initializing model...")
    model = TransformerModel(
        vocab_size=len(word_to_idx),
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=config.NUM_CLASSES,
        max_seq_len=config.MAX_SEQ_LEN,
        embedding_matrix=embedding_matrix
    ).to(config.DEVICE)
    print("Model initialized.")

    # Train model
    print("Training model...")
    results = train_model(model, train_loader, test_loader, config)
    print("Training completed.")

    # Optionally, print training results
    print("\nFinal Results:")
    print(f"Final Train Accuracy: {results['train_accuracies'][-1]:.2f}%")
    print(f"Final Test Accuracy: {results['test_accuracies'][-1]:.2f}%")

if __name__ == '__main__':
    main()
