# trainer.py
from time import time
import torch
from torch import nn, optim
from time import time

def train_model(model, train_loader, test_loader, config):
    device = config.DEVICE
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_accuracies, test_accuracies = [], []
    start_time = time()

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        correct, total = 0, 0

        for batch in train_loader:
            texts, labels = batch['text'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        # Evaluate on test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                texts, labels = batch['text'].to(device), batch['label'].to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Accuracy: {train_accuracy:.2f}% | Test Accuracy: {test_accuracy:.2f}%")

    end_time = time()
    total_time = end_time - start_time

    return {
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "total_time": total_time
    }