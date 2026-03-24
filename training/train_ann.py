import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from models.ann_model import ANN
from data.load_data import train_loader, test_loader

# Command-line arguments
parser = argparse.ArgumentParser(description="Train ANN on CIFAR-10")
parser.add_argument(
    "--epoch",
    type=int,
    default=10,  # default to 10 epochs
    help="Number of epochs to train (default: 10)"
)
args = parser.parse_args()
epochs = args.epoch


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"Training for {epochs} epochs")


# Model, loss, optimizer
model = ANN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)


    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")

# Save model
model.save("results/models/ann.pth")
print("Training finished. Model saved to results/models/ann.pth")