import torch
import torch.nn as nn
import torch.optim as optim

from models.ann_model import ANN
from data.load_data import train_loader, test_loader

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize model
model = ANN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Training parameters
epochs = 50
patience = 5

train_losses = []
test_accuracies = []

best_acc = 0
counter = 0

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

        running_loss += loss.item() * images.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)

    # Evaluate
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

    # Early stopping + save best model
    if accuracy > best_acc:
        best_acc = accuracy
        counter = 0
        model.save("results/models/ann_best.pth")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered.")
        break

print(f"Best Accuracy: {best_acc:.2f}%")

# Save last model (optional but useful)
model.save("results/models/ann_last.pth")

print("Training finished.")