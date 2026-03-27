import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from models.cnn_advanced import CNN_Advanced


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(
        root="../data/raw",
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root="../data/raw",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = CNN_Advanced().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30
    )

    epochs = 20
    best_accuracy = 0

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        # evaluation
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        avg_val_loss = val_loss / len(test_loader)

        scheduler.step()

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                model.state_dict(),
                "results/models/cnn_advanced_best.pth"
            )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {total_loss:.2f} | "
            f"Val Loss: {avg_val_loss:.2f} | "
            f"Best: {best_accuracy:.2f}%"
        )

    print(f"\nBest accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()