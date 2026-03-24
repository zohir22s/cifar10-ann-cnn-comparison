import torch
import numpy as np

from models.ann_model import ANN
from data.load_data import test_loader
from evaluation.metrics import accuracy, precision, recall, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    ann = ANN.from_pretrained("results/models/ann.pth", device)

    results = evaluate_model(ann)

    print("ANN Results")
    print("Accuracy :", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall   :", results["recall"])
    print("F1 Score :", results["f1"])