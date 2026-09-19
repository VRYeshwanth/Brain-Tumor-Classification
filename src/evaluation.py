from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix
)
import torch
import matplotlib.pyplot as plt

def evaluateModel(model, dataloader, device):

    accuracy = MulticlassAccuracy(num_classes=4).to(device)
    precision = MulticlassPrecision(num_classes=4, average="macro").to(device)
    recall = MulticlassRecall(num_classes=4, average="macro").to(device)

    cm = MulticlassConfusionMatrix(num_classes=4).to(device)

    model.eval()
    with torch.inference_mode():

        for images, labels in dataloader:

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            accuracy.update(outputs, labels)
            precision.update(outputs, labels)
            recall.update(outputs, labels)

            cm.update(outputs, labels)

    results = {
        "accuracy": accuracy.compute().item(),
        "precision": precision.compute().item(),
        "recall": recall.compute().item()
    }

    return results, cm


def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    cm.plot(labels=class_names, ax=ax)

    ax.set_title(f"{title}", fontsize=16, pad=15)
    ax.set_xlabel("Predicted class", fontsize=13)
    ax.set_ylabel("True class", fontsize=13)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    plt.setp(
        ax.get_yticklabels(),
        rotation=0
    )

    plt.tight_layout()
    plt.show()