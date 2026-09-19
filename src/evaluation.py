from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix
)
import torch
import matplotlib.pyplot as plt
import math

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


def get_misclassified_images(model, dataloader, dataset, device, class_names):
    model.eval()

    misclassified = []
    index = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            confidences, predictions = torch.max(
                probabilities, dim=1
            )

            for i in range(len(images)):

                if predictions[i] != labels[i]:
                    misclassified.append({
                        "image": images[i].cpu(),
                        "true_label": class_names[labels[i].item()],
                        "predicted_label": class_names[predictions[i].item()],
                        "confidence": confidences[i].item(),
                        "path": dataset.samples[index][0]
                    })

                index += 1
    return misclassified


mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

def show_misclassified_images(misclassified, n=20):

    n = min(n, len(misclassified))

    rows = math.ceil(n / 5)

    plt.figure(figsize=(15, rows * 3))

    for i in range(n):

        item = misclassified[i]

        image = item["image"]
        image = image * std[:, None, None] + mean[:, None, None]

        image = image.permute(1, 2, 0)
        image = image.clamp(0, 1)

        plt.subplot(rows, 5, i + 1)
        plt.imshow(image)

        plt.title(
            f"True: {item['true_label']}\n"
            f"Pred: {item['predicted_label']}\n"
            f"Conf: {item['confidence']:.2%}"
        )

        plt.axis("off")

    plt.suptitle("Misclassified Test Images", fontsize=20)

    plt.tight_layout()
    plt.show()