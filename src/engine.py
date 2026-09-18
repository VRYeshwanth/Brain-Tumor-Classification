import torch
import pandas as pd

def train_one_epoch(model, dataloader, loss_fn, optimizer, metrics, device):
    model.train()

    running_loss = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        for metric in metrics.values():
            metric.update(output, labels)

    epoch_loss = running_loss / len(dataloader)

    epoch_metrics = {}
    for name, metric in metrics.items():
        epoch_metrics[name] = metric.compute().item()
        metric.reset()

    return epoch_loss, epoch_metrics


def validate_one_epoch(model, dataloader, loss_fn, metrics, device):
    model.eval()
    running_loss = 0

    with torch.inference_mode():

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss = loss_fn(output, labels)
            running_loss += loss.item()

            for metric in metrics.values():
                metric.update(output, labels)

    epoch_loss = running_loss / len(dataloader)
    
    epoch_metrics = {}
    for name, metric in metrics.items():
        epoch_metrics[name] = metric.compute().item()
        metric.reset()
    
    return epoch_loss, epoch_metrics


def fit(model, train_loader, val_loader, loss_fn, optimizer, train_metrics, val_metrics, epochs, device, file_path):
    history = []
    best_val_acc = 0

    for epoch in range(epochs):
        train_loss, train_results = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=train_metrics,
            device=device
        )

        val_loss, val_results = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            metrics=val_metrics,
            device=device
        )

        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_results["accuracy"],
            "train_precision": train_results["precision"],
            "train_recall": train_results["recall"],

            "val_loss": val_loss,
            "val_accuracy": val_results["accuracy"],
            "val_precision": val_results["precision"],
            "val_recall": val_results["recall"],
        }
        history.append(epoch_results)

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"Training Loss: {train_loss:.4f} | Training Accuracy: {train_results['accuracy'] * 100:.2f}% | Training Precision: {train_results['precision'] * 100:.2f}% | Training Recall: {train_results['recall'] * 100:.2f}%")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_results['accuracy'] * 100:.2f}% | Validation Precision: {val_results['precision'] * 100:.2f}% | Validation Recall: {val_results['recall'] * 100:.2f}%")


        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            torch.save(
                model.state_dict(),
                file_path
            )

            print(f"Best model saved! Val Acc: {best_val_acc * 100:.2f}%")

        print("\n")

    df = pd.DataFrame(history)

    return df
