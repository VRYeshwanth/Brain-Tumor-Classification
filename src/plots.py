import matplotlib.pyplot as plt

# Plots Training v/s validation for 1 metric
def plot_metric(history, metric_name):
    epoch_range = history["epoch"]
    train_data = history[f"train_{metric_name}"]
    val_data = history[f"val_{metric_name}"]

    plt.title(f"Training v/s Validation {metric_name.capitalize()}")
    plt.xlabel("Epochs")
    plt.ylabel(f"{metric_name.capitalize()}")

    plt.plot(epoch_range, train_data, label=f"Training {metric_name.capitalize()}")
    plt.plot(epoch_range, val_data, label=f"Validation {metric_name.capitalize()}")

    plt.legend()
    plt.show()

# Plots Training v/s Validation plots for all the 4 metrics
def plot_all_metrics(history):
    metrics = ["loss", "accuracy", "precision", "recall"]
    epoch_range = history["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):

        train_data = history[f"train_{metric}"]
        val_data = history[f"val_{metric}"]

        ax.plot(epoch_range, train_data, label=f"Training {metric.capitalize()}")
        ax.plot(epoch_range, val_data, label=f"Validation {metric.capitalize()}")

        ax.set_title(f"Training vs Validation {metric.capitalize()}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric.capitalize())

        ax.legend()

    fig.suptitle("Model Training Performance", fontsize=16)

    plt.tight_layout()
    plt.show()