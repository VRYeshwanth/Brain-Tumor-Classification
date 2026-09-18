from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall
)

def create_metrics(num_classes):

    metrics = {
        "accuracy": MulticlassAccuracy(num_classes=num_classes),
        "precision": MulticlassPrecision(
            num_classes=num_classes,
            average='macro'
        ),
        "recall": MulticlassRecall(
            num_classes=num_classes,
            average='macro'
        )
    }

    return metrics