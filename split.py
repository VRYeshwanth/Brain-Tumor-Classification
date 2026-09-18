import os
import random
import shutil

source_dir = r".\archive"

training_dir = os.path.join(source_dir, "Training")
testing_dir = os.path.join(source_dir, "Testing")

output_dir = r".\dataset"

train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")



classes = os.listdir(training_dir)
classes = [
    cls for cls in classes
    if os.path.isdir(os.path.join(training_dir, cls))
]

print("Classes:", classes)


for split_dir in [train_dir, val_dir, test_dir]:

    os.makedirs(split_dir, exist_ok=True)

    for cls in classes:
        os.makedirs(
            os.path.join(split_dir, cls),
            exist_ok=True
        )


random.seed(42)

for cls in classes:

    class_path = os.path.join(training_dir, cls)

    images = [
        file
        for file in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, file))
    ]

    random.shuffle(images)

    split_index = int(0.8 * len(images))

    train_images = images[:split_index]
    val_images = images[split_index:]

    print(
        f"{cls}: "
        f"{len(train_images)} train, "
        f"{len(val_images)} validation"
    )

    for image in train_images:
        src = os.path.join(class_path, image)
        dst = os.path.join(train_dir, cls, image)

        shutil.copy2(src, dst)

    for image in val_images:
        src = os.path.join(class_path, image)
        dst = os.path.join(val_dir, cls, image)

        shutil.copy2(src, dst)


for cls in classes:

    source_class_path = os.path.join(testing_dir, cls)
    destination_class_path = os.path.join(test_dir, cls)

    images = [
        file
        for file in os.listdir(source_class_path)
        if os.path.isfile(os.path.join(source_class_path, file))
    ]

    print(f"{cls}: {len(images)} test")

    for image in images:

        src = os.path.join(source_class_path, image)
        dst = os.path.join(destination_class_path, image)

        shutil.copy2(src, dst)


print("\nDataset split completed!")