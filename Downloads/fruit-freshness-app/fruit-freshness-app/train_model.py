"""
train_model.py
----------------
Trains a fresh-vs-rotten fruit classifier using transfer learning
(a pretrained ResNet18 with a new final layer).

HOW TO RUN
    python train_model.py

BEFORE YOU RUN IT
    Your folders must look like this:

    dataset/
        train/
            freshapples/
            freshbanana/
            freshoranges/
            rottenapples/
            rottenbanana/
            rottenoranges/
        test/
            freshapples/
            freshbanana/
            freshoranges/
            rottenapples/
            rottenbanana/
            rottenoranges/

    (See README.md -> "Step 4: Get the dataset" for where to download this.)
    You can use fewer or different folder names -- the script automatically
    detects whatever subfolders exist. Just make sure every folder name
    starts with either "fresh" or "rotten" so the app can tell them apart.

WHAT IT PRODUCES
    model/fruit_model.pth       <- the trained weights
    model/class_names.json      <- the list of class names, in the right order
    model/training_curve.png    <- accuracy/loss plot
"""

import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# 1. Settings -- feel free to tweak these
# ---------------------------------------------------------------------------
DATA_DIR = "dataset/dataset"
MODEL_DIR = "model"
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 0.001
IMAGE_SIZE = 224  # ResNet18 expects 224x224 images

os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cpu":
    print("No GPU found -- training will work but be slower. "
          "If you're on Google Colab, go to Runtime > Change runtime type > GPU.")

# ---------------------------------------------------------------------------
# 2. Data loading & augmentation
# ---------------------------------------------------------------------------
# Training images get randomly flipped/rotated slightly so the model
# generalizes better. Test images are only resized -- we want to evaluate
# on realistic, untouched images.
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

image_datasets = {
    split: datasets.ImageFolder(os.path.join(DATA_DIR, split), data_transforms[split])
    for split in ["train", "test"]
}

dataloaders = {
    split: DataLoader(
        image_datasets[split],
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=2,
    )
    for split in ["train", "test"]
}

dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "test"]}
class_names = image_datasets["train"].classes
print(f"Found {len(class_names)} classes: {class_names}")
print(f"Training images: {dataset_sizes['train']} | Test images: {dataset_sizes['test']}")

# Save the class names now -- the web app needs this exact list, in this
# exact order, to turn a prediction number back into a class name.
with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)

# ---------------------------------------------------------------------------
# 3. Build the model: pretrained ResNet18 + a new final layer
# ---------------------------------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze every existing layer so we only train the new final layer
# (and the few layers we unfreeze below). This is "transfer learning":
# we reuse everything ResNet18 already learned about edges, shapes and
# textures from 1.2 million ImageNet photos, and only teach it the new task.
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last residual block too -- gives the model a bit more
# flexibility to adapt to fruit-specific texture (bruising, mold) without
# retraining the whole network.
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace the final classification layer. The original layer outputs
# 1000 ImageNet classes; ours needs to output len(class_names) classes.
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
best_acc = 0.0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 30)

    for phase in ["train", "test"]:
        model.train() if phase == "train" else model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

        history[f"{phase}_loss"].append(epoch_loss)
        history[f"{phase}_acc"].append(epoch_acc)

        print(f"{phase:>5} | loss: {epoch_loss:.4f} | accuracy: {epoch_acc:.4f}")

        if phase == "test" and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fruit_model.pth"))
            print(f"      -> new best model saved ({best_acc:.4f} accuracy)")

    scheduler.step()

elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
print(f"Best test accuracy: {best_acc:.4f}")

# ---------------------------------------------------------------------------
# 5. Plot and save the training curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["test_loss"], label="test")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["test_acc"], label="test")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curve.png"))
print(f"Saved training curve to {MODEL_DIR}/training_curve.png")
print(f"\nDone! Your model is ready at {MODEL_DIR}/fruit_model.pth")
print("Next step: run the web app with `python app.py`")
