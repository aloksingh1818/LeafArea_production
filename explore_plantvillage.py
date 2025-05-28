import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from PIL import Image
from collections import Counter

# Path to PlantVillage dataset
DATASET_PATH = "/workspaces/LeafArea_production/plantvillage_data"

# Get all image file paths and their labels
image_paths = glob(os.path.join(DATASET_PATH, '*', '*.JPG'))
labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

# Show class distribution
label_counts = Counter(labels)
print("Number of classes:", len(label_counts))
print("Class distribution:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# Show a few sample images
plt.figure(figsize=(12, 8))
for i, img_path in enumerate(np.random.choice(image_paths, 6, replace=False)):
    img = Image.open(img_path)
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(os.path.basename(os.path.dirname(img_path)))
    plt.axis('off')
plt.tight_layout()
plt.show()

# Prepare for model training: create train/val split
def prepare_data(image_paths, labels, val_split=0.2):
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=val_split, stratify=labels, random_state=42)
    return train_paths, val_paths, train_labels, val_labels

train_paths, val_paths, train_labels, val_labels = prepare_data(image_paths, labels)
print(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")
