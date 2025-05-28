import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Path to model and data
data_dir = 'plantvillage_data'
model_path = 'plant_disease_model.keras'

# Image parameters (update if your model uses different size)
img_height = 128
img_width = 128
batch_size = 32

# Load the model
model = keras.models.load_model(model_path)

# Print model summary for debugging input shape
print('Model Summary:')
model.summary()

# Prepare the data generator for validation/test set
# We'll use a small validation split from the data directory
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Predict
Y_pred = model.predict(val_generator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

# True labels
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Print and save classification report
report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
print('Classification Report:')
print(report)
with open('classification_report.txt', 'w') as f:
    f.write(report)

# Print and save confusion matrix
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)
np.savetxt('confusion_matrix.txt', cm, fmt='%d')

# Analyze misclassified images
filenames = val_generator.filenames
misclassified = []
for i, (true, pred) in enumerate(zip(y_true, y_pred)):
    if true != pred:
        misclassified.append({
            'filename': filenames[i],
            'true_label': class_labels[true],
            'predicted_label': class_labels[pred]
        })
if misclassified:
    df_mis = pd.DataFrame(misclassified)
    df_mis.to_csv('misclassified_images.csv', index=False)
    print(f"Saved {len(misclassified)} misclassified images to misclassified_images.csv")
else:
    print("No misclassified images found.")
