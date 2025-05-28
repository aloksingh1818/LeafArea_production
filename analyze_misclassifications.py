import pandas as pd
import collections

# Load misclassified images
misclassified = pd.read_csv('misclassified_images.csv')

# Count most common confusions
confusions = misclassified.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
confusions = confusions.sort_values('count', ascending=False)

print("Top 10 most common misclassifications:")
print(confusions.head(10))

# Optionally, show a summary per true label
print("\nMisclassification summary per true label:")
summary = misclassified.groupby('true_label').size().reset_index(name='misclassified_count')
print(summary.sort_values('misclassified_count', ascending=False))
