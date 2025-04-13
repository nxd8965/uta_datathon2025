import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
import os
import csv

df = pd.read_csv("ArtiFact_240K/metadata.csv")

# train_df = df[df['image_path'].str.startswith('ArtiFact_240K/train')]
# validate_df = df[df['image_path'].str.startswith('ArtiFact_240K/validation')]
# test_df = df[df['image_path'].str.startswith('ArtiFact_240K/test')]

def y_decoder(curr_arr):
    y_final = []
    for value in (curr_arr):
        match value:
            case 0:
                y_final.append([0, "vehicles"])
            case 1:
                y_final.append([0, "human_faces"])
            case 2:
                y_final.append([0, "animals"])
            case 3:
                y_final.append([1, "vehicles"])
            case 4:
                y_final.append([1, "human_faces"])
            case 5:
                y_final.append([1, "animals"])

    return y_final



def get_jpg_filepaths(folder_path):
    return [os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".jpg")]


test_arr = get_jpg_filepaths("ArtiFact_240K/test/")

model = models.load_model('imageDetect3.keras')


def predict_in_batches(filepaths, model, batch_size=256):
    num_samples = len(filepaths)
    predictions = []

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_paths = filepaths[start:end]

        batch_images = []
        for path in batch_paths:
            img = plt.imread(path)
            img = img / 255.0
            batch_images.append(img)

        batch_array = np.stack(batch_images) 

        batch_preds = model.predict(batch_array, verbose=0)
        batch_classes = np.argmax(batch_preds, axis=1)

        predictions.extend(batch_classes)

    return np.array(predictions)



y_pred = predict_in_batches(test_arr, model)

predictions = y_decoder(y_pred)

def save_predictions_to_csv(filepaths, predictions, output_csv="test.csv"):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label", "class"])  # CSV header
        for path, (label, class_name) in zip(filepaths, predictions):
            filename = os.path.basename(path)
            writer.writerow([filename, label, class_name])


save_predictions_to_csv(test_arr, predictions)


