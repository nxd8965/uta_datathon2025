import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models

df = pd.read_csv("ArtiFact_240K/metadata.csv")

# train_df = df[df['image_path'].str.startswith('ArtiFact_240K/train')]
validate_df = df[df['image_path'].str.startswith('ArtiFact_240K/validation')]
# test_df = df[df['image_path'].str.startswith('ArtiFact_240K/test')]


def y_encoder(curr_arr):
    y = []
    for row in (curr_arr):
        y_val = 0
        if row[2] == "human_faces":
            y_val+=1
        elif row[2] == "animals":
            y_val+=2
        y_val += (int(row[1]))*3
        y.append(y_val)
    return y


validate_array = validate_df.to_numpy()
X_validate = validate_array[:, 0]
y_validate = y_encoder(validate_array)

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



y_pred = predict_in_batches(X_validate, model)

accuracy = accuracy_score(y_validate, y_pred)

print(accuracy)

