import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from tensorflow.keras import layers, models

df = pd.read_csv("ArtiFact_240K/metadata.csv")

train_df = df[df['image_path'].str.startswith('ArtiFact_240K/train')]
# validate_df = df[df['image_path'].str.startswith('ArtiFact_240K/validation')]
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


train_array = train_df.to_numpy()
train_array = shuffle(train_array, random_state=42)
X_train = train_array[:, 0]
y_train = y_encoder(train_array)



def image_batch_generator(X_train, y_train, batch_size=32):
    while True:  
        for i in range(0, len(X_train), batch_size):  
            batch_paths = X_train[i:i + batch_size]
            batch_labels = y_train[i:i + batch_size]

            batch_images = []

            for path in batch_paths:
                img = plt.imread(path)
                img = img / 255.0
                batch_images.append(img)

            yield np.array(batch_images), np.array(batch_labels)




model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    # layers.Dense(256, activation='relu'),
    # layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    image_batch_generator(X_train, y_train, batch_size=32), 
    steps_per_epoch=len(X_train) // 32, 
    epochs=6
)


model.save('imageDetect3.keras')

