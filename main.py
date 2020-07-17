from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential


def get_model():
    model_new = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(150, 150, 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation="sigmoid")
    ])

    return model_new
