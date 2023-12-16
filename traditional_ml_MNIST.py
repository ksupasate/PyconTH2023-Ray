import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import time

# Force TensorFlow to use only the CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Define a simple neural network model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Traditional sequential training and Timer
start_time = time.time()

model = create_model()
model.fit(x_train, y_train, epochs=100)
results = model.evaluate(x_test, y_test)

end_time = time.time()
print("\nTraditional Sequential Results:", results)
print("Time taken:", end_time - start_time, "seconds")