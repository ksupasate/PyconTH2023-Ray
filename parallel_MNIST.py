import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import time

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
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

# Define training config
def train_model(data):
    (x_subset, y_subset), model = data
    model.fit(x_subset, y_subset, epochs=100, verbose=False)
    return model.evaluate(x_test, y_test)

# Parallel Computing and Timer
num_processors = multiprocessing.cpu_count()
data_chunks = (np.array_split(x_train, num_processors), np.array_split(y_train, num_processors))

start_time = time.time()

processes=num_processors
pool = multiprocessing.Pool(processes=num_processors)
print(processes)
results = pool.map(train_model, [(d, create_model()) for d in zip(*data_chunks)])
pool.close()
pool.join()

end_time = time.time()

print("\nParallel Computing Results:", results)
print("Time taken:", end_time - start_time, "seconds")