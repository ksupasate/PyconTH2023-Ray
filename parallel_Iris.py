import multiprocessing
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

# Load Iris data
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One hot encoding of the target variable
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple neural network model for the Iris dataset
def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 output units for 3 classes of Iris
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data):
    (x_subset, y_subset), model = data
    model.fit(x_subset, y_subset, epochs=100, verbose=False)
    return model.evaluate(X_test, y_test)

# Parallel Computing and Timer
num_processors = multiprocessing.cpu_count()
print(num_processors)
data_chunks = (np.array_split(X_train, num_processors), np.array_split(y_train, num_processors))

start_time = time.time()

pool = multiprocessing.Pool(processes=num_processors)
results = pool.map(train_model, [(d, create_model(X_train.shape[1])) for d in zip(*data_chunks)])
pool.close()
pool.join()

end_time = time.time()

print("\nParallel Computing Results:", results)
print("Time taken:", end_time - start_time, "seconds")