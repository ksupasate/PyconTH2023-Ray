import ray
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

ray.init()

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
@ray.remote
class IrisModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')  # 3 output units for 3 classes of Iris
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_subset, y_subset):
        self.model.fit(x_subset, y_subset, epochs=100, verbose=False)
        return self.model.evaluate(X_test, y_test)

# Split dataset for distributed processing
num_chunks = 16  # Adjust based on your cluster setup
data_chunks = (np.array_split(X_train, num_chunks), np.array_split(y_train, num_chunks))

# Measure time for distributed model training
start_time = time.time()

models = [IrisModel.remote(X_train.shape[1]) for _ in range(num_chunks)]
result_ids = [models[i].train.remote(x, y) for i, (x, y) in enumerate(zip(*data_chunks))]
results = ray.get(result_ids)

end_time = time.time()
ray.shutdown()

print("Ray Framework Results:", results)
print("Time taken:", end_time - start_time, "seconds")