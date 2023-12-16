import ray
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import time

ray.init()

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Define a simple neural network model
@ray.remote
class MNISTModel:
    def __init__(self):
        self.model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_subset, y_subset):
        self.model.fit(x_subset, y_subset, epochs=100, verbose=False)
        return self.model.evaluate(x_test, y_test)

# Split dataset for distributed processing
num_chunks = 16  # Adjust based on your cluster setup
data_chunks = (np.array_split(x_train, num_chunks), np.array_split(y_train, num_chunks))

# Measure the time for distributed model training
start_time = time.time()

models = [MNISTModel.remote() for _ in range(num_chunks)]
result_ids = [models[i].train.remote(x, y) for i, (x, y) in enumerate(zip(*data_chunks))]
results = ray.get(result_ids)

end_time = time.time()
ray.shutdown()

print("Ray Framework Results:", results)
print("Time taken:", end_time - start_time, "seconds")