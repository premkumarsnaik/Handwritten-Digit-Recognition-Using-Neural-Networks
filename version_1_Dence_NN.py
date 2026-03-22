import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------
# Load Dataset
# ----------------------------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ----------------------------------------
# Normalize pixel values (0-255 -> 0-1)
# ----------------------------------------

x_train = x_train / 255.0
x_test = x_test / 255.0

# ----------------------------------------
# Build Neural Network
# ----------------------------------------

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),     # 28x28 -> 784
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')    # 10 digits
])

# ----------------------------------------
# Compile Model
# ----------------------------------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------------------
# Train Model
# ----------------------------------------

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# ----------------------------------------
# Evaluate Accuracy
# ----------------------------------------

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"\nTest Accuracy: {test_acc:.4f}")

# ----------------------------------------
# Predict Sample Digit
# ----------------------------------------

prediction = model.predict(x_test)

sample_index = 0

print(f"\nPredicted Digit: {np.argmax(prediction[sample_index])}")
print(f"Actual Digit: {y_test[sample_index]}")

# ----------------------------------------
# Display Sample Image
# ----------------------------------------

plt.imshow(x_test[sample_index], cmap='gray')
plt.title(f"Predicted: {np.argmax(prediction[sample_index])}")
plt.axis('off')
plt.show()