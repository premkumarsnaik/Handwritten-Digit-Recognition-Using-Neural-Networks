import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------
# Load MNIST Dataset
# ----------------------------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ----------------------------------------
# Normalize pixel values
# ----------------------------------------

x_train = x_train / 255.0
x_test = x_test / 255.0

# ----------------------------------------
# Reshape for CNN
# CNN expects 4D input:
# (samples, height, width, channels)
# ----------------------------------------

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ----------------------------------------
# Build CNN Model
# ----------------------------------------

model = models.Sequential([
    
    # First Convolution Layer
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    
    # Pooling Layer
    layers.MaxPooling2D((2,2)),
    
    # Second Convolution Layer
    layers.Conv2D(64, (3,3), activation='relu'),
    
    # Second Pooling Layer
    layers.MaxPooling2D((2,2)),
    
    # Flatten before Dense Layers
    layers.Flatten(),
    
    # Dense Hidden Layer
    layers.Dense(128, activation='relu'),
    
    # Output Layer (10 digits)
    layers.Dense(10, activation='softmax')
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
# Evaluate Model
# ----------------------------------------

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"\nCNN Test Accuracy: {test_acc:.4f}")

# ----------------------------------------
# Predict Sample Digit
# ----------------------------------------

predictions = model.predict(x_test)

sample_index = 0

predicted_digit = np.argmax(predictions[sample_index])

print(f"\nPredicted Digit: {predicted_digit}")
print(f"Actual Digit: {y_test[sample_index]}")

# ----------------------------------------
# Display Sample Image
# ----------------------------------------

plt.imshow(x_test[sample_index].reshape(28,28), cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.axis('off')
plt.show()

# ----------------------------------------
# Save Model
# ----------------------------------------

model.save("mnist_cnn_model.h5")