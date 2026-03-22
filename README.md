# Handwritten Digit Recognition Using Neural Networks

This project implements handwritten digit recognition using the MNIST dataset with two different neural network approaches:

* **Version 1 → Dense Neural Network (Fully Connected Network)**
* **Version 2 → Convolutional Neural Network (CNN)**

The objective is to classify handwritten digits from **0 to 9** using deep learning.

## Dataset Used

The project uses the MNIST dataset, which contains handwritten digit images.

Dataset details:

* 60,000 training images
* 10,000 test images
* Image size: 28 × 28 grayscale pixels

The dataset is automatically loaded using TensorFlow:

```python id="0j0ktw"
from tensorflow.keras.datasets import mnist
```

## Project Versions

## Version 1 — Dense Neural Network

This version uses a fully connected neural network.

### Architecture

* Input Layer → 784 neurons (28×28 flattened image)
* Hidden Layer 1 → 128 neurons
* Hidden Layer 2 → 64 neurons
* Output Layer → 10 neurons

### Working Principle

The image is flattened into a 1D vector before being passed through dense layers.

### Expected Accuracy

```text id="imlz5q"
97% to 98%
```

### Advantages

* Easy to understand
* Good for learning neural network basics

### Limitation

Dense layers do not preserve spatial relationships between pixels.

## Version 2 — Convolutional Neural Network (CNN)

This version uses convolutional layers to learn image features.

### Architecture

* Convolution Layer 1 → 32 filters
* Max Pooling Layer
* Convolution Layer 2 → 64 filters
* Max Pooling Layer
* Flatten Layer
* Dense Layer → 128 neurons
* Output Layer → 10 neurons

### Working Principle

CNN learns local image features such as:

* edges
* curves
* strokes

before final classification.

### Expected Accuracy

```text id="wwk9m1"
99%+
```

### Advantages

* Better feature extraction
* Higher accuracy
* Standard image recognition approach

## Why CNN Performs Better

Dense neural networks treat all pixels equally after flattening.

CNN preserves spatial structure and learns visual patterns automatically.

## Project Structure

```bash id="jlwm1k"
handwritten-digit-recognition/
│── dense_nn_version.py
│── cnn_version.py
│── README.md
│── mnist_cnn_model.h5
```

## Training Process

Both models use:

* Adam optimizer
* Sparse categorical crossentropy loss
* Accuracy metric

## Learning Concepts Covered

* Neural Networks
* Deep Learning
* Image Classification
* Feature Extraction
* Convolutional Neural Networks

## Future Improvements

* Predict custom handwritten images
* Add confusion matrix
* Compare model performance visually
* Deploy as web application

## Practical Learning Outcome

This project demonstrates how model architecture affects performance in image recognition tasks.

Dense networks learn global patterns.

CNN learns hierarchical visual features.

## Recommended Progression

Study in this order:

1. Dense Neural Network
2. CNN
3. Custom image prediction
4. CNN optimization
