# MNIST Digit Classification with CNN

This project uses a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is trained to classify images of digits (0-9) using CNN layers, and the results are evaluated using accuracy and loss.


[Overview](#overview)

[Techniques Covered](#techniques-covered)

[Features](#features)

[Usage](#usage)

[Dependencies](#dependencies)

[Results](#results)


## OverView
The goal of this project is to build a Convolutional Neural Network (CNN) for the classification of handwritten digits from the MNIST dataset. The model uses layers such as convolution, pooling, dropout, and fully connected layers to classify the images. The network is trained for 5 epochs, and the model's performance is evaluated using accuracy and loss metrics.

## Techniques Covered
- Convolutional Neural Networks (CNN) for image classification.
- Data Preprocessing: Reshaping, normalizing, and one-hot encoding.
- Dropout for regularization to prevent overfitting.
- Model Evaluation: Using accuracy and loss as evaluation metrics.
- Training History Visualization: Plotting the training and validation accuracy and loss over epochs.

## Features
- Data Preprocessing: Normalizes the image pixels and reshapes them into a 4D array for CNN input.
- CNN Architecture:
  1. Convolutional layers for feature extraction.
  2. Max pooling for downsampling.
  3. Fully connected layers for classification.
  4. Dropout layers for regularization.
- Model Evaluation: The model’s performance is evaluated using accuracy and loss.
- Training History Visualization: Plots of training and validation accuracy/loss for insight into the model’s learning process.

## Usage
- Load and preprocess the MNIST dataset: The dataset is loaded, and the images are normalized and reshaped for CNN input. The labels are one-hot encoded.
- Build the CNN model: A Convolutional Neural Network is constructed with layers including convolution, pooling, dropout, and fully connected layers for image classification.
- Train the CNN model: The model is trained on the preprocessed MNIST dataset using training data and evaluated on the test data.
- Evaluate the model: After training, the model’s performance is evaluated using accuracy, loss, and visualized with training and validation accuracy/loss plots.

## Dependencies
```
keras
tensorflow
matplotlib
numpy

```
## Results
- Test accuracy: The accuracy of the model after training, evaluated on the test dataset.
- Training Accuracy & Loss: How well the model is performing on the training data during training.
- Validation Accuracy & Loss: How well the model generalizes to unseen data (validation set) during training.

### Sample Output
#### Test accuracy
```
Test accuracy: 98.45%
```
#### Training and Validation Accuracy Plot
```
precision    recall  f1-score   support

        0       0.90      0.86      0.88      1250
        1       0.87      0.92      0.89      1250

    accuracy                           0.89      2500
   macro avg       0.89      0.89      0.89      2500
weighted avg       0.89      0.89      0.89      2500
```
#### Training and Validation Loss Plot
```
[[1072  178]
 [ 100 1150]]
```
