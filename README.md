# Landmarks-Classification

## Project Overview
This Project aims to build a convolutional neural network that classify the landmark in the input image. The high level steps of the project include:

- Creating a CNN to Classify Landmarks (from Scratch) - visualize the dataset, process it for training, and then build a convolutional neural network from scratch to classify the landmarks. 

- Creating a CNN to Classify Landmarks (using Transfer Learning) - investigate different pre-trained models and decide on one to use for this classification task. Along with training and testing this transfer-learned network.

- Writing a Landmark Prediction Algorithm -  use the best model to create a simple interface for others to be able to use the model to find the most likely landmarks depicted in an image.

Both networks are trained and tested on a subset of `Google landmarks dataset v2`. Accordingly, this is considered as a multi-classification problem where we should train the network on **50 different classes**.

## Data Preperation

The following table explain the training, validation, and testing datasets sizes:

| Dataset | #_of_images |
| ----------- | ----------- |
| Training | 4497 |
| Validation | 499 |
| Testing | 1250 | 

Transformation is applied to all datasets to resize and normalize images. Also, the `DataLoader` is used to wrap the datasets.
The batch size used is 8 images, which means the model is trained on 8 images on each dataloader iteration.

## Network Architecture
The project includes creating, training, and testing 2 neural netoworks, one is built from the scratch and the other by applying transfer learning. 
Lets go in deep with each network's details:

### Building from the Scratch:
The network here consists of Convolutional, Linear, Maxpool, and Dropout  layers in addtion to ReLU activation function:
- 6 convolutional layers: Used to extract the main features from images.
- 3 linear layers: Used to classify the images by producing probability distribution, resulting in each class probability for the input image (Multi-class classification).
- 6 maxpool layers: Used to avoid overfitting after each convolutional layer
- 2 dropout layers: used after the linear layers to avoid overfitting
- ReLU function: Used as activation function for all layers except last one (the output layer)

Note that the input size should be (800, 800, 3) for each image.

### Transfer Learning:


## Training and Testing


