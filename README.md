# Landmarks-Classification

## :notebook_with_decorative_cover: Project Overview

<p align="justify">
 
This Project aims to build a convolutional neural network that classifies the landmark in the input image using `PyTorch`. The high-level steps of the project include:

- Creating a CNN to Classify Landmarks (from Scratch): Visualizing the dataset, process it for training, and then build a convolutional neural network from scratch to classify the landmarks. 

- Creating a CNN to Classify Landmarks (using Transfer Learning): Investigation of different pre-trained models and decide on one to use for this classification task. Along with training and testing this transfer-learned network.

- Writing a Landmark Prediction Algorithm:  By using the best model to create a simple interface for people to be able to use the model to find the most likely landmarks depicted in an image.

Both networks are trained and tested on a **subset** of `Google landmarks dataset v2`[^3]. Accordingly, this is considered as a multi-classification problem where we should train the network on **50 different classes**.

[Download the subset](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip) 
</p>

## :card_index_dividers: Files

- **landmark.ipynb**: The jupyter notebook that includes all the work done.
- **landmark.html**: The HTML version of the original notebook.
- **assets**: Contains images used in this README.
- **dataset**: The dataset used in this project is not uploaded into this repo. You can download it from [here](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip).

## :black_nib: Data Preparation

The following table explains the training, validation, and test datasets size:

<center>

| Dataset | #_of_images |
| ----------- | ----------- |
| Training | 4497 |
| Validation | 499 |
| Testing | 1250 | 
 </center>

<p align="justify">

Transformation is applied to all datasets to resize and normalize images. Also, the `DataLoader` is used to wrap the datasets.
The batch size is 8 images, which means the model is trained on 8 images on each iteration.
</p>

## :spider_web: Network Architecture
<p align="justify">
The project includes creating, training, and testing two neural networks with different architectures. The first one is built from the scratch and the other one is created by applying transfer learning. Let's go in deep with each network's details:
</p>

### Building from the Scratch:
<p align="justify">
 
The network here consists of Convolutional, Linear, Maxpool, and Dropout  layers in addition to ReLU activation function:

- 6 **Convolutional layers**: Used to extract the main and detailed features found in the images.
- 3 **Linear layers**: Used to classify the images by producing probability distribution, resulting in each class probability result for the input image.
- 6 **Max-pool layers**: Used to avoid overfitting after each convolutional layer.
- 2 **Dropout layers**: Used after each linear layer to avoid overfitting (Regularization).
- **ReLU function**: Used as the activation function for all layers except the last one (the output layer).

Note that the input size should be (800, 800, 3) for each image.
</p>

### Transfer Learning:
<p align="justify">
 
The second network is created by using the pre-trained version of `ResNet34`[^2] , which is a 34 layer convolutional neural network that can be utilized as a state-of-the-art image classification model. This network has been trained on the `ImageNet` dataset--a dataset that has 100,000+ images across 200 different classes. 

</p>

![ResNet-34 Architecture](assets/ResNet34.png)

<p align="justify">
 
In our case, we could replace the last FC (output) layer to give the desired output classes (50) instead of 1000, and freeze all the weights except for the last layer (FC output layer). Accordingly, the training goal was to update the FC layer weights. 

Note that The input image size for the network should be (224, 224, 3).

</p>

## :arrows_counterclockwise: Training and Testing
<p align="justify">
 
The training process was challenging as it required a powerful GPU to train the networks[^1]. We could use `Tesla T4` GPU with 12GB memory to perform training and testing. The training was run for a specific number of epochs for each network. The training process in each epoch can be illustrated in the following steps:

- Taking the batch (8 images) from the training loader.
- Calculating the output using the feed-forward process.
- Calculating the loss (error).
- Apply backpropagation to update the network weights.
- Updating the training loss for the current epoch.

After looping over the whole batches, the validation starts to evaluate the model performance and avoid overfitting:

- Taking the batch (8 images) from the validation loader.
- Finding the output using the feed-forward process by the trained network.
- Calculating the loss (error).
- Updating the validation loss for the current epoch.
- If the current validation loss is less than the minimum loss recorded, then save the model and update the minimum validation loss.

For both networks, the loss and optimization functions were the `CrossEntropyLoss` and `Adam`, respectively.
</p>

> The first (scratch) network is trained in 75 epochs. However, the validation loss started increasing after the 11th epoch. For the second one (transfer learning), the network is trained for 20 epochs and there was a chance for more training epochs as the validation loss was still decreasing.


<p align="center">
  <img src="assets/LossPlot.jpg">
</p>

<!-- ![Transfer learning loss plot](assets/LossPlot.jpg) -->

After testing both networks, the first one could only achieve 26% accuracy and 2.939369 as a test loss. On the other hand, the modified ResNet could perform better by achieving 74% accuracy and 0.968494 as a test loss. 

## :registered: References

[^1]: [Deep learning has a size problem: Shifting from state-of-the-art accuracy to state-of-the-art efficiency](https://heartbeat.comet.ml/deep-learning-has-a-size-problem-ea601304cd8) 

[^2]: [Understanding and visualizing ResNets](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)

[^3]: [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)
