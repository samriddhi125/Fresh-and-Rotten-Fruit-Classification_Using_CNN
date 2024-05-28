# Fresh-and-Rotten-Fruit-Classification Using CNN Model

This repository presents a deep learning model for classifying fruits as fresh or rotten using a pre-trained VGG16 model. The dataset comprises six categories: fresh apples, fresh oranges, fresh bananas, rotten apples, rotten oranges, and rotten bananas, sourced from [Kaggle](https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification).

## About the Project

This project aims to demonstrate the application of transfer learning in the field of image classification. By leveraging the features learned by the VGG16 model on the ImageNet dataset, we adapt it for the task of fruit classification.

## Steps

1. **Load Pre-trained VGG16 Model**

    We load the VGG16 model pre-trained on the ImageNet dataset to utilize its feature extraction capabilities.

2. **Freeze Base Model**

    Freezing the base model's layers prevents them from being updated during training, preserving the pre-trained weights.

3. **Customize Model for Fruit Classification**

    We add custom layers to the VGG16 model to adapt it for our specific classification task. This includes a global average pooling layer and a dense layer with softmax activation for multi-class classification.

4. **Compile Model**

    Compiling the model involves specifying the loss function and metrics for training. Here, we use categorical cross-entropy loss and accuracy as the metric.

5. **Data Augmentation**

    Data augmentation is employed to increase the diversity of the training dataset, enhancing the model's ability to generalize to unseen data.

6. **Load and Prepare Dataset**

    We load the fruit dataset and prepare it for training and validation using data generators.

7. **Train the Model**

    The model is trained on the augmented training dataset, with validation performed on a separate validation set.

8. **Fine-tune the Model**

    To further improve performance, we unfreeze the base model and fine-tune it with a lower learning rate.

9. **Evaluate the Model**

## Conclusion

This project serves as a practical example of utilizing pre-trained models for specific classification tasks using concepts of transfer learning. 
