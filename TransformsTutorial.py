## Transforms are used to manipulate data to make it suitable for training

## All TorchVision datasets have two parameters: transform and target_transform
    # transform is used to modify features
    # target_transform is used to modify labels that accept callables containing transformation logic.

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

## In this example, the FashionMNIST features are in PIL Image Format, labels are integers
    # For training, we need the features as normalized tensors, and labels as one-hot encoded tensors.
    # We use ToTensor and Lambda to make the transformations

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)


## ToTensor() converts a PIL image or NumPy ndarray into a FloatTensor and scales
    # the pixel intensity values in the range [0.,1.]

## Lambda Transforms apply any user-defined lambda function

#Example: this Lambda defines a function to turn the integer into a one-hot encoded tensor
    #creates a zero-tensor of size 10 (because we have 10 labels in the dataset) and calls scatter_
    # which assigns a value pof 1 to the index as given by the label y
target_transform =  Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))