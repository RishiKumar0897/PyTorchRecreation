import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.cuda

## Neural networks comprise of layers/modules that perform operations on data
## The torch.nn namespace provides building blocks for building neural networks
## Every module in PyTorch subclasses the nn.Module

# Checks to see if torch.vuda or torch.backends.mps is available (GPU)
# otherwise uses the CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")


## Defines our neural network by subclassing nn.Module, and initializes
## layers in __init__
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    ## Every nn.Module subclass implements the operations on input data in the forward method
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

## Creates an instance of the Neural Network and moves it to the device
model = NeuralNetwork().to(device)
## prints the structure of the Neural Network
print(model)


##Pass input data to use the model, which executs the model's overwritten "forward"
## Do not call model.forward() directly

##Calling the model on the input returns a 2-dim tensor with dim = 0 corresponding to each
## output of 10 raw predicted values for each class, and dim = 1 corresponding to the individual
## values of each output.

X = torch.rand(1,28,28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


## Taking a sample minibatch of 3 images 28x28 and passing it through the network

input_image = torch.rand(3,28,28)
print(input_image.size())


## nn.Flatten layer converts each 2D 28x28 image into a contiguous array of 784 pixels
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

## the linear layer is a module that applies a linear transformation on the input using its stored weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

## non-linear activations are what create the complex mappings between I/O of the model
## In this model, we use nn.ReLU between linear layers
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}\n\n")

## nn.Sequential is an ordered container of modules. THe data is passed through all the modules in the same order
## as defined.

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

## the last layer of the neural network returns logits (raw values from (-inf, inf)
## these are passed to nn.Softmax, which scales the logits to values [0,1] which
## represent the predicted probabilities for each class.

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

## Many layers in a neural network are paramaterized (they have associated weights and biases)
## Subclassing nn.Module automatically tracks all fields defined inside your model object
## You can access the paramaters using parameters() or named_parameters()

print(f"Model Structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")

