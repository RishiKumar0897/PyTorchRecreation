## When training neural networks, the most frequently used algorithm is back propagation.
## In this algorithm, paramaters are adjusted according to the gradient of the loss function
## To compute the gradients, we use torch.autograd

import torch


## The following is a one layer neural network with input x, output y, and parameters w and b
## using a loss function loss
x = torch.ones(5) # input tensor
y = torch.zeros(3) # output
w = torch.randn(5,3,requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x,w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

## In this network, w and b are paramaters that need to be optimized
## A function that we apply to tensors to construct computational graph is an object of class Function

print(f"Gradient function for = {z.grad_fn}")
print(f"Gradiest function for loss = {loss.grad_fn}")

## To optimize weights of parameters, we need to compute the derivatives of the loss function with respect to the parameters
loss.backward()
print(w.grad)
print(b.grad)


## You can disable Gradient tracking when you only want to do forward computations through the network
z = torch.matmul(x,w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x,w) + b
    print(z.requires_grad)


## the following achieves the same thing
z = torch.matmul(x,w) + b
z_det = z.detach()
print(z_det.requires_grad)

## You should disable gradient tracking to mark parameters as frozen or to speed up computations