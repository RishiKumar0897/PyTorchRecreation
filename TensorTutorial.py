import torch
import numpy as np

## Tensor created directly from the data; type is inferred as int when it is passed into tensor method
data = [[1,2] , [3,4]]
x_data = torch.tensor(data)
print(x_data)

## Tensor created from a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

## Forms another tensor while retaining the properties (shape, datatype)
## of the argument tensor unless explicity overridden.
x_ones = torch.ones_like(x_data) # retains properties; no override
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides datatype into float.
print(x_ones)
print(x_rand)


## shape is a tuple of tensor dimensions (2x3)
shape = (2,3,) ## passed in as arguments to tensor methods to create diff types of tensors
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


##Tensor attributes
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

###########################################################################

## Basic operations with tensors

## moving tensor from CPU -> GPU
if torch.cuda.is_available():
    tensor = tensor.to("cuda")


## Standard numpy-like indexing and slicing
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

## joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)

## Arithmetic operations

## Computes matrix multiplication between two tensors. y1, y2, y3
## have the same values

# tensor.T returns the transpose of a tensor.

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product.
# z1, z2, z3 have the same value

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)

torch.mul(tensor, tensor, out = z3)

## Single element tensor

# If you aggregate all the values of a tensor into one value, you can
# convert it to a Python numerical value

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


## Operations that store the result into the operand are "in place"
## denoted by a _ suffix
## use is discouraged

print(f"{tensor} \n")
tensor.add_(5) # in place operation
print(tensor)

## Bridging with NumPy

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy() # converts here
print(f"n: {n}")

## any change to the original tensor will reflect in the NumPy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

## and vice versa
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")