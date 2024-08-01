import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

##training data
training_data = datasets.FashionMNIST(
    root = "data" # path where the train/test data is stored
    , train = True # specifies training or test dataset
    , download = True # downloads the data from the internet if not available at root
    , transform = ToTensor() # specify the feature and label transformations
)

## testing data
test_data = datasets.FashionMNIST(
    root = "data"
    , train = False
    , download = True
    , transform = ToTensor()
)


##Indexing DataSets like a list using matplotlib

## the labels
labels_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

## some front-end work using matplotlib
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap = "gray")
plt.show()

