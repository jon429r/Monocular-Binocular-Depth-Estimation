import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision import datasets
from NN_model import NeuralNetwork
from visualize_depth import DepthMapVisualizer
import ssl
import matplotlib.pyplot as plt
from Kitti_dataset import CustomKittiDataset


# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context

new_height = 64
new_width = 192
max_width = 1280
max_height = 384

root = 'KITTI'

transforms = datasets.Kitti.transform = transforms.Compose([transforms.ToTensor(), 
    transforms.Resize((new_height, new_width))])

# Define transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((new_height, new_width))])

# Create datasets with transforms
custom_kitti_training_dataset = CustomKittiDataset(root=root, transform=transform)
custom_kitti_test_dataset = CustomKittiDataset(root=root, transform=transform)

kitti_training_dataset = datasets.Kitti(root=root, train=True, download=False, transform=transform)
kitti_test_dataset = datasets.Kitti(root=root, train=False, download=False, transform=transform)

# Create data loaders
batch_size = 1
train_dataloader = DataLoader(custom_kitti_training_dataset, batch_size=batch_size, shuffle=True)
print("Training data loaded")
test_dataloader = DataLoader(custom_kitti_test_dataset, batch_size=batch_size, shuffle=True)
print("Test data loaded")

print("Length of training data: ", len(custom_kitti_training_dataset))
print("Length of test data: ", len(custom_kitti_test_dataset))

# Define model
print("Model defined")
model = NeuralNetwork()


# Loss function and optimizer
learning_rate = 1e-3
print("Learning rate: ", learning_rate)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Device: ", device)

#print out one piece of data
#print(kitti_dataset[0])

# Training and testing loops
def train_loop(train_dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Batch {batch}, Loss: {loss.item()}")

def test_loop(test_dataloader, model, loss_fn):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, target in test_dataloader:
            pred = model(X)
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {100 * accuracy:.2f}%")

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Training finished!")