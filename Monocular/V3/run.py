import torch
import torch.nn as nn
from train import train_loop, test_loop

from NN_model import NeuralNetworkV3
import ssl
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets
from Dataset import Kitti
ssl._create_default_https_context = ssl._create_unverified_context


new_height = 64
new_width = 192
max_width = 1280
max_height = 384

root = 'KITTI_2'

def set_target_for_index(dataset, index, new_target):
    # Check if the index is within bounds
    if 0 <= index < len(dataset):
        # Modify the target for the specified index
        _, existing_target = dataset.__getitem__(index)
        existing_target[0] = new_target
    else:
        raise IndexError("Index out of bounds for the dataset.")


transforms = Kitti.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((new_height, new_width))])


kitti_training_dataset = Kitti(root=root, train=True, download=True, transform=transforms)
kitti_test_dataset = Kitti(root=root, train=False, download=True, transform=transforms)

batch_size = 1
train_dataloader = DataLoader(kitti_training_dataset, batch_size=batch_size, shuffle=True, collate_fn=None)
print("Training data loaded")
test_dataloader = DataLoader(kitti_test_dataset, batch_size=batch_size, shuffle=True, collate_fn=None)
print("Test data loaded")


print("Length of training data: ", len(kitti_training_dataset))
print("Length of test data: ", len(kitti_test_dataset))


# Define model

model = NeuralNetworkV3()
print("Model defined")
print(model)

# Loss function and optimizer
learning_rate = 1e-3
print("Learning rate: ", learning_rate)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Device: ", device)

'''for batch_idx, (data, target) in enumerate(train_dataloader):
    print(f"Batch {batch_idx}: Data type: {type(data)}, Target type: {type(target)}")

for batch_idx, (data, target) in enumerate(test_dataloader):
    print(f"Batch {batch_idx}: Data type: {type(data)}, Target type: {type(target)}")
'''


# Training and testing loops
num_epochs = 10
for epoch in range(num_epochs):
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done")