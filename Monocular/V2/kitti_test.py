import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


batch_size = 53

kitti_dataset = datasets.Kitti(root='KITTI', train=True, download=False, transform=transforms)
# Assuming you have already created the dataset and dataloader

train_dataloader = DataLoader(kitti_dataset, batch_size=batch_size, shuffle=True)

# Print the output of __getitem__ for the first item
sample = train_dataloader.__getattribute__('type')
print("Sample:", sample)
