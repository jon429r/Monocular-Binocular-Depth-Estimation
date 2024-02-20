import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomKittiDataset(Dataset):
    def __init__(self, root, transform=None):
        new_height = 64
        new_width = 192
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((new_height, new_width))])
        self.root = root
        self.data = []  # List to store your data, each item is a tuple (image, annotations)

        # Load or process your data and populate self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, annotations = self.data[idx]

        # Apply transformations if available
        if self.transform:
            image = self.transform(image)

        # Process annotations and extract class labels
        labels = [obj['type'] for obj in annotations]
        labels = torch.tensor(labels)

        return image, labels
    
    def __getlabel__(self, idx):
        image, annotations = self.data[idx]

        # Apply transformations if available
        if self.transform:
            image = self.transform(image)

        # Process annotations and extract class labels
        labels = [obj['type'] for obj in annotations]
        labels = torch.tensor(labels)

        return labels
    
    def __getimage__(self, idx):
        image, annotations = self.data[idx]

        # Apply transformations if available
        if self.transform:
            image = self.transform(image)

        # Process annotations and extract class labels
        labels = [obj['type'] for obj in annotations]
        labels = torch.tensor(labels)

        return image
