import torch
import torch.optim as optim
import torchvision.transforms as transforms
from model import DepthEstimationModel  
from Read_KITTI import KITTIDataset 

# Define your model, dataset, loss function, and optimizer
model = DepthEstimationModel()
dataset = KITTIDataset(basedir='KITTI', date='2011_09_26', drive='0002', frames=range(0, 50, 5))
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for idx in range(len(dataset)):
        # Load data
        input_data, target = dataset[idx]

        # Forward pass
        predictions = model(input_data)

        # Compute loss
        loss = criterion(predictions, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if idx % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{idx}/{len(dataset)}], Loss: {loss.item()}')
