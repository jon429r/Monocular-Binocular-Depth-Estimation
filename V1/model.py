import torch.nn.functional as F
import torch.nn as nn

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()

        # Encoder (Downsampling)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Add more encoder layers as needed

        # Decoder (Upsampling)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Add more encoder layers as needed

        # Decoder
        x = self.upconv1(x)
        x = self.relu3(x)

        x = self.upconv2(x)

        return x

# Instantiate the model
model = DepthEstimationModel()

# Print the model architecture
print(model)
