import matplotlib.pyplot as plt
import torch

class DepthMapVisualizer:
    def __init__(self, cmap='viridis'):
        self.cmap = cmap

    def visualize_depth_map(self, depth_map_tensor):
        """
        Visualize a depth map using Matplotlib.

        Parameters:
        - depth_map_tensor (torch.Tensor): The depth map tensor to visualize.
        """
        # Ensure the tensor is on CPU and convert it to a NumPy array
        depth_map_numpy = depth_map_tensor.cpu().numpy()

        # Plot the depth map using Matplotlib
        plt.imshow(depth_map_numpy, cmap=self.cmap)
        plt.colorbar()  # Add a colorbar for reference
        plt.title('Depth Map')
        plt.show()
