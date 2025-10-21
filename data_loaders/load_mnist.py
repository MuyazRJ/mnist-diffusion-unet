import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    # Skip the 16-byte header
    images = np.frombuffer(data, dtype=np.uint8, offset=16)

    # Reshape to (num_images, height, width)
    num_images = len(images) // (28*28)
    images = images.reshape(num_images, 28, 28)
    
    # Reshape to (num_images, height, width)
    # Normalize to [-1, 1] for diffusion model
    images = images.astype(np.float32) / 127.5 - 1.0

    images = torch.tensor(images)  
    images = images.unsqueeze(1) 
    return images

def get_mnist_loader(filename, batch_size=32, shuffle=True):
    images = load_mnist_images(filename)
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

# Example usage
# train_images = load_mnist_images('mnist/train-images-idx3-ubyte/train-images-idx3-ubyte')
