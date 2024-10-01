import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class PixelByPixelMNIST(Dataset):
    def __init__(self, train=True, noise=0):
        # flatten into a 784-element vector
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts image to PyTorch Tensor
            transforms.Lambda(lambda x: x.view(-1))  # Flattens the image
        ])
        
        self.mnist = datasets.MNIST(root='MNIST_data', train=train, download=True, transform=self.transform)
        self.noise = noise
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, index):
      
        image, label = self.mnist[index]
        image_sequence = image.unsqueeze(-1) 
        label = torch.tensor(label).repeat(image_sequence.shape[0])  
        
        n_noise = int(image_sequence.shape[0]*self.noise)
        noise_inx, _ = torch.randperm(image_sequence.shape[0])[:n_noise].sort()
        image_sequence[noise_inx, 0] = torch.rand(n_noise)
        
        return image_sequence, label
