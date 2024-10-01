import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class PixelByPixelCIFAR100(Dataset):
    def __init__(self, train=True, noise=0):
        self.transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Lambda(lambda x: x.view(-1, 3))  
        ])
        
        self.cifar = datasets.CIFAR100(root='CIFAR100_data', train=train, download=True, transform=self.transform)
        self.noise = noise
        
    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, index):
        image, label = self.cifar[index]
        image_sequence = image
        label = torch.tensor(label).repeat(image_sequence.shape[0])  
        
        n_noise = int(image_sequence.shape[0]*self.noise)
        noise_inx, _ = torch.randperm(image_sequence.shape[0])[:n_noise].sort()
        image_sequence[noise_inx, :] = torch.rand(n_noise,3)
                
        return image_sequence, label
