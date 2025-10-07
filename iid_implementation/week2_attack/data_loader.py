# Simple data loading - IID split initially
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from config import Config

def load_mnist():
    """Download and load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        Config.DATA_DIR, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        Config.DATA_DIR, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    return train_dataset, test_dataset

def split_iid(dataset, num_clients):
    """Split dataset into IID chunks for each client"""
    num_items = len(dataset) // num_clients
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    client_indices = []
    for i in range(num_clients):
        start = i * num_items
        end = start + num_items
        client_indices.append(indices[start:end])
    
    return client_indices

def get_client_loaders(num_clients):
    """Return data loaders for each client + test loader"""
    train_dataset, test_dataset = load_mnist()
    client_indices = split_iid(train_dataset, num_clients)
    
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True
        )
        client_loaders.append(loader)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    return client_loaders, test_loader
