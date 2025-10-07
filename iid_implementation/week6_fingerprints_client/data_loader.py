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
    """Return data loaders for each client + test loader + validation loader"""
    train_dataset, test_dataset = load_mnist()
    
    # Reserve validation set from training data
    validation_size = Config.VALIDATION_SIZE
    train_size = len(train_dataset) - validation_size
    
    # Split indices
    all_indices = list(range(len(train_dataset)))
    np.random.shuffle(all_indices)
    
    validation_indices = all_indices[:validation_size]
    client_train_indices = all_indices[validation_size:]
    
    # Split remaining data among clients (IID)
    num_items = len(client_train_indices) // num_clients
    client_loaders = []
    for i in range(num_clients):
        start = i * num_items
        end = start + num_items
        indices = client_train_indices[start:end]
        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True
        )
        client_loaders.append(loader)
    
    # Validation loader
    validation_subset = Subset(train_dataset, validation_indices)
    validation_loader = DataLoader(
        validation_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    # Test loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    return client_loaders, validation_loader, test_loader
