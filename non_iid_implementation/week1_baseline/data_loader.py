# Non-IID data loading with Dirichlet distribution
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

def split_non_iid_dirichlet(dataset, num_clients, alpha=0.5):
    """
    Split dataset using Dirichlet distribution for heterogeneous (Non-IID) data.
    
    Args:
        dataset: PyTorch dataset with labels
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
               - alpha = 0.1: Highly non-IID (each client has 1-2 dominant classes)
               - alpha = 0.5: Moderately non-IID (uneven class distribution)
               - alpha = 1.0: Slightly non-IID
               - alpha = 10+: Approaches IID
    
    Returns:
        List of indices for each client
    """
    num_classes = 10  # MNIST has 10 classes
    
    # Get labels from dataset
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Group indices by class
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
    
    # For each class, use Dirichlet to determine how to split among clients
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        np.random.shuffle(indices)
        
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Split indices according to proportions
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)
        
        # Assign to clients
        for client_id in range(num_clients):
            client_indices[client_id].extend(split_indices[client_id].tolist())
    
    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices

def get_client_loaders(num_clients, alpha=0.5):
    """
    Return data loaders for each client + test loader (Non-IID).
    
    Args:
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (default 0.5 for moderate heterogeneity)
    
    Returns:
        client_loaders: List of DataLoader objects for each client
        test_loader: DataLoader for test set
    """
    train_dataset, test_dataset = load_mnist()
    
    # Use Dirichlet distribution to split data among clients (Non-IID)
    print(f"\nCreating Non-IID data split with Dirichlet(α={alpha})...")
    
    # Create indices for the full training set
    client_indices_split = split_non_iid_dirichlet(train_dataset, num_clients, alpha)
    
    # Print data distribution statistics
    print(f"Data distribution per client:")
    for client_id in range(num_clients):
        # Get labels for this client
        client_labels = [train_dataset[idx][1] for idx in client_indices_split[client_id]]
        label_counts = np.bincount(client_labels, minlength=10)
        dominant_class = np.argmax(label_counts)
        print(f"  Client {client_id}: {len(client_indices_split[client_id])} samples, "
              f"dominant class={dominant_class} ({label_counts[dominant_class]} samples), "
              f"distribution={label_counts.tolist()}")
    
    # Create data loaders
    client_loaders = []
    for i in range(num_clients):
        subset = Subset(train_dataset, client_indices_split[i])
        loader = DataLoader(
            subset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True
        )
        client_loaders.append(loader)
    
    # Test loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    return client_loaders, test_loader
