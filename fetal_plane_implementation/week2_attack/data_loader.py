# Non-IID data loading with Dirichlet distribution for Fetal Plane dataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from config import Config
from PIL import Image
import os
import pandas as pd

class FetalPlaneDataset(Dataset):
    """
    Custom dataset for fetal plane images.
    
    Expected directory structure:
    data/fetal_planes/
        train/
            class_0/
                image1.png
                image2.png
                ...
            class_1/
                ...
        test/
            class_0/
                ...
            class_1/
                ...
    
    Or provide your own image paths and labels.
    """
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        
        # Load image paths and labels
        self.data = []
        self.targets = []
        
        # Read CSV file
        csv_path = os.path.join(root_dir, 'FETAL_PLANES_DB_data.csv')
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} does not exist. Dataset will be empty.")
            print("Please update Config.DATA_DIR to point to your FETAL dataset.")
            return
        
        # Load CSV with semicolon delimiter
        df = pd.read_csv(csv_path, sep=';')
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Filter by train/test split (Train column: 1 for train, 0 for test)
        train_value = 1 if train else 0
        df_split = df[df['Train'] == train_value].copy()
        
        # Map plane values to class indices (0-5)
        # Actual plane values from CSV: 'Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other'
        plane_to_class = {
            'Fetal abdomen': 0,
            'Fetal brain': 1,
            'Fetal femur': 2,
            'Fetal thorax': 3,
            'Maternal cervix': 4,
            'Other': 5
        }
        
        df_split['class_idx'] = df_split['Plane'].map(plane_to_class)
        
        # Drop rows with unmapped classes (if any)
        df_split = df_split.dropna(subset=['class_idx'])
        df_split['class_idx'] = df_split['class_idx'].astype(int)
        
        # Images directory
        images_dir = os.path.join(root_dir, 'Images')
        
        # Process each row
        for _, row in df_split.iterrows():
            image_name = row['Image_name']
            class_idx = row['class_idx']
            
            # Construct image path (add .png extension)
            img_path = os.path.join(images_dir, f'{image_name}.png')
            
            if os.path.exists(img_path):
                self.data.append(img_path)
                self.targets.append(class_idx)
        
        self.targets = np.array(self.targets)
        
        split_name = 'train' if train else 'test'
        print(f"Loaded {len(self.data)} images for {split_name} split from CSV")
        
        # Show class distribution
        if len(self.targets) > 0:
            unique, counts = np.unique(self.targets, return_counts=True)
            print(f"  Class distribution: {dict(zip(unique, counts))}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        
        # Load image as RGB (for ResNet18)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label to long tensor (int64) for PyTorch cross_entropy
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

def load_fetal_plane_data():
    """Download/load fetal plane dataset"""
    
    # Data augmentation for training (RGB normalization for pretrained models)
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # No augmentation for test
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = FetalPlaneDataset(
        Config.DATA_DIR, 
        train=True, 
        transform=train_transform
    )
    
    test_dataset = FetalPlaneDataset(
        Config.DATA_DIR, 
        train=False, 
        transform=test_transform
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
    num_classes = Config.NUM_CLASSES
    
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
    """Create data loaders for all clients with Non-IID split"""
    
    print(f"\nCreating Non-IID data split with Dirichlet(α={alpha})...")
    
    # Load datasets
    train_dataset, test_dataset = load_fetal_plane_data()
    
    if len(train_dataset) == 0:
        print("\n" + "="*70)
        print("ERROR: No training data found!")
        print("Please update Config.DATA_DIR in config.py to point to your dataset.")
        print("Expected structure:")
        print("  data/fetal_planes/")
        print("    train/")
        print("      class_0/")
        print("        image1.png")
        print("      class_1/")
        print("        ...")
        print("="*70)
        print("\nFor testing without data, you can use the MNIST implementation instead:")
        print("  cd ../../../non_iid_implementation/week2_attack")
        print("  python main.py")
        print("="*70)
        raise ValueError("No training data found. Please provide fetal plane dataset.")
    
    # Split training data among clients (Non-IID)
    client_indices = split_non_iid_dirichlet(train_dataset, num_clients, alpha)
    
    # Create data loaders for each client
    client_loaders = []
    print("\nData distribution per client:")
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True
        )
        client_loaders.append(loader)
        
        # Show distribution statistics
        client_labels = [train_dataset.targets[i] for i in indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        dominant_class = unique[np.argmax(counts)]
        dominant_count = np.max(counts)
        
        print(f"  Client {client_id}: {len(indices)} samples, "
              f"dominant class={dominant_class} ({dominant_count} samples)")
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    return client_loaders, test_loader
