# Non-IID data loading with Dirichlet distribution + Validation set
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset, random_split
import numpy as np
from config import Config
from PIL import Image
import os
import pandas as pd

class FetalPlaneDataset(Dataset):
    """Custom dataset for fetal plane images from CSV"""
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
        plane_to_class = {
            'Fetal abdomen': 0,
            'Fetal brain': 1,
            'Fetal femur': 2,
            'Fetal thorax': 3,
            'Maternal cervix': 4,
            'Other': 5
        }
        
        df_split['class_idx'] = df_split['Plane'].map(plane_to_class)
        df_split = df_split.dropna(subset=['class_idx'])
        df_split['class_idx'] = df_split['class_idx'].astype(int)
        
        # Images directory
        images_dir = os.path.join(root_dir, 'Images')
        
        # Process each row
        for _, row in df_split.iterrows():
            image_name = row['Image_name']
            class_idx = row['class_idx']
            img_path = os.path.join(images_dir, f'{image_name}.png')
            
            if os.path.exists(img_path):
                self.data.append(img_path)
                self.targets.append(class_idx)
        
        self.targets = np.array(self.targets)
        
        split_name = 'train' if train else 'test'
        print(f"Loaded {len(self.data)} images for {split_name} split from CSV")
        if len(self.targets) > 0:
            unique, counts = np.unique(self.targets, return_counts=True)
            print(f"  Class distribution: {dict(zip(unique, counts))}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        image = Image.open(img_path).convert('RGB')  # RGB for ResNet18
        if self.transform:
            image = self.transform(image)
        # Convert label to long tensor (int64) for PyTorch cross_entropy
        label = torch.tensor(label, dtype=torch.long)
        return image, label

def load_fetal_plane_data():
    """Load fetal plane dataset with validation split"""
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_train_dataset = FetalPlaneDataset(Config.DATA_DIR, train=True, transform=train_transform)
    test_dataset = FetalPlaneDataset(Config.DATA_DIR, train=False, transform=test_transform)
    
    # Split train into train + validation
    val_size = min(Config.VALIDATION_SIZE, len(full_train_dataset) // 10)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    return train_dataset, val_dataset, test_dataset

def split_non_iid_dirichlet(dataset, num_clients, alpha=0.5):
    """Split dataset using Dirichlet distribution"""
    num_classes = Config.NUM_CLASSES
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'dataset'):  # For Subset
        labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Group indices by class
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
    
    # Dirichlet split
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)
        
        for client_id in range(num_clients):
            client_indices[client_id].extend(split_indices[client_id].tolist())
    
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices

def get_client_loaders(num_clients, alpha=0.5):
    """Create data loaders for clients, validation, and test"""
    print(f"\nCreating Non-IID data split with Dirichlet(α={alpha})...")
    
    train_dataset, val_dataset, test_dataset = load_fetal_plane_data()
    
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
        print("  cd ../../../non_iid_implementation/week6_full_defense")
        print("  python main.py")
        print("="*70)
        raise ValueError("No training data found. Please provide fetal plane dataset.")
    
    # Split training data among clients (Non-IID)
    client_indices = split_non_iid_dirichlet(train_dataset, num_clients, alpha)
    
    # Create client loaders
    client_loaders = []
    print("\nData distribution per client:")
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        
        # Create subset for this client
        if hasattr(train_dataset, 'dataset'):  # Subset from random_split
            actual_indices = [train_dataset.indices[i] for i in indices]
            subset = Subset(train_dataset.dataset, actual_indices)
        else:
            subset = Subset(train_dataset, indices)
        
        loader = DataLoader(subset, batch_size=Config.BATCH_SIZE, shuffle=True)
        client_loaders.append(loader)
        
        print(f"  Client {client_id}: {len(indices)} samples")
    
    # Create validation and test loaders
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f"\nValidation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    return client_loaders, val_loader, test_loader
