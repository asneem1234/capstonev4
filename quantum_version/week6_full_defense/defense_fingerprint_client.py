"""
Defense Layer 2 (Client-Side): Fingerprint Generation
Generates cryptographic fingerprints from client updates for identity verification
"""

import numpy as np
import torch


class ClientFingerprintDefense:
    """
    Client-side fingerprint generation using random projection.
    Creates a 512-D fingerprint from the update for server-side validation.
    """
    
    def __init__(self, projection_dim=512, seed=42):
        """
        Initialize fingerprint defense
        
        Args:
            projection_dim: Dimension of fingerprint vector (default: 512)
            seed: Random seed for reproducibility
        """
        self.projection_dim = projection_dim
        self.projection_matrix = None
        self.seed = seed
        np.random.seed(seed)
    
    def initialize_projection(self, model):
        """
        Initialize random projection matrix based on model size
        
        Args:
            model: torch.nn.Module to get parameter dimensions
        """
        # Calculate total parameter count
        total_params = sum(p.numel() for p in model.parameters())
        
        # Create random projection matrix: [total_params, projection_dim]
        # Using Gaussian random projection for dimensionality reduction
        self.projection_matrix = np.random.randn(total_params, self.projection_dim) / np.sqrt(total_params)
        
        print(f"Fingerprint projection initialized: {total_params} -> {self.projection_dim} dimensions")
    
    def compute_fingerprint(self, update_dict, train_loss, train_acc):
        """
        Compute fingerprint from update parameters
        
        Args:
            update_dict: Dictionary mapping parameter names to update tensors
            train_loss: Training loss value
            train_acc: Training accuracy value
        
        Returns:
            fingerprint: 512-D numpy array (normalized)
        """
        if self.projection_matrix is None:
            raise ValueError("Projection matrix not initialized. Call initialize_projection() first.")
        
        # Flatten all update tensors into a single vector
        update_values = []
        for name in sorted(update_dict.keys()):  # Sort for consistency
            tensor = update_dict[name]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().detach().numpy()
            update_values.append(tensor.flatten())
        
        update_flat = np.concatenate(update_values)
        
        # Apply random projection: [total_params] @ [total_params, 512] -> [512]
        projected = np.dot(update_flat, self.projection_matrix)
        
        # Add metadata features (loss, accuracy, norm)
        update_norm = np.linalg.norm(update_flat)
        metadata = np.array([train_loss, train_acc, update_norm])
        
        # Concatenate and normalize to unit vector
        fingerprint = np.concatenate([projected, metadata])
        fingerprint = fingerprint / (np.linalg.norm(fingerprint) + 1e-10)
        
        return fingerprint
