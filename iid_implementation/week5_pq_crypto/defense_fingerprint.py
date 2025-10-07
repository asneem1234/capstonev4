# Fingerprint-based clustering defense
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class FingerprintDefense:
    """Fast pre-filtering using gradient fingerprints"""
    
    def __init__(self, fingerprint_dim=512):
        self.fingerprint_dim = fingerprint_dim
        self.projection_matrix = None
    
    def _flatten_update(self, update):
        """Flatten update dict into 1D vector"""
        vectors = []
        for name in sorted(update.keys()):
            vectors.append(update[name].flatten())
        return torch.cat(vectors)
    
    def compute_fingerprint(self, update):
        """Project update to lower dimension and normalize"""
        # Flatten update
        flat_update = self._flatten_update(update)
        total_dim = flat_update.shape[0]
        
        # Create or reuse projection matrix
        if self.projection_matrix is None:
            # Random projection matrix (Gaussian)
            torch.manual_seed(42)  # Reproducible
            self.projection_matrix = torch.randn(
                self.fingerprint_dim, 
                total_dim
            ) / np.sqrt(self.fingerprint_dim)
        
        # Project to fingerprint space
        f_raw = self.projection_matrix @ flat_update
        
        # CRITICAL: Normalize to unit vector (remove magnitude, keep direction)
        f_norm = torch.norm(f_raw)
        if f_norm > 1e-8:  # Avoid division by zero
            fingerprint = f_raw / f_norm
        else:
            fingerprint = f_raw
        
        return fingerprint.numpy()
    
    def cluster_updates(self, client_updates):
        """Cluster using cosine similarity (direction-based, not magnitude)"""
        # Compute normalized fingerprints
        fingerprints = []
        for update in client_updates:
            fp = self.compute_fingerprint(update)  # Already normalized
            fingerprints.append(fp)
        
        fingerprints = np.array(fingerprints)  # Shape: [n_clients, 512]
        n_clients = len(fingerprints)
        
        # Compute pairwise cosine similarities (dot product of normalized vectors)
        similarity_matrix = fingerprints @ fingerprints.T  # [n_clients, n_clients]
        
        # Identify main cluster using density-based approach
        threshold = 0.7  # Cosine similarity threshold (angle < 45°)
        
        # For each client, count how many neighbors are similar
        densities = []
        for i in range(n_clients):
            neighbors = np.sum(similarity_matrix[i] > threshold) - 1  # Exclude self
            densities.append(neighbors)
        
        # Main cluster = clients with high density (many similar neighbors)
        # Expect honest clients to have density >= n/2
        density_threshold = max(2, n_clients // 2)  # At least 2 neighbors
        
        main_cluster_indices = [i for i, density in enumerate(densities) 
                               if density >= density_threshold]
        outlier_indices = [i for i, density in enumerate(densities) 
                          if density < density_threshold]
        
        return main_cluster_indices, outlier_indices
