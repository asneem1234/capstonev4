# Client-side Fingerprint Defense (Quantum Version)
# Adapted for quantum federated learning with 5 clients

import torch
import numpy as np

class ClientSideFingerprintDefense:
    """
    Client-side fingerprint computation with server-side verification.
    
    Key improvements over server-side:
    1. Client computes fingerprint of their update
    2. Fingerprint sent alongside update  
    3. Server verifies fingerprint matches update (integrity check)
    4. Prevents Byzantine attacks from manipulating fingerprint computation
    5. Enables cluster-based anomaly detection
    
    Optimized for small client count (5 clients) and quantum models.
    """
    
    def __init__(self, fingerprint_dim=512):
        """
        Args:
            fingerprint_dim: Dimension of projected fingerprint (512-D)
        """
        self.fingerprint_dim = fingerprint_dim
        self.projection_matrix = None
        self.model_param_size = None
    
    def _initialize_projection(self, param_size):
        """Initialize random projection matrix (shared across all clients)"""
        if self.projection_matrix is None or self.model_param_size != param_size:
            self.model_param_size = param_size
            # Use fixed seed for reproducibility across clients and server
            np.random.seed(42)
            self.projection_matrix = np.random.randn(
                self.fingerprint_dim, 
                param_size
            ).astype(np.float32)
            # Normalize columns
            self.projection_matrix /= np.linalg.norm(
                self.projection_matrix, axis=0, keepdims=True
            )
    
    def compute_fingerprint(self, update, train_loss=None, train_acc=None):
        """
        Compute fingerprint of update using random projection + normalization.
        
        Method:
        1. Flatten update: Δw ∈ R^d (d = model parameters)
        2. Project: f = P × Δw where P ∈ R^(512×d)
        3. Normalize: f_norm = f / ||f||
        
        Args:
            update: Dict of parameter updates {name: tensor}
            train_loss: Training loss (optional, for metadata)
            train_acc: Training accuracy (optional, for metadata)
            
        Returns:
            fingerprint: Normalized fingerprint vector (512,)
        """
        # Flatten update to vector
        flat_update = []
        for name in sorted(update.keys()):
            flat_update.append(update[name].cpu().numpy().flatten())
        flat_update = np.concatenate(flat_update)
        
        # Initialize projection matrix if needed
        self._initialize_projection(len(flat_update))
        
        # Project to low dimension: f = P × Δw
        fingerprint = self.projection_matrix @ flat_update
        
        # Normalize to unit vector: f_norm = f / ||f||
        norm = np.linalg.norm(fingerprint)
        if norm > 1e-10:
            fingerprint = fingerprint / norm
        
        return fingerprint
    
    def verify_fingerprint(self, update, claimed_fingerprint, tolerance=1e-3):
        """
        Verify that claimed fingerprint matches the actual update.
        
        Args:
            update: Dict of parameter updates
            claimed_fingerprint: Fingerprint sent by client
            tolerance: Maximum allowed difference
            
        Returns:
            (is_valid, actual_fingerprint, similarity)
        """
        actual_fingerprint = self.compute_fingerprint(update)
        
        # Compute cosine similarity
        similarity = np.dot(claimed_fingerprint, actual_fingerprint)
        
        # Fingerprints should be nearly identical (cosine sim ~1.0)
        is_valid = similarity >= (1.0 - tolerance)
        
        return is_valid, actual_fingerprint, similarity
    
    def cluster_fingerprints(self, fingerprints, threshold=0.85, metadata=None):
        """
        Cluster clients based on fingerprint similarity + optional metadata.
        
        For 5 clients with 40% malicious (2 malicious, 3 honest):
        - Threshold 0.85 = 31.8° angle separation (relaxed for quantum model variation)
        - With metadata: 50% fingerprint + 50% loss/acc pattern
        
        Args:
            fingerprints: List of fingerprint vectors
            threshold: Cosine similarity threshold (0.85 recommended for 5 clients)
            metadata: Optional dict with 'losses' and 'accuracies' lists
            
        Returns:
            (main_cluster, outliers): Indices of main cluster and outliers
        """
        if len(fingerprints) == 0:
            return [], []
        
        fingerprints = np.array(fingerprints)
        n_clients = len(fingerprints)
        
        # Compute pairwise cosine similarity matrix
        # Since fingerprints are normalized, dot product = cosine similarity
        similarity_matrix = fingerprints @ fingerprints.T
        
        # Enhanced clustering with metadata (loss/accuracy)
        if metadata is not None and 'losses' in metadata and 'accuracies' in metadata:
            losses = np.array(metadata['losses'])
            accuracies = np.array(metadata['accuracies'])
            
            # Normalize metadata to [0, 1]
            loss_range = losses.max() - losses.min()
            if loss_range > 1e-10:
                loss_norm = (losses - losses.min()) / loss_range
            else:
                loss_norm = np.zeros_like(losses)
            
            acc_norm = accuracies / 100.0
            
            # Compute metadata distance matrix
            # Malicious clients typically have HIGHER loss and LOWER accuracy
            metadata_dist = np.zeros((n_clients, n_clients))
            for i in range(n_clients):
                for j in range(n_clients):
                    # Distance = how different their loss/acc patterns are
                    metadata_dist[i, j] = abs(loss_norm[i] - loss_norm[j]) + abs(acc_norm[i] - acc_norm[j])
            
            # Combine: 50% gradient similarity, 50% metadata similarity
            # Higher weight on metadata to catch malicious clients with similar gradients
            combined_similarity = 0.5 * similarity_matrix - 0.5 * metadata_dist
        else:
            combined_similarity = similarity_matrix
        
        # Find main cluster using density-based approach
        # Count neighbors with similarity > threshold for each client
        neighbor_counts = (combined_similarity > threshold).sum(axis=1) - 1  # -1 to exclude self
        
        # Main cluster = clients with most neighbors
        max_neighbors = neighbor_counts.max()
        if max_neighbors < 1:
            # No clear cluster, all are outliers
            return [], list(range(len(fingerprints)))
        
        # For 5 clients, use 50% threshold for main cluster membership
        # (at least 50% of max neighbor count)
        main_cluster = np.where(neighbor_counts >= max_neighbors * 0.5)[0].tolist()
        outliers = [i for i in range(len(fingerprints)) if i not in main_cluster]
        
        return main_cluster, outliers
