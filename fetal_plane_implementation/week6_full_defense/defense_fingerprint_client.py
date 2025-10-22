# Client-side Fingerprint Defense for Fetal Plane Classification
import torch
import numpy as np
from sklearn.cluster import DBSCAN

class ClientSideFingerprintDefense:
    """
    Client-side fingerprint computation with server-side verification.
    
    Clients compute fingerprints locally, server verifies integrity and clusters.
    """
    
    def __init__(self, fingerprint_dim=512):
        """
        Args:
            fingerprint_dim: Dimension of projected fingerprint
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
        
        Args:
            update: Dict of parameter updates {name: tensor}
            train_loss: Training loss (optional, for metadata)
            train_acc: Training accuracy (optional, for metadata)
            
        Returns:
            fingerprint: Normalized fingerprint vector
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
        
        # Normalize to unit vector
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
    
    def cluster_fingerprints(self, fingerprints, train_losses=None, train_accs=None, 
                           cosine_threshold=0.90, min_cluster_size=2):
        """
        Cluster fingerprints to detect malicious updates using DBSCAN.
        
        Args:
            fingerprints: List of fingerprint vectors
            train_losses: Optional list of training losses for metadata
            train_accs: Optional list of training accuracies for metadata
            cosine_threshold: Cosine similarity threshold
            min_cluster_size: Minimum samples for a cluster
            
        Returns:
            (honest_indices, malicious_indices, cluster_labels)
        """
        if len(fingerprints) == 0:
            return [], [], []
        
        # Convert to numpy array
        fp_array = np.array(fingerprints)
        
        # Add metadata features if provided
        if train_losses is not None and train_accs is not None:
            # Normalize metadata to similar scale as fingerprints
            losses = np.array(train_losses).reshape(-1, 1)
            accs = np.array(train_accs).reshape(-1, 1)
            
            # Standardize
            losses = (losses - np.mean(losses)) / (np.std(losses) + 1e-10)
            accs = (accs - np.mean(accs)) / (np.std(accs) + 1e-10)
            
            # Concatenate (give less weight to metadata)
            fp_array = np.concatenate([fp_array, 0.1 * losses, 0.1 * accs], axis=1)
        
        # Convert cosine similarity threshold to distance threshold
        # Distance = 1 - cosine_similarity, so eps = 1 - threshold
        eps = 1.0 - cosine_threshold
        
        # DBSCAN clustering with cosine distance
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_cluster_size,
            metric='cosine'
        )
        
        cluster_labels = clustering.fit_predict(fp_array)
        
        # Find largest cluster (assumed honest)
        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], 
                                         return_counts=True)
        
        if len(unique_labels) == 0:
            # No clusters found - all outliers
            return [], list(range(len(fingerprints))), cluster_labels
        
        largest_cluster_label = unique_labels[np.argmax(counts)]
        
        # Honest: in largest cluster
        # Malicious: outliers (-1) or in small clusters
        honest_indices = np.where(cluster_labels == largest_cluster_label)[0].tolist()
        malicious_indices = np.where(cluster_labels != largest_cluster_label)[0].tolist()
        
        return honest_indices, malicious_indices, cluster_labels
