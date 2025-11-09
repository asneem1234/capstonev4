"""
Defense Layer 1: Adaptive Statistical Defense
Uses 6-feature statistical analysis to detect outlier updates
"""

import numpy as np
import torch


class AdaptiveDefense:
    """
    Adaptive defense using statistical analysis of update features.
    Detects outliers that differ significantly from the cluster of normal updates.
    """
    
    def __init__(self, threshold_std=2.0):
        """
        Initialize adaptive defense
        
        Args:
            threshold_std: Number of standard deviations for threshold
                          Threshold = mean_distance + std × threshold_std
        """
        self.threshold_std = threshold_std
        self.rejection_history = []
    
    def extract_features(self, update_params, global_params, metrics):
        """
        Extract 6 statistical features from an update
        
        Features:
        1. L2 norm of update
        2. L1 norm of update
        3. Max absolute value in update
        4. Cosine similarity to global model
        5. Training loss
        6. Training accuracy
        
        Args:
            update_params: List of numpy arrays (client update)
            global_params: List of numpy arrays (global model)
            metrics: Dict with 'loss' and 'accuracy'
        
        Returns:
            features: 6-element numpy array
        """
        # Convert to flat arrays for computation
        update_flat = np.concatenate([p.flatten() for p in update_params])
        global_flat = np.concatenate([p.flatten() for p in global_params])
        
        # Compute update delta
        delta = update_flat - global_flat
        
        # Feature 1: L2 norm
        l2_norm = np.linalg.norm(delta)
        
        # Feature 2: L1 norm
        l1_norm = np.sum(np.abs(delta))
        
        # Feature 3: Max absolute value
        max_abs = np.max(np.abs(delta))
        
        # Feature 4: Cosine similarity between update and global
        cos_sim = np.dot(update_flat, global_flat) / (
            np.linalg.norm(update_flat) * np.linalg.norm(global_flat) + 1e-10
        )
        
        # Feature 5: Training loss
        train_loss = metrics.get('loss', 0.0)
        
        # Feature 6: Training accuracy
        train_acc = metrics.get('accuracy', 0.0)
        
        features = np.array([l2_norm, l1_norm, max_abs, cos_sim, train_loss, train_acc])
        
        return features
    
    def filter_updates(self, client_updates, global_model):
        """
        Filter updates using statistical outlier detection
        
        Args:
            client_updates: List of update dicts
            global_model: torch.nn.Module (global model for comparison)
        
        Returns:
            accepted_updates: Updates that passed the filter
            rejected_ids: List of rejected client IDs
            stats: Dictionary with filtering statistics
        """
        if len(client_updates) == 0:
            return [], [], {'accepted': 0, 'rejected': 0}
        
        # Get global parameters
        global_params = [p.cpu().detach().numpy() for p in global_model.parameters()]
        
        # Extract features for all updates
        feature_matrix = []
        for update in client_updates:
            features = self.extract_features(
                update['params'],
                global_params,
                update['metrics']
            )
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix)  # Shape: [N_clients, 6]
        
        # Normalize features (z-score normalization)
        mean_features = np.mean(feature_matrix, axis=0)
        std_features = np.std(feature_matrix, axis=0) + 1e-10
        normalized_features = (feature_matrix - mean_features) / std_features
        
        # Calculate distance from center for each update
        center = np.mean(normalized_features, axis=0)
        distances = np.linalg.norm(normalized_features - center, axis=1)
        
        # Set threshold based on mean + std
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        threshold = mean_distance + self.threshold_std * std_distance
        
        # Filter updates
        accepted_updates = []
        rejected_ids = []
        
        for i, (update, distance) in enumerate(zip(client_updates, distances)):
            if distance <= threshold:
                accepted_updates.append(update)
            else:
                rejected_ids.append(update['client_id'])
        
        # Statistics
        stats = {
            'threshold': float(threshold),
            'mean_distance': float(mean_distance),
            'std_distance': float(std_distance),
            'distances': distances.tolist(),
            'accepted': len(accepted_updates),
            'rejected': len(rejected_ids),
            'rejected_ids': rejected_ids
        }
        
        self.rejection_history.append(stats)
        
        return accepted_updates, rejected_ids, stats
    
    def get_summary(self):
        """Get summary statistics across all rounds"""
        if not self.rejection_history:
            return {'total_accepted': 0, 'total_rejected': 0}
        
        total_accepted = sum(h['accepted'] for h in self.rejection_history)
        total_rejected = sum(h['rejected'] for h in self.rejection_history)
        
        return {
            'total_accepted': total_accepted,
            'total_rejected': total_rejected,
            'rejection_rate': total_rejected / (total_accepted + total_rejected) if (total_accepted + total_rejected) > 0 else 0.0
        }
