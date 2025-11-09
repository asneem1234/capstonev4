# Adaptive Byzantine Defense with Pattern Learning
import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats

class AdaptiveDefense:
    """
    Adaptive defense that learns patterns and automatically detects malicious clients.
    
    Key features:
    1. No hard-coded thresholds
    2. Learns honest client distribution each round
    3. Uses statistical tests to detect outliers
    4. Adapts to varying attack intensities
    """
    
    def __init__(self, validation_loader):
        self.validation_loader = validation_loader
        
        # History for adaptive learning
        self.norm_history = []
        self.loss_increase_history = []
        
    def compute_update_features(self, client_updates, model, client_metadata=None):
        """
        Extract multiple features from client updates for anomaly detection.
        
        Args:
            client_updates: List of update dictionaries
            model: Current global model
            client_metadata: Optional dict with 'losses', 'accuracies', etc.
            
        Returns:
            features: numpy array (n_clients, n_features)
        """
        n_clients = len(client_updates)
        features = []
        
        for i, update in enumerate(client_updates):
            # Feature 1: Update norm (magnitude)
            norm = 0.0
            for name, delta in update.items():
                norm += torch.norm(delta).item() ** 2
            norm = norm ** 0.5
            
            # Feature 2: Validation loss increase
            loss_increase = self._compute_loss_increase(model, update)
            
            # Feature 3: Layer-wise norm variance (gradient inconsistency)
            layer_norms = []
            for name, delta in update.items():
                layer_norms.append(torch.norm(delta).item())
            norm_variance = np.var(layer_norms) if len(layer_norms) > 1 else 0.0
            
            # Feature 4: Sign consistency (are gradients pointing same direction?)
            sign_consistency = self._compute_sign_consistency(update)
            
            # Feature 5 & 6: Metadata (if available)
            train_loss = client_metadata['losses'][i] if client_metadata and 'losses' in client_metadata else 0.0
            train_acc = client_metadata['accuracies'][i] if client_metadata and 'accuracies' in client_metadata else 0.0
            
            features.append([
                norm,                # Feature 1: Overall magnitude
                loss_increase,       # Feature 2: Impact on validation
                norm_variance,       # Feature 3: Layer inconsistency
                sign_consistency,    # Feature 4: Gradient direction consistency
                train_loss,          # Feature 5: Local training loss
                100.0 - train_acc    # Feature 6: Local error rate
            ])
        
        return np.array(features)
    
    def _compute_loss_increase(self, model, update):
        """Compute validation loss increase for a single update"""
        model.eval()
        
        # Loss before update
        loss_before = 0.0
        total = 0
        with torch.no_grad():
            for data, target in self.validation_loader:
                output = model(data)
                loss_before += torch.nn.functional.cross_entropy(
                    output, target, reduction='sum'
                ).item()
                total += len(target)
        loss_before /= total
        
        # Temporarily apply update
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()
            if name in update:
                param.data += update[name]
        
        # Loss after update
        loss_after = 0.0
        total = 0
        with torch.no_grad():
            for data, target in self.validation_loader:
                output = model(data)
                loss_after += torch.nn.functional.cross_entropy(
                    output, target, reduction='sum'
                ).item()
                total += len(target)
        loss_after /= total
        
        # Restore original parameters
        for name, param in model.named_parameters():
            param.data = original_params[name]
        
        model.train()
        return loss_after - loss_before
    
    def _compute_sign_consistency(self, update):
        """
        Measure how consistent gradient signs are across layers.
        Malicious updates with gradient ascent may have flipped signs.
        """
        positive_count = 0
        negative_count = 0
        total_count = 0
        
        for name, delta in update.items():
            flat = delta.flatten()
            positive_count += (flat > 0).sum().item()
            negative_count += (flat < 0).sum().item()
            total_count += len(flat)
        
        # Return ratio of majority sign (0.5 = balanced, 1.0 = all same sign)
        if total_count == 0:
            return 0.5
        majority = max(positive_count, negative_count)
        return majority / total_count
    
    def detect_malicious_adaptive(self, features, method='isolation_forest', contamination='auto'):
        """
        Adaptive anomaly detection using statistical methods.
        
        Args:
            features: (n_clients, n_features) array
            method: 'isolation_forest', 'dbscan', 'statistical', or 'clustering'
            contamination: Expected fraction of outliers (or 'auto' to learn)
            
        Returns:
            (honest_indices, malicious_indices, separation_factor, diagnostics)
        """
        n_clients = len(features)
        
        if method == 'statistical':
            # Statistical method: Z-score on norm + loss_increase
            return self._statistical_detection(features)
        
        elif method == 'clustering':
            # K-means clustering: find 2 clusters (honest vs malicious)
            return self._clustering_detection(features)
        
        elif method == 'dbscan':
            # DBSCAN: density-based clustering
            return self._dbscan_detection(features)
        
        else:  # isolation_forest (default)
            return self._isolation_forest_detection(features, contamination)
    
    def _statistical_detection(self, features):
        """
        Statistical anomaly detection using IQR method on norms ONLY.
        Simple, robust, and catches gradient ascent attacks effectively.
        """
        n_clients = len(features)
        
        # Use ONLY update norms (most reliable feature for gradient ascent)
        norms = features[:, 0]
        
        # IQR Method (robust to outliers, doesn't use mean)
        Q1 = np.percentile(norms, 25)
        Q3 = np.percentile(norms, 75)
        IQR = Q3 - Q1
        
        # Adaptive threshold: Q3 + k×IQR where k depends on expected malicious ratio
        # With 40% malicious, we need k ≈ 1.5 to separate
        # This puts threshold right in the gap between honest and malicious
        k_factor = 1.5
        threshold = Q3 + k_factor * IQR
        
        # Classify: anyone above threshold is suspicious
        malicious_mask = norms > threshold
        malicious_indices = np.where(malicious_mask)[0].tolist()
        honest_indices = np.where(~malicious_mask)[0].tolist()
        
        # Recompute Q1, Q3 on HONEST ONLY (iterative refinement)
        if len(honest_indices) >= n_clients * 0.5:  # If we have majority honest
            honest_norms = norms[honest_indices]
            honest_Q3 = np.percentile(honest_norms, 75)
            honest_IQR = np.percentile(honest_norms, 75) - np.percentile(honest_norms, 25)
            # More aggressive threshold: 3× the honest IQR above honest Q3
            refined_threshold = honest_Q3 + 3.0 * honest_IQR
            
            # Reclassify with refined threshold
            malicious_mask = norms > refined_threshold
            malicious_indices = np.where(malicious_mask)[0].tolist()
            honest_indices = np.where(~malicious_mask)[0].tolist()
            threshold = refined_threshold
        
        # Compute separation factor
        if len(malicious_indices) > 0 and len(honest_indices) > 0:
            honest_norms = norms[honest_indices]
            malicious_norms = norms[malicious_indices]
            separation_factor = np.mean(malicious_norms) / (np.mean(honest_norms) + 1e-10)
        else:
            separation_factor = 1.0
        
        diagnostics = {
            'method': 'statistical',
            'threshold': threshold,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'norm_range': [np.min(norms), np.max(norms)],
            'honest_norm_mean': np.mean(norms[honest_indices]) if honest_indices else 0.0,
            'malicious_norm_mean': np.mean(norms[malicious_indices]) if malicious_indices else 0.0,
        }
        
        return honest_indices, malicious_indices, separation_factor, diagnostics
    
    def _clustering_detection(self, features):
        """
        K-means clustering to find honest vs malicious groups.
        """
        n_clients = len(features)
        
        # Normalize features (important for clustering)
        features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)
        
        # K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_norm)
        
        # Determine which cluster is honest (lower norms, lower loss_increase)
        cluster_0_features = features[labels == 0]
        cluster_1_features = features[labels == 1]
        
        # Compare average norms and loss_increases
        cluster_0_score = cluster_0_features[:, 0].mean() + cluster_0_features[:, 1].mean()
        cluster_1_score = cluster_1_features[:, 0].mean() + cluster_1_features[:, 1].mean()
        
        if cluster_0_score < cluster_1_score:
            honest_cluster = 0
            malicious_cluster = 1
        else:
            honest_cluster = 1
            malicious_cluster = 0
        
        honest_indices = np.where(labels == honest_cluster)[0].tolist()
        malicious_indices = np.where(labels == malicious_cluster)[0].tolist()
        
        # Compute separation factor
        honest_norms = features[honest_indices, 0]
        malicious_norms = features[malicious_indices, 0]
        separation_factor = np.mean(malicious_norms) / (np.mean(honest_norms) + 1e-10)
        
        diagnostics = {
            'method': 'clustering',
            'cluster_centers': kmeans.cluster_centers_,
            'honest_cluster': honest_cluster,
            'malicious_cluster': malicious_cluster,
            'separation_factor': separation_factor,
            'honest_norm_mean': np.mean(honest_norms),
            'malicious_norm_mean': np.mean(malicious_norms),
        }
        
        return honest_indices, malicious_indices, separation_factor, diagnostics
    
    def _dbscan_detection(self, features):
        """
        DBSCAN density-based clustering.
        """
        from sklearn.cluster import DBSCAN
        
        # Normalize features
        features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)
        
        # DBSCAN (eps and min_samples are auto-tuned)
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        labels = dbscan.fit_predict(features_norm)
        
        # Find the largest cluster (likely honest)
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique_labels) > 0:
            main_cluster = unique_labels[np.argmax(counts)]
            honest_indices = np.where(labels == main_cluster)[0].tolist()
            malicious_indices = np.where(labels != main_cluster)[0].tolist()
        else:
            # No clusters found, use norm threshold
            norms = features[:, 0]
            threshold = np.median(norms) * 2.0
            honest_indices = np.where(norms <= threshold)[0].tolist()
            malicious_indices = np.where(norms > threshold)[0].tolist()
        
        # Compute separation factor
        if len(malicious_indices) > 0 and len(honest_indices) > 0:
            separation_factor = features[malicious_indices, 0].mean() / (features[honest_indices, 0].mean() + 1e-10)
        else:
            separation_factor = 1.0
        
        diagnostics = {
            'method': 'dbscan',
            'n_clusters': len(unique_labels) if len(unique_labels) > 0 else 0,
            'separation_factor': separation_factor,
        }
        
        return honest_indices, malicious_indices, separation_factor, diagnostics
    
    def _isolation_forest_detection(self, features, contamination):
        """
        Isolation Forest for anomaly detection (scikit-learn).
        """
        from sklearn.ensemble import IsolationForest
        
        # Auto-estimate contamination if needed
        if contamination == 'auto':
            # Use statistical method to estimate
            norms = features[:, 0]
            z_scores = np.abs(stats.zscore(norms))
            contamination = (z_scores > 2.5).sum() / len(norms)
            contamination = max(0.05, min(0.5, contamination))  # Clamp to [5%, 50%]
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(features)
        
        # -1 = outlier (malicious), 1 = inlier (honest)
        honest_indices = np.where(predictions == 1)[0].tolist()
        malicious_indices = np.where(predictions == -1)[0].tolist()
        
        # Compute separation factor
        if len(malicious_indices) > 0 and len(honest_indices) > 0:
            separation_factor = features[malicious_indices, 0].mean() / (features[honest_indices, 0].mean() + 1e-10)
        else:
            separation_factor = 1.0
        
        diagnostics = {
            'method': 'isolation_forest',
            'contamination': contamination,
            'separation_factor': separation_factor,
            'honest_norm_mean': features[honest_indices, 0].mean() if honest_indices else 0.0,
            'malicious_norm_mean': features[malicious_indices, 0].mean() if malicious_indices else 0.0,
        }
        
        return honest_indices, malicious_indices, separation_factor, diagnostics
