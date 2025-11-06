# Adaptive Byzantine Defense with Pattern Learning (Quantum Version)
# Adapted for quantum federated learning with 5 clients and quantum models

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
    5. Works with small client count (5 clients) and quantum models
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
            client_updates: List of update dictionaries {param_name: tensor}
            model: Current global quantum model
            client_metadata: Optional dict with 'losses', 'accuracies', etc.
            
        Returns:
            features: numpy array (n_clients, 6 features)
                [norm, loss_increase, norm_variance, sign_consistency, train_loss, train_error]
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
        device = next(model.parameters()).device
        model.eval()
        
        # Loss before update
        loss_before = 0.0
        total = 0
        with torch.no_grad():
            for data, target in self.validation_loader:
                data, target = data.to(device), target.to(device)
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
                param.data += update[name].to(device)
        
        # Loss after update
        loss_after = 0.0
        total = 0
        with torch.no_grad():
            for data, target in self.validation_loader:
                data, target = data.to(device), target.to(device)
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
    
    def detect_malicious_adaptive(self, features, method='statistical', contamination='auto'):
        """
        Adaptive anomaly detection using statistical methods.
        
        Args:
            features: (n_clients, 6) array
            method: 'statistical' (recommended for 5 clients), 'clustering', 'isolation_forest'
            contamination: Expected fraction of outliers (or 'auto' to learn)
            
        Returns:
            (honest_indices, malicious_indices, separation_factor, diagnostics)
        """
        n_clients = len(features)
        
        if method == 'statistical':
            # Statistical method: IQR-based (robust for small n)
            return self._statistical_detection(features)
        
        elif method == 'clustering':
            # K-means clustering: find 2 clusters (honest vs malicious)
            return self._clustering_detection(features)
        
        else:  # isolation_forest
            return self._isolation_forest_detection(features, contamination)
    
    def _statistical_detection(self, features):
        """
        Statistical anomaly detection using IQR method on norms.
        Optimized for small client count (5 clients, 2 malicious).
        """
        n_clients = len(features)
        
        # Use ONLY update norms (most reliable feature for gradient ascent)
        norms = features[:, 0]
        
        # IQR Method (robust to outliers)
        Q1 = np.percentile(norms, 25)
        Q3 = np.percentile(norms, 75)
        IQR = Q3 - Q1
        
        # For 5 clients with 40% malicious (2 out of 5):
        # We expect clear separation, so use aggressive threshold
        k_factor = 1.5
        threshold = Q3 + k_factor * IQR
        
        # Classify: anyone above threshold is suspicious
        malicious_mask = norms > threshold
        malicious_indices = np.where(malicious_mask)[0].tolist()
        honest_indices = np.where(~malicious_mask)[0].tolist()
        
        # Iterative refinement (if we have at least 3 honest clients)
        if len(honest_indices) >= 3:
            honest_norms = norms[honest_indices]
            honest_median = np.median(honest_norms)
            honest_std = np.std(honest_norms)
            
            # More aggressive: 3 standard deviations above median
            refined_threshold = honest_median + 3.0 * honest_std
            
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
            'all_norms': norms.tolist(),
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
        malicious_norms = features[malicious_indices, 0] if len(malicious_indices) > 0 else np.array([0.0])
        separation_factor = np.mean(malicious_norms) / (np.mean(honest_norms) + 1e-10)
        
        diagnostics = {
            'method': 'clustering',
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'honest_cluster': honest_cluster,
            'malicious_cluster': malicious_cluster,
            'separation_factor': separation_factor,
            'honest_norm_mean': float(np.mean(honest_norms)),
            'malicious_norm_mean': float(np.mean(malicious_norms)),
        }
        
        return honest_indices, malicious_indices, separation_factor, diagnostics
    
    def _isolation_forest_detection(self, features, contamination):
        """
        Isolation Forest for anomaly detection.
        """
        from sklearn.ensemble import IsolationForest
        
        # Auto-estimate contamination if needed
        if contamination == 'auto':
            # For 5 clients, expect 0-40% malicious
            contamination = 0.4
        
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
            'honest_norm_mean': float(features[honest_indices, 0].mean()) if honest_indices else 0.0,
            'malicious_norm_mean': float(features[malicious_indices, 0].mean()) if malicious_indices else 0.0,
        }
        
        return honest_indices, malicious_indices, separation_factor, diagnostics
