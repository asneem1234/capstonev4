"""
Norm-Based Filtering Defense for Quantum Federated Learning
Simple and effective defense against Byzantine attacks
"""

import numpy as np
from typing import List, Tuple, Dict


class NormBasedDefense:
    """
    Norm-based Byzantine defense using median filtering
    
    Concept: Malicious updates have abnormally large norms due to 
    gradient ascent attack (10× amplification). Use median-based 
    threshold to reject outliers.
    
    Algorithm:
    1. Collect all client update norms
    2. Calculate median norm
    3. Set threshold = median × multiplier
    4. Reject updates with norm > threshold
    5. Aggregate only accepted updates
    """
    
    def __init__(self, threshold_multiplier=3.0, verbose=True):
        """
        Initialize norm-based defense
        
        Args:
            threshold_multiplier: Multiplier for median norm (default: 3.0)
            verbose: Print defense statistics
        """
        self.threshold_multiplier = threshold_multiplier
        self.verbose = verbose
        
        # Statistics
        self.round_stats = []
    
    def filter_updates(
        self, 
        client_results: List[Tuple[any, int, Dict]]
    ) -> Tuple[List[Tuple[any, int, Dict]], List[int], Dict]:
        """
        Filter client updates based on norm threshold
        
        Args:
            client_results: List of (fit_result, num_samples, metrics) tuples
        
        Returns:
            filtered_results: Accepted client results
            rejected_indices: List of rejected client indices
            defense_stats: Dictionary with defense statistics
        """
        # Extract update norms from client metrics
        client_norms = []
        client_ids = []
        is_malicious_list = []
        
        for i, (fit_res, num_samples, metrics) in enumerate(client_results):
            norm = metrics.get('update_norm', 0.0)
            client_id = metrics.get('client_id', i)
            is_malicious = metrics.get('is_malicious', False)
            
            client_norms.append(norm)
            client_ids.append(client_id)
            is_malicious_list.append(is_malicious)
        
        # Calculate median norm
        median_norm = np.median(client_norms)
        
        # Set threshold
        threshold = median_norm * self.threshold_multiplier
        
        # Filter updates
        accepted_results = []
        rejected_indices = []
        
        for i, (fit_res, num_samples, metrics) in enumerate(client_results):
            norm = client_norms[i]
            
            if norm <= threshold:
                # Accept update
                accepted_results.append((fit_res, num_samples, metrics))
            else:
                # Reject update
                rejected_indices.append(i)
        
        # Calculate defense statistics
        num_total = len(client_results)
        num_accepted = len(accepted_results)
        num_rejected = len(rejected_indices)
        
        # Detection metrics (if ground truth available)
        true_positives = sum(1 for i in rejected_indices if is_malicious_list[i])
        false_positives = num_rejected - true_positives
        true_negatives = sum(1 for i in range(num_total) if i not in rejected_indices and not is_malicious_list[i])
        false_negatives = sum(1 for i in range(num_total) if i not in rejected_indices and is_malicious_list[i])
        
        precision = true_positives / num_rejected if num_rejected > 0 else 0.0
        recall = true_positives / sum(is_malicious_list) if sum(is_malicious_list) > 0 else 0.0
        
        defense_stats = {
            'median_norm': median_norm,
            'threshold': threshold,
            'num_total': num_total,
            'num_accepted': num_accepted,
            'num_rejected': num_rejected,
            'rejected_clients': [client_ids[i] for i in rejected_indices],
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'min_norm': min(client_norms) if len(client_norms) > 0 else 0.0,
            'max_norm': max(client_norms) if len(client_norms) > 0 else 0.0,
            'mean_norm': np.mean(client_norms) if len(client_norms) > 0 else 0.0,
            'std_norm': np.std(client_norms) if len(client_norms) > 0 else 0.0
        }
        
        # Store statistics
        self.round_stats.append(defense_stats)
        
        # Print statistics
        if self.verbose:
            self._print_defense_stats(defense_stats)
        
        return accepted_results, rejected_indices, defense_stats
    
    def _print_defense_stats(self, stats: Dict):
        """Print defense statistics"""
        print(f"\n{'='*60}")
        print(f"🛡️  Norm-Based Defense Statistics")
        print(f"{'='*60}")
        print(f"Norm statistics:")
        print(f"  Median: {stats['median_norm']:.4f}")
        print(f"  Threshold: {stats['threshold']:.4f} (median × {self.threshold_multiplier})")
        print(f"  Range: [{stats['min_norm']:.4f}, {stats['max_norm']:.4f}]")
        print(f"  Mean: {stats['mean_norm']:.4f}, Std: {stats['std_norm']:.4f}")
        print(f"\nFiltering results:")
        print(f"  Total clients: {stats['num_total']}")
        print(f"  ✓ Accepted: {stats['num_accepted']}")
        print(f"  ✗ Rejected: {stats['num_rejected']}")
        if stats['num_rejected'] > 0:
            print(f"  Rejected clients: {stats['rejected_clients']}")
        print(f"\nDetection metrics:")
        print(f"  True Positives (malicious caught): {stats['true_positives']}")
        print(f"  False Positives (honest rejected): {stats['false_positives']}")
        print(f"  True Negatives (honest accepted): {stats['true_negatives']}")
        print(f"  False Negatives (malicious missed): {stats['false_negatives']}")
        print(f"  Precision: {stats['precision']*100:.2f}%")
        print(f"  Recall: {stats['recall']*100:.2f}%")
        print(f"{'='*60}\n")
    
    def get_statistics(self) -> List[Dict]:
        """Get all round statistics"""
        return self.round_stats
    
    def get_summary(self) -> Dict:
        """Get overall defense summary"""
        if not self.round_stats:
            return {}
        
        total_rejected = sum(s['num_rejected'] for s in self.round_stats)
        total_clients = sum(s['num_total'] for s in self.round_stats)
        
        avg_precision = np.mean([s['precision'] for s in self.round_stats])
        avg_recall = np.mean([s['recall'] for s in self.round_stats])
        
        return {
            'num_rounds': len(self.round_stats),
            'total_clients_processed': total_clients,
            'total_rejected': total_rejected,
            'rejection_rate': total_rejected / total_clients if total_clients > 0 else 0.0,
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'perfect_detection_rounds': sum(1 for s in self.round_stats if s['precision'] == 1.0 and s['recall'] == 1.0)
        }


if __name__ == "__main__":
    print("Testing Norm-Based Defense...")
    
    # Simulate client results
    # 18 honest clients (norm ~0.5-1.5) + 12 malicious (norm ~5-20)
    import random
    random.seed(42)
    
    client_results = []
    
    # Honest clients (0.5-1.5 norm)
    for i in range(18):
        norm = random.uniform(0.5, 1.5)
        metrics = {
            'client_id': i + 12,
            'update_norm': norm,
            'is_malicious': False,
            'loss': 0.5,
            'accuracy': 85.0
        }
        client_results.append((None, 2000, metrics))
    
    # Malicious clients (5-20 norm)
    for i in range(12):
        norm = random.uniform(5.0, 20.0)
        metrics = {
            'client_id': i,
            'update_norm': norm,
            'is_malicious': True,
            'loss': 0.5,
            'accuracy': 85.0
        }
        client_results.append((None, 2000, metrics))
    
    # Shuffle
    random.shuffle(client_results)
    
    # Apply defense
    defense = NormBasedDefense(threshold_multiplier=3.0, verbose=True)
    accepted, rejected, stats = defense.filter_updates(client_results)
    
    print(f"\n✓ Defense test completed!")
    print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")
    print(f"Precision: {stats['precision']*100:.1f}%, Recall: {stats['recall']*100:.1f}%")
    
    # Summary
    summary = defense.get_summary()
    print(f"\nSummary: {summary}")
