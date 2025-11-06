"""
Krum Defense Implementation for Table 5
Byzantine-robust aggregation by selecting the update with smallest distance sum
"""

import numpy as np
from typing import List
from defense_base import DefenseBase


class KrumDefense(DefenseBase):
    """
    Krum: Select the client update that is closest to other updates
    
    Paper: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    Blanchard et al., NeurIPS 2017
    
    Algorithm:
    1. Compute pairwise distances between all updates
    2. For each client, sum distances to n-f-2 nearest neighbors
    3. Select client with minimum distance sum
    4. Use that client's update as the aggregated result
    
    where f is the number of Byzantine clients to tolerate
    """
    
    def __init__(self, f: int = 2, verbose: bool = False):
        """
        Initialize Krum defense
        
        Args:
            f: Number of Byzantine clients to tolerate (default: 2 for 40% of 5 clients)
            verbose: Print detailed logs
        """
        super().__init__("Krum", verbose)
        self.f = f
    
    def aggregate(self, updates: List[np.ndarray], weights: List[int], 
                  is_malicious: List[bool]) -> np.ndarray:
        """
        Aggregate using Krum selection
        """
        n = len(updates)
        
        if n < 2 * self.f + 3:
            raise ValueError(f"Krum requires n >= 2f + 3, got n={n}, f={self.f}")
        
        # Compute pairwise squared distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(updates[i] - updates[j]) ** 2
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each client, compute score (sum of distances to n-f-2 closest neighbors)
        scores = []
        k = n - self.f - 2  # Number of neighbors to consider
        
        for i in range(n):
            # Get distances to all other clients
            client_distances = distances[i].copy()
            client_distances[i] = np.inf  # Exclude self
            
            # Sort and take k smallest
            k_smallest = np.partition(client_distances, k-1)[:k]
            score = np.sum(k_smallest)
            scores.append(score)
        
        # Select client with minimum score
        selected_idx = np.argmin(scores)
        
        if self.verbose:
            print(f"\n[Krum] Scores: {[f'{s:.2f}' for s in scores]}")
            print(f"[Krum] Selected client {selected_idx} (malicious={is_malicious[selected_idx]})")
        
        # Detection: All others are considered potentially malicious
        # (Krum doesn't explicitly detect, but we can infer rejection)
        predicted_malicious = [i != selected_idx for i in range(n)]
        
        # However, for fair comparison, mark top f clients with highest scores as malicious
        top_f_indices = np.argsort(scores)[-self.f:]
        predicted_malicious = [i in top_f_indices for i in range(n)]
        
        metrics = self.compute_detection_metrics(predicted_malicious, is_malicious)
        
        return updates[selected_idx]


if __name__ == "__main__":
    # Test Krum defense
    print("Testing Krum Defense...")
    
    # Simulate 5 clients: 2 malicious, 3 honest
    np.random.seed(42)
    
    honest_updates = [np.random.randn(100) * 0.1 for _ in range(3)]
    malicious_updates = [np.random.randn(100) * 5.0 for _ in range(2)]  # 50x larger
    
    updates = malicious_updates + honest_updates
    weights = [100] * 5
    is_malicious = [True, True, False, False, False]
    
    krum = KrumDefense(f=2, verbose=True)
    aggregated = krum.aggregate(updates, weights, is_malicious)
    
    print(f"\nAggregated update norm: {np.linalg.norm(aggregated):.4f}")
    print(f"Expected: ~{np.mean([np.linalg.norm(u) for u in honest_updates]):.4f} (honest)")
    
    summary = krum.get_summary()
    print(f"\nSummary: {summary}")
