"""
Median Defense Implementation for Table 5
Coordinate-wise median aggregation (Byzantine-robust)
"""

import numpy as np
from typing import List
from defense_base import DefenseBase


class MedianDefense(DefenseBase):
    """
    Coordinate-wise Median: Take median value for each parameter coordinate
    
    Simple but effective Byzantine-robust aggregation:
    - For each parameter coordinate, compute median across all clients
    - Resistant to up to 50% Byzantine clients
    - No explicit detection, but implicitly filters outliers
    
    For detection metrics, we identify clients whose updates deviate
    significantly from the median.
    """
    
    def __init__(self, detection_threshold: float = 2.0, verbose: bool = False):
        """
        Initialize Median defense
        
        Args:
            detection_threshold: Distance threshold for malicious detection
                                 (in multiples of MAD - median absolute deviation)
            verbose: Print detailed logs
        """
        super().__init__("Median", verbose)
        self.detection_threshold = detection_threshold
    
    def aggregate(self, updates: List[np.ndarray], weights: List[int], 
                  is_malicious: List[bool]) -> np.ndarray:
        """
        Aggregate using coordinate-wise median
        """
        n = len(updates)
        
        # Stack updates (n_clients × n_params)
        stacked = np.stack(updates, axis=0)
        
        # Coordinate-wise median
        aggregated = np.median(stacked, axis=0)
        
        # Detection: Flag clients whose distance from median exceeds threshold
        # Use MAD (Median Absolute Deviation) for robust scale estimation
        distances = np.array([np.linalg.norm(update - aggregated) for update in updates])
        median_distance = np.median(distances)
        mad = np.median(np.abs(distances - median_distance))
        
        # Threshold based on MAD
        threshold = median_distance + self.detection_threshold * mad
        
        predicted_malicious = [dist > threshold for dist in distances]
        
        if self.verbose:
            print(f"\n[Median] Distances from median: {distances}")
            print(f"[Median] Median distance: {median_distance:.4f}, MAD: {mad:.4f}")
            print(f"[Median] Threshold: {threshold:.4f}")
            print(f"[Median] Detected malicious: {[i for i, m in enumerate(predicted_malicious) if m]}")
        
        metrics = self.compute_detection_metrics(predicted_malicious, is_malicious)
        
        return aggregated


if __name__ == "__main__":
    # Test Median defense
    print("Testing Median Defense...")
    
    # Simulate 5 clients: 2 malicious, 3 honest
    np.random.seed(42)
    
    honest_updates = [np.random.randn(100) * 0.1 for _ in range(3)]
    malicious_updates = [np.random.randn(100) * 5.0 for _ in range(2)]  # 50x larger
    
    updates = malicious_updates + honest_updates
    weights = [100] * 5
    is_malicious = [True, True, False, False, False]
    
    median_defense = MedianDefense(detection_threshold=2.0, verbose=True)
    aggregated = median_defense.aggregate(updates, weights, is_malicious)
    
    print(f"\nAggregated update norm: {np.linalg.norm(aggregated):.4f}")
    print(f"Expected: ~{np.mean([np.linalg.norm(u) for u in honest_updates]):.4f} (honest)")
    
    summary = median_defense.get_summary()
    print(f"\nSummary: {summary}")
