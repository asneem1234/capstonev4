"""
Trimmed-Mean Defense Implementation for Table 5
Remove top/bottom β fraction and average the rest
"""

import numpy as np
from typing import List
from defense_base import DefenseBase


class TrimmedMeanDefense(DefenseBase):
    """
    Trimmed-Mean: Remove β fraction from top and bottom, then average
    
    Byzantine-robust aggregation:
    - For each coordinate, sort values across clients
    - Remove β fraction from top and bottom
    - Compute mean of remaining values
    - Resistant to up to β fraction of Byzantine clients
    """
    
    def __init__(self, beta: float = 0.2, verbose: bool = False):
        """
        Initialize Trimmed-Mean defense
        
        Args:
            beta: Fraction to trim from each end (default: 0.2 = 20%)
            verbose: Print detailed logs
        """
        super().__init__("Trimmed-Mean", verbose)
        self.beta = beta
    
    def aggregate(self, updates: List[np.ndarray], weights: List[int], 
                  is_malicious: List[bool]) -> np.ndarray:
        """
        Aggregate using trimmed mean
        """
        n = len(updates)
        n_trim = int(n * self.beta)
        
        if n_trim >= n // 2:
            raise ValueError(f"Beta too large: {self.beta}, would remove all clients")
        
        # Stack updates (n_clients × n_params)
        stacked = np.stack(updates, axis=0)
        
        # Sort along client axis
        sorted_updates = np.sort(stacked, axis=0)
        
        # Trim top and bottom
        if n_trim > 0:
            trimmed = sorted_updates[n_trim:-n_trim, :]
        else:
            trimmed = sorted_updates
        
        # Mean of remaining
        aggregated = np.mean(trimmed, axis=0)
        
        # Detection: Flag clients in trimmed regions
        # Compute distance from aggregated for each client
        distances = np.array([np.linalg.norm(update - aggregated) for update in updates])
        distance_threshold = np.percentile(distances, (1 - self.beta) * 100)
        
        predicted_malicious = [dist > distance_threshold for dist in distances]
        
        if self.verbose:
            print(f"\n[Trimmed-Mean] Beta: {self.beta}, Trim count: {n_trim}")
            print(f"[Trimmed-Mean] Distances: {distances}")
            print(f"[Trimmed-Mean] Threshold: {distance_threshold:.4f}")
        
        metrics = self.compute_detection_metrics(predicted_malicious, is_malicious)
        
        return aggregated
