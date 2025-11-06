"""
RobustAvg (Geometric Median) Defense Implementation for Table 5
Compute geometric median using iterative Weiszfeld algorithm
"""

import numpy as np
from typing import List
from defense_base import DefenseBase


class RobustAvgDefense(DefenseBase):
    """
    RobustAvg: Geometric median computation
    
    Geometric median minimizes sum of Euclidean distances:
        argmin_x Σ ||x - x_i||
    
    More robust than arithmetic mean to outliers.
    Uses Weiszfeld's algorithm for iterative computation.
    """
    
    def __init__(self, max_iter: int = 100, tolerance: float = 1e-5, verbose: bool = False):
        """
        Initialize RobustAvg defense
        
        Args:
            max_iter: Maximum iterations for Weiszfeld algorithm
            tolerance: Convergence tolerance
            verbose: Print detailed logs
        """
        super().__init__("RobustAvg", verbose)
        self.max_iter = max_iter
        self.tolerance = tolerance
    
    def geometric_median(self, points: np.ndarray) -> np.ndarray:
        """
        Compute geometric median using Weiszfeld algorithm
        
        Args:
            points: (n_points, n_dims) array
        
        Returns:
            Geometric median (n_dims,)
        """
        # Initialize with arithmetic mean
        median = np.mean(points, axis=0)
        
        for iteration in range(self.max_iter):
            # Compute distances
            distances = np.linalg.norm(points - median, axis=1)
            
            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)
            
            # Weighted update
            weights = 1.0 / distances
            new_median = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)
            
            # Check convergence
            if np.linalg.norm(new_median - median) < self.tolerance:
                if self.verbose:
                    print(f"[RobustAvg] Converged in {iteration+1} iterations")
                break
            
            median = new_median
        
        return median
    
    def aggregate(self, updates: List[np.ndarray], weights: List[int], 
                  is_malicious: List[bool]) -> np.ndarray:
        """
        Aggregate using geometric median
        """
        # Stack updates (n_clients × n_params)
        stacked = np.stack(updates, axis=0)
        
        # Compute geometric median
        aggregated = self.geometric_median(stacked)
        
        # Detection: Flag clients far from geometric median
        distances = np.array([np.linalg.norm(update - aggregated) for update in updates])
        median_distance = np.median(distances)
        mad = np.median(np.abs(distances - median_distance))
        
        threshold = median_distance + 2.0 * mad
        predicted_malicious = [dist > threshold for dist in distances]
        
        if self.verbose:
            print(f"\n[RobustAvg] Distances from geometric median: {distances}")
            print(f"[RobustAvg] Threshold: {threshold:.4f}")
        
        metrics = self.compute_detection_metrics(predicted_malicious, is_malicious)
        
        return aggregated
