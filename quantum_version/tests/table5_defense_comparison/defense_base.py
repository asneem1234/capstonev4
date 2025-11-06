"""
Base class for all defense methods in Table 5 comparison
Provides common interface and evaluation metrics
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class DefenseBase(ABC):
    """
    Abstract base class for Byzantine defense methods
    
    All defense methods must implement:
    - aggregate(): Aggregate client updates with defense
    - get_detection_stats(): Return detection metrics
    """
    
    def __init__(self, name: str, verbose: bool = False):
        """
        Initialize defense method
        
        Args:
            name: Name of the defense method
            verbose: Print detailed logs
        """
        self.name = name
        self.verbose = verbose
        
        # Track statistics
        self.round_stats = []
        self.detection_history = []
        
    @abstractmethod
    def aggregate(self, updates: List[np.ndarray], weights: List[int], 
                  is_malicious: List[bool]) -> np.ndarray:
        """
        Aggregate client updates with defense mechanism
        
        Args:
            updates: List of client parameter updates (as flattened arrays)
            weights: List of number of samples per client
            is_malicious: Ground truth malicious flags (for evaluation)
        
        Returns:
            Aggregated update (flattened array)
        """
        pass
    
    def compute_detection_metrics(self, predicted_malicious: List[bool], 
                                   actual_malicious: List[bool]) -> Dict[str, float]:
        """
        Compute detection metrics
        
        Args:
            predicted_malicious: Predicted malicious flags
            actual_malicious: Ground truth malicious flags
        
        Returns:
            Dict with TP, FP, TN, FN, precision, recall, f1, detection_rate, fpr
        """
        n_clients = len(actual_malicious)
        
        # Confusion matrix
        tp = sum(1 for i in range(n_clients) if predicted_malicious[i] and actual_malicious[i])
        fp = sum(1 for i in range(n_clients) if predicted_malicious[i] and not actual_malicious[i])
        tn = sum(1 for i in range(n_clients) if not predicted_malicious[i] and not actual_malicious[i])
        fn = sum(1 for i in range(n_clients) if not predicted_malicious[i] and actual_malicious[i])
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        detection_rate = recall * 100.0  # Same as recall, in percentage
        fpr = fp / (fp + tn) * 100.0 if (fp + tn) > 0 else 0.0
        
        metrics = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "detection_rate": detection_rate,
            "fpr": fpr
        }
        
        return metrics
    
    def log_round(self, round_num: int, metrics: Dict[str, float]):
        """Log detection metrics for a round"""
        self.round_stats.append({
            "round": round_num,
            "method": self.name,
            **metrics
        })
        
        if self.verbose:
            print(f"\n[{self.name}] Round {round_num} Detection:")
            print(f"  Detection Rate: {metrics['detection_rate']:.1f}%")
            print(f"  FPR: {metrics['fpr']:.1f}%")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print(f"  TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all rounds
        
        Returns:
            Dict with averaged metrics
        """
        if not self.round_stats:
            return {
                "detection_rate": 0.0,
                "fpr": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
        
        # Average across rounds
        avg_detection_rate = np.mean([s["detection_rate"] for s in self.round_stats])
        avg_fpr = np.mean([s["fpr"] for s in self.round_stats])
        avg_f1 = np.mean([s["f1_score"] for s in self.round_stats])
        avg_precision = np.mean([s["precision"] for s in self.round_stats])
        avg_recall = np.mean([s["recall"] for s in self.round_stats])
        
        return {
            "method": self.name,
            "detection_rate": avg_detection_rate,
            "fpr": avg_fpr,
            "f1_score": avg_f1,
            "precision": avg_precision,
            "recall": avg_recall,
            "num_rounds": len(self.round_stats)
        }


class NoDefense(DefenseBase):
    """
    FedAvg with no defense (baseline)
    Simply aggregates all updates without filtering
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__("FedAvg (No Defense)", verbose)
    
    def aggregate(self, updates: List[np.ndarray], weights: List[int], 
                  is_malicious: List[bool]) -> np.ndarray:
        """
        Standard FedAvg: weighted average of all updates
        """
        total_samples = sum(weights)
        aggregated = np.zeros_like(updates[0])
        
        for update, weight in zip(updates, weights):
            aggregated += update * (weight / total_samples)
        
        # No detection happens - all clients accepted
        predicted_malicious = [False] * len(updates)
        metrics = self.compute_detection_metrics(predicted_malicious, is_malicious)
        
        return aggregated


def flatten_update(update_dict):
    """Flatten parameter dictionary to numpy array"""
    flat = []
    for name in sorted(update_dict.keys()):
        flat.append(update_dict[name].cpu().numpy().flatten())
    return np.concatenate(flat)


def unflatten_update(flat_array, template_dict):
    """Unflatten numpy array back to parameter dictionary"""
    import torch
    result = {}
    idx = 0
    for name in sorted(template_dict.keys()):
        shape = template_dict[name].shape
        size = template_dict[name].numel()
        result[name] = torch.from_numpy(flat_array[idx:idx+size].reshape(shape))
        idx += size
    return result
