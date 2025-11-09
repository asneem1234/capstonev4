"""
Defense Layer 0: Norm-based Filtering
Fast pre-filter that rejects updates with abnormally large norms
"""

import numpy as np


class NormFilter:
    """
    Norm-based defense that filters out updates with abnormally large gradient norms.
    This is the first line of defense against gradient-based attacks.
    """
    
    def __init__(self, threshold_multiplier=3.0):
        """
        Initialize norm filter
        
        Args:
            threshold_multiplier: Multiplier for median norm (default: 3.0)
                                 Threshold = median_norm × multiplier
        """
        self.threshold_multiplier = threshold_multiplier
        self.rejection_history = []
    
    def filter_updates(self, client_updates):
        """
        Filter client updates based on gradient norm
        
        Args:
            client_updates: List of dicts with keys:
                - 'client_id': int
                - 'params': list of numpy arrays
                - 'metrics': dict with 'update_norm'
                - 'fingerprint': optional numpy array
        
        Returns:
            accepted_updates: List of updates that passed the filter
            rejected_ids: List of rejected client IDs
            stats: Dict with filtering statistics
        """
        if len(client_updates) == 0:
            return [], [], {'threshold': 0.0, 'median_norm': 0.0, 'accepted': 0, 'rejected': 0}
        
        # Extract norms from metrics
        norms = []
        for update in client_updates:
            norm = update['metrics'].get('update_norm', 0.0)
            norms.append(norm)
        
        # Calculate threshold based on median
        median_norm = np.median(norms)
        threshold = median_norm * self.threshold_multiplier
        
        # Filter updates
        accepted_updates = []
        rejected_ids = []
        
        for update, norm in zip(client_updates, norms):
            if norm <= threshold:
                accepted_updates.append(update)
            else:
                rejected_ids.append(update['client_id'])
        
        # Statistics
        stats = {
            'threshold': float(threshold),
            'median_norm': float(median_norm),
            'accepted': len(accepted_updates),
            'rejected': len(rejected_ids),
            'rejected_ids': rejected_ids,
            'norms': norms
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
