"""
Defense Layer 2 (Server-Side): Fingerprint Validation
Validates client fingerprints against historical patterns to detect impersonation
"""

import numpy as np
from collections import defaultdict


class ServerFingerprintDefense:
    """
    Server-side fingerprint validation.
    Maintains historical fingerprints per client and validates new ones against history.
    """
    
    def __init__(self, similarity_threshold=0.7, max_history=10):
        """
        Initialize fingerprint validation
        
        Args:
            similarity_threshold: Minimum cosine similarity to accept (default: 0.7)
            max_history: Maximum number of historical fingerprints to store per client
        """
        self.similarity_threshold = similarity_threshold
        self.max_history = max_history
        self.fingerprint_history = defaultdict(list)  # {client_id: [fp1, fp2, ...]}
        self.rejection_history = []
    
    def validate_fingerprint(self, client_id, new_fingerprint):
        """
        Validate a single fingerprint against client's history
        
        Args:
            client_id: Client identifier
            new_fingerprint: numpy array (fingerprint to validate)
        
        Returns:
            is_valid: Boolean indicating if fingerprint is valid
            similarity_score: Average similarity to historical fingerprints
        """
        if new_fingerprint is None:
            return False, 0.0
        
        # If no history, accept and store (first round or new client)
        if len(self.fingerprint_history[client_id]) == 0:
            self.fingerprint_history[client_id].append(new_fingerprint)
            return True, 1.0
        
        # Calculate similarity to all historical fingerprints
        similarities = []
        for historic_fp in self.fingerprint_history[client_id]:
            # Cosine similarity
            similarity = np.dot(new_fingerprint, historic_fp) / (
                np.linalg.norm(new_fingerprint) * np.linalg.norm(historic_fp) + 1e-10
            )
            similarities.append(similarity)
        
        # Average similarity
        avg_similarity = np.mean(similarities)
        
        # Validate against threshold
        is_valid = avg_similarity >= self.similarity_threshold
        
        # If valid, add to history (maintain max_history size)
        if is_valid:
            self.fingerprint_history[client_id].append(new_fingerprint)
            if len(self.fingerprint_history[client_id]) > self.max_history:
                self.fingerprint_history[client_id].pop(0)  # Remove oldest
        
        return is_valid, float(avg_similarity)
    
    def validate_batch(self, client_updates):
        """
        Validate fingerprints for a batch of client updates
        
        Args:
            client_updates: List of dicts with 'client_id' and 'fingerprint'
        
        Returns:
            accepted_updates: Updates with valid fingerprints
            rejected_ids: List of rejected client IDs
            stats: Dictionary with validation statistics
        """
        if len(client_updates) == 0:
            return [], [], {'accepted': 0, 'rejected': 0}
        
        accepted_updates = []
        rejected_ids = []
        similarities = []
        
        for update in client_updates:
            client_id = update['client_id']
            fingerprint = update.get('fingerprint', None)
            
            is_valid, similarity = self.validate_fingerprint(client_id, fingerprint)
            similarities.append(similarity)
            
            if is_valid:
                accepted_updates.append(update)
            else:
                rejected_ids.append(client_id)
        
        # Statistics
        stats = {
            'threshold': self.similarity_threshold,
            'similarities': similarities,
            'avg_similarity': float(np.mean(similarities)) if similarities else 0.0,
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
            'rejection_rate': total_rejected / (total_accepted + total_rejected) if (total_accepted + total_rejected) > 0 else 0.0,
            'clients_tracked': len(self.fingerprint_history)
        }
