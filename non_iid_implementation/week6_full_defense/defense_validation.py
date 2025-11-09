# Enhanced validation-based defense with norm filtering
import torch
import torch.nn.functional as F
import copy
import numpy as np

class ValidationDefense:
    """Validate client updates using held-out validation set + norm filtering"""
    
    def __init__(self, validation_loader, threshold=0.05, use_norm_filtering=True, norm_multiplier=3.0):
        self.validation_loader = validation_loader
        self.threshold = threshold  # Max acceptable loss increase
        self.use_norm_filtering = use_norm_filtering
        self.norm_multiplier = norm_multiplier  # Reject if norm > multiplier * median
    
    def validate_update(self, global_model, client_update):
        """Test if applying update improves or degrades model"""
        # Create temporary model with update applied
        temp_model = copy.deepcopy(global_model)
        temp_model.eval()
        
        # Apply client update
        for name, param in temp_model.named_parameters():
            param.data += client_update[name]
        
        # Compute loss before (global model)
        loss_before = self._compute_loss(global_model)
        
        # Compute loss after (with update)
        loss_after = self._compute_loss(temp_model)
        
        # Accept if loss doesn't increase too much
        loss_increase = loss_after - loss_before
        is_valid = loss_increase <= self.threshold
        
        return is_valid, loss_before, loss_after, loss_increase
    
    def _compute_loss(self, model):
        """Compute average loss on validation set"""
        model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.validation_loader:
                output = model(data)
                loss = F.cross_entropy(output, target, reduction='sum')
                total_loss += loss.item()
                total_samples += len(target)
        
        return total_loss / total_samples
