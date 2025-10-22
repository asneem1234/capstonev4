# Simple validation-based defense for fetal plane classification
import torch
import torch.nn.functional as F
import copy

class ValidationDefense:
    """Validate client updates using held-out validation set"""
    
    def __init__(self, validation_loader, threshold=0.15):
        """
        Args:
            validation_loader: DataLoader with validation data
            threshold: Max acceptable loss increase (higher for medical imaging)
        """
        self.validation_loader = validation_loader
        self.threshold = threshold
    
    def validate_update(self, global_model, client_update):
        """Test if applying update improves or degrades model"""
        # Create temporary model with update applied
        temp_model = copy.deepcopy(global_model)
        temp_model.eval()
        
        # Apply client update
        for name, param in temp_model.named_parameters():
            if name in client_update:
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
        
        return total_loss / total_samples if total_samples > 0 else float('inf')
