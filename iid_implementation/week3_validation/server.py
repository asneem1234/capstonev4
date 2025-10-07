# Server with validation defense
import torch
import torch.nn.functional as F
from config import Config
from defense_validation import ValidationDefense

class Server:
    def __init__(self, model, validation_loader, test_loader):
        self.model = model
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        
        # Initialize defense
        if Config.DEFENSE_ENABLED:
            self.defense = ValidationDefense(
                validation_loader, 
                threshold=Config.VALIDATION_THRESHOLD
            )
        else:
            self.defense = None
    
    def aggregate(self, client_updates):
        """FedAvg with optional validation filtering"""
        validation_results = []
        
        # Filter updates if defense is enabled
        if self.defense:
            filtered_updates = []
            for i, update in enumerate(client_updates):
                is_valid, loss_before, loss_after, loss_increase = \
                    self.defense.validate_update(self.model, update)
                
                validation_results.append({
                    'client_id': i,
                    'valid': is_valid,
                    'loss_before': loss_before,
                    'loss_after': loss_after,
                    'loss_increase': loss_increase
                })
                
                if is_valid:
                    filtered_updates.append(update)
            
            updates_to_aggregate = filtered_updates
        else:
            updates_to_aggregate = client_updates
        
        # Check if we have any valid updates
        if len(updates_to_aggregate) == 0:
            print("  WARNING: No valid updates! Skipping aggregation.")
            return 0.0, validation_results
        
        # Initialize aggregated update
        aggregated_update = {}
        for name in updates_to_aggregate[0].keys():
            aggregated_update[name] = torch.zeros_like(
                updates_to_aggregate[0][name]
            )
        
        # Average valid updates
        num_valid = len(updates_to_aggregate)
        for update in updates_to_aggregate:
            for name in update.keys():
                aggregated_update[name] += update[name] / num_valid
        
        # Compute aggregated update norm
        agg_norm = 0.0
        for name in aggregated_update.keys():
            agg_norm += torch.norm(aggregated_update[name]).item() ** 2
        agg_norm = agg_norm ** 0.5
        
        # Apply to global model
        for name, param in self.model.named_parameters():
            param.data += aggregated_update[name]
        
        return agg_norm, validation_results
    
    def evaluate(self):
        """Test global model accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
        
        accuracy = 100. * correct / total
        return accuracy
