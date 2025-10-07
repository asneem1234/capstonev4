# Simple server - FedAvg aggregation only
import torch
import torch.nn.functional as F
from config import Config

class Server:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
    
    def aggregate(self, client_updates):
        """Simple FedAvg: average all updates"""
        # Initialize aggregated update
        aggregated_update = {}
        for name in client_updates[0].keys():
            aggregated_update[name] = torch.zeros_like(
                client_updates[0][name]
            )
        
        # Average updates
        num_clients = len(client_updates)
        for update in client_updates:
            for name in update.keys():
                aggregated_update[name] += update[name] / num_clients
        
        # Compute aggregated update norm
        agg_norm = 0.0
        for name in aggregated_update.keys():
            agg_norm += torch.norm(aggregated_update[name]).item() ** 2
        agg_norm = agg_norm ** 0.5
        
        # Apply to global model
        for name, param in self.model.named_parameters():
            param.data += aggregated_update[name]
        
        return agg_norm
    
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
