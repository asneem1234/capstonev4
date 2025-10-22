# Client training (Baseline - No Attack)
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from config import Config

class Client:
    def __init__(self, client_id, data_loader):
        self.client_id = client_id
        self.data_loader = data_loader
    
    def train(self, global_model):
        """Train on local data, return update and metrics"""
        # Copy global model
        model = copy.deepcopy(global_model)
        model.train()
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=Config.LEARNING_RATE
        )
        
        # Local training
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(Config.LOCAL_EPOCHS):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += len(target)
        
        # Compute update: Δw = w_local - w_global
        update = {}
        update_norm = 0.0
        for name, param in model.named_parameters():
            global_param = dict(global_model.named_parameters())[name]
            delta = param.data - global_param.data
            update[name] = delta
            update_norm += torch.norm(delta).item() ** 2
        
        update_norm = update_norm ** 0.5
        train_acc = 100. * total_correct / total_samples
        avg_loss = total_loss / total_samples
        
        return update, train_acc, avg_loss, update_norm
