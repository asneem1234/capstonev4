# Client training with optional attack (Non-IID data)
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from config import Config
from attack import ModelPoisoningAttack

class Client:
    def __init__(self, client_id, data_loader):
        self.client_id = client_id
        self.data_loader = data_loader
        # Attack will be initialized per round if needed
        self.attack = None
    
    def train(self, global_model, is_malicious_this_round=False):
        """
        Train on local data, return update and metrics
        
        Args:
            global_model: The current global model
            is_malicious_this_round: Whether this client is malicious for this round
        """
        # Initialize attack for this round if malicious
        if is_malicious_this_round and Config.ATTACK_ENABLED:
            # Use aggressive model poisoning with gradient ascent
            # Options: 'gradient_ascent', 'scaled_poison', 'random_noise'
            self.attack = ModelPoisoningAttack(
                num_classes=10,
                attack_type='gradient_ascent',  # Most aggressive
                scale_factor=10.0  # Amplify the damage
            )
        else:
            self.attack = None
        # Copy global model
        model = copy.deepcopy(global_model)
        model.train()
        
        optimizer = optim.SGD(
            model.parameters(), 
            lr=Config.LEARNING_RATE
        )
        
        # Local training
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(Config.LOCAL_EPOCHS):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                # Apply attack if malicious
                if self.attack is not None:
                    data, target = self.attack.apply(data, target)
                
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
        for name, param in model.named_parameters():
            global_param = dict(global_model.named_parameters())[name]
            delta = param.data - global_param.data
            update[name] = delta
        
        # POISON THE UPDATE if malicious
        if self.attack is not None:
            update = self.attack.poison_update(update)
        
        # Compute update norm (after poisoning)
        update_norm = 0.0
        for name, delta in update.items():
            update_norm += torch.norm(delta).item() ** 2
        update_norm = update_norm ** 0.5
        
        train_acc = 100. * total_correct / total_samples
        avg_loss = total_loss / total_samples
        
        return update, train_acc, avg_loss, update_norm
