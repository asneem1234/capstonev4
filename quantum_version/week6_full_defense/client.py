"""
Flower Client for Quantum Federated Learning with Full Defense
Each client trains a quantum neural network locally
Malicious clients apply gradient ascent attack
Clients generate fingerprints for server-side validation (Layer 2 Defense)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import NumPyClient
import numpy as np
from quantum_model import HybridQuantumNet
from attack import ModelPoisoningAttack
from defense_fingerprint_client import ClientFingerprintDefense


class QuantumFlowerClient(NumPyClient):
    """
    Flower client with quantum neural network
    Can be honest or malicious based on client_id
    """
    
    def __init__(self, client_id, train_loader, test_loader, config, is_malicious=False, scale_factor=None):
        """
        Initialize quantum client
        
        Args:
            client_id: Unique client identifier
            train_loader: DataLoader for client's training data
            test_loader: DataLoader for test data (shared)
            config: Configuration module
            is_malicious: Whether this client is malicious
            scale_factor: Attack intensity for this specific client (overrides config.SCALE_FACTOR)
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.is_malicious = is_malicious
        
        # Create quantum model
        self.model = HybridQuantumNet(
            n_qubits=config.N_QUBITS,
            n_layers=config.N_LAYERS
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE
        )
        
        # Initialize attack if client is malicious
        if self.is_malicious and config.ATTACK_ENABLED:
            # Use client-specific scale factor if provided, otherwise use config default
            attack_scale = scale_factor if scale_factor is not None else config.SCALE_FACTOR
            self.attack = ModelPoisoningAttack(
                num_classes=config.NUM_CLASSES,
                attack_type=config.ATTACK_TYPE,
                scale_factor=attack_scale
            )
        else:
            self.attack = None
        
        # Initialize fingerprint defense if enabled (Layer 2)
        if config.USE_FINGERPRINTS:
            self.fingerprint_defense = ClientFingerprintDefense(
                projection_dim=config.FINGERPRINT_DIM
            )
            self.fingerprint_defense.initialize_projection(self.model)
        else:
            self.fingerprint_defense = None
    
    def get_parameters(self, config=None):
        """
        Get model parameters as numpy arrays
        Returns:
            List of numpy arrays (weights)
        """
        return [param.cpu().detach().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters):
        """
        Set model parameters from numpy arrays
        Args:
            parameters: List of numpy arrays (weights)
        """
        params_dict = zip(self.model.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.from_numpy(new_param).to(self.device)
    
    def fit(self, parameters, config):
        """
        Train model locally (with attack if malicious)
        
        Args:
            parameters: Global model parameters
            config: Training configuration
        
        Returns:
            Updated parameters, number of samples, metrics dict
        """
        # Store old parameters for attack
        old_parameters = [p.copy() for p in parameters]
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Train locally
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.config.LOCAL_EPOCHS):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # Gradient clipping for stability
                if hasattr(self.config, 'GRADIENT_CLIP'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / (len(self.train_loader) * self.config.LOCAL_EPOCHS)
        accuracy = 100.0 * correct / total
        
        # Get updated parameters
        updated_parameters = self.get_parameters()
        
        # Apply attack if malicious
        if self.attack is not None:
            updated_parameters = self.attack.poison_update(old_parameters, updated_parameters)
        
        # Calculate update norm (for defense)
        update_norm = self._calculate_update_norm(old_parameters, updated_parameters)
        
        # Compute fingerprint for integrity verification (Layer 2 defense)
        fingerprint = None
        if self.fingerprint_defense is not None:
            # Convert update to dictionary format
            update_dict = {}
            for idx, (name, _) in enumerate(self.model.named_parameters()):
                old_tensor = torch.from_numpy(old_parameters[idx])
                new_tensor = torch.from_numpy(updated_parameters[idx])
                update_dict[name] = new_tensor - old_tensor
            
            # Compute fingerprint
            fingerprint = self.fingerprint_defense.compute_fingerprint(
                update_dict,
                train_loss=avg_loss,
                train_acc=accuracy
            )
        
        # Return results
        num_samples = len(self.train_loader.dataset)
        metrics = {
            "client_id": self.client_id,
            "loss": avg_loss,
            "accuracy": accuracy,
            "update_norm": float(update_norm),
            "is_malicious": self.is_malicious
        }
        
        # Add fingerprint to metrics if available
        if fingerprint is not None:
            metrics["fingerprint"] = fingerprint.tolist()
        
        return updated_parameters, num_samples, metrics
    
    def evaluate(self, parameters, config):
        """
        Evaluate model on test set
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
        
        Returns:
            Loss, number of samples, metrics dict
        """
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluate
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        metrics = {
            "accuracy": accuracy
        }
        
        return avg_loss, total, metrics
    
    def _calculate_update_norm(self, old_params, new_params):
        """
        Calculate L2 norm of parameter update
        Args:
            old_params: List of numpy arrays (old parameters)
            new_params: List of numpy arrays (new parameters)
        Returns:
            L2 norm of update
        """
        update_norm = 0.0
        for old, new in zip(old_params, new_params):
            update = new - old
            update_norm += np.sum(update ** 2)
        return np.sqrt(update_norm)


def create_client(client_id, train_loader, test_loader, config, is_malicious=False, scale_factor=None):
    """
    Factory function to create Flower client
    
    Args:
        client_id: Unique client identifier
        train_loader: Client's training data
        test_loader: Test data (shared)
        config: Configuration module
        is_malicious: Whether client should apply attacks
        scale_factor: Attack intensity for this specific client (overrides config.SCALE_FACTOR)
    
    Returns:
        QuantumFlowerClient instance
    """
    return QuantumFlowerClient(client_id, train_loader, test_loader, config, is_malicious, scale_factor)


if __name__ == "__main__":
    # Test client creation
    print("Testing Quantum Flower Client...")
    
    import config
    from data_loader import get_client_loaders
    
    # Load data
    client_loaders, test_loader = get_client_loaders(
        num_clients=5,
        alpha=config.DIRICHLET_ALPHA,
        batch_size=config.BATCH_SIZE
    )
    
    # Create client
    client = create_client(0, client_loaders[0], test_loader, config)
    
    # Test get_parameters
    params = client.get_parameters()
    print(f"Number of parameter arrays: {len(params)}")
    print(f"First param shape: {params[0].shape}")
    
    # Test fit
    print("\nTraining for 1 epoch...")
    new_params, num_samples, metrics = client.fit(params, {})
    print(f"Samples: {num_samples}")
    print(f"Metrics: {metrics}")
    
    # Test evaluate
    print("\nEvaluating...")
    loss, num_test, eval_metrics = client.evaluate(new_params, {})
    print(f"Test loss: {loss:.4f}")
    print(f"Test metrics: {eval_metrics}")
    
    print("\n✓ Client test passed!")
