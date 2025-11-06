"""
Flower Client for Quantum Federated Learning
Each client trains a quantum neural network locally
"""

import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import NumPyClient
import numpy as np
from quantum_model import HybridQuantumNet


class QuantumFlowerClient(NumPyClient):
    """
    Flower client with quantum neural network
    """
    
    def __init__(self, client_id, train_loader, test_loader, config):
        """
        Initialize quantum client
        
        Args:
            client_id: Unique client identifier
            train_loader: DataLoader for client's training data
            test_loader: DataLoader for test data (shared)
            config: Configuration module
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        
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
        Train model locally
        
        Args:
            parameters: Global model parameters
            config: Training configuration
        
        Returns:
            Updated parameters, number of samples, metrics dict
        """
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
        
        # Calculate update norm (for defense)
        update_norm = self._calculate_update_norm(parameters, updated_parameters)
        
        # Return results
        num_samples = len(self.train_loader.dataset)
        metrics = {
            "client_id": self.client_id,
            "loss": avg_loss,
            "accuracy": accuracy,
            "update_norm": float(update_norm)
        }
        
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


def create_client(client_id, train_loader, test_loader, config):
    """
    Factory function to create Flower client
    
    Args:
        client_id: Unique client identifier
        train_loader: Client's training data
        test_loader: Test data (shared)
        config: Configuration module
    
    Returns:
        QuantumFlowerClient instance
    """
    return QuantumFlowerClient(client_id, train_loader, test_loader, config)


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
