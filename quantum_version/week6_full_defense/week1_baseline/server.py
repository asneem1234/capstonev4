"""
Flower Server for Quantum Federated Learning
Coordinates training and aggregates updates using FedAvg
"""

import torch
import numpy as np
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, Scalar
from typing import List, Tuple, Dict, Optional, Union
from quantum_model import create_model


class QuantumFedAvg(FedAvg):
    """
    Custom FedAvg strategy for quantum federated learning
    Includes evaluation on test set after each round
    """
    
    def __init__(self, test_loader, config, **kwargs):
        """
        Initialize quantum FedAvg strategy
        
        Args:
            test_loader: Test dataset loader
            config: Configuration module
            **kwargs: Additional arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model for server-side evaluation
        self.model = create_model()
        self.model.to(self.device)
        
        # Track metrics
        self.round_accuracies = []
        self.round_losses = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[FitRes, int]],
        failures: List[Union[Tuple[FitRes, int], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results using FedAvg
        
        Args:
            server_round: Current round number
            results: List of (FitRes, num_samples) tuples
            failures: List of failed clients
        
        Returns:
            Aggregated parameters and metrics
        """
        # Print client metrics
        if self.config.VERBOSE:
            print(f"\n{'='*60}")
            print(f"Round {server_round} - Client Training Results")
            print(f"{'='*60}")
            
            for i, (fit_res, num_samples) in enumerate(results):
                metrics = fit_res.metrics
                client_id = metrics.get("client_id", i)
                loss = metrics.get("loss", 0)
                acc = metrics.get("accuracy", 0)
                norm = metrics.get("update_norm", 0)
                
                print(f"  Client {client_id}: "
                      f"Loss={loss:.4f}, Acc={acc:.2f}%, "
                      f"Samples={num_samples}, Norm={norm:.4f}")
        
        # Aggregate using parent FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Evaluate aggregated model on test set
        if aggregated_parameters is not None:
            accuracy, loss = self._evaluate_global_model(aggregated_parameters)
            
            self.round_accuracies.append(accuracy)
            self.round_losses.append(loss)
            
            if self.config.VERBOSE:
                print(f"\n{'='*60}")
                print(f"Round {server_round} - Global Model Evaluation")
                print(f"{'='*60}")
                print(f"  Test Accuracy: {accuracy:.2f}%")
                print(f"  Test Loss: {loss:.4f}")
                print(f"{'='*60}\n")
            
            # Add to aggregated metrics
            aggregated_metrics["test_accuracy"] = accuracy
            aggregated_metrics["test_loss"] = loss
        
        return aggregated_parameters, aggregated_metrics
    
    def _evaluate_global_model(self, parameters: Parameters) -> Tuple[float, float]:
        """
        Evaluate global model on test set
        
        Args:
            parameters: Global model parameters
        
        Returns:
            (accuracy, loss) tuple
        """
        # Convert parameters to model weights
        params_list = parameters.tensors
        params_dict = zip(self.model.parameters(), params_list)
        
        for param, new_param in params_dict:
            param.data = torch.from_numpy(np.frombuffer(new_param, dtype=np.float32).reshape(param.shape)).to(self.device)
        
        # Evaluate
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        return accuracy, avg_loss
    
    def get_final_results(self) -> Dict[str, any]:
        """
        Get final training results
        
        Returns:
            Dictionary with accuracy and loss history
        """
        return {
            "accuracies": self.round_accuracies,
            "losses": self.round_losses,
            "final_accuracy": self.round_accuracies[-1] if self.round_accuracies else 0.0,
            "final_loss": self.round_losses[-1] if self.round_losses else 0.0
        }


def create_server_config(num_rounds: int, clients_per_round: int) -> ServerConfig:
    """
    Create Flower server configuration
    
    Args:
        num_rounds: Number of federated learning rounds
        clients_per_round: Number of clients to sample per round
    
    Returns:
        ServerConfig object
    """
    return ServerConfig(num_rounds=num_rounds)


def create_strategy(test_loader, config) -> QuantumFedAvg:
    """
    Create federated learning strategy
    
    Args:
        test_loader: Test dataset loader
        config: Configuration module
    
    Returns:
        QuantumFedAvg strategy
    """
    # Initialize global model
    model = create_model()
    initial_parameters = [param.cpu().detach().numpy() for param in model.parameters()]
    
    # Convert to Flower Parameters format
    from flwr.common import ndarrays_to_parameters
    initial_parameters_fl = ndarrays_to_parameters(initial_parameters)
    
    strategy = QuantumFedAvg(
        test_loader=test_loader,
        config=config,
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=0.0,  # No client-side evaluation
        min_fit_clients=config.CLIENTS_PER_ROUND,
        min_available_clients=config.NUM_CLIENTS,
        initial_parameters=initial_parameters_fl,
    )
    
    return strategy


if __name__ == "__main__":
    # Test server strategy
    print("Testing Quantum Server Strategy...")
    
    import config
    from data_loader import get_client_loaders
    
    # Load data
    _, test_loader = get_client_loaders(
        num_clients=5,
        alpha=config.DIRICHLET_ALPHA,
        batch_size=config.BATCH_SIZE
    )
    
    # Create strategy
    strategy = create_strategy(test_loader, config)
    
    print(f"Strategy created: {type(strategy).__name__}")
    print(f"Initial parameters set: {strategy.initial_parameters is not None}")
    
    print("\n✓ Server strategy test passed!")
