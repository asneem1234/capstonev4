"""
Flower Server for Quantum Federated Learning with Full Defense
Coordinates training and aggregates updates using FedAvg
Implements 3-layer cascading defense: Norm Filter + Adaptive + Fingerprints
"""

import torch
import numpy as np
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, Scalar
from typing import List, Tuple, Dict, Optional, Union
from quantum_model import create_model
from defense_norm_filter import NormFilter
from defense_adaptive import AdaptiveDefense
from defense_fingerprint_server import ServerFingerprintDefense


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
        self.model = create_model(n_qubits=config.N_QUBITS, n_layers=config.N_LAYERS)
        self.model.to(self.device)
        
        # Track metrics
        self.round_accuracies = []
        self.round_losses = []
        
        # Initialize defense layers if enabled
        self.defense_enabled = config.DEFENSE_ENABLED
        if self.defense_enabled:
            # Layer 0: Norm Filter
            if config.USE_NORM_FILTERING:
                self.norm_filter = NormFilter(
                    threshold_multiplier=config.NORM_THRESHOLD_MULTIPLIER
                )
            else:
                self.norm_filter = None
            
            # Layer 1: Adaptive Defense
            if config.USE_ADAPTIVE_DEFENSE:
                self.adaptive_defense = AdaptiveDefense(
                    threshold_std=config.ADAPTIVE_THRESHOLD_STD
                )
            else:
                self.adaptive_defense = None
            
            # Layer 2: Fingerprint Validation
            if config.USE_FINGERPRINTS:
                self.fingerprint_defense = ServerFingerprintDefense(
                    similarity_threshold=config.FINGERPRINT_SIMILARITY_THRESHOLD
                )
            else:
                self.fingerprint_defense = None
        else:
            self.norm_filter = None
            self.adaptive_defense = None
            self.fingerprint_defense = None
    
    def apply_cascading_defense(self, client_updates):
        """
        Apply 3-layer cascading defense to client updates
        
        Args:
            client_updates: List of dicts with keys:
                - 'client_id': int
                - 'params': list of numpy arrays
                - 'metrics': dict with training metrics
                - 'fingerprint': numpy array (optional)
        
        Returns:
            accepted_updates: List of updates that passed all defense layers
            defense_stats: Dict with statistics from each layer
        """
        accepted = client_updates
        defense_stats = {
            'layer0': None,
            'layer1': None,
            'layer2': None,
            'initial_count': len(client_updates),
            'final_count': 0
        }
        
        if not self.defense_enabled or len(client_updates) == 0:
            defense_stats['final_count'] = len(accepted)
            return accepted, defense_stats
        
        # Layer 0: Norm Filter (Fast pre-filter)
        if self.norm_filter is not None:
            accepted, rejected_ids, layer0_stats = self.norm_filter.filter_updates(accepted)
            defense_stats['layer0'] = layer0_stats
            
            print(f"\nDEFENSE Layer 0 (Norm Filter): {layer0_stats['accepted']}/{len(client_updates)} accepted")
            print(f"   Threshold: {layer0_stats['threshold']:.4f} (median={layer0_stats['median_norm']:.4f} x {self.config.NORM_THRESHOLD_MULTIPLIER})")
            if layer0_stats['rejected'] > 0:
                print(f"   Rejected: {rejected_ids}")
        
        # Layer 1: Adaptive Defense (Statistical outlier detection)
        if self.adaptive_defense is not None and len(accepted) > 0:
            accepted, rejected_ids, layer1_stats = self.adaptive_defense.filter_updates(accepted, self.model)
            defense_stats['layer1'] = layer1_stats
            
            print(f"DEFENSE Layer 1 (Adaptive): {layer1_stats['accepted']} survived")
            if layer1_stats['rejected'] > 0:
                print(f"   Rejected: {rejected_ids}")
        
        # Layer 2: Fingerprint Validation (Identity verification)
        if self.fingerprint_defense is not None and len(accepted) > 0:
            accepted, rejected_ids, layer2_stats = self.fingerprint_defense.validate_batch(accepted)
            defense_stats['layer2'] = layer2_stats
            
            print(f"DEFENSE Layer 2 (Fingerprints): {layer2_stats['accepted']} final")
            if layer2_stats['rejected'] > 0:
                print(f"   Rejected: {rejected_ids}")
        
        defense_stats['final_count'] = len(accepted)
        print(f"\nFINAL: {len(accepted)}/{len(client_updates)} updates accepted for aggregation\n")
        
        return accepted, defense_stats
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[FitRes, int]],
        failures: List[Union[Tuple[FitRes, int], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results using FedAvg with 3-layer defense
        
        Args:
            server_round: Current round number
            results: List of (client_proxy, fit_res) tuples
            failures: List of failed clients
        
        Returns:
            Aggregated parameters and metrics
        """
        # Print client metrics
        if self.config.VERBOSE:
            print(f"\n{'='*60}")
            print(f"Round {server_round} - Client Training Results")
            print(f"{'='*60}")
            
            for i, (client_proxy, fit_res) in enumerate(results):
                # fit_res is the FitRes object containing metrics
                if hasattr(fit_res, 'metrics') and fit_res.metrics:
                    metrics = fit_res.metrics
                    client_id = metrics.get("client_id", i)
                    loss = metrics.get("loss", 0)
                    acc = metrics.get("accuracy", 0)
                    norm = metrics.get("update_norm", 0)
                    num_samples = fit_res.num_examples
                    
                    print(f"  Client {client_id}: "
                          f"Loss={loss:.4f}, Acc={acc:.2f}%, "
                          f"Samples={num_samples}, Norm={norm:.4f}")
        
        # Apply defense if enabled
        if self.defense_enabled and len(results) > 0:
            # Convert Flower results to defense format
            from flwr.common import parameters_to_ndarrays
            client_updates = []
            for client_proxy, fit_res in results:
                params_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                metrics = fit_res.metrics if fit_res.metrics else {}
                
                update_dict = {
                    'client_id': metrics.get('client_id', 0),
                    'params': params_ndarrays,
                    'metrics': metrics,
                    'fingerprint': np.array(metrics.get('fingerprint', [])) if 'fingerprint' in metrics else None,
                    'num_examples': fit_res.num_examples
                }
                client_updates.append(update_dict)
            
            # Apply cascading defense
            accepted_updates, defense_stats = self.apply_cascading_defense(client_updates)
            
            # Convert back to Flower format
            from flwr.common import ndarrays_to_parameters
            filtered_results = []
            for update in accepted_updates:
                # Find original result
                for client_proxy, fit_res in results:
                    if fit_res.metrics and fit_res.metrics.get('client_id') == update['client_id']:
                        filtered_results.append((client_proxy, fit_res))
                        break
            
            # Use filtered results for aggregation
            results = filtered_results
        
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
        # Convert parameters to numpy arrays
        from flwr.common import parameters_to_ndarrays
        params_list = parameters_to_ndarrays(parameters)
        
        # Load parameters into model
        for param, new_param in zip(self.model.parameters(), params_list):
            param.data = torch.from_numpy(new_param).to(self.device)
        
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
    # Initialize global model with config parameters
    model = create_model(n_qubits=config.N_QUBITS, n_layers=config.N_LAYERS)
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
