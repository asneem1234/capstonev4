"""
Flower Server for Quantum Federated Learning with QuantumDefend PLUS v2
3-Layer Cascading Defense: Norm filtering + Adaptive detection + Fingerprint verification
"""

import torch
import numpy as np
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, Scalar
from typing import List, Tuple, Dict, Optional, Union
from quantum_model import create_model
from defense_adaptive import AdaptiveDefense
from defense_fingerprint_client import ClientSideFingerprintDefense


class QuantumFedAvgDefended(FedAvg):
    """
    Custom FedAvg strategy for quantum federated learning with QuantumDefend PLUS v2
    
    3-Layer Cascading Defense Architecture:
    - Layer 0: Fast norm-based filtering (removes obvious 50x attacks)
    - Layer 1: Adaptive 6-feature anomaly detection (removes sophisticated 2-10x attacks)
    - Layer 2: Client fingerprint verification (removes stealthy mimicry attacks)
    """
    
    def __init__(self, validation_loader, test_loader, config, **kwargs):
        """
        Initialize quantum FedAvg strategy with multi-layer defense
        
        Args:
            validation_loader: Validation dataset for defense (loss increase computation)
            test_loader: Test dataset loader
            config: Configuration module
            **kwargs: Additional arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model for server-side evaluation
        self.model = create_model()
        self.model.to(self.device)
        
        # Initialize multi-layer defense
        if config.DEFENSE_ENABLED:
            if config.USE_ADAPTIVE_DEFENSE:
                self.adaptive_defense = AdaptiveDefense(validation_loader)
            else:
                self.adaptive_defense = None
            
            if config.USE_FINGERPRINTS:
                self.fingerprint_defense = ClientSideFingerprintDefense(
                    fingerprint_dim=config.FINGERPRINT_DIM
                )
            else:
                self.fingerprint_defense = None
        else:
            self.adaptive_defense = None
            self.fingerprint_defense = None
        
        # Track metrics
        self.round_accuracies = []
        self.round_losses = []
        self.defense_stats = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[FitRes, int]],
        failures: List[Union[Tuple[FitRes, int], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results using FedAvg with defense
        
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
            
            for i, (client_proxy, fit_res) in enumerate(results):
                # fit_res is the FitRes object containing metrics
                if hasattr(fit_res, 'metrics') and fit_res.metrics:
                    metrics = fit_res.metrics
                    client_id = metrics.get("client_id", i)
                    loss = metrics.get("loss", 0)
                    acc = metrics.get("accuracy", 0)
                    norm = metrics.get("update_norm", 0)
                    is_mal = metrics.get("is_malicious", False)
                    mal_str = " ⚠️ MALICIOUS" if is_mal else " ✓ HONEST"
                    
                    print(f"  Client {client_id}: "
                          f"Loss={loss:.4f}, Acc={acc:.2f}%, "
                          f"Norm={norm:.4f}{mal_str}")
        
        # ===== QUANTUMDEFEND PLUS v2: 3-Layer Cascading Defense =====
        defense_stats = {
            "round": server_round,
            "total_clients": len(results),
            "layer0_rejected": [],
            "layer1_rejected": [],
            "layer2_rejected": [],
            "final_accepted": [],
            "final_rejected": []
        }
        
        if self.config.USE_NORM_FILTERING or self.adaptive_defense is not None or self.fingerprint_defense is not None:
            # Extract client updates and metadata
            client_updates = []
            client_metadata = {"losses": [], "accuracies": [], "norms": []}
            client_fingerprints = []
            
            from flwr.common import parameters_to_ndarrays
            for client_proxy, fit_res in results:
                # Convert parameters to update dictionary
                params_list = parameters_to_ndarrays(fit_res.parameters)
                update_dict = {}
                for idx, (name, _) in enumerate(self.model.named_parameters()):
                    update_dict[name] = torch.from_numpy(params_list[idx])
                
                client_updates.append(update_dict)
                
                # Extract metadata
                metrics = fit_res.metrics
                client_metadata["losses"].append(metrics.get("loss", 0.0))
                client_metadata["accuracies"].append(metrics.get("accuracy", 0.0))
                client_metadata["norms"].append(metrics.get("update_norm", 0.0))
                
                # Extract fingerprint (if available)
                if "fingerprint" in metrics:
                    client_fingerprints.append(np.array(metrics["fingerprint"]))
            
            # ===== Layer 0: Fast Norm-Based Filtering =====
            honest_indices = list(range(len(results)))
            if self.config.USE_NORM_FILTERING:
                norms = np.array(client_metadata["norms"])
                median_norm = np.median(norms)
                threshold = median_norm * self.config.NORM_THRESHOLD_MULTIPLIER
                
                # Filter by norm
                layer0_accepted = []
                for idx in honest_indices:
                    if norms[idx] <= threshold:
                        layer0_accepted.append(idx)
                    else:
                        defense_stats["layer0_rejected"].append(idx)
                
                honest_indices = layer0_accepted
                
                if self.config.VERBOSE:
                    print(f"\n🛡️  Layer 0 (Norm Filter): {len(honest_indices)}/{len(results)} accepted")
                    print(f"   Threshold: {threshold:.4f} (median={median_norm:.4f} × {self.config.NORM_THRESHOLD_MULTIPLIER})")
                    print(f"   Rejected: {defense_stats['layer0_rejected']}")
            
            # ===== Layer 1: Adaptive Anomaly Detection =====
            if self.adaptive_defense is not None and len(honest_indices) > 0:
                # Extract 6 features (only for clients that passed Layer 0)
                features = self.adaptive_defense.compute_update_features(
                    [client_updates[i] for i in honest_indices], 
                    self.model, 
                    {
                        "losses": [client_metadata["losses"][i] for i in honest_indices],
                        "accuracies": [client_metadata["accuracies"][i] for i in honest_indices]
                    }
                )
                
                # Detect anomalies
                honest_idx_local, malicious_idx_local, sep_factor, diagnostics = \
                    self.adaptive_defense.detect_malicious_adaptive(
                        features,
                        method=self.config.ADAPTIVE_METHOD
                    )
                
                # Convert local indices back to global indices
                layer1_accepted = [honest_indices[i] for i in honest_idx_local]
                layer1_rejected_global = [honest_indices[i] for i in malicious_idx_local]
                defense_stats["layer1_rejected"] = layer1_rejected_global
                defense_stats["layer1_diagnostics"] = diagnostics
                defense_stats["separation_factor"] = sep_factor
                
                honest_indices = layer1_accepted
                
                if self.config.VERBOSE:
                    print(f"\n🛡️  Layer 1 (Adaptive): {len(honest_indices)}/{len(results)} accepted")
                    print(f"   Method: {self.config.ADAPTIVE_METHOD}")
                    print(f"   Rejected: {layer1_rejected_global}")
                    print(f"   Separation Factor: {sep_factor:.2f}x")
            
            # ===== Layer 2: Fingerprint Verification =====
            if self.fingerprint_defense is not None and len(client_fingerprints) > 0:
                # Verify fingerprints for honest clients from Layer 1
                verified_indices = []
                for idx in honest_indices:
                    is_valid, actual_fp, similarity = self.fingerprint_defense.verify_fingerprint(
                        client_updates[idx],
                        client_fingerprints[idx],
                        tolerance=self.config.FINGERPRINT_TOLERANCE
                    )
                    
                    if is_valid:
                        verified_indices.append(idx)
                    else:
                        defense_stats["layer2_rejected"].append(idx)
                
                # Cluster fingerprints among verified clients
                if len(verified_indices) >= 2:
                    verified_fingerprints = [client_fingerprints[i] for i in verified_indices]
                    verified_metadata = {
                        "losses": [client_metadata["losses"][i] for i in verified_indices],
                        "accuracies": [client_metadata["accuracies"][i] for i in verified_indices]
                    }
                    
                    main_cluster, outliers_local = self.fingerprint_defense.cluster_fingerprints(
                        verified_fingerprints,
                        threshold=self.config.FINGERPRINT_THRESHOLD,
                        metadata=verified_metadata
                    )
                    
                    # Convert local indices back to global indices
                    final_accepted = [verified_indices[i] for i in main_cluster]
                    layer2_outliers = [verified_indices[i] for i in outliers_local]
                    defense_stats["layer2_rejected"].extend(layer2_outliers)
                else:
                    final_accepted = verified_indices
                
                honest_indices = final_accepted
                
                if self.config.VERBOSE:
                    print(f"🛡️  Layer 2 (Fingerprints): {len(honest_indices)}/{len(results)} accepted")
                    print(f"   Verification rejected: {len(defense_stats['layer2_rejected'])}")
            
            # Update results to only include accepted clients
            defense_stats["final_accepted"] = honest_indices
            defense_stats["final_rejected"] = [i for i in range(len(results)) if i not in honest_indices]
            self.defense_stats.append(defense_stats)
            
            results = [results[i] for i in honest_indices]
            
            if self.config.VERBOSE:
                print(f"\n✅ Final: {len(results)}/{defense_stats['total_clients']} updates accepted for aggregation")
        
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
        results = {
            "accuracies": self.round_accuracies,
            "losses": self.round_losses,
            "final_accuracy": self.round_accuracies[-1] if self.round_accuracies else 0.0,
            "final_loss": self.round_losses[-1] if self.round_losses else 0.0
        }
        
        # Add defense statistics if available
        if self.adaptive_defense is not None or self.fingerprint_defense is not None:
            results["defense_stats"] = self.defense_stats
            
            # Compute defense summary
            total_rounds = len(self.defense_stats)
            total_clients = sum(stat["total_clients"] for stat in self.defense_stats)
            layer0_rejections = sum(len(stat["layer0_rejected"]) for stat in self.defense_stats)
            layer1_rejections = sum(len(stat["layer1_rejected"]) for stat in self.defense_stats)
            layer2_rejections = sum(len(stat["layer2_rejected"]) for stat in self.defense_stats)
            total_accepted = sum(len(stat["final_accepted"]) for stat in self.defense_stats)
            
            results["defense_summary"] = {
                "total_rounds": total_rounds,
                "total_clients": total_clients,
                "layer0_rejections": layer0_rejections,
                "layer1_rejections": layer1_rejections,
                "layer2_rejections": layer2_rejections,
                "total_accepted": total_accepted,
                "rejection_rate": 1.0 - (total_accepted / total_clients) if total_clients > 0 else 0.0
            }
        
        return results


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


def create_strategy(validation_loader, test_loader, config) -> QuantumFedAvgDefended:
    """
    Create federated learning strategy with QuantumDefend PLUS
    
    Args:
        validation_loader: Validation dataset for defense
        test_loader: Test dataset loader
        config: Configuration module
    
    Returns:
        QuantumFedAvgDefended strategy
    """
    # Initialize global model
    model = create_model()
    initial_parameters = [param.cpu().detach().numpy() for param in model.parameters()]
    
    # Convert to Flower Parameters format
    from flwr.common import ndarrays_to_parameters
    initial_parameters_fl = ndarrays_to_parameters(initial_parameters)
    
    strategy = QuantumFedAvgDefended(
        validation_loader=validation_loader,
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
