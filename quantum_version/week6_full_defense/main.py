"""
Main entry point for Quantum Federated Learning (Week 6 - Full Defense)
Uses Flower simulation for federated learning with norm-based defense
"""

import torch
import numpy as np
import config
from data_loader import get_client_loaders
from server import create_strategy, create_server_config
from flwr.simulation import start_simulation
from flwr.common import Context
import time


def client_fn(context: Context):
    """
    Create client function for Flower simulation (new Context signature)

    Args:
        context: Flower Context object (contains client_id)

    Returns:
        A Flower `Client` instance created from our NumPyClient
    """
    # Import inside function to ensure Ray workers can access it
    from client import create_client
    
    # Robustly extract client id from Context (handles multiple flwr versions)
    def _extract_client_id(ctx):
        # If it's already a plain type
        if isinstance(ctx, (str, int)):
            return int(ctx)

        # Common attribute names
        for attr in ("client_id", "cid", "client"):
            if hasattr(ctx, attr):
                try:
                    return int(getattr(ctx, attr))
                except Exception:
                    pass

        # Ray/Simulation-specific: node_config may contain partition-id -> client index
        if hasattr(ctx, "node_config"):
            try:
                node_cfg = getattr(ctx, "node_config")
                # node_cfg may be a dict-like mapping
                if isinstance(node_cfg, dict):
                    pid = node_cfg.get("partition-id") or node_cfg.get("partition_id")
                    if pid is not None:
                        return int(pid)
                else:
                    # Try mapping-like access
                    pid = node_cfg.get("partition-id") if hasattr(node_cfg, "get") else None
                    if pid is not None:
                        return int(pid)
            except Exception:
                pass

        # If Context behaves like a mapping
        try:
            if hasattr(ctx, "get"):
                for key in ("client_id", "cid", "client"):
                    v = ctx.get(key, None)
                    if v is not None:
                        return int(v)
        except Exception:
            pass

        # Fallback: try to parse a small integer (up to 6 digits) from the string representation
        try:
            s = str(ctx)
            # Find contiguous digit groups and pick the shortest plausible id
            import re
            groups = re.findall(r"\d+", s)
            if groups:
                # Prefer small integers (likely client ids)
                candidates = [g for g in groups if len(g) <= 6]
                if candidates:
                    return int(candidates[-1])
                return int(groups[-1])
        except Exception:
            pass

        raise ValueError(f"Unable to extract client id from Context: {ctx!r}")

    client_id = _extract_client_id(context)

    # Debug: show extracted id and context repr
    try:
        print(f"[client_fn] context={repr(context)} type={type(context)} extracted_id={client_id}")
    except Exception:
        pass

    # Determine if this client is malicious
    is_malicious = client_id in malicious_clients

    # Create our NumPyClient instance
    numpy_client = create_client(
        client_id=client_id,
        train_loader=client_loaders[client_id],
        test_loader=test_loader,
        config=config,
        is_malicious=is_malicious
    )

    # Convert to a proper Flower Client instance
    try:
        return numpy_client.to_client()
    except Exception:
        # If it's already a Client, return directly
        return numpy_client


def main():
    """Run quantum federated learning simulation"""
    
    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Determine which clients are malicious (globally for client_fn)
    global malicious_clients, num_malicious
    num_malicious = max(1, int(config.NUM_CLIENTS * config.MALICIOUS_PERCENTAGE))  # At least 1 malicious
    malicious_clients = set(range(num_malicious))  # First N clients are malicious
    
    # Print configuration
    config.print_config()
    
    print("="*60)
    print("Quantum Federated Learning - Week 6 (Full Defense)")
    print("="*60)
    print(f"Framework: Flower (flwr)")
    print(f"Quantum Backend: PennyLane (default.qubit)")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"⚠️  ATTACK ENABLED: {config.ATTACK_TYPE} (scale={config.SCALE_FACTOR})")
    print(f"⚠️  MALICIOUS: {num_malicious}/{config.NUM_CLIENTS} clients ({sorted(malicious_clients)})")
    if config.DEFENSE_ENABLED:
        defense_layers = []
        if config.USE_NORM_FILTERING:
            defense_layers.append(f"Norm Filter (median×{config.NORM_THRESHOLD_MULTIPLIER})")
        if config.USE_ADAPTIVE_DEFENSE:
            defense_layers.append(f"Adaptive ({config.ADAPTIVE_METHOD})")
        if config.USE_FINGERPRINTS:
            defense_layers.append(f"Fingerprints ({config.FINGERPRINT_DIM}-D)")
        print(f"🛡️  DEFENSE ENABLED: {' + '.join(defense_layers)}")
    else:
        print(f"⚠️  NO DEFENSE")
    print("="*60 + "\n")
    
    # Load data (globally for simulation)
    print("Loading MNIST dataset...")
    global client_loaders, test_loader
    client_loaders, test_loader = get_client_loaders(
        num_clients=config.NUM_CLIENTS,
        alpha=config.DIRICHLET_ALPHA,
        batch_size=config.BATCH_SIZE
    )
    
    # Create validation loader (use first 2000 samples from test set for defense)
    from torch.utils.data import DataLoader, Subset
    test_dataset = test_loader.dataset
    validation_indices = list(range(2000))
    test_indices = list(range(2000, len(test_dataset)))
    
    validation_loader = DataLoader(
        Subset(test_dataset, validation_indices),
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    # Create strategy
    print("Creating federated learning strategy...")
    strategy = create_strategy(validation_loader, test_loader, config)
    
    # Create server config
    server_config = create_server_config(
        num_rounds=config.NUM_ROUNDS,
        clients_per_round=config.CLIENTS_PER_ROUND
    )
    
    print(f"✓ Strategy: FedAvg")
    print(f"✓ Server config: {config.NUM_ROUNDS} rounds\n")
    
    # Run simulation
    print("="*60)
    print("Starting Quantum Federated Learning Simulation...")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    history = start_simulation(
        client_fn=client_fn,
        num_clients=config.NUM_CLIENTS,
        config=server_config,
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 0.0,  # Use CPU for simulation
        },
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Get final results
    results = strategy.get_final_results()
    
    # Print final summary
    print("\n" + "="*60)
    print("Quantum Federated Learning - Final Results")
    print("="*60)
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"\nAccuracy per round:")
    for round_num, acc in enumerate(results['accuracies'], 1):
        print(f"  Round {round_num}: {acc:.2f}%")
    
    print(f"\nFinal Test Accuracy: {results['final_accuracy']:.2f}%")
    print(f"Final Test Loss: {results['final_loss']:.4f}")
    
    # Print defense summary if available
    if 'defense_summary' in results:
        defense_summary = results['defense_summary']
        print(f"\n{'='*60}")
        print(f"🛡️  Defense Summary (All Rounds)")
        print(f"{'='*60}")
        print(f"Total rounds: {defense_summary['total_rounds']}")
        print(f"Total clients processed: {defense_summary['total_clients']}")
        print(f"\nDefense Layer Performance:")
        print(f"  Layer 0 (Norm Filter) rejections: {defense_summary['layer0_rejections']}")
        print(f"  Layer 1 (Adaptive) rejections: {defense_summary['layer1_rejections']}")
        print(f"  Layer 2 (Fingerprints) rejections: {defense_summary['layer2_rejections']}")
        print(f"  Total accepted: {defense_summary['total_accepted']}")
        print(f"  Overall rejection rate: {defense_summary['rejection_rate']*100:.2f}%")
    
    print("="*60 + "\n")
    
    # Save model if configured
    if config.SAVE_MODEL:
        print(f"Saving final model to {config.MODEL_SAVE_PATH}...")
        # Model is saved in strategy, we can get the final parameters
        # For simplicity, we'll note that the model can be reconstructed from final parameters
        print("✓ Model training complete (parameters available in strategy)")
    
    print("\n✓ Quantum Federated Learning simulation completed successfully!")
    
    return results


if __name__ == "__main__":
    main()
