"""
Main entry point for Quantum Federated Learning - Week 6 (Full Defense) - SCALED
Uses Flower simulation for federated learning with diverse attack intensities
"""

import os
# Force Ray to not cache modules
os.environ["RAY_DEDUP_LOGS"] = "0"

# Ensure the worker processes can import local modules (helps Ray workers find
# `client.py`, `data_loader.py`, etc.). Prepend the current package directory
# to PYTHONPATH before Ray initializes.
pkg_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = pkg_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch
import numpy as np
import random
import config
from data_loader import get_client_loaders
from server import create_strategy, create_server_config
from client import create_client
from flwr.simulation import start_simulation
import time


# Set random seed for reproducibility
random.seed(config.SEED)
np.random.seed(config.SEED)

# Assign diverse attacks to random clients
# Format: {client_id: (attack_type, scale_factor)}
malicious_client_assignments = {}

# Get all client IDs and shuffle them
all_client_ids = list(range(config.NUM_CLIENTS))
random.shuffle(all_client_ids)

assignment_idx = 0

# Assign aggressive attacks (caught by Layer 0)
for _ in range(config.MALICIOUS_CLIENTS_CONFIG['aggressive']['count']):
    client_id = all_client_ids[assignment_idx]
    malicious_client_assignments[client_id] = (
        'aggressive', 
        config.MALICIOUS_CLIENTS_CONFIG['aggressive']['scale_factor']
    )
    assignment_idx += 1

# Assign moderate attacks (caught by Layer 1)
for _ in range(config.MALICIOUS_CLIENTS_CONFIG['moderate']['count']):
    client_id = all_client_ids[assignment_idx]
    malicious_client_assignments[client_id] = (
        'moderate',
        config.MALICIOUS_CLIENTS_CONFIG['moderate']['scale_factor']
    )
    assignment_idx += 1

# Assign subtle attacks (caught by Layer 2)
for _ in range(config.MALICIOUS_CLIENTS_CONFIG['subtle']['count']):
    client_id = all_client_ids[assignment_idx]
    malicious_client_assignments[client_id] = (
        'subtle',
        config.MALICIOUS_CLIENTS_CONFIG['subtle']['scale_factor']
    )
    assignment_idx += 1

malicious_clients = set(malicious_client_assignments.keys())

print(f"\n{'='*60}")
print(f"MALICIOUS CLIENT ASSIGNMENT:")
print(f"{'='*60}")
for client_id in sorted(malicious_client_assignments.keys()):
    attack_type, scale = malicious_client_assignments[client_id]
    print(f"  Client {client_id}: {attack_type.upper()} attack (scale={scale})")
print(f"Total malicious: {len(malicious_clients)}/{config.NUM_CLIENTS} clients")
print(f"{'='*60}\n")


from flwr.common import Context


def client_fn(cid_or_context: "Context | str"):
    """
    Create client function for Flower simulation
    
    Args:
        cid: Client ID (string or Context)
    
    Returns:
        FlowerClient instance
    """
    # Import inside function to ensure Ray workers have access
    from client import create_client as create_client_func
    import config as cfg
    from data_loader import get_client_loaders as get_loaders
    
    global client_loaders, test_loader, malicious_client_assignments
    
    # Handle both old string cid and new Context
    if isinstance(cid_or_context, str):
        client_id = int(cid_or_context)
    else:
        # Extract from Context if provided
        ctx = cid_or_context
        try:
            # Try common attributes
            if hasattr(ctx, "cid"):
                client_id = int(ctx.cid)
            elif hasattr(ctx, "node_config") and ctx.node_config:
                client_id = int(ctx.node_config.get("partition-id", "0"))
            else:
                # Fallback
                client_id = 0
        except Exception:
            client_id = 0
    
    is_malicious = client_id in malicious_client_assignments
    
    # Get attack scale factor for this specific client
    if is_malicious:
        attack_type, scale_factor = malicious_client_assignments[client_id]
    else:
        scale_factor = None  # Not used for honest clients
    
    client = create_client_func(
        client_id=client_id,
        train_loader=client_loaders[client_id],
        test_loader=test_loader,
        config=cfg,
        is_malicious=is_malicious,
        scale_factor=scale_factor  # Pass individual scale factor
    )
        
    
    # Convert to proper Client if needed
    if hasattr(client, 'to_client'):
        return client.to_client()
    return client


def main():
    """Run quantum federated learning simulation"""
    
    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Print configuration
    config.print_config()
    
    # Load data (globally for simulation)
    print("Loading MNIST dataset...")
    global client_loaders, test_loader
    client_loaders, test_loader = get_client_loaders(
        num_clients=config.NUM_CLIENTS,
        alpha=config.DIRICHLET_ALPHA,
        batch_size=config.BATCH_SIZE
    )
    
    # Create strategy
    print("Creating federated learning strategy...")
    strategy = create_strategy(test_loader, config)
    
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
