# Main training loop - Non-IID with Label Flipping Attack
import torch
import numpy as np
from config import Config
from model import get_model
from data_loader import get_client_loaders
from client import Client
from server import Server

def select_random_malicious_clients(num_clients, malicious_percentage):
    """
    Randomly select a percentage of clients to be malicious.
    
    Args:
        num_clients: Total number of clients
        malicious_percentage: Percentage of clients to be malicious (e.g., 0.2 for 20%)
    
    Returns:
        List of malicious client IDs
    """
    num_malicious = int(num_clients * malicious_percentage)
    malicious_clients = np.random.choice(num_clients, num_malicious, replace=False).tolist()
    return sorted(malicious_clients)

def main():
    print("=" * 70)
    print("Federated Learning - FETAL PLANE CLASSIFICATION")
    print("NON-IID with LABEL FLIPPING ATTACK")
    print("=" * 70)
    print(f"Clients: {Config.NUM_CLIENTS}")
    print(f"Rounds: {Config.NUM_ROUNDS}")
    print(f"Local epochs: {Config.LOCAL_EPOCHS}")
    print(f"Data Distribution: NON-IID (Dirichlet α={Config.DIRICHLET_ALPHA})")
    print(f"Attack enabled: {Config.ATTACK_ENABLED}")
    if Config.RANDOM_MALICIOUS:
        print(f"Malicious clients: RANDOM {Config.MALICIOUS_PERCENTAGE*100:.0f}% per round "
              f"(~{int(Config.NUM_CLIENTS * Config.MALICIOUS_PERCENTAGE)} clients)")
    else:
        print(f"Malicious clients: {Config.MALICIOUS_CLIENTS}")
    print(f"Defense enabled: {Config.DEFENSE_ENABLED}")
    print("=" * 70)
    print("⚠️  WARNING: Malicious clients will flip labels!")
    print("   Expected: Model accuracy will degrade significantly")
    print("=" * 70)
    
    # Load data (Non-IID with Dirichlet distribution)
    print("\nLoading fetal plane data...")
    client_loaders, test_loader = get_client_loaders(
        Config.NUM_CLIENTS, 
        alpha=Config.DIRICHLET_ALPHA
    )
    
    # Initialize global model
    print("\nInitializing global model...")
    global_model = get_model(
        num_classes=Config.NUM_CLASSES, 
        pretrained=Config.PRETRAINED
    )
    
    # Create clients
    print("Creating clients...")
    clients = [Client(i, client_loaders[i]) for i in range(Config.NUM_CLIENTS)]
    
    # Create server
    server = Server(global_model, test_loader)
    
    # Initial evaluation
    initial_acc = server.evaluate()
    print(f"\nInitial Test Accuracy: {initial_acc:.2f}%")
    print("=" * 70)
    
    # Training rounds
    prev_accuracy = initial_acc
    for round_num in range(Config.NUM_ROUNDS):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}/{Config.NUM_ROUNDS}")
        print(f"{'='*70}")
        
        # Randomly select malicious clients for this round
        if Config.RANDOM_MALICIOUS:
            malicious_clients_this_round = select_random_malicious_clients(
                Config.NUM_CLIENTS, 
                Config.MALICIOUS_PERCENTAGE
            )
            print(f"\n[MALICIOUS SELECTION]")
            print(f"  Randomly selected {len(malicious_clients_this_round)} malicious clients: {malicious_clients_this_round}")
        else:
            malicious_clients_this_round = Config.MALICIOUS_CLIENTS
        
        # Clients train
        client_updates = []
        print("\n[CLIENT TRAINING]")
        for i, client in enumerate(clients):
            is_malicious = i in malicious_clients_this_round
            update, train_acc, train_loss, update_norm = client.train(global_model, is_malicious)
            client_updates.append(update)
            malicious_tag = " [MALICIOUS]" if is_malicious else ""
            print(f"  Client {i}{malicious_tag}: Train Acc={train_acc:.2f}%, "
                  f"Loss={train_loss:.4f}, Update Norm={update_norm:.4f}")
        
        # Server aggregates (no defense - accepts all updates)
        print("\n[SERVER AGGREGATION]")
        agg_norm = server.aggregate(client_updates)
        print(f"  Aggregated {len(client_updates)} client updates")
        print(f"  ⚠️  WARNING: No defense - malicious updates included!")
        print(f"  Aggregated Update Norm: {agg_norm:.4f}")
        
        # Evaluate
        print("\n[GLOBAL EVALUATION]")
        test_accuracy = server.evaluate()
        print(f"  Global Test Accuracy: {test_accuracy:.2f}%")
        
        change = test_accuracy - prev_accuracy
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"  Change from previous round: {direction} {change:+.2f}%")
        prev_accuracy = test_accuracy
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Initial Test Accuracy: {initial_acc:.2f}%")
    print(f"Final Test Accuracy:   {test_accuracy:.2f}%")
    improvement = test_accuracy - initial_acc
    print(f"Total Change:          {improvement:+.2f}%")
    print("=" * 70)
    
    if improvement < 0:
        print("\n❌ Model degraded due to attacks!")
        print(f"   {Config.MALICIOUS_PERCENTAGE*100:.0f}% malicious clients successfully poisoned the model")
    else:
        print("\n⚠️  Model still improved despite attacks")
        print(f"   But improvement is likely much less than baseline")
    
    print("=" * 70)
    print("\n📊 Compare this with Week 1 Baseline to see attack impact!")
    print("   Next step: Week 6 - Implement defenses to recover performance")
    print("=" * 70)

if __name__ == "__main__":
    main()
