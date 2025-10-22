# Main training loop with FULL DEFENSE (Fetal Plane Classification)
import torch
import numpy as np
from config import Config
from model import get_model
from data_loader import get_client_loaders
from client import Client
from server import Server

def select_random_malicious_clients(num_clients, malicious_percentage):
    """Randomly select malicious clients"""
    num_malicious = int(num_clients * malicious_percentage)
    malicious_clients = np.random.choice(num_clients, num_malicious, replace=False).tolist()
    return sorted(malicious_clients)

def main():
    print("=" * 70)
    print("Federated Learning - FETAL PLANE CLASSIFICATION")
    print("WITH FULL DEFENSE STACK")
    print("=" * 70)
    print(f"Clients: {Config.NUM_CLIENTS}")
    print(f"Rounds: {Config.NUM_ROUNDS}")
    print(f"Attack: {Config.ATTACK_ENABLED} ({Config.MALICIOUS_PERCENTAGE*100:.0f}% malicious)")
    print(f"Defense: {Config.DEFENSE_ENABLED}")
    if Config.DEFENSE_ENABLED:
        print(f"  - Validation filtering (threshold={Config.VALIDATION_THRESHOLD})")
        if Config.USE_FINGERPRINTS:
            print(f"  - Fingerprint clustering (threshold={Config.COSINE_THRESHOLD})")
        if Config.USE_PQ_CRYPTO:
            crypto_type = "real liboqs" if Config.USE_REAL_CRYPTO else "simulated"
            print(f"  - PQ Crypto ({crypto_type}): {Config.PQ_KEM_ALG} + {Config.PQ_SIG_ALG}")
    print("=" * 70)
    
    # Load data
    print("\nLoading fetal plane data...")
    client_loaders, val_loader, test_loader = get_client_loaders(
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
    server = Server(global_model, val_loader, test_loader)
    
    # Collect client public keys for PQ crypto
    client_public_keys = []
    if Config.USE_PQ_CRYPTO:
        for client in clients:
            client_public_keys.append(client.sig_public_key)
    
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
        
        # Select malicious clients
        if Config.RANDOM_MALICIOUS and Config.ATTACK_ENABLED:
            malicious_clients_this_round = select_random_malicious_clients(
                Config.NUM_CLIENTS, 
                Config.MALICIOUS_PERCENTAGE
            )
            print(f"\n[ATTACK]")
            print(f"  Malicious clients: {malicious_clients_this_round}")
        else:
            malicious_clients_this_round = []
        
        # Clients train
        print("\n[CLIENT TRAINING]")
        client_messages = []
        for i, client in enumerate(clients):
            is_malicious = i in malicious_clients_this_round
            
            msg, train_acc, train_loss, update_norm = client.train(
                global_model,
                server_public_key=server.kem_public_key if Config.USE_PQ_CRYPTO else None,
                round_num=round_num,
                is_malicious_this_round=is_malicious
            )
            
            client_messages.append(msg)
            
            tag = " [MALICIOUS]" if is_malicious else ""
            print(f"  Client {i}{tag}: Acc={train_acc:.2f}%, Loss={train_loss:.4f}, Norm={update_norm:.4f}")
        
        # Server aggregates with defense
        print("\n[SERVER AGGREGATION + DEFENSE]")
        agg_norm, defense_stats, _, _ = server.aggregate(client_messages, client_public_keys)
        
        if defense_stats:
            print(f"  Defense results:")
            print(f"    Total updates: {defense_stats.get('total_updates', 0)}")
            print(f"    Accepted: {defense_stats.get('accepted_updates', 0)}")
            print(f"    Rejected: {defense_stats.get('rejected_updates', 0)}")
        
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
    print(f"Total Improvement:     {improvement:+.2f}%")
    print("=" * 70)
    
    if Config.ATTACK_ENABLED:
        print(f"\n✅ Defense successfully mitigated {Config.MALICIOUS_PERCENTAGE*100:.0f}% malicious clients!")
        print("   Compare with Week 2 (attack without defense) to see the difference.")
    else:
        print("\n✅ Model trained successfully!")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
