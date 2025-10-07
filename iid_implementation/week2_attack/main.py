# Main training loop - with attack
import torch
from config import Config
from model import get_model
from data_loader import get_client_loaders
from client import Client
from server import Server

def main():
    print("=" * 70)
    print("Federated Learning - WITH LABEL FLIPPING ATTACK")
    print("=" * 70)
    print(f"Clients: {Config.NUM_CLIENTS}")
    print(f"Rounds: {Config.NUM_ROUNDS}")
    print(f"Local epochs: {Config.LOCAL_EPOCHS}")
    print(f"Attack enabled: {Config.ATTACK_ENABLED}")
    print(f"Malicious clients: {Config.MALICIOUS_CLIENTS}")
    print(f"Defense enabled: {Config.DEFENSE_ENABLED}")
    print("=" * 70)
    
    # Setup
    print("\nLoading data...")
    client_loaders, test_loader = get_client_loaders(Config.NUM_CLIENTS)
    
    print("Creating clients...")
    clients = [
        Client(i, client_loaders[i]) 
        for i in range(Config.NUM_CLIENTS)
    ]
    
    print("Initializing global model...")
    global_model = get_model()
    server = Server(global_model, test_loader)
    
    # Initial evaluation
    initial_acc = server.evaluate()
    print(f"\nInitial Test Accuracy: {initial_acc:.2f}%")
    print("=" * 70)
    
    # Training rounds
    for round_num in range(Config.NUM_ROUNDS):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}/{Config.NUM_ROUNDS}")
        print(f"{'='*70}")
        
        # Clients train
        client_updates = []
        print("\n[CLIENT TRAINING]")
        for i, client in enumerate(clients):
            update, train_acc, train_loss, update_norm = client.train(global_model)
            client_updates.append(update)
            malicious_tag = " [MALICIOUS]" if client.is_malicious else ""
            print(f"  Client {i}{malicious_tag}: Train Acc={train_acc:.2f}%, Loss={train_loss:.4f}, Update Norm={update_norm:.4f}")
        
        # Server aggregates
        print("\n[SERVER AGGREGATION]")
        agg_norm = server.aggregate(client_updates)
        print(f"  Aggregated {len(client_updates)} client updates")
        print(f"  Aggregated Update Norm: {agg_norm:.4f}")
        print(f"  Applied FedAvg to global model")
        
        # Evaluate
        print("\n[GLOBAL EVALUATION]")
        test_accuracy = server.evaluate()
        print(f"  Global Test Accuracy: {test_accuracy:.2f}%")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Initial Test Accuracy: {initial_acc:.2f}%")
    print(f"Final Test Accuracy:   {test_accuracy:.2f}%")
    print(f"Improvement:           {test_accuracy - initial_acc:+.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
