# Main training loop - Non-IID Baseline (No Attack)
import torch
from config import Config
from model import get_model
from data_loader import get_client_loaders
from client import Client
from server import Server

def main():
    print("=" * 70)
    print("Federated Learning - FETAL PLANE CLASSIFICATION")
    print("NON-IID BASELINE (No Attack)")
    print("=" * 70)
    print(f"Clients: {Config.NUM_CLIENTS} (simulating hospitals/clinics)")
    print(f"Rounds: {Config.NUM_ROUNDS}")
    print(f"Local epochs: {Config.LOCAL_EPOCHS}")
    print(f"Data Distribution: NON-IID (Dirichlet α={Config.DIRICHLET_ALPHA})")
    print(f"Model: {Config.MODEL_TYPE}")
    print(f"Number of classes: {Config.NUM_CLASSES}")
    print(f"Attack enabled: {Config.ATTACK_ENABLED}")
    print(f"Defense enabled: {Config.DEFENSE_ENABLED}")
    print("=" * 70)
    print("This is the BASELINE - all clients are honest!")
    print("Expected: Model should improve steadily over training rounds")
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
    
    # Create clients (all honest)
    print("Creating clients (all honest)...")
    clients = [Client(i, client_loaders[i]) for i in range(Config.NUM_CLIENTS)]
    
    # Create server
    server = Server(global_model, test_loader)
    
    # Initial evaluation
    print("\nEvaluating initial model...")
    initial_acc = server.evaluate()
    print(f"Initial Test Accuracy: {initial_acc:.2f}%")
    print("=" * 70)
    
    # Training rounds
    prev_accuracy = initial_acc
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
            print(f"  Client {i}: Train Acc={train_acc:.2f}%, "
                  f"Loss={train_loss:.4f}, Update Norm={update_norm:.4f}")
        
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
        
        if round_num > 0:
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
    print("\n✅ Baseline established!")
    print(f"   With Non-IID data and no attacks, model reaches ~{test_accuracy:.1f}%")
    print("   This is the UPPER BOUND for comparison with attack scenarios.")
    print("=" * 70)

if __name__ == "__main__":
    main()
