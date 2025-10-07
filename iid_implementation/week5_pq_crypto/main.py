# Main training loop - with PQ crypto + attack + defense
import torch
from config import Config
from model import get_model
from data_loader import get_client_loaders
from client import Client
from server import Server

def main():
    print("=" * 70)
    print("Federated Learning - FULL THREE-LAYER DEFENSE")
    print("=" * 70)
    print(f"Clients: {Config.NUM_CLIENTS}")
    print(f"Rounds: {Config.NUM_ROUNDS}")
    print(f"Local epochs: {Config.LOCAL_EPOCHS}")
    print(f"Attack enabled: {Config.ATTACK_ENABLED}")
    print(f"Malicious clients: {Config.MALICIOUS_CLIENTS}")
    print(f"PQ Crypto: {'ENABLED' if Config.USE_PQ_CRYPTO else 'DISABLED'} " + 
          f"({'Real' if Config.USE_REAL_CRYPTO else 'Simulated'} mode)")
    print(f"Fingerprint Defense: {'ENABLED' if Config.USE_FINGERPRINTS else 'DISABLED'}")
    print(f"Validation Defense: {'ENABLED' if Config.DEFENSE_ENABLED else 'DISABLED'}")
    print("=" * 70)
    
    # Setup
    print("\nLoading data...")
    client_loaders, validation_loader, test_loader = get_client_loaders(Config.NUM_CLIENTS)
    print(f"Validation set size: {len(validation_loader.dataset)} samples")
    
    print("Creating clients...")
    clients = [
        Client(i, client_loaders[i]) 
        for i in range(Config.NUM_CLIENTS)
    ]
    
    print("Initializing global model...")
    global_model = get_model()
    server = Server(global_model, validation_loader, test_loader)
    
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
        client_messages = []
        client_public_keys = []
        print("\n[CLIENT TRAINING]")
        for i, client in enumerate(clients):
            message, train_acc, train_loss, update_norm = client.train(
                global_model, 
                server.kem_public_key if Config.USE_PQ_CRYPTO else None,
                round_num
            )
            client_messages.append(message)
            if Config.USE_PQ_CRYPTO:
                client_public_keys.append(client.sig_public_key)
            
            malicious_tag = " [MALICIOUS]" if client.is_malicious else ""
            crypto_tag = " (encrypted)" if Config.USE_PQ_CRYPTO else ""
            print(f"  Client {i}{malicious_tag}: Train Acc={train_acc:.2f}%, Loss={train_loss:.4f}, Update Norm={update_norm:.4f}{crypto_tag}")
        
        # Server aggregates with three-layer defense
        print("\n[SERVER AGGREGATION]")
        
        # Show PQ crypto layer
        if Config.USE_PQ_CRYPTO:
            print("  [LAYER 1: POST-QUANTUM CRYPTO]")
            print(f"    Algorithm: Kyber512 (encryption) + Dilithium2 (signatures)")
            print(f"    Mode: {'Real' if Config.USE_REAL_CRYPTO else 'Simulated'}")
        
        agg_norm, validation_results, fingerprint_results, crypto_results = server.aggregate(
            client_messages,
            client_public_keys if Config.USE_PQ_CRYPTO else None
        )
        
        if Config.USE_PQ_CRYPTO:
            print(f"    Verified & Decrypted: {crypto_results['decrypted']}/{len(client_messages)}")
            if crypto_results['failed'] > 0:
                print(f"    Failed: {crypto_results['failed']}")
        
        # Show fingerprint clustering results
        if Config.USE_FINGERPRINTS and fingerprint_results:
            print("  [LAYER 2: FINGERPRINT CLUSTERING (Cosine Similarity)]")
            main_cluster = fingerprint_results['main_cluster']
            outliers = fingerprint_results['outliers']
            print(f"    Main cluster (likely honest): {main_cluster} [{len(main_cluster)} clients]")
            print(f"    Outliers (suspicious): {outliers} [{len(outliers)} clients]")
            print(f"    → Main cluster auto-accepted (skipped validation)")
            print(f"    → Outliers will be validated")
        
        # Show validation results
        if Config.DEFENSE_ENABLED and validation_results:
            print("  [LAYER 3: VALIDATION FILTERING]")
            accepted = sum(1 for r in validation_results if r['valid'])
            rejected = len(validation_results) - accepted
            print(f"    Accepted: {accepted}/{len(validation_results)} updates")
            print(f"    Rejected: {rejected}/{len(validation_results)} updates")
            
            for result in validation_results:
                status = "✓ ACCEPT" if result['valid'] else "✗ REJECT"
                method = f"[{result['method']}]"
                
                if result['method'] == 'validation':
                    print(f"    Client {result['client_id']}: {status} {method} "
                          f"(Δloss={result['loss_increase']:+.4f})")
                else:
                    print(f"    Client {result['client_id']}: {status} {method} "
                          f"(in main cluster)")
        
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
