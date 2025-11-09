# Main training loop - CLIENT-SIDE FINGERPRINTS with PQ crypto
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
    print("Federated Learning - NON-IID with Full Defense (Week 6)")
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
    print(f"PQ Crypto: {'ENABLED' if Config.USE_PQ_CRYPTO else 'DISABLED'} " + 
          f"({'Real' if Config.USE_REAL_CRYPTO else 'Simulated'} mode)")
    print(f"Fingerprint Defense: {'ENABLED (CLIENT-SIDE)' if Config.USE_FINGERPRINTS else 'DISABLED'}")
    print(f"Validation Defense: {'ENABLED' if Config.DEFENSE_ENABLED else 'DISABLED'}")
    print("=" * 70)
    
    # Load data (Non-IID with Dirichlet distribution)
    print("\nLoading data...")
    client_loaders, validation_loader, test_loader = get_client_loaders(
        Config.NUM_CLIENTS, 
        alpha=Config.DIRICHLET_ALPHA
    )
    
    # Initialize global model
    print("Initializing global model...")
    global_model = get_model()
    
    # Create clients
    clients = [Client(i, client_loaders[i]) for i in range(Config.NUM_CLIENTS)]
    
    # Create server
    server = Server(global_model, validation_loader, test_loader)
    
    # Initial evaluation
    test_loss, test_acc = server.evaluate()
    print(f"\nInitial Test Accuracy: {test_acc:.2f}%")
    
    # Training rounds
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
        
        # Clients train and compute fingerprints
        client_messages = []
        client_public_keys = []
        print(f"\n[CLIENT TRAINING & FINGERPRINT COMPUTATION]")
        n_clients = len(clients)  # Store for later use in detection accuracy
        for i, client in enumerate(clients):
            is_malicious = i in malicious_clients_this_round
            message, train_acc, train_loss, update_norm = client.train(
                global_model, 
                server.kem_public_key if Config.USE_PQ_CRYPTO else None,
                round_num,
                is_malicious
            )
            # Add update_norm to message for defense filtering
            message['update_norm'] = update_norm
            client_messages.append(message)
            if Config.USE_PQ_CRYPTO:
                client_public_keys.append(client.sig_public_key)
            
            malicious_tag = " [MALICIOUS]" if is_malicious else ""
            crypto_tag = " (encrypted)" if Config.USE_PQ_CRYPTO else ""
            fp_tag = " [fingerprint computed]" if Config.USE_FINGERPRINTS else ""
            print(f"  Client {i}{malicious_tag}: Train Acc={train_acc:.2f}%, "
                  f"Loss={train_loss:.4f}, Update Norm={update_norm:.4f}{crypto_tag}{fp_tag}")
        
        # Server aggregates with three-layer defense
        print("\n[SERVER AGGREGATION]")
        
        # Show PQ crypto layer
        if Config.USE_PQ_CRYPTO:
            print("  [LAYER 1: POST-QUANTUM CRYPTO]")
            print(f"    Algorithm: Kyber512 (encryption) + Dilithium2 (signatures)")
            print(f"    Mode: {'Real' if Config.USE_REAL_CRYPTO else 'Simulated'}")
        
        agg_norm, validation_results, fingerprint_results, crypto_results, integrity_results = server.aggregate(
            client_messages,
            client_public_keys if Config.USE_PQ_CRYPTO else None
        )
        
        # Show crypto results
        if Config.USE_PQ_CRYPTO:
            print(f"    Verified & Decrypted: {crypto_results['decrypted']}/{len(client_messages)}")
            if crypto_results['failed'] > 0:
                print(f"    Failed: {crypto_results['failed']}")
        
        # Show fingerprint integrity verification
        if Config.USE_FINGERPRINTS and Config.USE_PQ_CRYPTO:
            print(f"\n  [LAYER 2a: FINGERPRINT INTEGRITY VERIFICATION (Client-Side)]")
            print(f"    Verified: {integrity_results['verified']}/{len(client_messages)}")
            if integrity_results['failed'] > 0:
                print(f"    ✗ Tampered/Mismatched: {integrity_results['failed']} "
                      f"(clients: {integrity_results['tampered']})")
            print(f"    ✓ Integrity checks protect against:")
            print(f"      - Network tampering (MITM attacks)")
            print(f"      - Malicious server modifying updates")
            print(f"      - Update corruption during transmission")
        
        # Show ADAPTIVE defense results
        if fingerprint_results and 'adaptive' in fingerprint_results.get('method', ''):
            print(f"\n  [LAYER 2: ADAPTIVE PATTERN-LEARNING DEFENSE]")
            honest = fingerprint_results['main_cluster']
            malicious = fingerprint_results['outliers']
            sep_factor = fingerprint_results.get('separation_factor', 0.0)
            diagnostics = fingerprint_results.get('diagnostics', {})
            
            print(f"    Method: {diagnostics.get('method', 'unknown').upper()}")
            print(f"    ✓ Detected honest clients: {honest} [{len(honest)} clients]")
            print(f"    ✗ Detected malicious clients: {malicious} [{len(malicious)} clients]")
            print(f"    📊 Separation Factor: {sep_factor:.2f}x")
            print(f"       (malicious updates are {sep_factor:.2f}× larger than honest)")
            
            # Show key statistics
            if 'honest_norm_mean' in diagnostics:
                print(f"    📈 Honest norm (avg): {diagnostics['honest_norm_mean']:.4f}")
            if 'malicious_norm_mean' in diagnostics:
                print(f"    📈 Malicious norm (avg): {diagnostics['malicious_norm_mean']:.4f}")
            if 'threshold' in diagnostics:
                print(f"    🎯 Adaptive threshold: {diagnostics['threshold']:.4f} (learned from data)")
            
            # Compare with actual malicious clients (ground truth)
            actual_malicious = malicious_clients_this_round
            detected_malicious = set(malicious)
            actual_malicious_set = set(actual_malicious)
            
            true_positives = len(detected_malicious & actual_malicious_set)
            false_positives = len(detected_malicious - actual_malicious_set)
            false_negatives = len(actual_malicious_set - detected_malicious)
            true_negatives = n_clients - len(actual_malicious_set) - false_positives
            
            precision = true_positives / len(detected_malicious) if detected_malicious else 0.0
            recall = true_positives / len(actual_malicious_set) if actual_malicious_set else 0.0
            accuracy = (true_positives + true_negatives) / n_clients
            
            print(f"\n    🎯 DETECTION ACCURACY:")
            print(f"       True Positives (caught malicious): {true_positives}/{len(actual_malicious_set)}")
            print(f"       False Positives (honest flagged): {false_positives}")
            print(f"       False Negatives (malicious missed): {false_negatives}")
            print(f"       Precision: {precision:.1%} | Recall: {recall:.1%} | Accuracy: {accuracy:.1%}")
        
        # Show aggregation results
        if Config.DEFENSE_ENABLED and validation_results:
            print(f"\n  [LAYER 3: AGGREGATION]")
            accepted = sum(1 for r in validation_results if r['valid'])
            rejected = len(validation_results) - accepted
            print(f"    ✓ Aggregated: {accepted}/{len(validation_results)} updates")
            print(f"    ✗ Rejected: {rejected}/{len(validation_results)} updates")
        
        print(f"\n  Aggregated Update Norm: {agg_norm:.4f}")
        print(f"  Applied FedAvg to global model")
        
        # Evaluate
        test_loss, test_acc = server.evaluate()
        print(f"\n[GLOBAL MODEL EVALUATION]")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("\nKEY IMPROVEMENTS (Client-Side Fingerprints):")
    print("  ✓ Clients compute their own fingerprints (not server)")
    print("  ✓ Server verifies fingerprint matches decrypted update")
    print("  ✓ Detects network tampering and malicious server attacks")
    print("  ✓ Stronger integrity guarantees than server-side fingerprints")
    print("  ✓ Three-layer defense: PQ Crypto → Fingerprint → Validation")

if __name__ == "__main__":
    main()
