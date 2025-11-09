"""
Test script to verify gradient attack configuration
Shows which clients will be malicious and their attack parameters
"""

import config
import random
import numpy as np

# Set random seed for reproducibility
random.seed(config.SEED)
np.random.seed(config.SEED)

def test_attack_setup():
    """Test and display the attack setup"""
    
    print("\n" + "="*70)
    print("GRADIENT ATTACK TEST SETUP - Week 6 Full Defense")
    print("="*70)
    
    # Show configuration
    config.print_config()
    
    # Simulate client assignment (same logic as main.py)
    malicious_client_assignments = {}
    
    # Get all client IDs and shuffle them
    all_client_ids = list(range(config.NUM_CLIENTS))
    random.shuffle(all_client_ids)
    
    assignment_idx = 0
    
    # Assign aggressive attacks
    for _ in range(config.MALICIOUS_CLIENTS_CONFIG['aggressive']['count']):
        client_id = all_client_ids[assignment_idx]
        malicious_client_assignments[client_id] = (
            'aggressive', 
            config.MALICIOUS_CLIENTS_CONFIG['aggressive']['scale_factor']
        )
        assignment_idx += 1
    
    # Assign moderate attacks
    for _ in range(config.MALICIOUS_CLIENTS_CONFIG['moderate']['count']):
        client_id = all_client_ids[assignment_idx]
        malicious_client_assignments[client_id] = (
            'moderate',
            config.MALICIOUS_CLIENTS_CONFIG['moderate']['scale_factor']
        )
        assignment_idx += 1
    
    # Assign subtle attacks
    for _ in range(config.MALICIOUS_CLIENTS_CONFIG['subtle']['count']):
        client_id = all_client_ids[assignment_idx]
        malicious_client_assignments[client_id] = (
            'subtle',
            config.MALICIOUS_CLIENTS_CONFIG['subtle']['scale_factor']
        )
        assignment_idx += 1
    
    malicious_clients = set(malicious_client_assignments.keys())
    
    # Display client assignments
    print("\n" + "="*70)
    print("CLIENT ROLE ASSIGNMENTS")
    print("="*70)
    print(f"Total Clients: {config.NUM_CLIENTS}")
    print(f"Honest Clients: {config.NUM_CLIENTS - len(malicious_clients)}")
    print(f"Malicious Clients: {len(malicious_clients)}")
    print()
    
    for client_id in range(config.NUM_CLIENTS):
        if client_id in malicious_clients:
            attack_type, scale = malicious_client_assignments[client_id]
            print(f"  Client {client_id}: 🔴 MALICIOUS - {attack_type.upper()} attack (gradient scale factor: {scale}x)")
        else:
            print(f"  Client {client_id}: 🟢 HONEST")
    
    print("\n" + "="*70)
    print("ATTACK DETAILS")
    print("="*70)
    print(f"Attack Type: {config.ATTACK_TYPE}")
    print(f"Attack Mechanism: Gradient Ascent (reverses gradient direction)")
    print()
    
    for client_id in sorted(malicious_clients):
        attack_type, scale = malicious_client_assignments[client_id]
        print(f"  Client {client_id} ({attack_type}):")
        print(f"    - Scale Factor: {scale}x")
        print(f"    - Effect: Gradient is reversed and amplified by {scale}x")
        print(f"    - Formula: poisoned_update = old_params - {scale} * (new_params - old_params)")
        print(f"    - Expected Defense Layer: ", end="")
        if scale >= 50.0:
            print("Layer 0 (Norm Filter) - Very high norm")
        elif scale >= 10.0:
            print("Layer 1 (Adaptive Defense) - Statistical outlier")
        else:
            print("Layer 2 (Fingerprint) - Behavioral anomaly")
        print()
    
    print("="*70)
    print("DEFENSE CONFIGURATION")
    print("="*70)
    
    if config.DEFENSE_ENABLED:
        if config.USE_NORM_FILTERING:
            print(f"✓ Layer 0 - Norm Filter:")
            print(f"    Threshold: median_norm × {config.NORM_THRESHOLD_MULTIPLIER}")
            print(f"    Catches: Updates with abnormally large norms")
        
        if config.USE_ADAPTIVE_DEFENSE:
            print(f"\n✓ Layer 1 - Adaptive Statistical Defense:")
            print(f"    Method: {config.ADAPTIVE_METHOD}")
            print(f"    Threshold: mean + {config.ADAPTIVE_THRESHOLD_STD}×std")
            print(f"    Catches: Statistical outliers in update distributions")
        
        if config.USE_FINGERPRINTS:
            print(f"\n✓ Layer 2 - Fingerprint Validation:")
            print(f"    Dimension: {config.FINGERPRINT_DIM}")
            print(f"    Similarity Threshold: {config.FINGERPRINT_SIMILARITY_THRESHOLD}")
            print(f"    Catches: Updates with suspicious behavioral patterns")
    else:
        print("⚠️  NO DEFENSE ACTIVE - All updates will be aggregated")
    
    print("\n" + "="*70)
    print("EXPECTED BEHAVIOR")
    print("="*70)
    print("With 2 malicious clients using gradient attacks:")
    print("  1. Malicious clients will reverse and amplify gradients")
    print("  2. High-scale attack (50x) should be caught by Norm Filter (Layer 0)")
    print("  3. Medium-scale attack (10x) should be caught by Adaptive Defense (Layer 1)")
    print("  4. Defense should maintain model accuracy despite attacks")
    print("  5. Server will reject malicious updates and only aggregate honest ones")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_attack_setup()
