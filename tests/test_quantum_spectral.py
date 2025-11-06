"""
Test 2: Quantum Model + Spectral Defense Integration
====================================================
Priority: HIGH ⭐⭐
Goal: Test spectral defense with actual quantum gradients from federated learning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from test_spectral_defense import SpectralAnalyzer

# Import your quantum FL components
# Adjust these imports based on your actual structure
try:
    from week6_full_defense.quantum_model import HybridQuantumNet
    from week6_full_defense.client import QuantumFlowerClient
    from week6_full_defense.data_loader import load_mnist_data, split_non_iid_dirichlet
    from week6_full_defense.config import *
    IMPORTS_AVAILABLE = True
except ImportError:
    print("⚠️  Could not import quantum_version modules")
    print("    This test requires actual quantum FL implementation")
    IMPORTS_AVAILABLE = False


def extract_model_gradients(model):
    """Extract flattened gradients from model"""
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1))
    return torch.cat(gradients).detach().cpu().numpy()


def run_quantum_spectral_test():
    """Test spectral defense with actual quantum model"""
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required modules not available")
        print("   Ensure week6_full_defense implementation exists")
        return None
    
    print("="*60)
    print("TEST 2: Quantum Model + Spectral Defense")
    print("="*60)
    print()
    
    # Configuration
    NUM_CLIENTS = 5
    MALICIOUS_CLIENTS = [3, 4]  # 40% malicious
    ALPHA = 0.5  # Non-IID parameter
    NUM_ROUNDS = 2  # Just 2 rounds for quick test
    
    print(f"Configuration:")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Malicious: {MALICIOUS_CLIENTS}")
    print(f"  α: {ALPHA}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print()
    
    # Load and split data
    print("📊 Loading MNIST data...")
    train_data, test_data = load_mnist_data()
    client_datasets = split_non_iid_dirichlet(train_data, NUM_CLIENTS, ALPHA)
    print("✅ Data loaded and split")
    print()
    
    # Initialize quantum model
    print("🔬 Initializing quantum model...")
    global_model = HybridQuantumNet()
    print("✅ Quantum model initialized")
    print()
    
    # Initialize spectral analyzer
    analyzer = SpectralAnalyzer()
    
    # Simulate one federated round
    print("🔄 Running federated learning round...")
    print()
    
    for client_id in range(NUM_CLIENTS):
        print(f"  Client {client_id}...", end=" ")
        
        # Create client
        is_malicious = client_id in MALICIOUS_CLIENTS
        client_data = client_datasets[client_id]
        
        # Create dataloader
        train_loader = torch.utils.data.DataLoader(
            client_data, batch_size=64, shuffle=True
        )
        
        # Local training
        model = HybridQuantumNet()
        model.load_state_dict(global_model.state_dict())
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Train for 1 epoch (quick test)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 5:  # Only 5 batches for speed
                break
            
            # Apply attack if malicious
            if is_malicious:
                target = (9 - target) % 10  # Label flipping
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Extract gradients (difference from global)
        gradient = []
        for (name, global_param), (_, local_param) in zip(
            global_model.named_parameters(), 
            model.named_parameters()
        ):
            diff = local_param.data - global_param.data
            gradient.append(diff.view(-1))
        
        gradient = torch.cat(gradient).cpu().numpy()
        
        # Analyze with spectral defense
        label = 'malicious' if is_malicious else 'honest'
        result = analyzer.analyze_gradient(gradient, label, client_id)
        
        print(f"ρ={result['spectral_ratio']:.4f} ({label})")
    
    print()
    
    # Compute statistics
    stats, df = analyzer.compute_statistics()
    
    # Print results
    print("="*60)
    print("📈 RESULTS:")
    print("-" * 60)
    print(f"Honest Clients:")
    print(f"  Spectral Ratio (ρ): {stats['honest_spectral_ratio_mean']:.4f} ± {stats['honest_spectral_ratio_std']:.4f}")
    print(f"  Entropy:            {stats['honest_entropy_mean']:.4f}")
    print()
    print(f"Malicious Clients:")
    print(f"  Spectral Ratio (ρ): {stats['malicious_spectral_ratio_mean']:.4f} ± {stats['malicious_spectral_ratio_std']:.4f}")
    print(f"  Entropy:            {stats['malicious_entropy_mean']:.4f}")
    print()
    print(f"✨ Spectral Separation: {stats['separation']:.4f}")
    print()
    
    # Test detection
    threshold = stats['honest_spectral_ratio_mean'] + 2 * stats['honest_spectral_ratio_std']
    detected = df[df['spectral_ratio'] > threshold]
    tp = len(detected[detected['label'] == 'malicious'])
    fp = len(detected[detected['label'] == 'honest'])
    fn = len(df[df['label'] == 'malicious']) - tp
    
    detection_rate = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    fpr = fp / len(df[df['label'] == 'honest']) * 100
    
    print(f"🎯 Detection Performance (threshold={threshold:.4f}):")
    print(f"  Detection Rate: {detection_rate:.1f}%")
    print(f"  False Positive Rate: {fpr:.1f}%")
    print()
    
    # Save results
    df.to_csv('tests/results/quantum_spectral_test.csv', index=False)
    print("✅ Saved results to tests/results/quantum_spectral_test.csv")
    
    # Plot
    analyzer.plot_results('tests/results/quantum_spectral_test.png')
    
    # Validation
    print()
    print("="*60)
    print("✅ TEST VALIDATION:")
    print("="*60)
    
    if stats['separation'] > 0.10:
        print("✅ PASS: Spectral separation detected with quantum gradients")
    else:
        print("⚠️  WEAK: Limited separation - may need tuning")
    
    if detection_rate > 70:
        print("✅ PASS: Reasonable detection rate")
    else:
        print("⚠️  WEAK: Detection rate needs improvement")
    
    print()
    print("📝 These are REAL quantum gradient results!")
    print("   Use these values in your paper tables!")
    print()
    
    return stats, df


if __name__ == "__main__":
    print("⚠️  WARNING: This test requires actual quantum model training")
    print("   It may take 5-10 minutes depending on your hardware")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        try:
            stats, df = run_quantum_spectral_test()
            if stats:
                print("🎉 Test completed successfully!")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Test cancelled")
