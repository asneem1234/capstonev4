"""Quick test to verify model creation works with 2 qubits"""
import sys
sys.path.insert(0, 'quantum_version/week6_full_defense')

import config
from quantum_model import create_model
import torch

print(f"Config N_QUBITS: {config.N_QUBITS}")
print(f"Config N_LAYERS: {config.N_LAYERS}")

print("\nCreating model...")
model = create_model()

print(f"\nModel created successfully!")
print(f"Model architecture:")
print(model)

# Test forward pass
print("\nTesting forward pass with batch of 2 images...")
x = torch.randn(2, 1, 28, 28)
try:
    output = model(x)
    print(f"✓ Forward pass successful!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: (2, 10)")
    assert output.shape == (2, 10), f"Output shape mismatch! Got {output.shape}, expected (2, 10)"
    print("✓ All tests passed!")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
