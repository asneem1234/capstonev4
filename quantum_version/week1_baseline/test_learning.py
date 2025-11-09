"""
Test if the quantum model can actually learn
"""
import torch
import torch.nn as nn
import torch.optim as optim
from quantum_model import HybridQuantumNet
from data_loader import get_client_loaders
import config

print("=" * 60)
print("Testing Quantum Model Learning Capability")
print("=" * 60)

# Load data
print("\nLoading data...")
client_loaders, test_loader = get_client_loaders(
    num_clients=1,
    alpha=1.0,  # IID for easier learning
    batch_size=32
)
train_loader = client_loaders[0]

# Create model
print("Creating model...")
model = HybridQuantumNet(n_qubits=4, n_layers=4)
device = torch.device("cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower LR

print(f"\nModel has {sum(p.numel() for p in model.parameters())} parameters")
print(f"Quantum weights shape: {model.quantum_weights.shape}")

# Test gradient flow
print("\n" + "=" * 60)
print("Testing Gradient Flow")
print("=" * 60)

model.train()
data, target = next(iter(train_loader))
data, target = data[:4].to(device), target[:4].to(device)  # Small batch

print(f"Input shape: {data.shape}")

# Forward pass
output = model(data)
print(f"Output shape: {output.shape}")
print(f"Output values (first sample): {output[0].detach().numpy()}")

# Backward pass
loss = criterion(output, target)
print(f"\nLoss: {loss.item():.4f}")

loss.backward()

# Check gradients
print("\nGradient statistics:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  {name}: grad_norm={grad_norm:.6f}, param_norm={param.data.norm().item():.6f}")
    else:
        print(f"  {name}: NO GRADIENT!")

# Train for a few batches
print("\n" + "=" * 60)
print("Training for 10 batches...")
print("=" * 60)

losses = []
accuracies = []

for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx >= 10:
        break
    
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Calculate accuracy
    _, predicted = output.max(1)
    accuracy = 100.0 * predicted.eq(target).sum().item() / target.size(0)
    
    losses.append(loss.item())
    accuracies.append(accuracy)
    
    print(f"Batch {batch_idx+1}: Loss={loss.item():.4f}, Acc={accuracy:.2f}%")

print("\n" + "=" * 60)
print("Results:")
print("=" * 60)
print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss change: {losses[0] - losses[-1]:.4f}")
print(f"\nInitial accuracy: {accuracies[0]:.2f}%")
print(f"Final accuracy: {accuracies[-1]:.2f}%")
print(f"Accuracy change: {accuracies[-1] - accuracies[0]:.2f}%")

if losses[0] > losses[-1] and accuracies[-1] > accuracies[0]:
    print("\n✓ Model is learning!")
else:
    print("\n✗ Model is NOT learning properly!")
    print("\nPossible issues:")
    print("  - Quantum gradients vanishing")
    print("  - Learning rate too high/low")
    print("  - Circuit architecture issues")
