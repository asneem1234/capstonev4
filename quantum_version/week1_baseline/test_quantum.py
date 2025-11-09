import torch
import pennylane as qml

dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev, interface='torch')
def circuit(inputs, weights):
    for i in range(4):
        qml.RY(inputs[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

inputs = torch.randn(4)
weights = torch.randn(4, 4, 2)
result = circuit(inputs, weights)

print(f'Result type: {type(result)}')
print(f'Result: {result}')
print(f'Result shape: {result.shape if hasattr(result, "shape") else "N/A"}')
print(f'Result len: {len(result) if hasattr(result, "__len__") else "N/A"}')

# Try stacking
if isinstance(result, (list, tuple)):
    stacked = torch.tensor([r.item() if hasattr(r, 'item') else float(r) for r in result])
    print(f'Stacked shape: {stacked.shape}')
    print(f'Stacked: {stacked}')
