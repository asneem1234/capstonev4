"""
Hybrid Quantum-Classical Neural Network for MNIST Classification
Uses PennyLane for quantum circuit and PyTorch for classical layers
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class QuantumCircuit:
    """
    Quantum circuit implementation using PennyLane
    - 4 qubits
    - Angle encoding for input features
    - 4 variational layers with RY rotations and CNOT entanglement
    """
    
    def __init__(self, n_qubits=4, n_layers=4):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device (simulator)
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Create quantum node
        self.qnode = qml.QNode(self._circuit, self.dev, interface='torch')
        
    def _circuit(self, inputs, weights):
        """
        Quantum circuit architecture:
        1. Angle encoding (input features → RY rotations)
        2. Variational layers (trainable RY + CNOT entanglement)
        3. Measurement (Pauli-Z expectation)
        """
        # 1. Encode input features (angle encoding)
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # 2. Variational layers
        for layer in range(self.n_layers):
            # Trainable rotations
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            
            # Entanglement (CNOT cascade)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Close the loop
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        
        # 3. Measure all qubits (Pauli-Z expectation)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, inputs, weights):
        """
        Forward pass through quantum circuit
        Args:
            inputs: [batch_size, n_qubits]
            weights: [n_layers, n_qubits, 2]
        Returns:
            outputs: [batch_size, n_qubits]
        """
        batch_size = inputs.shape[0]
        outputs = torch.zeros((batch_size, self.n_qubits))
        
        for i in range(batch_size):
            result = self.qnode(inputs[i], weights)
            outputs[i] = torch.stack(result)
        
        return outputs


class HybridQuantumNet(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network
    
    Architecture:
    1. Classical feature extraction: 28x28 → 16 features (4x4 spatial)
    2. Quantum processing: 16 → 4 qubits → 4 quantum features
    3. Classical output: 4 → 10 classes
    """
    
    def __init__(self, n_qubits=4, n_layers=4):
        super(HybridQuantumNet, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical feature extraction (CNN)
        # 28x28x1 → 14x14x8 → 7x7x16 → 4x4x16 → 16 features
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # 28→14
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # 14→14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14→7
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),  # 7→7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 7x7→4x4
        )
        
        # Dimension reduction: 16*4*4=256 → n_qubits (4)
        self.classical_to_quantum = nn.Linear(256, n_qubits)
        
        # Quantum circuit weights: [n_layers, n_qubits, 2]
        # Each qubit has 2 rotation angles (RY, RZ) per layer
        self.quantum_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2) * 0.1
        )
        
        # Initialize quantum circuit
        self.quantum_circuit = QuantumCircuit(n_qubits, n_layers)
        
        # Classical output layer: n_qubits (4) → 10 classes
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 10)
        )
    
    def forward(self, x):
        """
        Forward pass through hybrid network
        Args:
            x: [batch_size, 1, 28, 28]
        Returns:
            output: [batch_size, 10]
        """
        # 1. Classical feature extraction
        features = self.feature_extractor(x)  # [batch, 16, 4, 4]
        features = features.view(features.size(0), -1)  # [batch, 256]
        
        # 2. Reduce to quantum input dimension
        quantum_input = self.classical_to_quantum(features)  # [batch, n_qubits]
        quantum_input = torch.tanh(quantum_input) * np.pi  # Scale to [-π, π]
        
        # 3. Quantum processing
        quantum_output = self.quantum_circuit.forward(
            quantum_input, 
            self.quantum_weights
        )  # [batch, n_qubits]
        
        # 4. Classical classification
        output = self.classifier(quantum_output)  # [batch, 10]
        
        return output
    
    def get_weights(self):
        """Get model parameters as list"""
        return [param.data.clone() for param in self.parameters()]
    
    def set_weights(self, weights):
        """Set model parameters from list"""
        for param, new_param in zip(self.parameters(), weights):
            param.data = new_param.clone()


def create_model():
    """Factory function to create quantum model"""
    return HybridQuantumNet(n_qubits=4, n_layers=4)


if __name__ == "__main__":
    # Test quantum model
    print("Testing Quantum Model...")
    model = create_model()
    
    # Test input
    x = torch.randn(2, 1, 28, 28)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    quantum_params = model.quantum_weights.numel()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Quantum parameters: {quantum_params}")
    print(f"Classical parameters: {total_params - quantum_params:,}")
    
    print("\n✓ Quantum model test passed!")
