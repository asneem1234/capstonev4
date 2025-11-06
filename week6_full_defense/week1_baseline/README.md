# Week 1: Quantum Federated Learning Baseline

## Overview
Baseline implementation of quantum federated learning using **PennyLane** quantum circuits and **Flower** framework. No attacks or defenses - pure honest federated training.

## Architecture

### Hybrid Quantum-Classical Model
```
Input: 28x28 MNIST images
    ↓
Classical CNN Feature Extractor (28x28 → 4x4x16 = 256 features)
    ↓
Classical-to-Quantum Interface (256 → 4 features)
    ↓
Quantum Circuit (4 qubits, 4 variational layers)
    ↓
Classical Classifier (4 → 10 classes)
```

### Quantum Circuit Details
- **Qubits**: 4
- **Layers**: 4 variational layers
- **Encoding**: Angle encoding (RY rotations)
- **Ansatz**: Hardware-efficient (RY + RZ + CNOT)
- **Measurement**: Pauli-Z expectation values
- **Backend**: `default.qubit` (PennyLane simulator)

### Federated Learning
- **Framework**: Flower (flwr)
- **Strategy**: FedAvg
- **Clients**: 30 total, all participate per round
- **Rounds**: 5
- **Data**: Non-IID (Dirichlet α=0.5)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Expected Results
- Initial accuracy: ~10% (random)
- Final accuracy: ~85-90% (after 5 rounds)
- Training time: ~10-15 minutes (CPU, 30 clients)

## Configuration
Edit `config.py` to change:
- `NUM_CLIENTS`: Number of federated clients
- `NUM_ROUNDS`: Number of training rounds
- `N_QUBITS`: Number of qubits (default: 4)
- `N_LAYERS`: Number of quantum layers (default: 4)
- `DIRICHLET_ALPHA`: Non-IID intensity (default: 0.5)

## Files
- `main.py` - Entry point (Flower simulation)
- `quantum_model.py` - Hybrid quantum-classical model
- `client.py` - Flower client implementation
- `server.py` - Flower server strategy (FedAvg)
- `data_loader.py` - Non-IID MNIST data loading
- `config.py` - Configuration parameters
- `requirements.txt` - Python dependencies

## Notes
- Uses **Flower simulation** for efficient local testing
- Quantum circuits run on CPU simulator (no quantum hardware needed)
- Each client trains locally for 3 epochs per round
- Global model aggregated using FedAvg (weighted average)
