# Quantum Federated Learning with Byzantine Defense

## Overview
Quantum federated learning implementation using PennyLane quantum circuits and Flower framework, with Byzantine attack defense mechanisms.

## 📚 Documentation
Comprehensive documentation is available in the [`documentation/`](./documentation/) folder:

- **[QUANTUM_DEFEND_ARCHITECTURE.md](./documentation/QUANTUM_DEFEND_ARCHITECTURE.md)** - Complete visual guide with ASCII diagrams
- **[DOCUMENTATION_INDEX.md](./documentation/DOCUMENTATION_INDEX.md)** - Full documentation index
- **[QUICK_START.md](./documentation/QUICK_START.md)** - Quick start guide
- **[RESEARCH_OVERVIEW.md](./documentation/RESEARCH_OVERVIEW.md)** - Research methodology
- **[IMPLEMENTATION_SUMMARY.md](./documentation/IMPLEMENTATION_SUMMARY.md)** - Technical implementation
- **[EXPERIMENTAL_PROTOCOL.md](./documentation/EXPERIMENTAL_PROTOCOL.md)** - Experiment setup
- **[RESULTS_ANALYSIS.md](./documentation/RESULTS_ANALYSIS.md)** - Results and analysis

## Features
- **Quantum Neural Network**: PennyLane-based quantum circuits for MNIST classification
- **Federated Learning**: Flower (flwr) framework for distributed training
- **Non-IID Data**: Dirichlet distribution (α=0.5)
- **Byzantine Defense**: Norm-based filtering against gradient ascent attacks (40% malicious clients)
- **Parameter Efficiency**: 5,118 parameters (85% reduction vs classical CNNs)

## Architecture
```
28×28 MNIST Input
         ↓
Classical Feature Extraction (Conv → 256 features)
         ↓
Classical-to-Quantum Interface (256 → 4)
         ↓
Quantum Circuit (4 qubits, 4 variational layers)
         ↓
Classical Classifier (4 → 32 → 10 classes)
```

## Installation

```bash
pip install pennylane torch torchvision flower numpy scikit-learn
```

## Usage

### Run Baseline (No Attack)
```bash
cd week1_baseline
python main.py
```

### Run Attack (No Defense)
```bash
cd week2_attack
python main.py
```

### Run Full Defense
```bash
cd week6_full_defense
python main.py
```

## Quantum Circuit Details
- **Qubits**: 4 (encodes 16-dimensional classical features)
- **Layers**: 4 variational layers
- **Encoding**: Angle encoding (RY rotations)
- **Entanglement**: CNOT gates between neighboring qubits
- **Measurement**: Pauli-Z expectation values
- **Backend**: default.qubit (simulator)

## Results
See `QUANTUM_RESULTS.md` for detailed experimental results comparing quantum vs classical federated learning with Byzantine attacks.

## Directory Structure
```
quantum_version/
├── README.md                  (this file)
├── QUANTUM_RESULTS.md         (experimental results)
├── week1_baseline/            (honest federated learning)
│   ├── main.py
│   ├── quantum_model.py
│   ├── client.py
│   ├── server.py
│   ├── data_loader.py
│   └── config.py
├── week2_attack/              (gradient ascent attack)
│   └── ... (same structure)
└── week6_full_defense/        (full Byzantine defense)
    └── ... (same structure + defense modules)
```

## References
- PennyLane: https://pennylane.ai/
- Flower: https://flower.dev/
- Quantum Machine Learning: https://pennylane.ai/qml/
