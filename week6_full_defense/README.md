# Week 6: Quantum Federated Learning with Full Defense# Week 1: Quantum Federated Learning Baseline



## Overview## Overview

Complete implementation with **attack + defense**. Malicious clients apply gradient ascent attack (40%), and the server uses **norm-based filtering** to detect and reject poisoned updates.Baseline implementation of quantum federated learning using **PennyLane** quantum circuits and **Flower** framework. No attacks or defenses - pure honest federated training.



## Defense Mechanism## Architecture



### Norm-Based Filtering### Hybrid Quantum-Classical Model

**Concept**: Malicious updates from gradient ascent attack have 10× larger norms than honest updates.```

Input: 28x28 MNIST images

**Algorithm**:    ↓

```pythonClassical CNN Feature Extractor (28x28 → 4x4x16 = 256 features)

1. Collect all client update norms: [n₁, n₂, ..., n₃₀]    ↓

2. Calculate median: median_norm = median(norms)Classical-to-Quantum Interface (256 → 4 features)

3. Set threshold: threshold = median_norm × 3.0    ↓

4. For each client:Quantum Circuit (4 qubits, 4 variational layers)

   - If norm > threshold → REJECT (malicious)    ↓

   - Else → ACCEPT (honest)Classical Classifier (4 → 10 classes)

5. Aggregate only accepted updates using FedAvg```

```

### Quantum Circuit Details

**Why it works**:- **Qubits**: 4

- Honest client norms: ~0.5-1.5- **Layers**: 4 variational layers

- Malicious client norms: ~5-20 (10× larger!)- **Encoding**: Angle encoding (RY rotations)

- Median is robust to 50% corruption (we have 40%)- **Ansatz**: Hardware-efficient (RY + RZ + CNOT)

- Simple, fast O(n log n), no machine learning needed- **Measurement**: Pauli-Z expectation values

- **Backend**: `default.qubit` (PennyLane simulator)

## Expected Results

### Federated Learning

Based on classical non-IID implementation:- **Framework**: Flower (flwr)

- **Detection**: 100% precision, 100% recall (perfect!)- **Strategy**: FedAvg

- **Accuracy**: 85-90% (vs 10% in week2 without defense)- **Clients**: 30 total, all participate per round

- **Overhead**: <1% computational cost- **Rounds**: 5

- **Data**: Non-IID (Dirichlet α=0.5)

## Usage

## Installation

```bash

python main.py```bash

```pip install -r requirements.txt

```

## Configuration

## Usage

```python

# Attack settings (enabled)```bash

ATTACK_ENABLED = Truepython main.py

MALICIOUS_PERCENTAGE = 0.4```

SCALE_FACTOR = 10.0

## Expected Results

# Defense settings (enabled)- Initial accuracy: ~10% (random)

DEFENSE_ENABLED = True- Final accuracy: ~85-90% (after 5 rounds)

NORM_THRESHOLD_MULTIPLIER = 3.0- Training time: ~10-15 minutes (CPU, 30 clients)

```

## Configuration

## ComparisonEdit `config.py` to change:

- `NUM_CLIENTS`: Number of federated clients

| Metric | Week 2 (No Defense) | Week 6 (With Defense) |- `NUM_ROUNDS`: Number of training rounds

|--------|---------------------|------------------------|- `N_QUBITS`: Number of qubits (default: 4)

| Final Accuracy | 10-15% | 85-90% |- `N_LAYERS`: Number of quantum layers (default: 4)

| Detection Rate | N/A | 100% |- `DIRICHLET_ALPHA`: Non-IID intensity (default: 0.5)

| Model Recovery | ✗ Failed | ✓ Success |

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
