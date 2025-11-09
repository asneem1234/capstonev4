# Quantum Federated Learning - Quick Start Guide

## What's Implemented

✅ **Week 1 - Baseline**: Honest quantum federated learning
✅ **Week 2 - Attack**: Byzantine attack (gradient ascent, 40% malicious)  
🚧 **Week 6 - Defense**: Full defense with norm filtering (to be completed)

## Architecture

### Hybrid Quantum-Classical Model
```
MNIST 28x28 → Classical CNN (feature extraction)
           → Dimension reduction (256 → 4)
           → Quantum Circuit (4 qubits, 4 layers, PennyLane)
           → Classical Classifier (4 → 10 classes)
```

### Key Technologies
- **Quantum**: PennyLane (variational quantum circuits)
- **Federated Learning**: Flower (flwr framework)
- **Data**: MNIST Non-IID (Dirichlet α=0.5)
- **Attack**: Gradient ascent (scale_factor=10.0, 40% malicious)
- **Defense**: Median-based norm filtering (week6)

## Installation

```powershell
# Navigate to quantum_version
cd c:\Users\admin\OneDrive\Desktop\capstonev3\mnist_implementation\new_approach\quantum_version

# Install dependencies
cd week1_baseline
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
torchvision>=0.15.0
pennylane>=0.33.0
flwr>=1.6.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

## Usage

### Week 1: Baseline (No Attack)
```powershell
cd week1_baseline
python main.py
```
**Expected**: 85-90% accuracy after 5 rounds

### Week 2: Attack (No Defense)
```powershell
cd week2_attack
python main.py
```
**Expected**: 10-15% accuracy (model collapses due to attack)

### Week 6: Full Defense (to be completed)
```powershell
cd week6_full_defense
python main.py
```
**Expected**: 85-90% accuracy (defense blocks attack)

## Configuration

Edit `config.py` in each week's folder:

```python
NUM_CLIENTS = 30              # Number of federated clients
NUM_ROUNDS = 5                # Training rounds
DIRICHLET_ALPHA = 0.5         # Non-IID intensity
N_QUBITS = 4                  # Quantum circuit size
N_LAYERS = 4                  # Variational layers
MALICIOUS_PERCENTAGE = 0.4    # 40% malicious (week2, week6)
SCALE_FACTOR = 10.0           # Attack intensity
```

## Directory Structure

```
quantum_version/
├── README.md (this file)
├── week1_baseline/
│   ├── main.py                 # Entry point (Flower simulation)
│   ├── quantum_model.py        # Hybrid quantum-classical model
│   ├── client.py               # Flower client
│   ├── server.py               # Flower server (FedAvg)
│   ├── data_loader.py          # Non-IID MNIST
│   ├── config.py               # Configuration
│   └── requirements.txt        # Dependencies
│
├── week2_attack/
│   ├── attack.py               # *** Model poisoning attack ***
│   ├── (same structure as week1, with attack enabled)
│   └── config.py               # ATTACK_ENABLED=True
│
└── week6_full_defense/
    ├── (to be completed)
    └── defense modules (norm filtering, PQ crypto, fingerprints)
```

## Next Steps

To complete the quantum version:

1. ✅ Week1 baseline - **DONE**
2. ✅ Week2 attack structure - **DONE** (needs client.py update)
3. 🚧 Week2 client integration - Update `client.py` to apply attacks
4. 🚧 Week6 defense - Port norm filtering defense from non-IID
5. 🚧 Testing - Run all three weeks and compare results

## Notes

- **Quantum simulation** runs on CPU (no quantum hardware needed)
- **Flower simulation** allows efficient local testing
- Each client has ~2000 samples (60000 total / 30 clients)
- Training time: ~10-15 minutes per week on CPU
- Week2 requires client.py update to integrate attack.py

## Comparison with Non-IID Implementation

| Feature | Non-IID (Classical) | Quantum Version |
|---------|---------------------|-----------------|
| Model | Classical CNN | Hybrid Quantum-Classical |
| Framework | Custom FL | Flower (flwr) |
| Quantum | ❌ No | ✅ PennyLane (4 qubits) |
| Attack | Gradient Ascent | Same (gradient ascent) |
| Defense | Norm Filtering | Same (to be ported) |
| Non-IID | Dirichlet α=0.5 | Same |
| Accuracy | ~93% (with defense) | TBD (expected similar) |

## Research Questions

1. Does quantum circuit improve Byzantine resilience?
2. How do quantum features affect attack detection?
3. Is norm filtering effective for quantum updates?
4. Performance: quantum vs classical FL under attack

## References

- **PennyLane**: https://pennylane.ai/
- **Flower**: https://flower.dev/
- **Non-IID Implementation**: `../non_iid_implementation/`
- **Defense Results**: `../non_iid_implementation/DEFENSE_RESULTS.md`
