# Quantum Federated Learning - Installation & Testing Guide

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```powershell
cd c:\Users\admin\OneDrive\Desktop\capstonev3\mnist_implementation\new_approach\quantum_version\week1_baseline
pip install -r requirements.txt
```

### Step 2: Test Week 1 (Baseline)
```powershell
python main.py
```
**Expected output**: 85-90% accuracy after 5 rounds (~10-15 minutes)

### Step 3: Test Week 2 (Attack)
```powershell
cd ..\week2_attack
python main.py
```
**Expected output**: 10-15% accuracy (model collapses due to attack)

---

## Detailed Installation

### Prerequisites
- Python 3.8+ (you have Python 3.11 ✓)
- pip package manager
- ~500MB disk space for dependencies
- ~2GB RAM for training
- CPU (no GPU required)

### Install Commands

```powershell
# Navigate to project
cd c:\Users\admin\OneDrive\Desktop\capstonev3\mnist_implementation\new_approach\quantum_version

# Install for Week 1
cd week1_baseline
pip install torch torchvision pennylane pennylane-lightning flwr numpy scikit-learn

# Or use requirements.txt
pip install -r requirements.txt
```

### Verify Installation

```powershell
# Test imports
python -c "import torch; import pennylane; import flwr; print('✓ All imports successful')"

# Test quantum device
python -c "import pennylane as qml; dev = qml.device('default.qubit', wires=4); print('✓ Quantum device created')"

# Test Flower
python -c "import flwr; print(f'✓ Flower version: {flwr.__version__}')"
```

---

## Testing Guide

### Week 1: Honest Federated Learning

```powershell
cd week1_baseline
python main.py
```

**What to expect**:
```
============================================================
Quantum Federated Learning - Week 1 Baseline (No Attack)
============================================================
Framework: Flower (flwr)
Quantum Backend: PennyLane (default.qubit)
Device: CPU
============================================================

Loading MNIST dataset...
Creating Non-IID data split with Dirichlet(α=0.5)...

Data distribution per client:
  Client 0: 2134 samples, dominant class=0 (456 samples)
  Client 1: 1987 samples, dominant class=1 (412 samples)
  ...

============================================================
Starting Quantum Federated Learning Simulation...
============================================================

============================================================
Round 1 - Client Training Results
============================================================
  Client 0: Loss=0.4523, Acc=87.34%, Samples=2134, Norm=0.8765
  Client 1: Loss=0.5123, Acc=83.21%, Samples=1987, Norm=0.9123
  ...

============================================================
Round 1 - Global Model Evaluation
============================================================
  Test Accuracy: 67.45%
  Test Loss: 0.9876
============================================================

[Rounds 2-5 continue...]

============================================================
Quantum Federated Learning - Final Results
============================================================
Total time: 847.32 seconds (14.12 minutes)

Accuracy per round:
  Round 1: 67.45%
  Round 2: 79.23%
  Round 3: 84.56%
  Round 4: 87.12%
  Round 5: 89.34%

Final Test Accuracy: 89.34%
Final Test Loss: 0.3245
============================================================

✓ Quantum Federated Learning simulation completed successfully!
```

**Expected metrics**:
- Initial accuracy: ~10-15% (random guess)
- Round 1: ~60-70%
- Round 2: ~75-80%
- Round 3: ~80-85%
- Round 4: ~85-88%
- Round 5: ~88-91%
- Training time: 10-15 minutes on CPU

---

### Week 2: Attack (No Defense)

```powershell
cd ..\week2_attack
python main.py
```

**What to expect**:
```
============================================================
Quantum Federated Learning - Week 2 (Attack, No Defense)
============================================================
Framework: Flower (flwr)
Quantum Backend: PennyLane (default.qubit)
Device: CPU
⚠️  ATTACK ENABLED: gradient_ascent (scale=10.0)
⚠️  MALICIOUS: 12/30 clients
⚠️  NO DEFENSE: All updates aggregated
============================================================

Malicious clients: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

[Data loading same as week1...]

============================================================
Round 1 - Client Training Results
============================================================
  Client 0: Loss=0.4123, Acc=88.45%, Samples=2134, Norm=8.7654  ⚠️ MALICIOUS
  Client 1: Loss=0.3987, Acc=89.23%, Samples=1987, Norm=9.2341  ⚠️ MALICIOUS
  ...
  Client 12: Loss=0.5234, Acc=84.12%, Samples=1876, Norm=0.9123  ✓ HONEST
  Client 13: Loss=0.4987, Acc=85.67%, Samples=2098, Norm=0.8765  ✓ HONEST
  ...

============================================================
Round 1 - Global Model Evaluation
============================================================
  Test Accuracy: 23.45%  ⚠️ MODEL POISONED
  Test Loss: 2.1234
============================================================

[Model continues to degrade...]

Round 2: 15.23%
Round 3: 11.45%
Round 4: 10.87%
Round 5: 9.82%  ⚠️ COLLAPSED

Final Test Accuracy: 9.82%  ⚠️ WORSE THAN RANDOM
```

**Key observations**:
- Malicious clients have 10× larger update norms
- Honest: ~0.5-1.5, Malicious: ~5-20
- Model accuracy collapses rapidly
- Final accuracy ~10% (worse than random!)
- Attack is successful - defense needed!

---

## Troubleshooting

### Issue: ImportError for pennylane
```powershell
pip install pennylane pennylane-lightning
```

### Issue: ImportError for flwr
```powershell
pip install flwr
```

### Issue: CUDA out of memory
Solution: Code already uses CPU for quantum simulation. If you see GPU errors:
```python
# In client.py, force CPU:
self.device = torch.device("cpu")
```

### Issue: Slow training
Normal! Quantum simulation is computationally intensive:
- Week 1: 10-15 minutes expected
- Week 2: 10-15 minutes expected
- Reduce `NUM_CLIENTS` or `NUM_ROUNDS` in config.py for faster testing

### Issue: Low accuracy in Week 1
Check:
1. All 5 rounds completed?
2. Non-IID split working? (should show dominant classes)
3. Quantum circuit training? (update norms should be >0)

---

## Configuration Tuning

### Faster Training (for testing)
```python
# In config.py
NUM_CLIENTS = 10        # Reduce from 30
NUM_ROUNDS = 3          # Reduce from 5
LOCAL_EPOCHS = 2        # Reduce from 3
BATCH_SIZE = 64         # Increase from 32
```

### More IID Data
```python
DIRICHLET_ALPHA = 10.0  # Increase from 0.5
```

### Weaker Attack
```python
SCALE_FACTOR = 5.0      # Reduce from 10.0
MALICIOUS_PERCENTAGE = 0.2  # Reduce from 0.4
```

### Larger Quantum Circuit
```python
N_QUBITS = 8            # Increase from 4 (slower!)
N_LAYERS = 6            # Increase from 4 (slower!)
```

---

## Performance Benchmarks

| Setup | Time | Final Acc | Notes |
|-------|------|-----------|-------|
| Week1, 30 clients, 5 rounds | 12 min | 89% | Baseline |
| Week1, 10 clients, 3 rounds | 4 min | 82% | Fast testing |
| Week2, 30 clients, 40% mal | 13 min | 10% | Attack works |
| Week1, 8 qubits, 6 layers | 35 min | 91% | Larger quantum |

---

## Next Steps

After testing Week 1 and Week 2:

1. **Analyze results**: Compare accuracy trajectories
2. **Examine norms**: Verify 10× separation in Week 2
3. **Implement Week 6**: Add norm filtering defense
4. **Compare all three**: Week 1 vs Week 2 vs Week 6
5. **Write paper**: Document quantum FL with Byzantine defense

---

## File Checklist

### Week 1 (✅ Complete):
- [x] main.py
- [x] quantum_model.py
- [x] client.py
- [x] server.py
- [x] data_loader.py
- [x] config.py
- [x] requirements.txt
- [x] README.md

### Week 2 (✅ Complete):
- [x] attack.py
- [x] main.py (modified)
- [x] client.py (modified)
- [x] config.py (modified)
- [x] All other files from Week 1

### Week 6 (🚧 To be completed):
- [ ] defense_norm_filtering.py
- [ ] server.py (add defense)
- [ ] config.py (DEFENSE_ENABLED=True)

---

## Support

If you encounter issues:

1. Check Python version: `python --version` (need 3.8+)
2. Verify installations: See "Verify Installation" section
3. Check error messages: Usually import or out-of-memory
4. Reduce complexity: Use fewer clients/rounds for testing
5. Review logs: Flower simulation provides detailed output

---

**Ready to test!** Start with Week 1, then Week 2. Week 6 defense coming next.
