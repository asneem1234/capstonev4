# Week 2: Federated Learning with Label Flipping Attack
# Fetal Plane Classification

## Goal
Demonstrate the impact of **label flipping attacks** on federated learning for fetal plane classification. This shows what happens when **malicious clients** poison the training process.

---

## What This Is

This implements a federated learning system with:
- ✅ **Non-IID data** (Dirichlet distribution, α=0.5)
- ⚠️ **Malicious clients** (20% of clients flip labels)
- ❌ **No defenses** (server blindly accepts all updates)
- 📉 **Expected result**: Significant accuracy degradation

**Purpose**: Shows the vulnerability of federated learning to poisoning attacks.

---

## Attack Mechanism

### Label Flipping

Malicious clients flip labels before training:
```python
# For 6 classes: 0→5, 1→4, 2→3, 3→2, 4→1, 5→0
Trans-thalamic → Others
Trans-cerebellum → Femur
Trans-ventricular → Maternal cervix
...
```

### Random Selection

Each round, **20%** of clients are randomly selected to be malicious:
```python
MALICIOUS_PERCENTAGE = 0.2  # 20%
RANDOM_MALICIOUS = True     # Different malicious clients each round
```

This simulates a realistic scenario where you don't know which hospitals/clients are compromised.

---

## Files Included

```
week2_attack/
├── config.py           # Configuration with attack enabled
├── attack.py           # Label flipping attack implementation
├── data_loader.py      # Same as week1 (Non-IID split)
├── model.py            # Same as week1 (ResNet18)
├── client.py           # Client with attack capability
├── server.py           # Server with NO defense
├── main.py             # Training loop with attack
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Use the same dataset structure as Week 1:
```
data/fetal_planes/
    train/
        class_0/
            ...
        class_1/
            ...
    test/
        ...
```

### 3. Run Training with Attack

```bash
cd fetal_plane_implementation/week2_attack
python main.py
```

---

## Expected Output

### Malicious Client Selection
```
ROUND 1:
[MALICIOUS SELECTION]
  Randomly selected 2 malicious clients: [3, 7]

[CLIENT TRAINING]
  Client 0: Train Acc=65.20%, Loss=1.12, Update Norm=2.34
  Client 1: Train Acc=68.45%, Loss=1.08, Update Norm=2.28
  Client 2: Train Acc=64.10%, Loss=1.15, Update Norm=2.41
  Client 3 [MALICIOUS]: Train Acc=22.30%, Loss=2.85, Update Norm=4.12
  Client 4: Train Acc=67.50%, Loss=1.10, Update Norm=2.35
  ...
  Client 7 [MALICIOUS]: Train Acc=18.70%, Loss=3.02, Update Norm=4.35
```

**Key Observations**:
- **Malicious clients** have much **lower accuracy** (~20% vs ~65%)
- **Malicious clients** have much **higher loss** (~3.0 vs ~1.1)
- **Malicious clients** produce **larger updates** (~4.2 vs ~2.3)

### Accuracy Degradation
```
ROUND 1:
  Global Test Accuracy: 42.15%

ROUND 2:
  ⚠️  WARNING: No defense - malicious updates included!
  Global Test Accuracy: 38.50% ↓ -3.65%

ROUND 3:
  Global Test Accuracy: 35.20% ↓ -3.30%

...

ROUND 10:
  Global Test Accuracy: 28.40% ↓ -2.15%
```

**Key Observation**: Accuracy **decreases** over time instead of improving!

### Final Results
```
Initial Test Accuracy: 18.50%
Final Test Accuracy:   28.40%
Total Change:          +9.90%

❌ Model degraded compared to baseline!
   20% malicious clients successfully poisoned the model
```

**Compare with Week 1 Baseline**:
- **Week 1 (No Attack)**: ~82% accuracy ✅
- **Week 2 (With Attack)**: ~28% accuracy ❌
- **Impact**: **-54%** accuracy loss!

---

## Understanding the Attack

### Why Does This Work?

1. **Label Flipping Confuses the Model**
   - Malicious clients teach: "Trans-thalamic looks like Others"
   - Honest clients teach: "Trans-thalamic looks like Trans-thalamic"
   - Result: Model gets conflicting signals

2. **Averaging Amplifies Confusion**
   ```python
   # FedAvg with 10 clients, 2 malicious
   global_update = 0.8 * honest_updates + 0.2 * malicious_updates
   ```
   Even 20% malicious data significantly corrupts the model.

3. **No Defense Mechanism**
   - Server blindly accepts all updates
   - No validation or filtering
   - Malicious updates directly poison the global model

### Attack Indicators

Malicious clients exhibit suspicious patterns:
- **Low training accuracy** (training on wrong labels)
- **High loss** (model struggles with flipped labels)
- **Large update norms** (dramatic weight changes)

These indicators will be used for detection in Week 6!

---

## Configuration

In `config.py`:

```python
# Enable attack
ATTACK_ENABLED = True

# 20% malicious clients
MALICIOUS_PERCENTAGE = 0.2

# Randomly select different malicious clients each round
RANDOM_MALICIOUS = True

# No defense (for this week)
DEFENSE_ENABLED = False
```

### Experiment with Different Attack Rates

```python
MALICIOUS_PERCENTAGE = 0.1   # 10% malicious (less impact)
MALICIOUS_PERCENTAGE = 0.3   # 30% malicious (more impact)
MALICIOUS_PERCENTAGE = 0.5   # 50% malicious (severe impact)
```

### Fixed vs Random Malicious Clients

```python
# Random (default): Different malicious clients each round
RANDOM_MALICIOUS = True

# Fixed: Same malicious clients every round
RANDOM_MALICIOUS = False
MALICIOUS_CLIENTS = [2, 5, 8]  # Always clients 2, 5, 8
```

---

## Medical Imaging Context

### Real-World Threat Scenario

In federated medical imaging, attacks could come from:
1. **Compromised Hospital**: Hacked system sending corrupted data
2. **Malicious Insider**: Employee deliberately sabotaging training
3. **Data Quality Issues**: Mislabeled data that looks like an attack

### Impact on Healthcare

A poisoned fetal plane classifier could:
- ❌ Misclassify critical anatomical planes
- ❌ Lead to incorrect diagnoses
- ❌ Undermine trust in AI-assisted ultrasound

**This is why defenses are critical!**

---

## Comparison with Baseline

| Metric | Week 1 (Baseline) | Week 2 (Attack) | Difference |
|--------|-------------------|-----------------|------------|
| Initial Acc | 18.5% | 18.5% | 0% |
| Final Acc | **82.4%** | **28.4%** | **-54.0%** |
| Improvement | +63.9% | +9.9% | -54.0% |
| Malicious % | 0% | 20% | +20% |

**Conclusion**: Just 20% malicious clients can **destroy** model performance!

---

## Attack Types (Implemented and Potential)

### Implemented: Label Flipping
```python
class LabelFlippingAttack:
    def flip_labels(self, labels):
        # Flip to complementary classes
        return num_classes - 1 - labels
```

### Potential: Gradient Poisoning
```python
class GradientPoisoningAttack:
    def apply_to_update(self, update):
        # Scale up gradients to amplify effect
        return update * scale_factor
```

Gradient poisoning is included in `attack.py` but not used by default.

---

## Next Steps

### Week 6: Full Defense

To recover from attacks, we'll implement:
1. **Validation-based filtering**: Detect malicious updates using server validation data
2. **Client fingerprinting**: Identify suspicious client behavior patterns
3. **Post-quantum cryptography**: Ensure update integrity and authenticity

**Goal**: Recover baseline performance (~82%) even with 20% malicious clients!

---

## Troubleshooting

### Model Still Improving?

If accuracy still increases despite attacks:
- Increase `MALICIOUS_PERCENTAGE` (try 0.3 or 0.4)
- Reduce `NUM_ROUNDS` to see short-term impact
- Check that `ATTACK_ENABLED = True` in config.py

### Accuracy Too Low?

If starting accuracy is very low:
- Check dataset is loaded correctly
- Verify model is using pretrained weights (`PRETRAINED = True`)
- Ensure normalization matches your dataset

---

## Key Takeaways

1. **Federated learning is vulnerable** to even small percentages of malicious clients
2. **Malicious updates** can be detected by monitoring accuracy, loss, and update norms
3. **Without defenses**, model performance degrades significantly
4. **Medical applications** need robust defense mechanisms to be trustworthy

---

## References

- **Label Flipping Attacks**: Tolpegin et al., "Data Poisoning Attacks Against Federated Learning Systems"
- **Byzantine Attacks**: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
- **Medical FL Security**: Kumar et al., "Security and Privacy in Federated Learning for Healthcare"
