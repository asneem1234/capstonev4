# Non-IID Federated Learning Implementation

This folder contains the **Non-IID (heterogeneous data)** implementation of the federated learning system with full three-layer defense against Byzantine attacks.

## 🎯 Key Difference from IID Implementation

**Data Distribution:**
- **IID Implementation**: Data is uniformly distributed across clients (each client has balanced samples from all 10 MNIST classes)
- **Non-IID Implementation**: Data is heterogeneously distributed using **Dirichlet Distribution** (each client has imbalanced class distribution, some classes dominate)

**Why Non-IID Matters:**
- Real-world federated learning scenarios often have heterogeneous data (e.g., hospitals with different patient demographics, mobile devices with personalized usage patterns)
- Non-IID data makes Byzantine detection harder (legitimate clients may have very different gradient patterns)
- Tests robustness of defense mechanisms under realistic conditions

---

## 📊 Dirichlet Distribution for Non-IID Split

### What is Dirichlet Distribution?

The Dirichlet distribution is used to create heterogeneous data splits where each client receives an imbalanced subset of classes.

**Parameter α (Alpha):**
- **α = 0.1**: Highly non-IID (each client has 1-2 dominant classes, 90% concentration)
- **α = 0.5**: Moderately non-IID (uneven distribution, used in this implementation)
- **α = 1.0**: Slightly non-IID (some imbalance)
- **α = 10.0+**: Nearly IID (approaches uniform distribution)

### Example with α = 0.5:
```
Client 0: [8, 120, 5, 890, 15, 4, 3, 200, 10, 20] → Dominated by class 3 (890 samples)
Client 1: [500, 50, 10, 5, 600, 20, 8, 3, 100, 15] → Dominated by classes 0 & 4
Client 2: [20, 700, 5, 10, 50, 3, 800, 15, 20, 8] → Dominated by classes 1 & 6
Client 3: [100, 10, 850, 20, 5, 50, 10, 3, 600, 15] → Dominated by classes 2 & 8
Client 4: [15, 20, 10, 5, 100, 700, 8, 600, 20, 850] → Dominated by classes 5, 7, 9
```

---

## 🗂️ Folder Structure

```
non_iid_implementation/
├── README.md                       # This file
├── RESULTS.md                      # Experimental results (to be created)
└── week6_full_defense/             # Full three-layer defense with Non-IID data
    ├── config.py                   # Configuration with DIRICHLET_ALPHA=0.5
    ├── model.py                    # SimpleCNN model
    ├── data_loader.py              # Non-IID data split using Dirichlet
    ├── client.py                   # Client-side training + fingerprint computation
    ├── server.py                   # Server with PQ crypto + fingerprint + validation
    ├── attack.py                   # Label flipping attack
    ├── defense_fingerprint_client.py  # Client-side fingerprint defense
    ├── defense_validation.py       # Validation defense
    ├── pq_crypto.py                # Post-quantum cryptography (simulated)
    └── main.py                     # Training loop
```

---

## 🚀 Running the Non-IID Experiment

### Prerequisites
Same as IID implementation:
```bash
pip install torch torchvision numpy
```

### Run Week 6 (Full Defense with Non-IID Data)
```bash
cd non_iid_implementation/week6_full_defense
python main.py
```

### Expected Output:
```
======================================================================
Federated Learning - NON-IID with Full Defense (Week 6)
======================================================================
Clients: 5
Rounds: 5
Local epochs: 3
Data Distribution: NON-IID (Dirichlet α=0.5)
Attack enabled: True
Malicious clients: [0, 1]
PQ Crypto: ENABLED (Simulated mode)
Fingerprint Defense: ENABLED (CLIENT-SIDE)
Validation Defense: ENABLED
======================================================================

Loading data...

Creating Non-IID data split with Dirichlet(α=0.5)...
Data distribution per client:
  Client 0: 11800 samples, dominant class=3 (2156 samples), distribution=[...]
  Client 1: 11800 samples, dominant class=7 (2489 samples), distribution=[...]
  Client 2: 11800 samples, dominant class=1 (2345 samples), distribution=[...]
  Client 3: 11800 samples, dominant class=5 (2012 samples), distribution=[...]
  Client 4: 11800 samples, dominant class=9 (2287 samples), distribution=[...]

[Training proceeds with three-layer defense...]
```

---

## 🔬 Research Questions

### Key Questions to Answer:

1. **Does Non-IID data affect defense performance?**
   - Compare malicious detection rate: IID vs Non-IID
   - Expected: Similar or slightly lower detection rate due to higher gradient diversity

2. **Does accuracy degrade with Non-IID data?**
   - Compare final accuracy: IID (97-98%) vs Non-IID (?)
   - Expected: 2-5% accuracy drop due to heterogeneity

3. **Are fingerprint thresholds still effective?**
   - Do honest clients cluster together despite different data distributions?
   - Does threshold 0.90 still work or need adjustment?

4. **Does metadata (loss/accuracy) help more in Non-IID?**
   - With diverse gradients, do loss patterns become more important?
   - Test with/without metadata enhancement

5. **Convergence speed comparison**
   - IID: Fast convergence (2-3 rounds to 95%+)
   - Non-IID: Expected slower convergence (4-5 rounds?)

---

## 📈 Expected Results (Hypothesis)

### Baseline Expectations:

| Metric | IID | Non-IID (α=0.5) | Impact |
|--------|-----|-----------------|--------|
| Final Accuracy (no attack) | 98.57% | 94-96% | -2 to -4% |
| Final Accuracy (with attack, no defense) | 90.32% | 85-88% | -3 to -5% |
| Final Accuracy (full defense) | 97.20% | 92-95% | -3 to -5% |
| Malicious Detection Rate | 100% | 95-100% | Slightly lower |
| Convergence Rounds | 2-3 | 3-4 | Slower |

### Defense Performance:

| Defense Layer | IID Performance | Expected Non-IID | Notes |
|---------------|----------------|------------------|-------|
| PQ Crypto | 100% verified | 100% verified | Unaffected by data |
| Fingerprints (threshold 0.90) | All outliers initially | More outliers | Higher gradient diversity |
| Validation | Δloss +9 (malicious) vs -0.08 (honest) | Δloss +8 (malicious) vs -0.2 (honest) | Similar separation |

---

## 🎯 Experimental Protocol

### Step 1: Baseline Run (α=0.5)
```bash
cd week6_full_defense
python main.py
```
- Record all 5 rounds
- Note malicious detection per round
- Record final accuracy

### Step 2: Vary Dirichlet Alpha (Optional)
Edit `config.py`:
```python
DIRICHLET_ALPHA = 0.1  # Highly non-IID
# or
DIRICHLET_ALPHA = 1.0  # Slightly non-IID
```
Compare results across different heterogeneity levels.

### Step 3: Compare with IID
- Load IID results from `iid_implementation/RESULTS.md`
- Create comparison table in `non_iid_implementation/RESULTS.md`

---

## 📊 Data Distribution Visualization

### What to Observe:
When running `main.py`, observe the data distribution output:
```
Data distribution per client:
  Client 0: 11800 samples, dominant class=X (YYYY samples)
```

**Key Metrics:**
- **Dominant class percentage**: YYYY/11800 (should be 15-25% for α=0.5)
- **Class diversity**: How many classes have >500 samples? (typically 4-6 for α=0.5)
- **Imbalance ratio**: Max class / Min class (typically 10-50x for α=0.5)

---

## 🔧 Configuration Parameters

### Key Settings in `config.py`:

```python
# Non-IID specific
USE_NON_IID = True              # Enable Non-IID split
DIRICHLET_ALPHA = 0.5           # Concentration parameter

# Same as IID
NUM_CLIENTS = 5
MALICIOUS_CLIENTS = [0, 1]      # 40% malicious
LOCAL_EPOCHS = 3
NUM_ROUNDS = 5
COSINE_THRESHOLD = 0.90         # May need tuning for Non-IID
```

### Tunable Parameters:
1. **DIRICHLET_ALPHA**: Try 0.1, 0.5, 1.0 to vary heterogeneity
2. **COSINE_THRESHOLD**: May need to lower to 0.85 if too many false positives
3. **LOCAL_EPOCHS**: Increase to 5 if convergence is too slow

---

## 📚 Academic Context

### Why Test Non-IID?

**From Federated Learning Papers:**
- McMahan et al. (2017): "FedAvg performance degrades with non-IID data"
- Li et al. (2020): "Non-IID data causes weight divergence"
- Karimireddy et al. (2020): "Byzantine robustness harder with heterogeneous data"

**Our Contribution:**
- Test three-layer defense (PQ crypto + fingerprints + validation) under Non-IID
- Compare detection rates: IID vs Non-IID
- Evaluate if client-side fingerprints + metadata maintain effectiveness

### Expected Findings for Paper:

> "Our three-layer defense maintains 95%+ accuracy and >95% malicious detection rate 
> even under moderately heterogeneous data (Dirichlet α=0.5), demonstrating robustness 
> to real-world federated learning scenarios."

---

## 🐛 Troubleshooting

### Issue: "Division by zero" or "Empty indices"
**Solution**: Some clients may have 0 samples for certain classes. This is expected with Dirichlet. The code handles this gracefully.

### Issue: Accuracy much lower than IID
**Solution**: This is expected! Non-IID data inherently makes learning harder. Expect 2-5% accuracy drop.

### Issue: All clients marked as outliers
**Solution**: With Non-IID, gradient diversity is higher. Consider lowering `COSINE_THRESHOLD` from 0.90 to 0.85 or 0.80.

### Issue: Malicious clients not detected
**Solution**: Check validation layer (Layer 3). Even if fingerprints fail, validation should catch malicious updates (Δloss > 0.1).

---

## 📖 Next Steps

1. ✅ Run `week6_full_defense/main.py`
2. ⏳ Collect results for all 5 rounds
3. ⏳ Create `RESULTS.md` with IID vs Non-IID comparison
4. ⏳ (Optional) Test with different α values (0.1, 1.0)
5. ⏳ (Optional) Visualize data distribution per client
6. ⏳ Update paper with Non-IID findings

---

## 📞 Reference

For IID implementation and baseline comparisons, see:
- `../iid_implementation/README.md`
- `../iid_implementation/RESULTS.md`

For detailed defense mechanisms, see:
- IID Week 6: `../iid_implementation/week6_fingerprints_client/`
