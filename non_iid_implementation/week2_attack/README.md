# Week 2: Label Flipping Attack with Non-IID Data

## Goal
Show that the system is **vulnerable** to Byzantine attacks when malicious clients flip labels, **even with heterogeneous (Non-IID) data**.

**Expected result:** Accuracy drops significantly (from ~85% to ~40-50%)

---

## What's New

### Files Added:
- `attack.py` - Simple label flipping attack (0→9, 1→8, 2→7, etc.)
- `data_loader.py` - Non-IID data split using Dirichlet distribution
- `config.py` - Attack enabled, 2/5 clients are malicious, Non-IID settings

### Key Difference from IID Implementation:
- **Non-IID Data Distribution**: Uses Dirichlet distribution (α=0.5) to create heterogeneous data
- Each client has a different class distribution (e.g., Client 0 might have mostly digits 0-2, Client 1 has 3-5, etc.)
- This is more realistic for real-world federated learning scenarios

### Files Modified:
- `client.py` - Clients apply attack if malicious
- `main.py` - Shows which clients are malicious and data distribution

---

## How It Works

### Non-IID Data Split
```python
# Dirichlet parameter α controls heterogeneity:
# α = 0.1: Highly non-IID (each client has 1-2 dominant classes)
# α = 0.5: Moderately non-IID (uneven distribution) ← Default
# α = 1.0: Slightly non-IID
# α = 10+: Approaches IID
```

Example distribution with α=0.5:
```
Client 0: 12000 samples, dominant class=3, distribution=[452, 398, 1205, 3542, ...]
Client 1: 11800 samples, dominant class=7, distribution=[102, 234, 445, 234, ...]
Client 2: 12200 samples, dominant class=1, distribution=[234, 4123, 345, 123, ...]
```

### Label Flipping Attack
- Malicious clients (0, 1) flip all labels: `0→9, 1→8, 2→7, 3→6, 4→5, 5→4, 6→3, 7→2, 8→1, 9→0`
- This creates poisoned gradients that degrade the global model
- Honest clients (2, 3, 4) still train correctly
- Server aggregates ALL updates equally (no defense yet)

### Why Non-IID Makes This Interesting
- In IID setting, all clients see similar data, so malicious updates stand out
- In Non-IID setting, clients naturally have different gradient patterns
- This makes Byzantine attacks **harder to detect** but still **effective at poisoning**

---

## Running the Code

```bash
cd non_iid_implementation/week2_attack
python main.py
```

---

## What to Observe

### 1. Data Distribution (Start of Training)
```
Creating Non-IID data split with Dirichlet(α=0.5)...
Data distribution per client:
  Client 0: 12000 samples, dominant class=3 (3542 samples)
  Client 1: 11800 samples, dominant class=7 (4123 samples)
  ...
```

### 2. Malicious Client Behavior (During Training)
```
[CLIENT TRAINING]
  Client 0 [MALICIOUS]: Train Acc=12.34%, Loss=2.15, Update Norm=2.45
  Client 1 [MALICIOUS]: Train Acc=10.87%, Loss=2.31, Update Norm=2.67
  Client 2: Train Acc=89.23%, Loss=0.35, Update Norm=0.87
  Client 3: Train Acc=87.45%, Loss=0.41, Update Norm=0.92
  Client 4: Train Acc=90.12%, Loss=0.32, Update Norm=0.85
```

**Notice:**
- Malicious clients have **LOW accuracy** (~10-15%) and **HIGH loss** (~2.0-2.5)
- Honest clients have **HIGH accuracy** (~85-90%) and **LOW loss** (~0.3-0.5)
- Malicious clients have **LARGER update norms** (poisoned gradients are more aggressive)

### 3. Global Model Degradation (Each Round)
```
ROUND 1:
  Global Test Accuracy: 72.34% (dropped from 85.2%)

ROUND 2:
  Global Test Accuracy: 58.67% (dropped further)

ROUND 3:
  Global Test Accuracy: 45.23% (severe degradation)
```

### 4. Final Results
```
Initial Test Accuracy: 85.20%
Final Test Accuracy:   42.15%
Change:                -43.05%

⚠️  Notice: Accuracy should DROP significantly due to malicious clients!
```

---

## Key Insights

### 1. Attack is Effective Despite Non-IID Data
- Even though clients have heterogeneous data, the attack still degrades the model
- Malicious updates from 40% of clients (2 out of 5) are enough to poison the global model

### 2. Non-IID Makes Detection Harder
- In IID setting, all honest clients have similar gradients → outlier detection easier
- In Non-IID setting, honest clients have diverse gradients → outlier detection harder
- This motivates advanced defenses that account for Non-IID heterogeneity

### 3. Vulnerability is Clear
- Without defense, federated learning is **highly vulnerable** to Byzantine attacks
- A single malicious hospital, bank, or IoT device can sabotage the entire system
- This proves the need for Byzantine-robust aggregation (Week 3+)

---

## Comparison: IID vs Non-IID

| Aspect | IID (Week 2 IID) | Non-IID (This Week) |
|--------|------------------|---------------------|
| **Data Distribution** | Uniform across clients | Heterogeneous (Dirichlet) |
| **Honest Gradient Similarity** | High (easy to cluster) | Lower (diverse patterns) |
| **Attack Effectiveness** | Very effective | **Still very effective** |
| **Detection Difficulty** | Moderate | **Higher** (natural diversity) |
| **Final Accuracy Drop** | ~40-45% | ~40-45% (similar impact) |

**Key Takeaway**: Non-IID data makes detection harder but doesn't reduce attack effectiveness!

---

## Next Steps

### Week 3: Validation Defense (Non-IID)
Add a simple validation-based defense that:
- Tests each client's update on a held-out validation set
- Rejects updates that degrade model performance
- Must account for Non-IID data (honest clients might have diverse updates)

### Week 4-6: Advanced Defenses (Non-IID)
- Week 4: Fingerprint-based clustering (gradient similarity)
- Week 5: Post-quantum cryptography layer
- Week 6: Full three-layer defense (client-side fingerprints)

---

## Technical Notes

### Why Malicious Clients Have Low Accuracy
When training on flipped labels:
- Image of "3" → Model tries to predict "6"
- This is **impossible** to learn correctly
- Model parameters move in wrong directions
- Training accuracy stays low (~10-15%)
- Loss stays high (~2.0-2.5)

### Why Update Norm is Large
- Malicious gradients try to reverse correct patterns
- This creates large parameter changes (high norm)
- Honest gradients make small adjustments (low norm)
- Update norm is a simple signal, but not sufficient alone

### Why Server Still Aggregates
- Server has no defense mechanism (Week 2 baseline)
- Simple FedAvg: average ALL updates equally
- Malicious updates (2/5 = 40%) poison the average
- Global model degrades over time

---

## Code Structure

```
week2_attack/
├── config.py           # Configuration (Non-IID, attack enabled)
├── data_loader.py      # Non-IID Dirichlet data split
├── model.py            # Simple CNN (unchanged)
├── attack.py           # Label flipping attack
├── client.py           # Client training with attack
├── server.py           # Simple FedAvg aggregation (no defense)
├── main.py             # Training loop with logging
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## Troubleshooting

### If accuracy doesn't drop much:
- Check that `ATTACK_ENABLED = True` in config.py
- Check that `MALICIOUS_CLIENTS = [0, 1]` is set
- Verify malicious clients show low accuracy in logs

### If data distribution looks IID:
- Check that `DIRICHLET_ALPHA = 0.5` (or lower)
- Lower α = more heterogeneous
- Try α=0.1 for highly non-IID

### If code crashes:
- Install dependencies: `pip install -r requirements.txt`
- Check data directory: should auto-download MNIST
- Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`

---

## For Your Presentation

When explaining this to your audience:

1. **Start with motivation**: "Real-world federated learning uses Non-IID data—hospitals have different patient demographics, banks have different customer types."

2. **Show the attack**: "Even with heterogeneous data, Byzantine attacks are devastatingly effective."

3. **Highlight the challenge**: "Non-IID makes detection harder because honest clients naturally have diverse gradients."

4. **Build suspense**: "This sets up the need for advanced defenses that work under realistic Non-IID conditions."

---

## References

- **Dirichlet Distribution**: Commonly used to model Non-IID data in federated learning research
- **Label Flipping**: Classic Byzantine attack (Bagdasaryan et al., 2018)
- **FedAvg**: McMahan et al., 2017 - "Communication-Efficient Learning of Deep Networks from Decentralized Data"

---

**Ready for Week 3!** 🚀

Next: Add validation-based defense that works with Non-IID data.
