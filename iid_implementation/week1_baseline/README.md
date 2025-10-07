# Federated Learning with Two-Layer Defense

## Week 1: Simple Baseline (Current Stage)

**Goal:** Get basic FedAvg working on IID MNIST → ~70% accuracy

**What's implemented:**
- Simple CNN model (2 conv + 2 fc layers)
- IID data split across 10 clients
- Basic FedAvg aggregation
- No attacks, no defenses

**How to run:**
```bash
pip install -r requirements.txt
python main.py
```

**Expected result:** ~70% test accuracy after 50 rounds

## Next Steps

- **Week 2:** Add label flipping attack (3/10 malicious clients)
- **Week 3:** Add validation-based defense
- **Week 4:** Add fingerprint pre-filtering
- **Week 5:** Add PQ crypto (Kyber + Dilithium)
- **Week 6:** Test on Non-IID data

## File Structure

```
├── config.py          # All hyperparameters
├── model.py           # Simple CNN
├── data_loader.py     # MNIST loading + IID split
├── client.py          # Local training
├── server.py          # FedAvg aggregation
└── main.py            # Training loop
```

## Design Principles

1. **Small files** - Each file does ONE thing
2. **Simple first** - Get baseline working before adding complexity
3. **Incremental** - Add features one at a time
4. **Measurable** - Track accuracy every round
