# Week 3: Validation-Based Defense

## Goal
Add **Layer 2 defense** to detect and reject Byzantine updates using validation set testing.

**Expected result:** Accuracy recovers from ~40-50% (Week 2) back to ~70-80%

## What's New

### Files Added:
- `defense_validation.py` - Validates updates using held-out validation set

### Files Modified:
- `config.py` - Defense enabled, validation threshold set
- `data_loader.py` - Creates validation set (1000 samples)
- `server.py` - Filters updates before aggregation
- `main.py` - Shows validation results in logs

## How It Works

**Validation Defense:**
1. Server holds out 1000 samples for validation
2. For each client update:
   - Apply update to temporary model
   - Test on validation set
   - Compare loss before vs after
   - **Accept** if loss doesn't increase too much (threshold = 0.1)
   - **Reject** if loss increases significantly (indicates poisoning)
3. Aggregate only accepted updates

**Why It Works:**
- Malicious updates (label flipping) increase validation loss
- Honest updates decrease validation loss
- Simple threshold catches obvious attacks

## Run It

```bash
cd c:\Users\admin\OneDrive\Desktop\capstonev3\mnist_implementation\new_approach\week3
python main.py
```

## What to Observe

1. **Validation filtering** shows which clients are accepted/rejected
2. **Malicious clients** (0, 1) should be **rejected** most rounds
3. **Honest clients** (2, 3, 4) should be **accepted**
4. **Global accuracy** should **recover** to ~70-80%

Expected output:
```
[VALIDATION FILTERING]
  Accepted: 3/5 updates
  Rejected: 2/5 updates
  Client 0: ✗ REJECT (Δloss=+0.2341)  ← Malicious
  Client 1: ✗ REJECT (Δloss=+0.1876)  ← Malicious
  Client 2: ✓ ACCEPT (Δloss=-0.0123)  ← Honest
  Client 3: ✓ ACCEPT (Δloss=-0.0089)  ← Honest
  Client 4: ✓ ACCEPT (Δloss=-0.0145)  ← Honest
```

## Next: Week 4 - Fingerprint Pre-Filtering
Add fast clustering to reduce validation cost.
