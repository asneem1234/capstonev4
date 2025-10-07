# Week 2: Label Flipping Attack

## Goal
Show that the system is **vulnerable** to Byzantine attacks when malicious clients flip labels.

**Expected result:** Accuracy drops significantly (from ~85% to ~40-50%)

## What's New

### Files Added:
- `attack.py` - Simple label flipping attack (0→9, 1→8, 2→7, etc.)

### Files Modified:
- `config.py` - Attack enabled, 2/5 clients are malicious
- `client.py` - Clients apply attack if malicious
- `main.py` - Shows which clients are malicious in logs

## How It Works

**Label Flipping:**
- Malicious clients flip all labels: `0→9, 1→8, 2→7, 3→6, 4→5, 5→4, 6→3, 7→2, 8→1, 9→0`
- This creates poisoned gradients that degrade the global model
- Honest clients (2, 3, 4) still train correctly
- But the server aggregates ALL updates equally (no defense yet)

## Run It

```bash
cd c:\Users\admin\OneDrive\Desktop\capstonev3\mnist_implementation\new_approach\week2
python main.py
```

## What to Observe

1. **Malicious clients** will have different training patterns
2. **Global accuracy** should DROP significantly each round
3. **Final accuracy** should be much worse than Week 1 (~40-50% vs ~85%)

This proves the attack works and motivates Week 3 (defenses)!

## Next: Week 3 - Validation Defense
Add a simple defense that validates client updates before aggregation.
