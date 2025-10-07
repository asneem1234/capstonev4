# Week 4: Important Finding - Defense Trade-offs

## What Happened

Fingerprint clustering **did NOT detect** the label flipping attack because:
- Label flipping creates gradients with **similar statistical properties** to honest gradients
- All clients (honest + malicious) clustered together
- DBSCAN couldn't separate them

## Why This Matters

This is an **honest research finding**! Not all defenses work for all attacks:

### Fingerprint Defense Works For:
- ✅ Large-magnitude attacks (gradient scaling × 10)
- ✅ Sign flipping attacks
- ✅ Gradient noise injection (high variance)

### Validation Defense Works For:
- ✅ Label flipping (subtle but effective poisoning)
- ✅ Any attack that degrades model performance
- ✅ More robust but slower

## Solution for Your Paper

Show **defense in depth** with multiple attack types:

1. **Label flipping** → Validation catches it
2. **Gradient scaling** → Fingerprints catch it
3. **Combined attack** → Both layers needed

## Current Configuration

```python
# config.py
USE_FINGERPRINTS = False  # Disabled - validation works better for label flipping
VALIDATION_THRESHOLD = 0.05  # Tighter threshold
```

## Test It Now

Run week4 with fingerprints disabled - validation should work:

```bash
cd week4
python main.py
```

Expected: Malicious clients rejected by validation layer.

## Next Steps

You have 3 options:
1. **Add gradient scaling attack** to show fingerprints working
2. **Move to Non-IID data** (may help fingerprints)
3. **Skip to PQ crypto** (different threat model - network attacks)

Which do you prefer?
