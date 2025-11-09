# Quantum Federated Learning - Week 2 Attack Results

## Configuration

| Parameter | Value |
|-----------|-------|
| **Clients** | 5 total, 5 per round |
| **Rounds** | 10 (stopped at 5) |
| **Data Distribution** | Non-IID (Dirichlet α=0.5) |
| **Batch Size** | 64 |
| **Local Epochs** | 2 |
| **Learning Rate** | 0.001 |
| **Gradient Clipping** | 1.0 |
| **Quantum Qubits** | 4 |
| **Quantum Layers** | 4 |
| **Framework** | Flower (flwr) |
| **Quantum Backend** | PennyLane (default.qubit) |
| **Device** | CPU |

## Attack Configuration

| Parameter | Value |
|-----------|-------|
| **Attack Enabled** | ✅ YES |
| **Attack Type** | Gradient Ascent |
| **Scale Factor** | 50.0 (very aggressive) |
| **Malicious Clients** | 2/5 (40%) |
| **Malicious Client IDs** | [0, 1] |
| **Honest Client IDs** | [2, 3, 4] |
| **Defense Enabled** | ❌ NO (all updates aggregated) |

## Attack Mechanism

**Gradient Ascent Attack**: Reverses gradient direction to maximize loss instead of minimizing it.

```
poisoned_update = old_params - scale_factor × (new_params - old_params)
                = old_params - 50.0 × gradient_update
```

This attack:
- Pushes model parameters away from the optimum
- Creates very large update norms (50× normal)
- Corrupts the global model when aggregated with honest updates

## Data Distribution

| Client | Samples | Dominant Class | Samples in Dominant Class | Status |
|--------|---------|----------------|---------------------------|---------|
| Client 0 | 13,910 | 6 | 4,786 (34.4%) | ⚠️ **MALICIOUS** |
| Client 1 | 6,144 | 7 | 2,361 (38.4%) | ⚠️ **MALICIOUS** |
| Client 2 | 14,834 | 3 | 4,530 (30.5%) | ✅ Honest |
| Client 3 | 14,432 | 4 | 5,046 (35.0%) | ✅ Honest |
| Client 4 | 10,680 | 1 | 4,270 (40.0%) | ✅ Honest |
| **Total** | **60,000** | - | - | - |

## Training Results

### Round-by-Round Performance

| Round | Test Accuracy | Test Loss | Change from Previous |
|-------|---------------|-----------|---------------------|
| 1 | 10.10% | 5.2228 | - |
| 2 | 11.35% | 14.0248 | +1.25% acc, loss ×2.7 |
| 3 | 11.35% | 101.8411 | +0.00% acc, loss ×7.3 |
| 4 | 9.82% | 658.6760 | -1.53% acc, loss ×6.5 |
| 5 | 10.10% | 1109.9002 | +0.28% acc, loss ×1.7 |

### Client Training Results by Round

#### Round 1
| Client | Loss | Accuracy | Samples | Update Norm | Status | Attack Impact |
|--------|------|----------|---------|-------------|---------|--------------|
| Client 0 | 1.1477 | 59.15% | 13,910 | **248.4933** | ⚠️ Malicious | **48× normal** |
| Client 1 | 1.3270 | 61.37% | 6,144 | **156.1249** | ⚠️ Malicious | **50× normal** |
| Client 2 | 1.1305 | 64.80% | 14,834 | 5.8697 | ✅ Honest | Normal |
| Client 3 | 1.1629 | 56.54% | 14,432 | 5.1558 | ✅ Honest | Normal |
| Client 4 | 1.1892 | 64.88% | 10,680 | 4.7997 | ✅ Honest | Normal |

**Observation**: Malicious clients have update norms **30-50× larger** than honest clients!

#### Round 2
| Client | Loss | Accuracy | Samples | Update Norm | Status |
|--------|------|----------|---------|-------------|---------|
| Client 0 | 3.3575 | 7.43% | 13,910 | **157.2087** | ⚠️ Malicious |
| Client 1 | 3.6127 | 10.85% | 6,144 | **84.0022** | ⚠️ Malicious |
| Client 2 | 2.1917 | 30.41% | 14,834 | 2.1731 | ✅ Honest |
| Client 3 | 2.2755 | 33.37% | 14,432 | 3.0235 | ✅ Honest |
| Client 4 | 2.1186 | 36.93% | 10,680 | 1.8535 | ✅ Honest |

**Observation**: Honest clients' accuracy dropping due to poisoned global model. Malicious clients showing very low accuracy (correctly failing).

#### Round 3
| Client | Loss | Accuracy | Samples | Update Norm | Status |
|--------|------|----------|---------|-------------|---------|
| Client 0 | 14.4015 | 0.06% | 13,910 | **100.3953** | ⚠️ Malicious |
| Client 1 | 15.4086 | 1.79% | 6,144 | **57.7298** | ⚠️ Malicious |
| Client 2 | 5.3567 | 26.37% | 14,834 | 1.6362 | ✅ Honest |
| Client 3 | 5.5236 | 29.55% | 14,432 | 3.2269 | ✅ Honest |
| Client 4 | 5.6703 | 39.98% | 10,680 | 1.3862 | ✅ Honest |

**Observation**: Loss exploding! Malicious clients near 0% accuracy (attack working perfectly).

#### Round 4
| Client | Loss | Accuracy | Samples | Update Norm | Status |
|--------|------|----------|---------|-------------|---------|
| Client 0 | 96.0565 | 0.06% | 13,910 | **227.7604** | ⚠️ Malicious |
| Client 1 | 116.3064 | 1.67% | 6,144 | **118.4899** | ⚠️ Malicious |
| Client 2 | 39.0196 | 22.96% | 14,834 | 4.6159 | ✅ Honest |
| Client 3 | 32.7625 | 26.81% | 14,432 | 4.3371 | ✅ Honest |
| Client 4 | 45.3878 | 37.24% | 10,680 | 3.7247 | ✅ Honest |

**Observation**: Catastrophic model failure. Even honest clients struggling with poisoned model.

#### Round 5
| Client | Loss | Accuracy | Samples | Update Norm | Status |
|--------|------|----------|---------|-------------|---------|
| Client 0 | 280.8978 | 0.41% | 13,910 | **205.4394** | ⚠️ Malicious |
| Client 1 | 513.7954 | 0.33% | 6,144 | **103.2446** | ⚠️ Malicious |
| Client 2 | 117.0473 | 24.53% | 14,834 | 3.9576 | ✅ Honest |
| Client 3 | 119.5568 | 32.20% | 14,432 | 3.7442 | ✅ Honest |
| Client 4 | 165.7700 | 26.64% | 10,680 | 3.5588 | ✅ Honest |

**Observation**: Total model collapse. Losses in hundreds, accuracy near random (10%).

## Attack Effectiveness Analysis

### 1. **Update Norm Signature**

**Clear Separation Between Malicious and Honest Clients:**

| Round | Malicious Avg Norm | Honest Avg Norm | Ratio |
|-------|-------------------|-----------------|-------|
| 1 | 202.31 | 5.28 | **38.3×** |
| 2 | 120.61 | 2.34 | **51.5×** |
| 3 | 79.06 | 2.07 | **38.2×** |
| 4 | 173.12 | 4.23 | **40.9×** |
| 5 | 154.34 | 3.74 | **41.3×** |

**Finding**: Malicious clients have consistently **38-51× larger** update norms than honest clients!

### 2. **Global Model Degradation**

| Metric | Week 1 (No Attack) | Week 2 (Attack) | Degradation |
|--------|-------------------|-----------------|-------------|
| **Round 1 Accuracy** | 10.62% | 10.10% | -0.52% |
| **Round 5 Accuracy** | **90.78%** | **10.10%** | **-80.68%** ❌ |
| **Round 5 Loss** | **0.3353** | **1109.9002** | **×3309** ❌ |
| **Convergence** | Smooth ✅ | Failed ❌ | - |

**Key Finding**: Attack prevents learning completely! Model stuck at random guess level (~10%).

### 3. **Honest vs Malicious Client Behavior**

**Honest Clients (2, 3, 4):**
- Initial accuracy: 56-65% (learning from data)
- Final accuracy: 24-32% (degraded by poisoned global model)
- Update norms: 1.4-5.9 (stable, reasonable)
- Loss: 1.1-165 (increasing due to poisoned model)

**Malicious Clients (0, 1):**
- Initial accuracy: 59-61% (before attack applied)
- Final accuracy: 0.33-0.41% (attack destroying model)
- Update norms: 57-248 (extreme, 50× normal)
- Loss: 1.1-514 (exploding)

### 4. **Loss Explosion Pattern**

```
Round 1:     5.22
Round 2:    14.02  (×2.7)
Round 3:   101.84  (×7.3)
Round 4:   658.68  (×6.5)
Round 5: 1,109.90  (×1.7)
```

Loss grows **exponentially** due to gradient ascent attack!

## Attack Success Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Model Destruction** | Accuracy stuck at 10% | ✅ **Complete Success** |
| **Detectability** | 38-51× norm difference | ✅ **Highly Detectable** |
| **Honest Client Impact** | 60% accuracy loss | ✅ **Severe Damage** |
| **Loss Explosion** | ×3309 increase | ✅ **Catastrophic** |
| **Defense Bypass** | No defense present | N/A |

## Comparison: Week 1 (No Attack) vs Week 2 (Attack)

### Global Model Performance

| Round | Week 1 Accuracy | Week 2 Accuracy | Difference |
|-------|----------------|-----------------|------------|
| 1 | 10.62% | 10.10% | -0.52% |
| 2 | 48.65% | 11.35% | **-37.30%** |
| 3 | 74.33% | 11.35% | **-62.98%** |
| 4 | 85.82% | 9.82% | **-76.00%** |
| 5 | 90.78% | 10.10% | **-80.68%** |

### Visual Comparison

```
Week 1 (No Attack): 10.62% → 48.65% → 74.33% → 85.82% → 90.78% ✅
Week 2 (Attack):    10.10% → 11.35% → 11.35% →  9.82% → 10.10% ❌
```

## Key Findings

### 1. **Attack is Devastating Without Defense**
- Model completely fails to learn
- Accuracy remains at random guess level (10%)
- Loss explodes to 1,100+ (vs 0.34 baseline)

### 2. **Attack Signature is Obvious**
- Malicious client update norms are **38-51× larger** than honest clients
- This makes detection trivial with norm-based filtering
- Defense mechanisms in Week 6 should easily catch this

### 3. **Gradient Ascent is Extremely Effective**
- Scale factor of 50.0 completely destroys the model
- Even 60% honest clients cannot overcome 40% malicious
- FedAvg aggregation amplifies the attack

### 4. **Honest Clients Suffer Collateral Damage**
- Honest clients' accuracy degrades from 65% to 25%
- Poisoned global model makes local training ineffective
- Creates negative feedback loop

## Defense Requirements (for Week 6)

Based on these results, Week 6 defense must:

1. **Detect Anomalous Update Norms**
   - Filter updates with norms >3× median
   - Clear 38-51× separation makes this easy

2. **Validate Update Direction**
   - Check if updates actually improve validation loss
   - Reject updates that increase loss

3. **Use Robust Aggregation**
   - Median, Krum, or Trimmed Mean instead of FedAvg
   - Reduce influence of outlier updates

4. **Add Cryptographic Verification**
   - Post-quantum signatures (Dilithium2)
   - Client fingerprinting to track malicious behavior

## Conclusions

### Attack Success: ✅ **Complete**

The gradient ascent attack with scale=50.0 and 40% malicious clients:
- ✅ Completely prevents model learning
- ✅ Reduces final accuracy by 80.68 percentage points
- ✅ Increases final loss by 3,309×
- ✅ Creates obvious detection signature (38-51× norm difference)
- ✅ Demonstrates critical need for Byzantine defenses

### Next Steps: Week 6 - Full Defense

Week 6 will implement:
1. Norm-based filtering (median absolute deviation)
2. Spectral gradient analysis
3. Post-quantum cryptography (Kyber512 + Dilithium2)
4. Client fingerprinting
5. Adaptive entropy-based thresholding

**Expected Result**: Restore model accuracy to 85-90% despite 40% malicious clients.

---

**Date:** November 6, 2025  
**Status:** ✅ Attack Successfully Demonstrated  
**Next:** Week 6 - Full Defense Implementation
