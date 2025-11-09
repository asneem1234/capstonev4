# Byzantine Attack Strategies: Comprehensive Analysis

## Table of Contents
1. [Threat Model](#threat-model)
2. [Attack Taxonomy](#attack-taxonomy)
3. [Gradient Ascent Attack](#gradient-ascent-attack)
4. [Attack Implementation](#attack-implementation)
5. [Attack Analysis](#attack-analysis)
6. [Comparative Study](#comparative-study)

---

## Threat Model

### Federated Learning Security Model

```
┌─────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING SYSTEM                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐                                   │
│  │   Central Server     │  ← TRUSTED                        │
│  │  (Aggregator)        │    Assumed honest                 │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             │ Model Parameters                               │
│             │                                                │
│    ┌────────┴────────┬──────────┬──────────┐               │
│    ▼                 ▼          ▼          ▼                │
│  ┌────┐           ┌────┐     ┌────┐     ┌────┐            │
│  │ C1 │ ✓ HONEST  │ C2 │ ⚠️   │ C3 │ ✓   │ C4 │ ✓          │
│  └────┘           └────┘     └────┘     └────┘            │
│    │    MALICIOUS   │          │          │                │
│    ↓                ↓          ↓          ↓                │
│  Local            Local      Local      Local               │
│  Data             Data       Data       Data                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Adversary Assumptions

**Byzantine Threat Model**:

1. **Adversary Capability**:
   - Controls a subset of clients (25% in our case)
   - Has full access to local training data
   - Can observe global model parameters
   - Can arbitrarily modify local updates before sending

2. **Adversary Goal**:
   - **Primary**: Maximize global model loss (reduce accuracy)
   - **Secondary**: Avoid detection by defense mechanisms
   - **Tertiary**: Remain persistent across rounds

3. **Adversary Constraints**:
   - Cannot directly modify server aggregation
   - Cannot access other clients' data or models
   - Cannot break cryptographic primitives (if used)
   - Limited to model poisoning attacks

4. **Attack Vector**:
   - Manipulate model updates (gradients/parameters)
   - Train on poisoned data
   - Send crafted malicious updates

### Security Properties at Risk

| Property | Description | Threat |
|----------|-------------|--------|
| **Integrity** | Global model accuracy | Degraded by poisoned updates |
| **Availability** | Model convergence | Prevented by divergent updates |
| **Robustness** | Performance under attack | Compromised without defense |
| **Fairness** | Equal client contribution | Disrupted by malicious clients |

---

## Attack Taxonomy

### Classification of Attacks

```
Model Poisoning Attacks
├── Data Poisoning
│   ├── Label Flipping
│   ├── Backdoor Injection
│   └── Data Corruption
│
└── Gradient/Update Poisoning ← OUR FOCUS
    ├── Gradient Ascent (Implemented)
    ├── Sign Flipping
    ├── Scaled Poisoning
    ├── Random Noise
    └── Targeted Attacks
```

### Attack Types Implemented

#### 1. **Gradient Ascent** (Primary)
**Concept**: Reverse the gradient direction to maximize loss
```
Normal Update:     θnew = θold - η∇L      (minimize loss)
Malicious Update:  θnew = θold + η∇L      (maximize loss)
Amplified:         θnew = θold + 10η∇L    (aggressive)
```

**Effect**: Forces model to learn opposite of correct patterns

#### 2. **Scaled Poisoning** (Alternative)
**Concept**: Train on flipped labels, then amplify update
```
Step 1: Flip labels (0→9, 1→8, 2→7, ...)
Step 2: Train normally on flipped data
Step 3: Amplify update by scale factor
```

**Effect**: Pushes model toward wrong classifications

#### 3. **Random Noise** (Baseline)
**Concept**: Add large random noise to destabilize training
```
Update = Normal Update + scale_factor × random_noise
```

**Effect**: Introduces chaos, prevents convergence

### Threat Severity Levels

| Attack Type | Severity | Detection Difficulty | Impact |
|-------------|----------|---------------------|--------|
| Label Flipping | Low | Easy | Gradual degradation |
| Scaled Poisoning | Medium | Medium | Significant damage |
| **Gradient Ascent** | **High** | **Medium** | **Model collapse** |
| Random Noise | Medium | Easy | Instability |
| Backdoor | High | Hard | Targeted corruption |

---

## Gradient Ascent Attack

### Mathematical Formulation

#### Normal Federated Learning Update

**Client-side**:
```
1. Receive global model: θglobal(t)
2. Train locally: θlocal = θglobal - η∑∇L(θ, xi, yi)
3. Compute update: Δθ = θlocal - θglobal
4. Send update to server: Δθ
```

**Server-side**:
```
5. Aggregate: θglobal(t+1) = θglobal(t) + (1/K)∑Δθk
```

#### Gradient Ascent Attack

**Malicious Client**:
```
1. Receive global model: θglobal(t)
2. Train locally (appears normal): θlocal = θglobal - η∑∇L
3. Compute honest update: Δθhonest = θlocal - θglobal
4. Reverse and amplify: Δθmalicious = -λ × Δθhonest
5. Poisoned parameters: θpoisoned = θglobal + Δθmalicious
                                   = θglobal - λ(θlocal - θglobal)
                                   = θglobal(1 + λ) - λθlocal
6. Send poisoned update: Δθmalicious
```

**Where**:
- λ = scale_factor = 10.0 (amplification)
- Negative sign reverses gradient
- Amplification increases damage

### Visual Representation

```
Parameter Space Visualization:

                  Loss Landscape
                        
      High Loss ←─────────────────→ Low Loss
                        
        ╔═══════════════════════════════════╗
        ║                                   ║
        ║    Honest                         ║
        ║    Update ──→  θglobal ─→ θhonest ║
        ║                   │               ║
        ║                   │               ║
        ║                   ↓               ║
        ║                 Malicious         ║
        ║                   Update          ║
        ║                   (10×)           ║
        ║                   ↓               ║
        ║    θmalicious ←───┘               ║
        ║    (High Loss)                    ║
        ║                                   ║
        ╚═══════════════════════════════════╝

Distance from θglobal:
- Honest:    ||Δθhonest||    ≈ 0.1  (small, productive)
- Malicious: ||Δθmalicious|| ≈ 1.1  (large, destructive)

Ratio: 1.1 / 0.1 = 11× larger norm
```

### Attack Effectiveness

**Why Gradient Ascent Works**:

1. **Direct Opposition**: Moves model in exactly wrong direction
2. **Amplification**: 10× multiplier creates large, harmful updates
3. **Immediate Impact**: Affects model in single round
4. **Cumulative Damage**: Each round compounds the effect

**Expected Behavior**:

| Round | No Attack (Accuracy) | With Attack (Accuracy) |
|-------|---------------------|------------------------|
| 0 | 10% (random) | 10% (random) |
| 1 | 65% ↗ | 15% ↘ |
| 2 | 80% ↗ | 12% ↘ |
| 3 | 85% ↗ | 11% ↘ |
| 5 | 88% ↗ | 10% ↘ (collapse) |

### Attack Parameters

```python
# Attack Configuration
ATTACK_ENABLED = True
ATTACK_TYPE = "gradient_ascent"
SCALE_FACTOR = 10.0              # Amplification (10×)
MALICIOUS_PERCENTAGE = 0.25      # 25% of clients (1 out of 4)

# Attack Behavior
- Malicious clients: [0]         # Client IDs
- Attack every round: Yes
- Persistent: Yes (unless detected)
```

---

## Attack Implementation

### Code Structure

```python
class ModelPoisoningAttack:
    """
    Implements gradient ascent attack
    """
    
    def __init__(self, 
                 num_classes=10,
                 attack_type='gradient_ascent',
                 scale_factor=10.0):
        self.attack_type = attack_type
        self.scale_factor = scale_factor
    
    def poison_update(self, old_params, new_params):
        """
        Transform honest update into malicious update
        
        Args:
            old_params: θglobal (before training)
            new_params: θlocal (after training)
        
        Returns:
            poisoned_params: θmalicious
        """
        if self.attack_type == 'gradient_ascent':
            # Compute honest update
            # Δθ = θlocal - θglobal
            
            # Reverse and amplify
            # θmalicious = θglobal - scale_factor × Δθ
            #            = θglobal - scale_factor × (θlocal - θglobal)
            #            = θglobal(1 + scale_factor) - scale_factor × θlocal
            
            poisoned = []
            for old, new in zip(old_params, new_params):
                update = new - old
                poisoned_param = old - self.scale_factor * update
                poisoned.append(poisoned_param)
            
            return poisoned
        
        # Other attack types...
```

### Client Integration

```python
class QuantumFlowerClient(NumPyClient):
    def __init__(self, ..., is_malicious=False):
        self.is_malicious = is_malicious
        
        if is_malicious:
            self.attack = ModelPoisoningAttack(
                attack_type='gradient_ascent',
                scale_factor=10.0
            )
    
    def fit(self, parameters, config):
        # Store old parameters
        old_parameters = [p.copy() for p in parameters]
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Train locally (honestly)
        for epoch in range(self.config.LOCAL_EPOCHS):
            for batch_x, batch_y in self.train_loader:
                # Normal training...
        
        # Get new parameters
        new_parameters = self.get_parameters()
        
        # ATTACK: Poison if malicious
        if self.is_malicious and self.attack:
            poisoned_parameters = self.attack.poison_update(
                old_parameters, 
                new_parameters
            )
            
            # Override with poisoned parameters
            new_parameters = poisoned_parameters
        
        # Return (possibly poisoned) parameters
        return new_parameters, num_samples, metrics
```

### Attack Metrics

**Tracked Metrics**:
```python
metrics = {
    'client_id': self.client_id,
    'is_malicious': self.is_malicious,
    'loss': avg_loss,
    'accuracy': accuracy,
    'update_norm': update_norm,  # ||Δθ||
}
```

**Update Norm Calculation**:
```python
def calculate_update_norm(old_params, new_params):
    """
    Compute L2 norm of update: ||Δθ||₂
    """
    norm = 0.0
    for old, new in zip(old_params, new_params):
        delta = new - old
        norm += np.sum(delta ** 2)
    
    return np.sqrt(norm)
```

**Expected Norms**:
- Honest client: ||Δθ|| ≈ 0.05 - 0.15
- Malicious client: ||Δθ|| ≈ 0.5 - 1.5 (10× larger)

---

## Attack Analysis

### Effectiveness Metrics

#### 1. **Impact on Global Model**

**Accuracy Degradation**:
```
Accuracy Drop = Accuracy(No Attack) - Accuracy(With Attack)

Expected:
- Round 5: 88% → 10% = 78% drop
- Final: Complete model collapse
```

**Loss Increase**:
```
Loss Increase = Loss(With Attack) - Loss(No Attack)

Expected:
- Normal: Loss ≈ 0.3 - 0.5
- Attack: Loss ≈ 2.0 - 2.5 (cross-entropy ceiling)
```

#### 2. **Attack Success Rate**

**Success Criteria**:
- ✓ Success: Global accuracy < 20% (worse than random with defense)
- ⚠ Partial: Global accuracy 20-50%
- ✗ Failure: Global accuracy > 50%

**Factors Affecting Success**:
1. **Malicious Percentage**: More malicious clients → stronger attack
2. **Scale Factor**: Larger amplification → more damage
3. **Data Heterogeneity**: Non-IID data → easier to disrupt
4. **Defense Presence**: Defense mechanisms → reduced effectiveness

#### 3. **Stealth Analysis**

**Detectability**:
```
Detection Difficulty = Low (Easy to detect)

Reasons:
1. Large update norm (10-20× normal)
2. Consistent pattern across rounds
3. Opposite direction from honest clients
4. Causes visible accuracy drop
```

**Stealth Score**: 2/10 (highly detectable)

### Update Distribution Analysis

**Visual Comparison**:

```
Update Norm Distribution

Honest Clients:          Malicious Clients:
                         
Frequency                Frequency
    │                        │
    │  ███                   │
    │ █████                  │
    │███████                 │              ███
    │████████                │             █████
────┴────────────→       ────┴──────────────────→
    0.1  0.2              0.5          1.0  1.5

Mean: 0.12               Mean: 1.1
Std:  0.03               Std:  0.2
```

**Statistical Separation**:
```
Cohen's d = (μmalicious - μhonest) / σpooled
          = (1.1 - 0.12) / 0.15
          ≈ 6.5 (very large effect size)

Interpretation: Highly separable distributions
→ Easy to detect with norm-based filtering
```

### Attack Traces

**Round-by-Round Analysis**:

```
Round 1:
  Client 0 (Malicious): Norm = 1.234, Loss = 0.156 ↑, Acc = 12.3% ↓
  Client 1 (Honest):    Norm = 0.098, Loss = 0.345 ↓, Acc = 65.2% ↑
  Client 2 (Honest):    Norm = 0.112, Loss = 0.332 ↓, Acc = 67.1% ↑
  Client 3 (Honest):    Norm = 0.105, Loss = 0.341 ↓, Acc = 66.5% ↑
  
  Aggregated (No Defense):
    Global: Loss = 0.543, Acc = 52.8% (damaged by Client 0)

Round 2:
  Client 0 (Malicious): Norm = 1.456, Loss = 0.123 ↑, Acc = 11.2% ↓
  [Similar pattern...]
  
  Aggregated (No Defense):
    Global: Loss = 1.234, Acc = 28.3% (further damaged)

Round 3:
  ...continues until collapse...
```

---

## Comparative Study

### Attack Comparison

| Attack Type | Norm Ratio | Stealth | Effectiveness | Implementation |
|-------------|-----------|---------|---------------|----------------|
| **Gradient Ascent** | **10-20×** | **Low** | **Very High** | **Simple** |
| Scaled Poisoning | 5-10× | Medium | High | Moderate |
| Label Flipping | 2-3× | High | Medium | Simple |
| Random Noise | 3-5× | Low | Medium | Simple |
| Sign Flipping | 1-2× | Very High | Low | Simple |
| Backdoor | 1× | Very High | High (targeted) | Complex |

### Scale Factor Sensitivity

**Effect of Different Scale Factors**:

| Scale Factor | Attack Strength | Detection Difficulty | Final Accuracy |
|--------------|----------------|---------------------|----------------|
| 1.0 | Weak | Hard | 70-75% |
| 2.0 | Weak | Hard | 60-65% |
| 5.0 | Medium | Medium | 30-40% |
| **10.0** | **Strong** | **Easy** | **10-15%** |
| 20.0 | Very Strong | Very Easy | 10-12% |
| 50.0 | Extreme | Trivial | 10% |

**Optimal Choice**: Scale factor = 10.0
- Strong enough to collapse model
- Not so extreme as to be immediately obvious
- Represents realistic worst-case scenario

### Defense Bypass Attempts

**Potential Evasion Strategies** (not implemented, for research):

1. **Gradual Scaling**: Start with low scale, increase over rounds
   ```python
   scale_factor = min(2.0 + round * 0.5, 10.0)
   ```

2. **Probabilistic Attack**: Attack only with probability p < 1
   ```python
   if random.random() < 0.7:  # 70% of rounds
       apply_attack()
   ```

3. **Adaptive Magnitude**: Match norm to honest clients
   ```python
   target_norm = median(honest_norms) * 1.5
   scale_factor = target_norm / honest_norm
   ```

4. **Round Robin**: Different malicious clients activate in different rounds
   ```python
   if round % num_malicious == client_id:
       apply_attack()
   ```

**Research Note**: These evasion techniques would be interesting future work to test defense robustness.

---

## Experimental Validation

### Validation Metrics

**Primary Metrics**:
1. **Attack Success Rate**: % of rounds where accuracy < 20%
2. **Accuracy Degradation**: Final accuracy with vs without attack
3. **Convergence Disruption**: Number of rounds until collapse

**Secondary Metrics**:
1. **Update Norm Ratio**: Malicious/Honest norm ratio
2. **Loss Trajectory**: Loss over rounds
3. **Parameter Drift**: Distance from baseline model

### Expected Results (Week 2 - No Defense)

```
Configuration:
- Clients: 4 total, 1 malicious (25%)
- Rounds: 5
- Attack: Gradient ascent, scale_factor=10.0

Expected Outcome:

Round | Honest Accuracy | Malicious Accuracy | Global Accuracy | Global Loss
------|-----------------|--------------------|-----------------|-----------
  0   |      10%       |        10%         |       10%       |    2.30
  1   |      65%       |        12%         |       52%       |    1.20
  2   |      75%       |        11%         |       35%       |    1.65
  3   |      80%       |        10%         |       22%       |    1.95
  4   |      83%       |        10%         |       15%       |    2.15
  5   |      85%       |        10%         |       11%       |    2.25

Final Result: MODEL COLLAPSE (accuracy ≈ random guessing)
```

### Real-World Implications

**Why This Matters**:

1. **Shows Vulnerability**: FL is fragile without defense
2. **Motivates Defense**: Even 25% malicious clients can destroy model
3. **Realistic Scenario**: Gradient ascent is simple to implement
4. **Generalizable**: Attack works across architectures (quantum/classical)

**Limitations**:

1. **Obvious Attack**: Real adversaries might be stealthier
2. **Fixed Strategy**: No adaptive behavior
3. **No Coordination**: Malicious clients don't collaborate
4. **Perfect Knowledge**: Assumes attacker knows aggregation

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Status**: Research Documentation
