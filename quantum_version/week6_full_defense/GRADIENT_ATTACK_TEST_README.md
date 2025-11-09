# Week 6 - Gradient Attack Testing Configuration

## Overview
This setup tests the 3-layer defense system against gradient scale attacks in a federated learning scenario.

## Configuration Summary

### Client Setup
- **Total Clients**: 5
- **Honest Clients**: 3 (Clients 0, 2, 4)
- **Malicious Clients**: 2 (Clients 1, 3)

### Attack Configuration
- **Attack Type**: Gradient Ascent (gradient_ascent)
- **Attack Mechanism**: Reverses gradient direction and amplifies by scale factor

#### Malicious Client Details
1. **Client 1 - MODERATE Attack**
   - Scale Factor: 10.0x
   - Description: Medium-intensity gradient scale attack
   - Expected to be caught by: Layer 1 (Adaptive Defense)
   - Formula: `poisoned_update = old_params - 10.0 * (new_params - old_params)`

2. **Client 3 - AGGRESSIVE Attack**
   - Scale Factor: 50.0x
   - Description: High-intensity gradient scale attack
   - Expected to be caught by: Layer 0 (Norm Filter)
   - Formula: `poisoned_update = old_params - 50.0 * (new_params - old_params)`

### Defense Layers

#### Layer 0: Norm Filter
- **Purpose**: Catch updates with abnormally large norms
- **Threshold**: `median_norm × 3.0`
- **Target**: Aggressive attacks (50x scale factor)
- **Mechanism**: Computes L2 norm of each update and rejects outliers

#### Layer 1: Adaptive Statistical Defense
- **Purpose**: Catch statistical outliers
- **Method**: Statistical analysis
- **Threshold**: `mean + 2.0 × std`
- **Target**: Moderate attacks (10x scale factor)
- **Mechanism**: Analyzes update distribution and rejects statistical anomalies

#### Layer 2: Fingerprint Validation
- **Purpose**: Catch behavioral anomalies
- **Dimension**: 512-D fingerprint projection
- **Similarity Threshold**: 0.7
- **Target**: Subtle adaptive attacks
- **Mechanism**: Validates update consistency with historical behavior

## Training Parameters

- **Rounds**: 5
- **Clients per Round**: 5 (all clients participate)
- **Data Distribution**: Non-IID (Dirichlet α=0.5)
- **Batch Size**: 64
- **Local Epochs**: 2
- **Learning Rate**: 0.01
- **Quantum Circuit**: 2 qubits, 1 layer

## Expected Behavior

1. **Round Start**: Server broadcasts global model to all 5 clients
2. **Local Training**: 
   - Honest clients (0, 2, 4) train normally
   - Malicious clients (1, 3) apply gradient attacks with their respective scale factors
3. **Defense Application**:
   - Layer 0 rejects Client 3 (50x attack - high norm)
   - Layer 1 rejects Client 1 (10x attack - statistical outlier)
   - Only honest updates (Clients 0, 2, 4) are aggregated
4. **Model Update**: Server aggregates 3 honest updates using FedAvg
5. **Evaluation**: Server evaluates global model accuracy

## Running the Test

### 1. Verify Configuration
```bash
cd c:\Users\admin\OneDrive\Desktop\capstonev3\mnist_implementation\new_approach\quantum_version\week6_full_defense
python test_gradient_attack_setup.py
```

### 2. Run Full Training
```bash
python main.py
```

## Success Metrics

1. **Defense Effectiveness**:
   - Layer 0 should catch Client 3 (50x attack)
   - Layer 1 should catch Client 1 (10x attack)
   - Both malicious clients should be rejected

2. **Model Performance**:
   - Accuracy should improve over rounds
   - Final accuracy should be reasonable (>80% for MNIST)
   - No catastrophic model degradation despite attacks

3. **Logging**:
   - Clear indication of which clients are rejected
   - Defense layer statistics for each round
   - Per-round accuracy metrics

## File Structure

```
week6_full_defense/
├── config.py                          # Configuration (UPDATED)
├── main.py                            # Main entry point
├── client.py                          # Client implementation
├── server.py                          # Server with 3-layer defense
├── attack.py                          # Gradient attack implementation
├── quantum_model.py                   # Quantum neural network
├── data_loader.py                     # Non-IID data distribution
├── defense_norm_filter.py             # Layer 0 defense
├── defense_adaptive.py                # Layer 1 defense
├── defense_fingerprint_server.py      # Layer 2 defense (server-side)
├── defense_fingerprint_client.py      # Layer 2 defense (client-side)
└── test_gradient_attack_setup.py      # Configuration test script (NEW)
```

## Next Steps

After testing gradient scale attacks:
1. Analyze defense layer effectiveness
2. Check which layer catches which attack
3. Consider testing other attack types if needed
4. Tune defense thresholds if false positives/negatives occur

## Notes

- Random seed is set to 42 for reproducibility
- Client assignments are randomized but deterministic
- Attack scale factors can be adjusted in `config.py`
- Defense thresholds can be tuned if needed
