# Gradient Attack Test Setup - Summary

## ✅ Configuration Complete

Your Week 6 full defense implementation is now configured to test **gradient scale attacks** with the following setup:

### 🎯 Test Scenario

**Total Clients**: 5
- **3 Honest Clients** (IDs: 0, 2, 4)
- **2 Malicious Clients** (IDs: 1, 3)

### ⚔️ Attack Configuration

Both malicious clients use **Gradient Ascent Attack** with different intensities:

#### Client 1 - MODERATE Attack
- **Scale Factor**: 10.0x
- **Mechanism**: Reverses gradient and amplifies by 10x
- **Target Defense Layer**: Layer 1 (Adaptive Statistical Defense)
- **Formula**: `poisoned_params = old_params - 10.0 × (new_params - old_params)`

#### Client 3 - AGGRESSIVE Attack  
- **Scale Factor**: 50.0x
- **Mechanism**: Reverses gradient and amplifies by 50x
- **Target Defense Layer**: Layer 0 (Norm Filter)
- **Formula**: `poisoned_params = old_params - 50.0 × (new_params - old_params)`

### 🛡️ Defense Layers (All Active)

1. **Layer 0 - Norm Filter**
   - Catches updates with abnormally high L2 norms
   - Threshold: median_norm × 3.0
   - Will catch: Client 3 (50x attack)

2. **Layer 1 - Adaptive Statistical Defense**
   - Catches statistical outliers in update distribution
   - Threshold: mean + 2.0×std  
   - Will catch: Client 1 (10x attack)

3. **Layer 2 - Fingerprint Validation**
   - Catches behavioral anomalies
   - 512-D fingerprint projection
   - Similarity threshold: 0.7

### 📋 Training Parameters

- **Rounds**: 5
- **Clients per Round**: 5 (all participate)
- **Data**: Non-IID (Dirichlet α=0.5)
- **Batch Size**: 64
- **Local Epochs**: 2
- **Learning Rate**: 0.01
- **Quantum**: 2 qubits, 1 layer

## 🚀 How to Run

### Step 1: Verify Configuration
```powershell
python test_gradient_attack_setup.py
```
This shows which clients are malicious and their attack parameters.

### Step 2: Run Pre-Flight Check
```powershell
python quick_check.py
```
Verifies all dependencies and settings are correct.

### Step 3: Run Full Training
```powershell
python main.py
```
Runs the full federated learning with attacks and defense.

## 📊 Expected Results

### What Should Happen:

1. **Round Execution**:
   - All 5 clients receive global model
   - Clients 0, 2, 4 train honestly
   - Clients 1 and 3 apply gradient attacks
   - All clients send updates to server

2. **Defense Application**:
   - Layer 0 rejects Client 3 (norm too high)
   - Layer 1 rejects Client 1 (statistical outlier)
   - Only Clients 0, 2, 4 updates are aggregated

3. **Model Performance**:
   - Accuracy should improve over rounds
   - No catastrophic degradation despite attacks
   - Final accuracy should be reasonable (>80%)

### What to Look For in Output:

```
==========================================================
ROUND 1 - CASCADING DEFENSE REPORT
==========================================================
Layer 0 (Norm Filter):
  ✓ Accepted: 4 updates
  ✗ Rejected: 1 update (Client 3)
  
Layer 1 (Adaptive Defense):  
  ✓ Accepted: 3 updates
  ✗ Rejected: 1 update (Client 1)
  
Layer 2 (Fingerprint):
  ✓ Accepted: 3 updates
  ✗ Rejected: 0 updates
  
Final aggregation: 3 honest updates
Round 1 Accuracy: 85.32%
==========================================================
```

## 📁 Files Created/Modified

### Modified:
- ✏️ `config.py` - Updated attack configuration for 2 gradient attacks

### Created:
- ✨ `test_gradient_attack_setup.py` - Configuration verification script
- ✨ `quick_check.py` - Pre-flight check script  
- ✨ `GRADIENT_ATTACK_TEST_README.md` - Detailed documentation
- ✨ `SETUP_SUMMARY.md` - This file

## 🔍 Key Differences from Previous Config

**Before**: Mixed attack types (gradient ascent, scaled poison, etc.)
**Now**: Only gradient scale attacks with 2 intensities (10x and 50x)

This focused approach lets you:
- Test gradient attacks specifically
- Validate defense layer effectiveness
- Understand which layer catches which intensity

## 📝 Notes

- All clients use the same attack type (`gradient_ascent`)
- Only the scale factor differs (10x vs 50x)
- Client IDs are randomly assigned but deterministic (seed=42)
- You can adjust scale factors in `config.py` if needed
- Defense thresholds can be tuned based on results

## ⚠️ Troubleshooting

If defense doesn't catch attacks:
1. Check defense thresholds in `config.py`
2. Verify `DEFENSE_ENABLED = True`
3. Ensure all defense layers are active
4. Consider adjusting threshold multipliers

If training is too slow:
1. Reduce `NUM_ROUNDS` (currently 5)
2. Reduce `LOCAL_EPOCHS` (currently 2)
3. Use fewer quantum layers/qubits

## 🎓 Next Steps

After running this test:
1. ✅ Analyze defense effectiveness per layer
2. ✅ Check rejection rates for each client
3. ✅ Compare accuracy with/without defense
4. ✅ Consider testing other attack types if needed
5. ✅ Tune defense parameters based on results

---

**Ready to test?** Run `python main.py` to start! 🚀
