# Week 6 Full Defense - Fixes Applied

## Date: November 6, 2025

## Issues Fixed

### 1. Ray Serialization Error: `ModuleNotFoundError: No module named 'client'`

**Problem:**
- Flower's Ray-based simulation couldn't serialize the client function
- Ray workers couldn't find the `client` module when trying to deserialize
- Error: `ModuleNotFoundError: No module named 'client'`

**Root Cause:**
- The `client` module was imported at the top level of `main.py`
- Ray workers can't access top-level imports when deserializing functions
- The `client_fn` function also tried to access global variables (`client_loaders`, `test_loader`, `malicious_clients`)

**Solution:**
1. Moved `from client import create_client` **inside** the `client_fn` function
2. Changed `client_fn` signature from `client_fn(cid: str)` to `client_fn(context: Context)` to match Flower's new API
3. Added robust client ID extraction logic that handles multiple Flower versions
4. Made `malicious_clients` and `num_malicious` global variables initialized in `main()` so they can be accessed by `client_fn`

**Files Modified:**
- `quantum_version/week6_full_defense/main.py`
  - Lines 1-110: Rewrote client_fn with proper Context handling and internal import

---

### 2. KeyError in Defense Summary: `'num_rounds'`

**Problem:**
- After simulation completed, the code tried to access `defense_summary['num_rounds']`
- Actual key returned by server was `defense_summary['total_rounds']`
- Also tried to access other non-existent keys like `total_clients_processed`, `average_precision`, etc.

**Root Cause:**
- Mismatch between `server.py`'s `get_final_results()` return values and `main.py` expectations
- Server returns: `total_rounds`, `total_clients`, `layer0_rejections`, `layer1_rejections`, `layer2_rejections`, `total_accepted`, `rejection_rate`
- Main.py expected: `num_rounds`, `total_clients_processed`, `total_rejected`, `average_precision`, `average_recall`, `perfect_detection_rounds`

**Solution:**
- Updated `main.py` to use the correct keys from server's defense_summary
- Changed summary printing to show 3-layer defense statistics properly

**Files Modified:**
- `quantum_version/week6_full_defense/main.py`
  - Lines 228-239: Fixed defense summary keys and display

---

### 3. Model Dimension Mismatch: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x2 and 4x32)`

**Problem:**
- Quantum circuit output was 2-dimensional (from 2 qubits)
- Classifier expected 4-dimensional input (designed for 4 qubits)
- Error during matrix multiplication: `(64x2) @ (4x32)` is invalid

**Root Cause:**
- `config.py` had `N_QUBITS = 2` and `N_LAYERS = 1` (likely for quick testing)
- `quantum_model.py` has a classifier that expects 4-qubit output: `nn.Linear(n_qubits, 32)`
- When `n_qubits=2`, the classifier tries `Linear(2, 32)` but receives `n_qubits=4` from architecture

**Solution:**
- Changed `config.py` to use correct quantum configuration:
  ```python
  N_QUBITS = 4  # Fixed: Must match quantum_model.py architecture
  N_LAYERS = 4  # Fixed: Match baseline architecture
  ```

**Files Modified:**
- `quantum_version/week6_full_defense/config.py`
  - Line 20: Changed `N_QUBITS = 2` → `N_QUBITS = 4`
  - Line 21: Changed `N_LAYERS = 1` → `N_LAYERS = 4`

---

## Summary of Changes

### Files Modified:
1. **`main.py`**
   - Rewrote `client_fn` with Context API and internal imports
   - Fixed defense summary key access
   - Made malicious_clients global

2. **`config.py`**
   - Fixed N_QUBITS from 2 → 4
   - Fixed N_LAYERS from 1 → 4

### Current Status:
✅ **Ray serialization fixed** - Client function properly serializable
✅ **Defense summary fixed** - Keys match server implementation  
✅ **Model dimensions fixed** - 4 qubits match classifier expectations
🏃 **Experiments running** - Week 6 full defense simulation in progress

### Expected Results:
With 3-layer cascading defense (Norm Filter → Adaptive → Fingerprints):
- Detection Rate: >95%
- False Positive Rate: <5%
- Final Accuracy: 80-90% (vs 10.10% in week2 attack-only)

### Next Steps:
1. Wait for week6 experiments to complete (running in background)
2. Run Table 5 comparison tests: `cd quantum_version/tests/table5_defense_comparison && python run_all_tests.py`
3. Update paper with actual results
4. Remove PQ crypto references from paper (focus on Byzantine defense)
