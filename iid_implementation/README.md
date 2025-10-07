# Federated Learning with Three-Layer Byzantine Defense (IID Implementation)

This folder contains a complete incremental implementation of a Byzantine-resilient federated learning system with post-quantum cryptography.

## 📁 Project Structure

```
iid_implementation/
├── week1_baseline/          # Basic FedAvg (no attacks, no defenses)
├── week2_attack/            # Label flipping attack added
├── week3_validation/        # Validation-based defense
├── week4_fingerprints_server/ # Server-side fingerprint clustering
├── week5_pq_crypto/         # Post-quantum cryptography layer
└── week6_fingerprints_client/ # Client-side fingerprints (BEST VERSION)
```

## 🎯 Week-by-Week Evolution

### **Week 1: Baseline FedAvg**
- **Goal**: Establish baseline performance
- **Setup**: 5 clients, IID MNIST, 3 local epochs, 5 rounds
- **Results**: ~85-90% test accuracy
- **Files**: `main.py`, `model.py`, `data_loader.py`, `client.py`, `server.py`, `config.py`

**Run:**
```bash
cd week1_baseline
python main.py
```

---

### **Week 2: Label Flipping Attack**
- **Goal**: Demonstrate vulnerability to Byzantine attacks
- **Attack**: 2/5 clients flip labels (0↔9, 1↔8, 2↔7, etc.)
- **Results**: Accuracy drops to ~40-50%
- **New Files**: `attack.py`

**Run:**
```bash
cd week2_attack
python main.py
```

---

### **Week 3: Validation Defense**
- **Goal**: Filter malicious updates using held-out validation set
- **Defense**: Test each update on 1000-sample validation set
- **Threshold**: Reject if validation loss increases by >0.1
- **Results**: Accuracy recovers to ~70-80%
- **New Files**: `defense_validation.py`

**Run:**
```bash
cd week3_validation
python main.py
```

---

### **Week 4: Server-Side Fingerprint Clustering**
- **Goal**: Fast pre-filtering using gradient fingerprints
- **Method**: 
  - Random projection: 225K parameters → 512D
  - L2 normalization for unit vectors
  - Cosine similarity clustering (threshold=0.7)
- **Two-layer defense**: Fingerprints (fast) → Validation (expensive)
- **Results**: Similar to Week 3, but faster
- **New Files**: `defense_fingerprint.py`

**Run:**
```bash
cd week4_fingerprints_server
python main.py
```

---

### **Week 5: Post-Quantum Cryptography**
- **Goal**: Add network-level security against quantum attacks
- **Algorithms**:
  - **Kyber512**: Key encapsulation for encryption
  - **Dilithium2**: Digital signatures for authentication
- **Mode**: Simulated (academically valid for papers)
- **Three-layer defense**: PQ Crypto → Fingerprints → Validation
- **New Files**: `pq_crypto.py`, `requirements.txt`

**Run:**
```bash
cd week5_pq_crypto
pip install -r requirements.txt
python main.py
```

---

### **Week 6: Client-Side Fingerprints (⭐ BEST VERSION)**
- **Goal**: Move fingerprint computation to client-side for integrity
- **Key Improvements**:
  1. **Client computes** fingerprint of their update
  2. **Server verifies** fingerprint matches decrypted update (integrity check)
  3. **Metadata enhancement**: Loss + accuracy features (50% weight)
  4. **Stricter threshold**: 0.90 cosine similarity (26° angle)
- **Complete three-layer defense**:
  - **Layer 1**: PQ crypto (Kyber512 + Dilithium2)
  - **Layer 2a**: Fingerprint integrity verification
  - **Layer 2b**: Clustering with metadata (loss/accuracy)
  - **Layer 3**: Validation filtering on outliers
- **Results**: ~97% accuracy, malicious clients consistently rejected
- **New Files**: `defense_fingerprint_client.py`

**Run:**
```bash
cd week6_fingerprints_client
python main.py
```

---

## 📊 Performance Comparison

| Week | Defense Mechanism | Test Accuracy | Malicious Detected |
|------|------------------|---------------|-------------------|
| Week 1 | None (baseline) | ~85-90% | N/A |
| Week 2 | None (attack) | ~40-50% | No ❌ |
| Week 3 | Validation only | ~70-80% | Partial ⚠️ |
| Week 4 | Fingerprints + Validation (server-side) | ~70-80% | Partial ⚠️ |
| Week 5 | PQ Crypto + Fingerprints + Validation | ~70-80% | Partial ⚠️ |
| Week 6 | **Client-side Fingerprints + Metadata** | **~97%** | **Yes ✅** |

---

## 🔬 Technical Details

### Attack Configuration
- **Type**: Label flipping (bidirectional)
- **Malicious clients**: 2 out of 5 (40%)
- **Mapping**: {0↔9, 1↔8, 2↔7, 3↔6, 4↔5}

### Model Architecture
```python
SimpleCNN(
  Conv2d(1, 32, kernel_size=3) → ReLU → MaxPool2d(2)
  Conv2d(32, 64, kernel_size=3) → ReLU → MaxPool2d(2)
  Flatten → Linear(1600, 128) → ReLU
  Linear(128, 10)
)
Total parameters: ~225,000
```

### Training Setup
- **Dataset**: MNIST (60K train, 10K test)
- **Distribution**: IID (equal split across clients)
- **Clients**: 5
- **Local epochs**: 3
- **Rounds**: 5
- **Batch size**: 32
- **Learning rate**: 0.01 (SGD)

### Defense Parameters (Week 6)
- **Fingerprint dimension**: 512
- **Cosine threshold**: 0.90 (very strict)
- **Metadata weight**: 50% (loss + accuracy)
- **Validation threshold**: 0.1 (loss increase)
- **Validation set size**: 1000 samples

---

## 📝 For Academic Papers

### Citation of Algorithms
```bibtex
@inproceedings{kyber,
  title={CRYSTALS-Kyber: a CCA-secure module-lattice-based KEM},
  author={Bos, Joppe and others},
  booktitle={IEEE EuroS\&P},
  year={2018}
}

@inproceedings{dilithium,
  title={CRYSTALS-Dilithium: A lattice-based digital signature scheme},
  author={Ducas, L{\'e}o and others},
  booktitle={IACR TCHES},
  year={2018}
}
```

### Key Contributions
1. **Three-layer Byzantine defense**: PQ crypto + fingerprinting + validation
2. **Client-side fingerprints**: Prevents malicious server from framing honest clients
3. **Metadata-enhanced clustering**: Uses loss/accuracy patterns to improve detection
4. **High accuracy under attack**: Maintains 97% accuracy despite 40% malicious clients

---

## 🚀 Future Work

### Non-IID Implementation
Create `non_iid_implementation/` folder with:
- Dirichlet distribution (α=0.5) for heterogeneous data
- Test defense robustness under data heterogeneity

### Additional Attacks
- Model poisoning (targeted backdoor attacks)
- Gradient inversion attacks
- Free-rider attacks

### Advanced Defenses
- Adaptive thresholds based on round number
- Multi-round reputation systems
- Differential privacy integration

---

## 📦 Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

**Optional** (for real PQ crypto):
```txt
liboqs-python>=0.7.0  # Requires C library installation
```

---

## 🎓 Authors & License

This implementation was created for academic research purposes.

**Note**: Simulated PQ crypto mode is used by default for reproducibility and ease of deployment. For production systems, enable real PQ crypto with `USE_REAL_CRYPTO=True` in `config.py`.

---

## 🔍 Troubleshooting

### Issue: Malicious clients not detected
- **Solution**: Increase `COSINE_THRESHOLD` to 0.90 or 0.95
- **Solution**: Increase `USE_METADATA_FEATURES` weight in clustering

### Issue: Too many false positives
- **Solution**: Decrease `COSINE_THRESHOLD` to 0.85 or 0.80
- **Solution**: Increase `VALIDATION_THRESHOLD` to 0.15

### Issue: liboqs installation fails
- **Solution**: Use simulated mode (`USE_REAL_CRYPTO=False`)
- **Alternative**: Install via conda: `conda install -c conda-forge liboqs-python`

---

**Best implementation**: Use `week6_fingerprints_client/` for your final experiments and paper results!
