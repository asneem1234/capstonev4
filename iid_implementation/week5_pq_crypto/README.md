# Week 5: Post-Quantum Cryptography Layer

## Goal
Add **Layer 1 defense** against network attackers (MITM) using PQ-secure encryption and signatures.

**Expected result:** Same accuracy as Week 4, but with cryptographic protection against:
- Man-in-the-Middle (MITM) attacks
- Eavesdropping
- Tampering
- Impersonation

---

## What's New

### Files Added:
- `pq_crypto.py` - Simple PQ crypto wrapper (Kyber512 + Dilithium2)

### Files Modified:
- `config.py` - PQ crypto settings
- `requirements.txt` - Added liboqs-python (optional)

---

## Two Operating Modes

### Mode 1: Simulated (Default - Easy Testing)
```python
USE_PQ_CRYPTO = True
USE_REAL_CRYPTO = False  # No liboqs needed
```

**What happens:**
- ✅ Shows crypto workflow (encrypt/decrypt/sign/verify)
- ✅ Measures overhead of serialization
- ❌ NOT actually secure (just for testing)
- ✅ No external dependencies

**Use for:** Testing the integration, measuring performance impact

### Mode 2: Real PQ Crypto (Optional - Actual Security)
```python
USE_PQ_CRYPTO = True
USE_REAL_CRYPTO = True  # Requires liboqs
```

**What happens:**
- ✅ Real Kyber512 encryption (quantum-resistant)
- ✅ Real Dilithium2 signatures (quantum-resistant)
- ✅ Actually secure against MITM
- ❌ Requires installing liboqs C library + Python bindings

**Use for:** Final version, security analysis, paper results

---

## How It Works

### Current Flow (Week 4):
```
Client → [Update] → Server
```
**Vulnerable to:** MITM can intercept, modify, replay

### New Flow (Week 5):
```
Client:
1. Train → Δw
2. Sign: σ = Dilithium.Sign(Hash(Δw || client_id || round))
3. Encrypt: c = Kyber.Encrypt(Δw, pk_server)
4. Send: (c, σ, client_id, round)

Server:
5. Verify signature (authentic?)
6. Decrypt update
7. Continue with fingerprint + validation defenses
```

**Protected against:**
- ✅ MITM modification (signature invalid)
- ✅ Eavesdropping (update encrypted)
- ✅ Replay attacks (round number in signature)
- ✅ Impersonation (need client's private key)

---

## Cryptographic Algorithms

### Kyber512 (KEM - Key Encapsulation)
- **Purpose:** Encrypt model updates
- **Security:** 128-bit quantum security (equivalent to AES-128)
- **Key sizes:** 
  - Public key: 800 bytes
  - Secret key: 1632 bytes
  - Ciphertext: 768 bytes
- **Speed:** ~0.1ms encapsulation, ~0.1ms decapsulation

### Dilithium2 (Digital Signatures)
- **Purpose:** Authenticate updates, prevent tampering
- **Security:** 128-bit quantum security
- **Key sizes:**
  - Public key: 1312 bytes
  - Secret key: 2528 bytes
  - Signature: 2420 bytes
- **Speed:** ~0.5ms sign, ~0.1ms verify

**Total overhead per update:**
- Communication: ~3.2 KB (signature + KEM ciphertext)
- Computation: ~0.7ms per client

---

## Defense Layers Summary

Now you have **THREE layers**:

### Layer 1: Post-Quantum Crypto (Network Security)
- **Threat:** External attacker on network
- **Defense:** Kyber + Dilithium
- **Catches:** MITM, eavesdropping, tampering, impersonation

### Layer 2: Fingerprint Clustering (Fast Byzantine Detection)
- **Threat:** Malicious insider client
- **Defense:** Cosine similarity clustering
- **Catches:** Obvious poisoning (large-scale attacks)

### Layer 3: Validation Filtering (Deep Byzantine Detection)
- **Threat:** Sophisticated malicious client
- **Defense:** Test on validation set
- **Catches:** Subtle poisoning (label flipping, backdoors)

---

## Run It

### Simulated Mode (No dependencies):
```bash
cd c:\Users\admin\OneDrive\Desktop\capstonev3\mnist_implementation\new_approach\week5
python main.py
```

### Real PQ Crypto Mode (Requires setup):
```bash
# Install liboqs (platform-specific)
# Windows: https://github.com/open-quantum-safe/liboqs/releases
# Linux: sudo apt install liboqs-dev
# Mac: brew install liboqs

# Install Python bindings
pip install liboqs-python

# Enable in config.py
USE_REAL_CRYPTO = True

# Run
python main.py
```

---

## Expected Output

```
======================================================================
Federated Learning - FULL THREE-LAYER DEFENSE
======================================================================
Clients: 5
PQ Crypto: ENABLED (Simulated mode)
Fingerprint Defense: ENABLED
Validation Defense: ENABLED
======================================================================

[CLIENT TRAINING]
  Client 0 [MALICIOUS]: Update signed and encrypted
  ...

[SERVER: LAYER 1 - PQ CRYPTO]
  Verified signatures: 5/5
  Decrypted updates: 5/5
  Overhead: ~5ms

[SERVER: LAYER 2 - FINGERPRINT]
  Main cluster: [2, 3, 4]
  Outliers: [0, 1]

[SERVER: LAYER 3 - VALIDATION]
  Client 0: ✗ REJECT (Δloss=+0.15)
  Client 1: ✗ REJECT (Δloss=+0.12)
  
[GLOBAL EVALUATION]
  Test Accuracy: 88.5%
```

---

## Performance Impact

| Layer | Overhead per Round |
|-------|-------------------|
| PQ Crypto (5 clients) | ~5-10ms |
| Fingerprint Clustering | ~10ms |
| Validation (2 outliers) | ~200ms |
| **Total** | **~220ms** |

Still under 500ms target! ✅

---

## For Your Paper

This demonstrates **defense in depth**:

1. **Network layer:** PQ crypto protects communication
2. **Detection layer:** Fingerprints + validation catch malicious updates
3. **Complete threat model:** External attackers + Byzantine insiders

---

## Next: Week 6 - Refactor to Client-Side Fingerprints

Move fingerprint computation to clients for:
- Better integrity checking
- More realistic architecture
- Additional MITM detection
