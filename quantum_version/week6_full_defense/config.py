"""
Configuration for Quantum Federated Learning - Week 6 (Full Defense)
"""

# ===== Federated Learning Settings =====
NUM_CLIENTS = 5
CLIENTS_PER_ROUND = 5
NUM_ROUNDS = 5

# ===== Data Distribution =====
NON_IID = True
DIRICHLET_ALPHA = 0.5

# ===== Training Hyperparameters =====
BATCH_SIZE = 64
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.01

# ===== Quantum Circuit Settings =====
N_QUBITS = 4  # Fixed: Must match quantum_model.py architecture (classifier expects 4-D input)
N_LAYERS = 4  # Fixed: Match baseline architecture

# ===== Attack Settings =====
ATTACK_ENABLED = True
MALICIOUS_PERCENTAGE = 0.4
ATTACK_TYPE = "gradient_ascent"  # Only gradient ascent supported
SCALE_FACTOR = 50.0  # Attack intensity (λ = 50)

# ===== Defense Settings =====
# QuantumDefend PLUS v2: 3-Layer Cascading Defense System
# Layer 0: Fast norm-based filtering (catches obvious attacks)
# Layer 1: Adaptive 6-feature anomaly detection (catches sophisticated attacks)
# Layer 2: Client-side fingerprint verification (catches stealthy attacks)
DEFENSE_ENABLED = True

# Layer 0: Norm-Based Filtering (Fast Pre-filter)
USE_NORM_FILTERING = True
NORM_THRESHOLD_MULTIPLIER = 3.0  # median × 3.0 (catches 50x norm attacks)

# Layer 1: Adaptive Defense (Multi-feature Detection)
USE_ADAPTIVE_DEFENSE = True
ADAPTIVE_METHOD = 'statistical'  # Options: 'statistical', 'clustering', 'isolation_forest'
# Features: norm, loss_increase, layer_variance, sign_consistency, train_loss, train_acc

# Layer 2: Fingerprint Defense (Integrity Verification)
USE_FINGERPRINTS = True
FINGERPRINT_DIM = 512  # 512-D random projection
FINGERPRINT_THRESHOLD = 0.85  # Cosine similarity threshold (31.8° angle)
FINGERPRINT_TOLERANCE = 1e-3  # Verification tolerance

# ===== Model Settings =====
NUM_CLASSES = 10

# ===== Logging =====
VERBOSE = True
SAVE_MODEL = True
MODEL_SAVE_PATH = "quantum_model_defended.pth"

# ===== Random Seed =====
SEED = 42

# ===== Display Configuration =====
def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("Quantum Federated Learning - Week 6 (Full Defense)")
    print("="*60)
    print(f"Clients: {NUM_CLIENTS} total, {CLIENTS_PER_ROUND} per round")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"Data: Non-IID (α={DIRICHLET_ALPHA})" if NON_IID else "Data: IID")
    print(f"Batch size: {BATCH_SIZE}, Local epochs: {LOCAL_EPOCHS}, LR: {LEARNING_RATE}")
    print(f"Quantum: {N_QUBITS} qubits, {N_LAYERS} layers")
    if ATTACK_ENABLED:
        num_malicious = int(NUM_CLIENTS * MALICIOUS_PERCENTAGE)
        print(f"⚠️  ATTACK: {ATTACK_TYPE} (scale={SCALE_FACTOR})")
        print(f"⚠️  MALICIOUS: {num_malicious}/{NUM_CLIENTS} clients ({int(MALICIOUS_PERCENTAGE*100)}%)")
    if DEFENSE_ENABLED:
        if USE_NORM_FILTERING:
            print(f"🛡️  DEFENSE LAYER 0: Norm Filter (median × {NORM_THRESHOLD_MULTIPLIER})")
        if USE_ADAPTIVE_DEFENSE:
            print(f"🛡️  DEFENSE LAYER 1: Adaptive ({ADAPTIVE_METHOD}) with 6 features")
        if USE_FINGERPRINTS:
            print(f"🛡️  DEFENSE LAYER 2: Fingerprints ({FINGERPRINT_DIM}-D projection)")
        if not USE_NORM_FILTERING and not USE_ADAPTIVE_DEFENSE and not USE_FINGERPRINTS:
            print(f"⚠️  DEFENSE ENABLED but no methods configured")
    else:
        print(f"⚠️  NO DEFENSE - All updates aggregated")
    print("="*60 + "\n")
