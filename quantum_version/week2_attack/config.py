"""
Configuration for Quantum Federated Learning - Week 2 (Attack)
"""

# ===== Federated Learning Settings =====
NUM_CLIENTS = 5
CLIENTS_PER_ROUND = 5
NUM_ROUNDS = 10  # Increased to 10 rounds to see attack impact

# ===== Data Distribution =====
NON_IID = True
DIRICHLET_ALPHA = 0.5

# ===== Training Hyperparameters =====
BATCH_SIZE = 64
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.001  # Lower LR for quantum circuits
GRADIENT_CLIP = 1.0  # Gradient clipping for stability

# ===== Quantum Circuit Settings =====
N_QUBITS = 4
N_LAYERS = 4

# ===== Attack Settings =====
ATTACK_ENABLED = True
MALICIOUS_PERCENTAGE = 0.4  # 40% malicious clients (2 out of 5)
ATTACK_TYPE = "gradient_ascent"  # More destructive than label_flip
SCALE_FACTOR = 50.0  # Very aggressive attack intensity (was 10.0)

# ===== Defense Settings (disabled in week2) =====
DEFENSE_ENABLED = False
DEFENSE_TYPE = "norm_filtering"
NORM_THRESHOLD_MULTIPLIER = 3.0  # median × 3.0

# ===== Post-Quantum Crypto Settings =====
PQ_CRYPTO_ENABLED = False  # Set to True in week6
KEM_ALGORITHM = "Kyber512"
SIGNATURE_ALGORITHM = "Dilithium2"

# ===== Fingerprint Settings (for week6 only) =====
FINGERPRINT_ENABLED = False  # Set to True in week6

# ===== Model Settings =====
NUM_CLASSES = 10

# ===== Logging =====
VERBOSE = True
SAVE_MODEL = True
MODEL_SAVE_PATH = "quantum_model.pth"

# ===== Random Seed =====
SEED = 42

# ===== Display Configuration =====
def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("Quantum Federated Learning - Week 2 (Attack)")
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
        print(f"🛡️  DEFENSE: {DEFENSE_TYPE}")
    else:
        print(f"⚠️  NO DEFENSE - All updates aggregated")
    print("="*60 + "\n")
