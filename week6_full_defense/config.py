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
N_QUBITS = 2
N_LAYERS = 1

# ===== Attack Settings =====
ATTACK_ENABLED = True
MALICIOUS_PERCENTAGE = 0.4
ATTACK_TYPE = "label_flip"
SCALE_FACTOR = 10.0  # Attack intensity

# ===== Defense Settings =====
DEFENSE_ENABLED = True
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

DEFENSE_TYPE = "norm_filtering"
NORM_THRESHOLD_MULTIPLIER = 3.0  # median × 3.0

# ===== Post-Quantum Crypto Settings =====
PQ_CRYPTO_ENABLED = False  # Simulated (optional)
KEM_ALGORITHM = "Kyber512"
SIGNATURE_ALGORITHM = "Dilithium2"

# ===== Fingerprint Settings =====
FINGERPRINT_ENABLED = False  # Optional feature

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
        print(f"🛡️  DEFENSE: {DEFENSE_TYPE} (threshold=median×{NORM_THRESHOLD_MULTIPLIER})")
    else:
        print(f"⚠️  NO DEFENSE - All updates aggregated")
    print("="*60 + "\n")
