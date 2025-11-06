"""
Configuration for Quantum Federated Learning
Based on non-IID implementation settings
"""

# ===== Federated Learning Settings =====
NUM_CLIENTS = 30
CLIENTS_PER_ROUND = 30  # All clients participate
NUM_ROUNDS = 5

# ===== Data Distribution =====
NON_IID = True
DIRICHLET_ALPHA = 0.5  # Lower = more non-IID (0.1-1.0)

# ===== Training Hyperparameters =====
BATCH_SIZE = 32
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.01

# ===== Quantum Circuit Settings =====
N_QUBITS = 4
N_LAYERS = 4

# ===== Attack Settings (for week2 and week6) =====
ATTACK_ENABLED = False  # Set to True in week2 and week6
MALICIOUS_PERCENTAGE = 0.4  # 40% malicious clients
ATTACK_TYPE = "gradient_ascent"
SCALE_FACTOR = 10.0  # Attack intensity

# ===== Defense Settings (for week6 only) =====
DEFENSE_ENABLED = False  # Set to True in week6
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
    print("Quantum Federated Learning Configuration")
    print("="*60)
    print(f"Clients: {NUM_CLIENTS} total, {CLIENTS_PER_ROUND} per round")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"Data: Non-IID (α={DIRICHLET_ALPHA})" if NON_IID else "Data: IID")
    print(f"Batch size: {BATCH_SIZE}, Local epochs: {LOCAL_EPOCHS}, LR: {LEARNING_RATE}")
    print(f"Quantum: {N_QUBITS} qubits, {N_LAYERS} layers")
    if ATTACK_ENABLED:
        print(f"Attack: {ATTACK_TYPE} (scale={SCALE_FACTOR}, {int(MALICIOUS_PERCENTAGE*100)}% malicious)")
    if DEFENSE_ENABLED:
        print(f"Defense: {DEFENSE_TYPE} (threshold=median×{NORM_THRESHOLD_MULTIPLIER})")
    if PQ_CRYPTO_ENABLED:
        print(f"PQ Crypto: {KEM_ALGORITHM} + {SIGNATURE_ALGORITHM}")
    print("="*60 + "\n")
