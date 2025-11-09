"""
Configuration for Quantum Federated Learning - Week 6 (Full Defense)
EXPERIMENT: 5 clients with diverse attack intensities
"""

# ===== Federated Learning Settings =====
NUM_CLIENTS = 5
CLIENTS_PER_ROUND = 5
NUM_ROUNDS = 5  # 5 rounds for testing

# ===== Data Distribution =====
NON_IID = True
DIRICHLET_ALPHA = 0.5

# ===== Training Hyperparameters =====
BATCH_SIZE = 64
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.01  # Increased from 0.001 for faster convergence
GRADIENT_CLIP = 1.0  # Gradient clipping for stability

# ===== Quantum Circuit Settings =====
N_QUBITS = 2  # Reduced from 4 for faster training
N_LAYERS = 1  # Reduced from 4 for faster training

# ===== Attack Settings =====
ATTACK_ENABLED = True
ATTACK_TYPE = "gradient_ascent"  # Base attack type

# DIVERSE ATTACK CONFIGURATION (2 malicious clients with gradient scale attacks)
# Client IDs are randomly selected, but here we define the attack profiles:
MALICIOUS_CLIENTS_CONFIG = {
    # Aggressive gradient scale attack (high intensity)
    'aggressive': {
        'count': 1,
        'scale_factor': 50.0,  # Very high gradient scaling - tests Layer 0 norm filter
        'description': 'High-intensity gradient scale attack (50x amplification)'
    },
    # Moderate gradient scale attack (medium intensity)
    'moderate': {
        'count': 1,
        'scale_factor': 10.0,  # Medium gradient scaling - tests Layer 1 adaptive defense
        'description': 'Medium-intensity gradient scale attack (10x amplification)'
    },
    # No subtle attacks for now - testing gradient scale first
    'subtle': {
        'count': 0,
        'scale_factor': 1.5,  # Reserved for future testing
        'description': 'Reserved for future subtle attack testing'
    }
}

# Total malicious clients
TOTAL_MALICIOUS = sum(config['count'] for config in MALICIOUS_CLIENTS_CONFIG.values())
MALICIOUS_PERCENTAGE = TOTAL_MALICIOUS / NUM_CLIENTS  # 2/5 = 40%

# Legacy single scale factor (for backward compatibility, uses aggressive)
SCALE_FACTOR = MALICIOUS_CLIENTS_CONFIG['aggressive']['scale_factor']

# ===== Defense Settings (3-Layer Cascade) =====
DEFENSE_ENABLED = True

# Layer 0: Norm Filter
USE_NORM_FILTERING = True
NORM_THRESHOLD_MULTIPLIER = 3.0  # median × 3.0

# Layer 1: Adaptive Statistical Defense
USE_ADAPTIVE_DEFENSE = True
ADAPTIVE_METHOD = "statistical"  # Statistical outlier detection
ADAPTIVE_THRESHOLD_STD = 2.0  # mean + 2×std

# Layer 2: Fingerprint Validation
USE_FINGERPRINTS = True
FINGERPRINT_DIM = 512  # 512-D fingerprint projection
FINGERPRINT_SIMILARITY_THRESHOLD = 0.7  # Minimum similarity to history

# ===== Post-Quantum Crypto Settings =====
PQ_CRYPTO_ENABLED = False  # Not used in Week 6
KEM_ALGORITHM = "Kyber512"
SIGNATURE_ALGORITHM = "Dilithium2"

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
    print("Week 6: Testing Defense Against Gradient Scale Attacks")
    print("="*60)
    print(f"Clients: {NUM_CLIENTS} total, {CLIENTS_PER_ROUND} per round")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"Data: Non-IID (alpha={DIRICHLET_ALPHA})" if NON_IID else "Data: IID")
    print(f"Batch size: {BATCH_SIZE}, Local epochs: {LOCAL_EPOCHS}, LR: {LEARNING_RATE}")
    print(f"Quantum: {N_QUBITS} qubits, {N_LAYERS} layers")
    if ATTACK_ENABLED:
        print(f"\n⚠️  ATTACK SCENARIO: Gradient Scale Attack")
        print(f"    Attack Type: {ATTACK_TYPE}")
        for attack_type, attack_config in MALICIOUS_CLIENTS_CONFIG.items():
            if attack_config['count'] > 0:
                print(f"    - {attack_config['count']}x {attack_type.upper()}: scale={attack_config['scale_factor']} ({attack_config['description']})")
        print(f"    Total Malicious: {TOTAL_MALICIOUS}/{NUM_CLIENTS} clients ({int(MALICIOUS_PERCENTAGE*100)}%)")
    if DEFENSE_ENABLED:
        defenses = []
        if USE_NORM_FILTERING:
            defenses.append("Layer 0: Norm Filter")
        if USE_ADAPTIVE_DEFENSE:
            defenses.append("Layer 1: Adaptive")
        if USE_FINGERPRINTS:
            defenses.append("Layer 2: Fingerprints")
        print(f"\n🛡️  DEFENSE LAYERS ACTIVE:")
        for defense in defenses:
            print(f"    ✓ {defense}")
    else:
        print(f"\n⚠️  WARNING: NO DEFENSE - All updates aggregated")
    print("="*60 + "\n")
