"""
Configuration for Table 5 Defense Comparison Tests
All methods use identical settings for fair comparison
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
N_QUBITS = 4
N_LAYERS = 4

# ===== Attack Settings (Fixed for all tests) =====
ATTACK_ENABLED = True
MALICIOUS_PERCENTAGE = 0.4  # 40% malicious
ATTACK_TYPE = "gradient_ascent"
SCALE_FACTOR = 50.0  # λ = 50

# ===== Model Settings =====
NUM_CLASSES = 10

# ===== Random Seed (for reproducibility) =====
SEED = 42

# ===== Output Settings =====
VERBOSE = True
SAVE_RESULTS = True
RESULTS_DIR = "./results"

# ===== Defense Method Configurations =====

# Krum: Select update closest to others (Byzantine-robust aggregation)
KRUM_F = 2  # Number of Byzantine clients to tolerate

# Median: Coordinate-wise median
MEDIAN_METHOD = "coordinate_wise"

# Trimmed-Mean: Remove top/bottom β fraction
TRIMMED_MEAN_BETA = 0.2  # Remove 20% from each end

# RobustAvg: Geometric median (iterative Weiszfeld algorithm)
ROBUST_AVG_MAX_ITER = 100
ROBUST_AVG_TOLERANCE = 1e-5

# QuantumDefend PLUS v2: 3-Layer Cascading Defense
QUANTUMDEFEND_LAYER0_ENABLED = True  # Norm filtering
QUANTUMDEFEND_LAYER0_MULTIPLIER = 3.0  # median × 3.0

QUANTUMDEFEND_LAYER1_ENABLED = True  # Adaptive 6-feature
QUANTUMDEFEND_LAYER1_METHOD = "statistical"  # IQR-based

QUANTUMDEFEND_LAYER2_ENABLED = True  # Fingerprints
QUANTUMDEFEND_LAYER2_DIM = 512
QUANTUMDEFEND_LAYER2_THRESHOLD = 0.85
QUANTUMDEFEND_LAYER2_TOLERANCE = 1e-3


def print_config():
    """Print test configuration"""
    print("\n" + "="*60)
    print("Table 5: Defense Method Comparison Tests")
    print("="*60)
    print(f"Clients: {NUM_CLIENTS} ({int(MALICIOUS_PERCENTAGE*100)}% malicious)")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"Data: Non-IID (Dirichlet α={DIRICHLET_ALPHA})")
    print(f"Attack: {ATTACK_TYPE} (λ={SCALE_FACTOR})")
    print(f"Model: Quantum {N_QUBITS} qubits, {N_LAYERS} layers")
    print("="*60 + "\n")
