"""
Test Configuration for Quantum FL Experiments
==============================================
Optimized for fast quantum training while maintaining experimental validity
"""

# ============================================================================
# QUANTUM MODEL CONFIGURATION (Optimized for Speed)
# ============================================================================

# Quantum Circuit
NUM_QUBITS = 4  # Keep at 4 for reasonable training time
NUM_LAYERS = 4  # Reduced from potential higher values

# Model Architecture
FEATURE_DIM = 16  # Reduced from 256 in classical models

# ============================================================================
# FEDERATED LEARNING CONFIGURATION
# ============================================================================

# Clients
NUM_CLIENTS = 5
MALICIOUS_CLIENTS = [3, 4]  # 40% malicious (2 out of 5)
MALICIOUS_RATIO_20 = [4]     # 20% malicious (1 out of 5)

# Training
NUM_ROUNDS = 5  # Reduced from 10 for faster experiments
LOCAL_EPOCHS = 2  # Reduced from 3
BATCH_SIZE = 64
LEARNING_RATE = 0.01

# Data
NON_IID_ALPHA_EXTREME = 0.1
NON_IID_ALPHA_MODERATE = 0.5
NON_IID_ALPHA_MILD = 1.0

# ============================================================================
# ATTACK CONFIGURATION
# ============================================================================

# Label Flipping Attack
def label_flip_attack(y):
    """Flip labels: 0↔9, 1↔8, etc."""
    return (9 - y) % 10

# Gradient Scaling Attack
GRADIENT_SCALE_LAMBDA = 10.0  # Amplification factor

# Backdoor Attack
BACKDOOR_TRIGGER_SIZE = 3  # 3x3 white square
BACKDOOR_TARGET_CLASS = 1

# ============================================================================
# DEFENSE CONFIGURATION
# ============================================================================

# Spectral Defense
SPECTRAL_WEIGHTS = {
    'spectral_ratio': 0.5,
    'entropy': 0.3,
    'norm_distance': 0.2
}

# Adaptive Thresholding
MAD_BETA = 3.0  # Outlier factor for Median Absolute Deviation

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Statistical Rigor
NUM_RUNS = 3  # Reduced from 5 for faster experiments (increase to 5 for final)
RANDOM_SEEDS = [42, 123, 456]  # Deterministic seeds

# Metrics to Track
METRICS = [
    'test_accuracy',
    'train_loss',
    'detection_rate',
    'false_positive_rate',
    'f1_score',
    'spectral_ratio_honest',
    'spectral_ratio_malicious',
    'entropy_honest',
    'entropy_malicious',
]

# ============================================================================
# EXPECTED PERFORMANCE RANGES (For Validation)
# ============================================================================

EXPECTED_RANGES = {
    # Quantum model will have lower accuracy than classical
    'quantum_accuracy_no_attack': (70, 85),  # 4-qubit limitation
    'quantum_accuracy_with_attack': (30, 50),  # Without defense
    'quantum_accuracy_with_defense': (65, 80),  # With defense
    
    # Detection metrics (primary focus)
    'detection_rate': (85, 98),
    'false_positive_rate': (0, 10),
    'f1_score': (0.85, 0.97),
    
    # Spectral characteristics
    'spectral_ratio_honest': (0.05, 0.25),
    'spectral_ratio_malicious': (0.40, 0.80),
    
    # Overhead
    'defense_overhead_percent': (3, 15),
}

# ============================================================================
# FILE PATHS
# ============================================================================

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'tests', 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================================
# PAPER TABLE MAPPING
# ============================================================================

# Map experiments to paper tables
TABLE_EXPERIMENTS = {
    'table1_main_results': {
        'methods': ['NoAttack', 'FedAvg', 'QuantumDefend'],
        'alphas': [0.1, 0.5, 1.0],
        'malicious_ratios': [0.2, 0.4],
        'metrics': ['test_accuracy'],
        'priority': 'HIGH'
    },
    'table2_detection': {
        'methods': ['QuantumDefend'],
        'alphas': [0.5],
        'malicious_ratios': [0.4],
        'metrics': ['detection_rate', 'false_positive_rate', 'f1_score'],
        'priority': 'HIGHEST'
    },
    'table3_attack_types': {
        'methods': ['FedAvg', 'QuantumDefend'],
        'attacks': ['label_flip', 'gradient_scale', 'backdoor'],
        'alphas': [0.5],
        'malicious_ratios': [0.4],
        'metrics': ['test_accuracy', 'attack_success_rate'],
        'priority': 'HIGH'
    },
    'table4_overhead': {
        'components': ['baseline', 'quantum', 'dct', 'entropy', 'scoring'],
        'metrics': ['time_ms', 'overhead_percent'],
        'priority': 'MEDIUM'
    },
    'table5_ablation': {
        'configs': [
            'quantum_only',
            'spectral_only', 
            'quantum_spectral_no_entropy',
            'quantum_entropy_no_spectral',
            'full_system'
        ],
        'metrics': ['test_accuracy', 'detection_rate'],
        'priority': 'HIGH'
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_experiment_name(method, alpha, malicious_ratio, attack_type='label_flip', run_id=0):
    """Generate standardized experiment name"""
    return f"{method}_alpha{alpha}_mal{int(malicious_ratio*100)}_{attack_type}_run{run_id}"

def validate_result(metric_name, value):
    """Validate if result is within expected range"""
    if metric_name in EXPECTED_RANGES:
        min_val, max_val = EXPECTED_RANGES[metric_name]
        if min_val <= value <= max_val:
            return True, "PASS"
        else:
            return False, f"OUT_OF_RANGE (expected {min_val}-{max_val})"
    return True, "NO_VALIDATION"

def print_config_summary():
    """Print configuration summary"""
    print("="*60)
    print("QUANTUM FL TEST CONFIGURATION")
    print("="*60)
    print(f"Quantum Model:")
    print(f"  Qubits: {NUM_QUBITS}")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Feature Dim: {FEATURE_DIM}")
    print()
    print(f"Federated Learning:")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Local Epochs: {LOCAL_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print()
    print(f"Experiments:")
    print(f"  Runs per config: {NUM_RUNS}")
    print(f"  Total planned experiments: ~50-70")
    print()
    print(f"Expected Quantum Accuracy: 70-85% (vs 95-98% classical)")
    print(f"Focus: Defense effectiveness, not absolute accuracy")
    print("="*60)
    print()

if __name__ == "__main__":
    print_config_summary()
