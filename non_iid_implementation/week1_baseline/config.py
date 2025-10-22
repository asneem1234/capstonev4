# Configuration for Non-IID Federated Learning (Baseline - No Attack)
class Config:
    # Dataset
    DATASET = "MNIST"
    DATA_DIR = "./data"
    
    # Federated Learning basics
    NUM_CLIENTS = 30
    CLIENTS_PER_ROUND = 30
    NUM_ROUNDS = 5
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    
    # Non-IID data distribution
    USE_NON_IID = True  # Enable Non-IID data split
    DIRICHLET_ALPHA = 0.5  # Lower = more heterogeneous (0.1=highly non-IID, 1.0=slightly non-IID)
    
    # Attack settings (DISABLED for baseline)
    ATTACK_ENABLED = False
    MALICIOUS_CLIENTS = []  # No malicious clients in baseline
    
    # Defense settings (not needed for baseline)
    DEFENSE_ENABLED = False
    
    # Model
    MODEL_TYPE = "simple_cnn"  # Simple CNN for MNIST
