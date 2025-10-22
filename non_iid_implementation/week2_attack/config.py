# Configuration for Non-IID Federated Learning with Attack
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
    
    # Attack settings
    ATTACK_ENABLED = True
    MALICIOUS_PERCENTAGE = 0.2  # 20% of clients are malicious at each round
    RANDOM_MALICIOUS = True  # Randomly select malicious clients each round
    MALICIOUS_CLIENTS = []  # Will be populated dynamically each round
    
    # Defense settings (disabled for week 2)
    DEFENSE_ENABLED = False
    VALIDATION_SIZE = 1000  # samples for server validation (not used yet)
    
    # Model
    MODEL_TYPE = "simple_cnn"  # Simple CNN for MNIST
