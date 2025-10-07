# Configuration for Non-IID Federated Learning with Attack
class Config:
    # Dataset
    DATASET = "MNIST"
    DATA_DIR = "./data"
    
    # Federated Learning basics
    NUM_CLIENTS = 5
    CLIENTS_PER_ROUND = 5
    NUM_ROUNDS = 5
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    
    # Non-IID data distribution
    USE_NON_IID = True  # Enable Non-IID data split
    DIRICHLET_ALPHA = 0.5  # Lower = more heterogeneous (0.1=highly non-IID, 1.0=slightly non-IID)
    
    # Attack settings
    ATTACK_ENABLED = True
    MALICIOUS_CLIENTS = [0, 1]  # First 2 out of 5 clients are malicious
    
    # Defense settings (disabled for week 2)
    DEFENSE_ENABLED = False
    VALIDATION_SIZE = 1000  # samples for server validation (not used yet)
    
    # Model
    MODEL_TYPE = "simple_cnn"  # Simple CNN for MNIST
