# Simple configuration - keep it minimal
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
    
    # Attack settings (disabled initially)
    ATTACK_ENABLED = False
    MALICIOUS_CLIENTS = []  # e.g., [0, 1, 2] for first 3 clients
    
    # Defense settings (disabled initially)
    DEFENSE_ENABLED = False
    VALIDATION_SIZE = 1000  # samples for server validation
    
    # Model
    MODEL_TYPE = "simple_cnn"  # Simple CNN for MNIST
