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
    
    # Attack settings
    ATTACK_ENABLED = True
    MALICIOUS_CLIENTS = [0, 1]  # First 2 out of 5 clients are malicious
    
    # Defense settings
    DEFENSE_ENABLED = True
    VALIDATION_SIZE = 1000  # samples for server validation
    VALIDATION_THRESHOLD = 0.1  # Max acceptable loss increase
    
    # Model
    MODEL_TYPE = "simple_cnn"  # Simple CNN for MNIST
