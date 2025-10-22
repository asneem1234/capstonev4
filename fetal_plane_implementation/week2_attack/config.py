# Configuration for Non-IID Federated Learning with Attack
# Fetal Plane Classification
class Config:
    # Dataset
    DATASET = "FETAL_PLANE"
    DATA_DIR = "./data/fetal_planes"
    
    # Federated Learning basics
    NUM_CLIENTS = 10
    CLIENTS_PER_ROUND = 10
    NUM_ROUNDS = 10
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    
    # Non-IID data distribution
    USE_NON_IID = True
    DIRICHLET_ALPHA = 0.5
    
    # Attack settings
    ATTACK_ENABLED = True
    MALICIOUS_PERCENTAGE = 0.2  # 20% of clients are malicious at each round
    RANDOM_MALICIOUS = True  # Randomly select malicious clients each round
    MALICIOUS_CLIENTS = []  # Will be populated dynamically each round
    
    # Defense settings (disabled for week 2)
    DEFENSE_ENABLED = False
    VALIDATION_SIZE = 100  # samples for server validation (not used yet)
    
    # Model
    MODEL_TYPE = "resnet18"
    NUM_CLASSES = 6
    
    # Image settings
    IMAGE_SIZE = 224
    PRETRAINED = True
