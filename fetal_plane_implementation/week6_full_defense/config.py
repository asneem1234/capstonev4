# Configuration for Non-IID Federated Learning with Full Defense
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
    
    # Defense settings
    DEFENSE_ENABLED = True
    VALIDATION_SIZE = 100  # samples for server validation
    VALIDATION_THRESHOLD = 0.15  # Max acceptable loss increase
    
    # Fingerprint settings (CLIENT-SIDE with integrity verification)
    USE_FINGERPRINTS = True
    FINGERPRINT_DIM = 512  # Projection dimension for gradient space
    COSINE_THRESHOLD = 0.85  # Threshold for fingerprint similarity (stricter for medical)
    USE_METADATA_FEATURES = True  # Include loss/accuracy in fingerprints
    
    # Post-Quantum Cryptography settings
    USE_PQ_CRYPTO = True  # Enable PQ crypto layer
    USE_REAL_CRYPTO = False  # False = simulated (for testing), True = real liboqs
    PQ_KEM_ALG = "Kyber512"  # Key encapsulation (encryption)
    PQ_SIG_ALG = "Dilithium2"  # Digital signatures
    
    # Model
    MODEL_TYPE = "resnet18"
    NUM_CLASSES = 6
    
    # Image settings
    IMAGE_SIZE = 224
    PRETRAINED = True
