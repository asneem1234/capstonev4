# Configuration for Non-IID Federated Learning
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
    
    # Defense settings
    DEFENSE_ENABLED = True
    VALIDATION_SIZE = 1000  # samples for server validation
    VALIDATION_THRESHOLD = 0.1  # Max acceptable loss increase
    
    # Fingerprint settings
    USE_FINGERPRINTS = True  # CLIENT-SIDE with integrity verification
    FINGERPRINT_DIM = 512  # Projection dimension for gradient space
    COSINE_THRESHOLD = 0.90  # VERY STRICT threshold (angle < 26°, was 32°)
    USE_METADATA_FEATURES = True  # Include loss/accuracy in fingerprints
    
    # Post-Quantum Cryptography settings
    USE_PQ_CRYPTO = True  # Enable PQ crypto layer
    USE_REAL_CRYPTO = False  # False = simulated (for testing), True = real liboqs
    PQ_KEM_ALG = "Kyber512"  # Key encapsulation (encryption)
    PQ_SIG_ALG = "Dilithium2"  # Digital signatures
    
    # Model
    MODEL_TYPE = "simple_cnn"  # Simple CNN for MNIST
