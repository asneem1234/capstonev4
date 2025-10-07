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
    
    # Fingerprint settings
    USE_FINGERPRINTS = True  # Re-enabled with proper cosine similarity
    FINGERPRINT_DIM = 512  # Projection dimension
    COSINE_THRESHOLD = 0.7  # Similarity threshold for clustering (angle < 45°)
    
    # Post-Quantum Cryptography settings
    USE_PQ_CRYPTO = True  # Enable PQ crypto layer
    USE_REAL_CRYPTO = False  # False = simulated (for testing), True = real liboqs
    PQ_KEM_ALG = "Kyber512"  # Key encapsulation (encryption)
    PQ_SIG_ALG = "Dilithium2"  # Digital signatures
    
    # Model
    MODEL_TYPE = "simple_cnn"  # Simple CNN for MNIST
