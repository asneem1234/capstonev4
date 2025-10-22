# Configuration for Non-IID Federated Learning (Baseline - No Attack)
# Fetal Plane Classification
class Config:
    # Dataset
    DATASET = "FETAL_PLANE"
    DATA_DIR = "../FETAL"  # Path to FETAL dataset (one level up from week folders)
    
    # Federated Learning basics
    NUM_CLIENTS = 10  # Adjust based on your scenario (hospitals/clinics)
    CLIENTS_PER_ROUND = 10
    NUM_ROUNDS = 10  # More rounds for medical imaging
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 16  # Smaller batch size for medical images
    LEARNING_RATE = 0.001  # Lower learning rate for transfer learning
    
    # Non-IID data distribution
    USE_NON_IID = True  # Enable Non-IID data split
    DIRICHLET_ALPHA = 0.5  # Lower = more heterogeneous (0.1=highly non-IID, 1.0=slightly non-IID)
    
    # Attack settings (DISABLED for baseline)
    ATTACK_ENABLED = False
    MALICIOUS_CLIENTS = []  # No malicious clients in baseline
    
    # Defense settings (not needed for baseline)
    DEFENSE_ENABLED = False
    
    # Model
    MODEL_TYPE = "resnet18"  # ResNet18 for fetal plane classification
    NUM_CLASSES = 6  # Number of fetal plane classes
    # Classes: Trans-thalamic, Trans-cerebellum, Trans-ventricular, 
    #          Maternal cervix, Femur, Others
    
    # Image settings
    IMAGE_SIZE = 224  # Standard size for pretrained models
    PRETRAINED = True  # Use ImageNet pretrained weights
