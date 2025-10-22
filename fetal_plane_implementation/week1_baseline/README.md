# Week 1: Baseline Federated Learning with Non-IID Data
# Fetal Plane Classification

## Goal
Establish a **baseline** for Non-IID federated learning on fetal ultrasound plane classification without any attacks or defenses. This serves as the **upper bound** for comparison.

---

## What This Is

This is a **clean, working federated learning system** for fetal plane classification with:
- ✅ **Non-IID data** (Dirichlet distribution, α=0.5)
- ✅ **All clients are honest** (no malicious behavior)
- ✅ **Simple FedAvg** (standard aggregation)
- ✅ **No defenses needed** (baseline scenario)
- ✅ **ResNet18 backbone** (pretrained on ImageNet)

**Purpose**: Shows how well federated learning works in the **ideal case** (no attacks).

---

## Dataset Setup

### Expected Directory Structure

```
fetal_plane_implementation/
└── week1_baseline/
    └── data/
        └── fetal_planes/
            ├── train/
            │   ├── class_0/
            │   │   ├── image1.png
            │   │   ├── image2.png
            │   │   └── ...
            │   ├── class_1/
            │   │   └── ...
            │   └── ...
            └── test/
                ├── class_0/
                │   └── ...
                └── ...
```

### Fetal Plane Classes

The default configuration assumes 6 classes:
1. **Trans-thalamic plane** - Brain imaging at thalamus level
2. **Trans-cerebellum plane** - Brain imaging at cerebellum level  
3. **Trans-ventricular plane** - Brain imaging at ventricle level
4. **Maternal cervix** - Cervical region
5. **Femur** - Leg bone measurements
6. **Others** - Other anatomical views

**Note**: Update `Config.NUM_CLASSES` in `config.py` if your dataset has a different number of classes.

---

## Files Included

```
week1_baseline/
├── config.py           # Configuration (Non-IID, no attack)
├── data_loader.py      # Dirichlet-based Non-IID data split + custom dataset
├── model.py            # ResNet18 for fetal plane classification
├── client.py           # Client training (no attack)
├── server.py           # Simple FedAvg aggregation
├── main.py             # Training loop
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Place your fetal plane images in the expected directory structure (see above).

Update `Config.DATA_DIR` in `config.py` if needed:
```python
DATA_DIR = "./data/fetal_planes"  # Update this path
```

### 3. Configure Model Settings

In `config.py`, adjust these parameters:
```python
NUM_CLIENTS = 10        # Number of federated clients (hospitals/clinics)
NUM_ROUNDS = 10         # Training rounds
LOCAL_EPOCHS = 5        # Local training epochs per round
BATCH_SIZE = 16         # Batch size
LEARNING_RATE = 0.001   # Learning rate
NUM_CLASSES = 6         # Number of plane classes
IMAGE_SIZE = 224        # Input image size
PRETRAINED = True       # Use ImageNet pretrained weights
```

### 4. Run Training

```bash
cd fetal_plane_implementation/week1_baseline
python main.py
```

---

## Expected Output

### Data Distribution (Non-IID)
```
Creating Non-IID data split with Dirichlet(α=0.5)...
Data distribution per client:
  Client 0: 450 samples, dominant class=2 (185 samples)
  Client 1: 430 samples, dominant class=0 (192 samples)
  Client 2: 460 samples, dominant class=4 (178 samples)
  Client 3: 440 samples, dominant class=1 (201 samples)
  ...
```

**Key Observation**: Each client (hospital/clinic) has a **different dominant plane type** (heterogeneous data).

### Training Progress (All Clients Honest)
```
ROUND 1:
[CLIENT TRAINING]
  Client 0: Train Acc=35.20%, Loss=1.65, Update Norm=2.34
  Client 1: Train Acc=38.45%, Loss=1.58, Update Norm=2.28
  Client 2: Train Acc=32.10%, Loss=1.72, Update Norm=2.41
  ...

[GLOBAL EVALUATION]
  Global Test Accuracy: 42.15%

ROUND 2:
  ...
  Global Test Accuracy: 58.30% ↑ +16.15%
```

**Key Observation**: All clients show **steady improvement** as the global model learns.

### Final Results
```
Initial Test Accuracy: 18.50%  (pretrained but not fine-tuned)
Final Test Accuracy:   82.40%  (after 10 rounds)
Total Improvement:     +63.90%

✅ Baseline established!
   With Non-IID data and no attacks, model reaches ~82.4%
   This is the UPPER BOUND for comparison with attack scenarios.
```

---

## Key Characteristics

### 1. All Clients Are Honest
```python
ATTACK_ENABLED = False
MALICIOUS_CLIENTS = []  # Empty list
```
- No label flipping
- No gradient poisoning
- All clients train on correct labels

### 2. Non-IID Data Distribution
```python
USE_NON_IID = True
DIRICHLET_ALPHA = 0.5  # Moderately heterogeneous
```
Each client (hospital) has different proportions of plane types:
- **Hospital A**: Mostly brain scans (trans-thalamic, trans-cerebellum)
- **Hospital B**: Mostly femur measurements
- **Hospital C**: Mixed but with dominant cervix scans
- etc.

This mimics **real-world federated medical imaging** where different hospitals specialize in different procedures.

### 3. Simple FedAvg Aggregation
```python
# Average all client updates equally
aggregated_update = sum(client_updates) / num_clients
global_model = global_model + aggregated_update
```

No defenses, no filtering, just straightforward averaging.

---

## Customization

### Use RGB Images Instead of Grayscale

In `model.py`, change:
```python
# FROM:
self.backbone.conv1 = nn.Conv2d(1, 64, ...)  # Grayscale

# TO:
self.backbone.conv1 = nn.Conv2d(3, 64, ...)  # RGB
```

In `data_loader.py`, change:
```python
# FROM:
image = Image.open(img_path).convert('L')  # Grayscale

# TO:
image = Image.open(img_path).convert('RGB')  # RGB
```

Also update normalization:
```python
# FROM:
transforms.Normalize(mean=[0.5], std=[0.5])

# TO:
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Change Backbone Model

In `model.py`, you can use different architectures:
```python
# ResNet50 (more capacity)
self.backbone = models.resnet50(pretrained=pretrained)

# EfficientNet (more efficient)
self.backbone = models.efficientnet_b0(pretrained=pretrained)

# VGG16 (simpler)
self.backbone = models.vgg16(pretrained=pretrained)
```

### Adjust Non-IID Level

In `config.py`:
```python
DIRICHLET_ALPHA = 0.1   # Highly non-IID (extreme heterogeneity)
DIRICHLET_ALPHA = 0.5   # Moderately non-IID (realistic)
DIRICHLET_ALPHA = 1.0   # Slightly non-IID
DIRICHLET_ALPHA = 10.0  # Nearly IID
```

---

## Next Steps

After establishing this baseline:

1. **Week 2**: Introduce attacks (label flipping, gradient poisoning)
2. **Week 6**: Implement full defenses (fingerprinting, validation, PQ crypto)

The baseline accuracy establishes the **performance ceiling** that should be recovered after implementing defenses.

---

## Troubleshooting

### No Data Found
```
WARNING: No training data found!
```
**Solution**: Check that `Config.DATA_DIR` points to the correct location and has the expected structure.

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `BATCH_SIZE` in `config.py`:
```python
BATCH_SIZE = 8  # or even 4
```

### Low Accuracy
- Check data labels are correct
- Increase `NUM_ROUNDS` or `LOCAL_EPOCHS`
- Verify `LEARNING_RATE` is appropriate (try 0.0001 for fine-tuning)
- Ensure images are properly normalized

---

## Medical Imaging Considerations

### Privacy
- This implementation trains locally on each client's data
- Only model updates (gradients) are shared, not raw images
- Suitable for privacy-preserving medical AI

### Data Heterogeneity
- Different hospitals may have different ultrasound machines
- Scan quality and protocols may vary
- Non-IID split reflects real-world distribution

### Performance Expectations
- Medical imaging tasks are harder than MNIST
- 80-85% accuracy is good for federated learning on medical images
- Compare with centralized training baseline if possible

---

## References

- **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- **Non-IID Distribution**: Hsu et al., "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification"
- **Medical Imaging FL**: Rieke et al., "The Future of Digital Health with Federated Learning"
