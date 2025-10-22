# Fetal Plane Implementation

This folder contains the implementation for fetal plane classification using federated learning.

## ⚠️ Dataset Required

**This implementation requires a fetal plane ultrasound dataset that is not included.**

The code is ready to use but needs you to provide the dataset. See "Dataset Setup" below.

## Overview

This implementation applies federated learning techniques to fetal ultrasound plane classification with three progressive weeks:

- **Week 1**: Baseline federated learning (no attacks)
- **Week 2**: Label flipping attacks (demonstrate vulnerability)
- **Week 6**: Full defense stack (fingerprinting + validation + PQ crypto)

## Structure

```
fetal_plane_implementation/
├── README.md                    # This file
├── week1_baseline/              # Baseline implementation
│   ├── config.py               # Configuration
│   ├── model.py                # ResNet18 for fetal planes
│   ├── data_loader.py          # Custom dataset + Non-IID split
│   ├── client.py               # Client training
│   ├── server.py               # FedAvg aggregation
│   ├── main.py                 # Training loop
│   ├── requirements.txt        # Dependencies
│   └── README.md               # Detailed documentation
├── week2_attack/                # With label flipping attack
│   └── [same structure + attack.py]
└── week6_full_defense/          # With full defense
    └── [same structure + defense modules]
```

## Dataset Setup

### Option 1: Use Your Own Fetal Plane Dataset

Create the following directory structure:

```
fetal_plane_implementation/
└── week1_baseline/
    └── data/
        └── fetal_planes/
            ├── train/
            │   ├── class_0/        # Trans-thalamic plane
            │   │   ├── img001.png
            │   │   ├── img002.png
            │   │   └── ...
            │   ├── class_1/        # Trans-cerebellum plane
            │   │   └── ...
            │   ├── class_2/        # Trans-ventricular plane
            │   │   └── ...
            │   ├── class_3/        # Maternal cervix
            │   │   └── ...
            │   ├── class_4/        # Femur
            │   │   └── ...
            │   └── class_5/        # Others
            │       └── ...
            └── test/
                ├── class_0/
                │   └── ...
                └── ...
```

**Supported image formats**: `.png`, `.jpg`, `.jpeg`, `.bmp`

### Option 2: Update Config for Different Location

If your dataset is elsewhere, update `config.py`:

```python
DATA_DIR = "/path/to/your/fetal_planes_dataset"
```

### Option 3: Test with MNIST Instead

If you don't have a fetal plane dataset, use the working MNIST implementation:

```bash
cd ../non_iid_implementation/week1_baseline
python main.py
```

The MNIST implementation has the same structure and demonstrates all the same concepts.

## Dataset Requirements

### Image Specifications
- **Format**: Grayscale (1 channel) or RGB (3 channels)
- **Size**: Any size (will be resized to 224×224)
- **Classes**: 6 fetal plane types (adjustable in `config.py`)

### Recommended Class Labels
1. **Class 0**: Trans-thalamic plane
2. **Class 1**: Trans-cerebellum plane  
3. **Class 2**: Trans-ventricular plane
4. **Class 3**: Maternal cervix
5. **Class 4**: Femur
6. **Class 5**: Others/Background

**Note**: You can use different class names or numbers. Just update `NUM_CLASSES` in `config.py`.

## Quick Start (Once Dataset is Ready)

### 1. Install Dependencies

```bash
cd week1_baseline
pip install -r requirements.txt
```

### 2. Verify Dataset

```bash
python -c "from data_loader import load_fetal_plane_data; train, test = load_fetal_plane_data(); print(f'Train: {len(train)}, Test: {len(test)}')"
```

### 3. Run Baseline

```bash
python main.py
```

### 4. Run Attack Scenario

```bash
cd ../week2_attack
python main.py
```

### 5. Run Full Defense

```bash
cd ../week6_full_defense
python main.py
```

## Configuration

In each week's `config.py`:

```python
# Dataset
DATA_DIR = "./data/fetal_planes"    # Update this path
NUM_CLASSES = 6                      # Number of plane classes

# Federated Learning
NUM_CLIENTS = 10                     # Hospitals/clinics
NUM_ROUNDS = 10                      # Training rounds
LOCAL_EPOCHS = 5                     # Local training epochs
BATCH_SIZE = 16                      # Batch size

# Model
MODEL_TYPE = "resnet18"              # Or resnet50, efficientnet_b0
IMAGE_SIZE = 224                     # Input image size
PRETRAINED = True                    # Use ImageNet weights
```

## Public Fetal Plane Datasets

If you need a dataset, consider these public sources:

1. **FETAL_PLANES_DB** (Kaggle)
   - Multiple fetal ultrasound plane types
   - ~12,000 images

2. **Fetal Plane Ultrasound Dataset** 
   - Various anatomical planes
   - Research-ready annotations

**Note**: Always check dataset licenses and comply with data usage policies.

## Features

### Week 1: Baseline
- ✅ Non-IID data distribution (Dirichlet)
- ✅ ResNet18 architecture
- ✅ Medical image augmentation
- ✅ All clients honest
- ✅ Expected: ~80-85% accuracy

### Week 2: Attack
- ⚠️ 20% malicious clients
- ⚠️ Label flipping attack
- ⚠️ No defenses
- ⚠️ Expected: Accuracy drops to ~30%

### Week 6: Full Defense
- 🛡️ Client-side fingerprinting
- 🛡️ Validation-based filtering
- 🛡️ Post-quantum cryptography
- 🛡️ Expected: ~78-82% accuracy (defense recovers performance!)

## Medical Imaging Considerations

### Privacy
- Raw images never leave local clients (hospitals)
- Only model updates are transmitted
- Suitable for HIPAA/GDPR compliance

### Data Heterogeneity
- Different hospitals have different equipment
- Scan quality varies by institution
- Non-IID split reflects real-world distribution

### Safety
- Critical medical application
- Requires high confidence in predictions
- Defense mechanisms ensure robustness

## Documentation

Each week's folder has a detailed README with:
- Setup instructions
- Expected output
- Configuration options
- Troubleshooting guide
- Medical imaging best practices

## Troubleshooting

### "No training data found"
→ **Solution**: Provide dataset or use MNIST implementation

### "CUDA out of memory"
→ **Solution**: Reduce `BATCH_SIZE` in `config.py`

### "Model accuracy is low"
→ **Solution**: Increase `NUM_ROUNDS` or check data labels

### "Import errors"
→ **Solution**: `pip install -r requirements.txt`

## Next Steps

1. **Obtain fetal plane dataset** (or use your own medical images)
2. **Place in correct directory structure**
3. **Run week1_baseline** to establish baseline
4. **Run week2_attack** to see vulnerability
5. **Run week6_full_defense** to see defense effectiveness

## References

- **Federated Learning**: McMahan et al., "Communication-Efficient Learning"
- **Medical FL**: Rieke et al., "The Future of Digital Health with Federated Learning"
- **Fetal Ultrasound**: Burgos-Artizzu et al., "Evaluation of Deep Learning for Automatic Fetal Plane Detection"

## Support

For issues or questions:
1. Check the week-specific README
2. Verify dataset structure and paths
3. Test with MNIST implementation first
4. Review error messages (they provide clear guidance)
