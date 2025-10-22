# Fetal Plane Implementation - Quick Start Guide

This folder contains federated learning implementations for fetal plane classification using ultrasound images.

## Dataset Structure

The implementation expects the FETAL dataset in this structure:

```
fetal_plane_implementation/
├── FETAL/
│   ├── FETAL_PLANES_DB_data.csv (metadata)
│   └── Images/
│       ├── Patient00001_Plane1_1_of_15.png
│       ├── Patient00001_Plane1_2_of_15.png
│       └── ... (12,402 images)
├── week1_baseline/
├── week2_attack/
└── week6_full_defense/
```

## Dataset Information

- **Total Images**: 12,402 ultrasound images
- **Number of Classes**: 6 fetal plane categories
- **Classes**:
  - 0: Fetal abdomen
  - 1: Fetal brain
  - 2: Fetal femur
  - 3: Fetal thorax
  - 4: Maternal cervix
  - 5: Other
- **Image Format**: Grayscale PNG images (converted to RGB for ResNet18)
- **Train/Test Split**: Provided in CSV (Train column: 1=train, 0=test)

## Quick Test

Before running experiments, test that the data loader works:

```bash
cd fetal_plane_implementation
python test_dataloader.py
```

This will:
- Load the CSV metadata
- Verify images exist
- Show class distribution
- Load a sample image

## Running Experiments

### Week 1: Baseline (No Attack)

```bash
cd week1_baseline
python main.py
```

**What it does:**
- Trains ResNet18 on Non-IID fetal plane data
- 10 clients with Dirichlet distribution (alpha=0.5)
- 10 federated rounds
- FedAvg aggregation

### Week 2: Label Flipping Attack

```bash
cd week2_attack
python main.py
```

**What it does:**
- Same as Week 1 BUT adds malicious clients
- 30% of clients perform label flipping
- Attack targets: Fetal brain ↔ Fetal thorax
- Shows degradation of global model

### Week 6: Full Defense

```bash
cd week6_full_defense
python main.py
```

**What it does:**
- Applies client-side fingerprinting (512-dim)
- Validation-based filtering (threshold=0.15)
- Post-quantum crypto (Kyber512 + Dilithium2)
- Rejects malicious updates

## Configuration

Each week has a `config.py` file with adjustable parameters:

```python
# Key settings
NUM_CLIENTS = 10
NUM_ROUNDS = 10
LOCAL_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DIRICHLET_ALPHA = 0.5  # Non-IID level (lower = more heterogeneous)

# Attack settings (week2_attack only)
ATTACK_TYPE = "label_flip"
ATTACK_RATIO = 0.3  # 30% malicious clients

# Defense settings (week6_full_defense only)
FINGERPRINT_DIM = 512
VALIDATION_THRESHOLD = 0.15
```

## Model Architecture

- **Base Model**: ResNet18 (pretrained on ImageNet)
- **Input**: 224×224 RGB images
- **Output**: 6 classes (fetal planes)
- **Modifications**:
  - First conv layer accepts grayscale (converted to RGB)
  - Final FC layer: 512 → 6 classes

## Results

Results are saved in each week's folder:

- `results.txt`: Final accuracy and metrics
- Training logs: Printed to console

Expected results (approximate):
- **Week 1 (Baseline)**: ~75-80% test accuracy
- **Week 2 (Attack)**: ~40-50% test accuracy (degraded)
- **Week 6 (Defense)**: ~70-75% test accuracy (recovered)

## Requirements

Install dependencies:

```bash
cd week1_baseline  # or week2_attack, week6_full_defense
pip install -r requirements.txt
```

Key packages:
- torch >= 2.0
- torchvision
- numpy
- pandas
- pillow
- liboqs-python (for week6_full_defense)

## Troubleshooting

### "Warning: CSV does not exist"

- Check that `FETAL_PLANES_DB_data.csv` is in `../FETAL/`
- Verify `Config.DATA_DIR` points to correct location

### "No images loaded"

- Check that `Images/` folder exists in `../FETAL/`
- Verify images are PNG format
- Run `ls FETAL/Images/ | wc -l` to count images (should be 12,402)

### Low accuracy

- Check data distribution with `python test_dataloader.py`
- Try adjusting `DIRICHLET_ALPHA` (higher = more IID)
- Increase `NUM_ROUNDS` or `LOCAL_EPOCHS`
- Verify images are loading correctly (check sample)

### Out of memory

- Reduce `BATCH_SIZE` in `config.py`
- Reduce `NUM_CLIENTS` or `CLIENTS_PER_ROUND`

## Differences from Non-IID MNIST Implementation

This implementation is adapted from `non_iid_implementation` with these changes:

1. **Dataset**: CSV-based fetal plane images instead of folder-based MNIST
2. **Model**: ResNet18 instead of SimpleCNN
3. **Image format**: Grayscale ultrasound (converted to RGB) instead of MNIST digits
4. **Classes**: 6 fetal plane categories instead of 10 digits
5. **Data loader**: Reads from CSV metadata instead of folder structure
6. **Normalization**: ImageNet stats instead of MNIST stats

## Next Steps

After successfully running all three weeks:

1. **Analyze Results**: Compare test accuracies across weeks
2. **Visualize**: Plot training curves, class-wise accuracy
3. **Tune Hyperparameters**: Adjust alpha, learning rate, etc.
4. **Experiment**: Try different attack ratios, defense thresholds
5. **Document**: Record findings in `RESULTS.md`

## References

- Original MNIST implementation: `../non_iid_implementation/`
- Model details: `week1_baseline/model.py`
- Data loader: `week1_baseline/data_loader.py`
- Configuration: `week1_baseline/config.py`
