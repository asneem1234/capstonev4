# Experimental Protocol and Methodology

## Table of Contents
1. [Research Questions](#research-questions)
2. [Experimental Design](#experimental-design)
3. [Data and Setup](#data-and-setup)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Statistical Analysis](#statistical-analysis)
6. [Reproducibility](#reproducibility)

---

## Research Questions

### Primary Research Questions

**RQ1**: Can quantum neural networks maintain effectiveness in federated learning settings?
- **Hypothesis**: Hybrid quantum-classical models achieve comparable accuracy to classical CNNs
- **Metrics**: Test accuracy, convergence rate, parameter efficiency

**RQ2**: How effective is norm-based filtering against gradient ascent attacks in quantum federated learning?
- **Hypothesis**: Norm filtering recovers >90% of baseline accuracy under 25% malicious clients
- **Metrics**: Accuracy recovery rate, detection precision, detection recall

**RQ3**: What is the computational overhead of quantum federated learning with defense?
- **Hypothesis**: Overhead is acceptable (<20%) for practical deployment
- **Metrics**: Training time, communication cost, memory usage

**RQ4**: How does data heterogeneity affect quantum federated learning robustness?
- **Hypothesis**: Non-IID data increases vulnerability but defense remains effective
- **Metrics**: Accuracy under varying Dirichlet α, defense effectiveness across distributions

### Secondary Research Questions

**RQ5**: How does quantum circuit depth affect model performance and attack resilience?
- **Variables**: Number of qubits (4, 8), number of layers (2, 4, 6)

**RQ6**: What is the optimal threshold multiplier for norm-based filtering?
- **Variables**: λ ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0}

**RQ7**: How does the attack scale factor affect defense effectiveness?
- **Variables**: Scale factor ∈ {5, 10, 15, 20}

---

## Experimental Design

### Experimental Conditions

**Three Main Conditions**:

```
┌──────────────────────────────────────────────────────────────┐
│ Condition 1: BASELINE (No Attack, No Defense)               │
├──────────────────────────────────────────────────────────────┤
│ - All clients honest (100%)                                  │
│ - No Byzantine attacks                                        │
│ - Standard FedAvg aggregation                                │
│ - Purpose: Establish upper bound performance                 │
│ - Expected Accuracy: 85-90%                                  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Condition 2: ATTACK (Attack, No Defense)                    │
├──────────────────────────────────────────────────────────────┤
│ - 25% malicious clients (1 out of 4)                        │
│ - Gradient ascent attack (scale_factor=10.0)                │
│ - No defense mechanisms                                       │
│ - Purpose: Demonstrate attack effectiveness                  │
│ - Expected Accuracy: 10-15% (collapse)                       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Condition 3: DEFENSE (Attack + Defense)                     │
├──────────────────────────────────────────────────────────────┤
│ - 25% malicious clients (1 out of 4)                        │
│ - Gradient ascent attack (scale_factor=10.0)                │
│ - Norm-based filtering defense (threshold=median×3.0)        │
│ - Purpose: Evaluate defense effectiveness                    │
│ - Expected Accuracy: 80-85% (recovery)                       │
└──────────────────────────────────────────────────────────────┘
```

### Experimental Variables

**Independent Variables**:

| Variable | Type | Levels | Control |
|----------|------|--------|---------|
| Attack Presence | Categorical | {No, Yes} | Experimental condition |
| Defense Presence | Categorical | {No, Yes} | Experimental condition |
| Malicious Percentage | Continuous | 0%, 25%, 50% | Fixed at 25% (primary) |
| Scale Factor | Continuous | 5, 10, 15, 20 | Fixed at 10 (primary) |
| Threshold Multiplier | Continuous | 1.5 - 4.0 | Fixed at 3.0 (primary) |
| Dirichlet Alpha | Continuous | 0.5, 1.0, 1.5 | Fixed at 1.5 (primary) |
| Number of Qubits | Discrete | 4, 8 | Fixed at 4 (primary) |
| Number of Layers | Discrete | 2, 4, 6 | Fixed at 4 (primary) |

**Dependent Variables**:

| Variable | Type | Measurement |
|----------|------|-------------|
| Test Accuracy | Continuous | % correct on test set |
| Test Loss | Continuous | Cross-entropy loss |
| Training Time | Continuous | Seconds per round |
| Update Norm | Continuous | L2 norm of parameter updates |
| Detection Precision | Continuous | TP / (TP + FP) |
| Detection Recall | Continuous | TP / (TP + FN) |
| Convergence Round | Discrete | Round reaching 80% accuracy |

**Control Variables** (kept constant):

| Variable | Value | Rationale |
|----------|-------|-----------|
| Dataset | MNIST | Standard benchmark |
| Number of Clients | 4 | Computational feasibility |
| Clients Per Round | 4 | All clients participate |
| Number of Rounds | 5 | Sufficient for convergence |
| Local Epochs | 2 | Balance between local/global |
| Batch Size | 128 | Standard for MNIST |
| Learning Rate | 0.01 | Empirically determined |
| Optimizer | Adam | State-of-the-art |
| Random Seed | 42 | Reproducibility |

### Experimental Protocol

**Single Experiment Run**:

```
Step 1: Environment Setup
    ├── Set random seeds (42)
    ├── Initialize quantum device (PennyLane default.qubit)
    ├── Configure logging
    └── Clear GPU cache

Step 2: Data Preparation
    ├── Load MNIST dataset
    ├── Apply normalization (μ=0.1307, σ=0.3081)
    ├── Split using Dirichlet (α=1.5)
    └── Create dataloaders (batch_size=128)

Step 3: Model Initialization
    ├── Create HybridQuantumNet
    │   ├── Classical feature extractor (CNN)
    │   ├── Quantum circuit (4 qubits, 4 layers)
    │   └── Classical classifier
    ├── Initialize parameters (Xavier/He initialization)
    └── Move to device (CPU/GPU)

Step 4: Federated Learning Simulation
    For round r = 1 to 5:
        ├── Server broadcasts global model
        │
        ├── For each client c = 1 to 4:
        │   ├── Download global model
        │   ├── Train locally (2 epochs)
        │   ├── If malicious: Apply attack
        │   ├── Calculate update norm
        │   └── Upload update + metrics
        │
        ├── Server collects updates
        │
        ├── If defense enabled:
        │   ├── Calculate norm statistics
        │   ├── Apply norm filtering
        │   ├── Log defense metrics
        │   └── Reject outliers
        │
        ├── Aggregate accepted updates (FedAvg)
        │
        └── Evaluate global model on test set

Step 5: Results Collection
    ├── Save accuracy trajectory
    ├── Save loss trajectory
    ├── Save defense statistics
    ├── Save timing information
    └── Save final model

Step 6: Statistical Analysis
    ├── Calculate mean and std dev
    ├── Perform significance tests
    └── Generate visualizations
```

### Sample Size and Replication

**Replication Strategy**:

```python
# Run each condition 5 times with different random seeds
RANDOM_SEEDS = [42, 123, 456, 789, 1024]

for seed in RANDOM_SEEDS:
    for condition in ['baseline', 'attack', 'defense']:
        run_experiment(condition, seed)
```

**Sample Size Calculation**:

```
For statistical power analysis:
- Effect size (Cohen's d): 2.0 (large, accuracy difference ~70%)
- Significance level (α): 0.05
- Power (1-β): 0.95
- Required sample size: n = 5 per condition

Total experiments: 3 conditions × 5 replications = 15 runs
```

---

## Data and Setup

### Dataset: MNIST

**Properties**:
- **Task**: Handwritten digit classification
- **Classes**: 10 (digits 0-9)
- **Training Samples**: 60,000
- **Test Samples**: 10,000
- **Image Size**: 28×28 grayscale
- **Normalization**: μ=0.1307, σ=0.3081

**Why MNIST**:
1. Standard benchmark (easy comparison)
2. Manageable size for quantum simulation
3. Well-understood convergence behavior
4. Fast training (rapid experimentation)

### Data Distribution

**Non-IID Split using Dirichlet**:

```python
def split_non_iid_dirichlet(dataset, num_clients=4, alpha=1.5):
    """
    Dirichlet(α) controls heterogeneity:
    - α → 0: Highly non-IID (each client has 1-2 classes)
    - α = 0.5: Very non-IID
    - α = 1.5: Moderately non-IID (our choice)
    - α → ∞: Approaches IID
    """
    num_classes = 10
    labels = np.array(dataset.targets)
    
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
    
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Split according to proportions
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)
        
        for client_id in range(num_clients):
            client_indices[client_id].extend(split_indices[client_id].tolist())
    
    return client_indices
```

**Example Split (α=1.5, 4 clients)**:

```
Client 0: Class Distribution
  0: ████████ (800)
  1: ██ (200)
  2: ██████ (600)
  3: ███ (300)
  4: █████ (500)
  5: ██ (200)
  6: ███████ (700)
  7: ██ (200)
  8: ████ (400)
  9: ███ (300)
  Total: 4,200 samples

Client 1: Class Distribution
  0: ██ (200)
  1: ████████ (800)
  2: ███ (300)
  3: ██████ (600)
  4: ██ (200)
  5: ████████ (800)
  6: ██ (200)
  7: ███████ (700)
  8: ███ (300)
  9: ████ (400)
  Total: 4,500 samples

[Similar for Clients 2 and 3]
```

### Hardware and Software Setup

**Hardware**:
```
Primary Development:
- CPU: AMD Ryzen 9 / Intel i7/i9
- RAM: 16GB minimum
- GPU: Optional (NVIDIA GTX 1660+)
- Storage: 10GB available

Cloud Alternative (Google Colab):
- CPU: Intel Xeon (2 cores)
- RAM: 12GB
- GPU: Tesla T4 (optional)
```

**Software**:
```python
# Core Dependencies
python==3.8+
pennylane==0.30+      # Quantum ML
torch==2.0+           # Deep learning
torchvision==0.15+    # MNIST dataset
flwr==1.4+            # Federated learning
numpy==1.23+          # Numerics
scikit-learn==1.2+    # Metrics

# Optional
matplotlib==3.6+      # Visualization
pandas==1.5+          # Data analysis
jupyter==1.0+         # Notebooks
```

**Installation**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pennylane torch torchvision flwr numpy scikit-learn matplotlib
```

### Configuration Files

**config.py** (Primary Configuration):
```python
# Federated Learning
NUM_CLIENTS = 4
CLIENTS_PER_ROUND = 4
NUM_ROUNDS = 5

# Data
NON_IID = True
DIRICHLET_ALPHA = 1.5

# Training
BATCH_SIZE = 128
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.01

# Quantum
N_QUBITS = 4
N_LAYERS = 4

# Attack
ATTACK_ENABLED = True  # False for baseline, True for attack/defense
MALICIOUS_PERCENTAGE = 0.25
ATTACK_TYPE = "gradient_ascent"
SCALE_FACTOR = 10.0

# Defense
DEFENSE_ENABLED = True  # False for attack, True for defense
DEFENSE_TYPE = "norm_filtering"
NORM_THRESHOLD_MULTIPLIER = 3.0

# Misc
SEED = 42
VERBOSE = True
```

---

## Evaluation Metrics

### Model Performance Metrics

#### 1. **Accuracy**

**Definition**:
```
Accuracy = (Number of Correct Predictions) / (Total Predictions)
         = (TP + TN) / (TP + TN + FP + FN)
```

**Measurement**:
```python
def evaluate_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy
```

**Interpretation**:
- 10%: Random guessing (baseline)
- 85-90%: Good performance (target)
- <20%: Model collapse (attack success)

#### 2. **Loss**

**Definition** (Cross-Entropy):
```
L = -(1/N) Σᵢ₌₁ᴺ Σⱼ₌₁ᶜ yᵢⱼ log(ŷᵢⱼ)

where:
- N: number of samples
- C: number of classes (10)
- yᵢⱼ: true label (one-hot)
- ŷᵢⱼ: predicted probability
```

**Measurement**:
```python
def evaluate_loss(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    avg_loss = total_loss / total_samples
    return avg_loss
```

**Interpretation**:
- 0.3-0.5: Good convergence
- 2.3: Random (log(10))
- >2.0: Poor performance

### Defense Performance Metrics

#### 1. **Detection Precision**

**Definition**:
```
Precision = TP / (TP + FP)
          = (Correctly Detected Malicious) / (All Detected as Malicious)
```

**Target**: >0.95 (few false positives)

#### 2. **Detection Recall**

**Definition**:
```
Recall = TP / (TP + FN)
       = (Correctly Detected Malicious) / (All Actually Malicious)
```

**Target**: >0.90 (catch most malicious clients)

#### 3. **F1-Score**

**Definition**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Target**: >0.92 (balanced performance)

#### 4. **Accuracy Recovery Rate**

**Definition**:
```
Recovery Rate = Accuracy(With Defense) / Accuracy(Baseline)
```

**Target**: >0.90 (recover 90%+ of baseline)

### Efficiency Metrics

#### 1. **Training Time**

**Measurement**:
```python
start_time = time.time()
# Run federated learning
end_time = time.time()
training_time = end_time - start_time
```

**Components**:
- Per-round time
- Per-client time
- Total time

#### 2. **Communication Cost**

**Measurement**:
```python
def calculate_communication_cost(parameters):
    total_bytes = 0
    for param in parameters:
        total_bytes += param.nbytes
    return total_bytes
```

**Metrics**:
- Bytes per round
- Total bytes transmitted
- Overhead from defense/crypto

#### 3. **Update Norm**

**Definition**:
```
||Δθ||₂ = √(Σᵢ (θᵢ_new - θᵢ_old)²)
```

**Usage**:
- Honest clients: Small norm (~0.1)
- Malicious clients: Large norm (~1.0+)
- Defense threshold setting

---

## Statistical Analysis

### Descriptive Statistics

**For Each Metric**:
```python
# Calculate across 5 replications
results = []
for seed in [42, 123, 456, 789, 1024]:
    accuracy = run_experiment(seed)
    results.append(accuracy)

mean_accuracy = np.mean(results)
std_accuracy = np.std(results)
min_accuracy = np.min(results)
max_accuracy = np.max(results)
median_accuracy = np.median(results)

print(f"Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
print(f"Range: [{min_accuracy:.2f}%, {max_accuracy:.2f}%]")
```

### Inferential Statistics

#### 1. **Paired t-test** (Compare conditions)

**Hypothesis**:
```
H₀: μ_defense = μ_attack (no difference)
H₁: μ_defense > μ_attack (defense improves accuracy)
```

**Test**:
```python
from scipy import stats

attack_results = [11.2, 10.5, 12.1, 11.8, 10.9]
defense_results = [84.3, 85.1, 83.7, 84.9, 85.5]

t_stat, p_value = stats.ttest_rel(defense_results, attack_results)

if p_value < 0.05:
    print(f"Significant improvement (p={p_value:.4f})")
else:
    print(f"No significant difference (p={p_value:.4f})")
```

#### 2. **Effect Size** (Cohen's d)

**Definition**:
```
d = (μ₁ - μ₂) / σ_pooled

where σ_pooled = √((σ₁² + σ₂²) / 2)
```

**Interpretation**:
- d < 0.2: Small effect
- 0.2 ≤ d < 0.8: Medium effect
- d ≥ 0.8: Large effect

**Calculation**:
```python
def cohens_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    return (mean1 - mean2) / pooled_std

d = cohens_d(defense_results, attack_results)
print(f"Effect size (Cohen's d): {d:.2f}")
```

#### 3. **Confidence Intervals**

**95% CI for mean**:
```python
from scipy import stats

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - margin, mean + margin)

ci = confidence_interval(defense_results)
print(f"95% CI: [{ci[0]:.2f}%, {ci[1]:.2f}%]")
```

### Visualization

**Key Plots**:

1. **Accuracy Trajectory**:
```python
plt.figure(figsize=(10, 6))
for condition in ['baseline', 'attack', 'defense']:
    plt.plot(rounds, accuracies[condition], label=condition)
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('Federated Learning Performance')
plt.legend()
plt.grid(True)
```

2. **Box Plot** (Compare conditions):
```python
plt.boxplot([baseline_results, attack_results, defense_results],
            labels=['Baseline', 'Attack', 'Defense'])
plt.ylabel('Final Accuracy (%)')
plt.title('Accuracy Comparison')
```

3. **Norm Distribution**:
```python
plt.hist(honest_norms, alpha=0.5, label='Honest', bins=20)
plt.hist(malicious_norms, alpha=0.5, label='Malicious', bins=20)
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Update Norm')
plt.ylabel('Frequency')
plt.legend()
```

---

## Reproducibility

### Reproducibility Checklist

**✓ Random Seed Control**:
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**✓ Version Pinning**:
```bash
# requirements.txt with exact versions
pennylane==0.30.0
torch==2.0.1
torchvision==0.15.2
flwr==1.4.0
numpy==1.23.5
scikit-learn==1.2.2
```

**✓ Configuration Documentation**:
```python
# Save configuration with results
config_dict = {
    'num_clients': config.NUM_CLIENTS,
    'num_rounds': config.NUM_ROUNDS,
    'learning_rate': config.LEARNING_RATE,
    # ... all parameters
}

with open('results/config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)
```

**✓ Data Split Saving**:
```python
# Save client indices for reproducibility
with open('data/client_splits.pkl', 'wb') as f:
    pickle.dump(client_indices, f)
```

**✓ Model Checkpoints**:
```python
# Save model at each round
torch.save({
    'round': round_num,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': accuracy,
    'loss': loss,
}, f'checkpoints/model_round_{round_num}.pth')
```

### Artifact Preservation

**Directory Structure**:
```
experiments/
├── run_20250105_143022/          # Timestamp
│   ├── config.json                # Configuration
│   ├── results.json               # Final results
│   ├── logs/
│   │   ├── round_1.log
│   │   ├── round_2.log
│   │   └── ...
│   ├── checkpoints/
│   │   ├── model_round_1.pth
│   │   ├── model_round_2.pth
│   │   └── ...
│   ├── figures/
│   │   ├── accuracy_trajectory.png
│   │   ├── loss_trajectory.png
│   │   └── norm_distribution.png
│   └── data/
│       └── client_splits.pkl
└── ...
```

### Open Science Practices

**✓ Code Repository**:
- GitHub: Public repository with full code
- README: Clear instructions
- Documentation: Comprehensive docs (this file!)
- License: MIT/Apache 2.0

**✓ Data Sharing**:
- Dataset: MNIST (publicly available)
- Splits: Save and share Dirichlet splits
- Results: CSV/JSON files

**✓ Pre-registration**:
- Hypotheses stated before experiments
- Analysis plan documented
- No p-hacking or cherry-picking

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Status**: Research Protocol
