# 📋 Quick Start Guide for Google Colab Notebooks

## 🎯 TL;DR - Fastest Way to Run

### 1. Upload to Google Drive (One Time)
```
MyDrive/fetal_plane_implementation/
├── week1_baseline/ (all .py files)
├── week2_attack/ (all .py files)
├── week6_full_defense/ (all .py files)
└── FETAL/ (CSV + Images folder)
```

### 2. Open Notebook in Colab
- Upload any of the `colab_week*.ipynb` files to Colab
- Or open directly from Drive

### 3. Update ONE Line (Section 1)
```python
DRIVE_BASE = '/content/drive/MyDrive/fetal_plane_implementation'  # ← CHANGE THIS
```

### 4. Run All Cells
- Runtime → Run all
- Enable GPU for 3x faster training

## 📊 Expected Runtimes

| Notebook | CPU | GPU (T4) | GPU (A100) |
|----------|-----|----------|------------|
| Week 1 Baseline | 60-90 min | 20-30 min | 10-15 min |
| Week 2 Attack | 60-90 min | 20-30 min | 10-15 min |
| Week 6 Defense | 70-100 min | 25-35 min | 12-18 min |

## 🎮 Enable GPU
1. Runtime → Change runtime type
2. Hardware accelerator → GPU (T4 or A100)
3. Save

## 📁 Files Created

All outputs save automatically to your Google Drive:

```
MyDrive/fetal_plane_implementation/
├── fetal_plane_baseline_model.pth
├── week1_baseline_results.pkl
├── week1_baseline_results.png
├── fetal_plane_poisoned_model.pth
├── week2_attack_results.pkl
├── week2_attack_results.png
├── baseline_vs_attack_comparison.png
├── fetal_plane_defended_model.pth
├── week6_defense_results.pkl
├── week6_defense_results.png
└── complete_comparison.png
```

## ✅ Troubleshooting One-Liners

**Can't find files?**
```python
# Add this cell to check paths
import os
print(os.listdir(DRIVE_BASE))
print(os.listdir(CODE_DIR))
print(os.listdir(DATA_DIR))
```

**Import errors?**
```python
# Verify Python path
import sys
print(sys.path)
print(f"CODE_DIR in path: {CODE_DIR in sys.path}")
```

**Out of memory?**
```python
# In config.py or override:
Config.BATCH_SIZE = 8  # Reduce from 16
```

## 🚀 Pro Workflow

1. **Run Week 1** → Get baseline accuracy (~70-80%)
2. **Run Week 2** → See attack degradation (~20-40%)
3. **Run Week 6** → See defense recovery (~60-75%)
4. **Compare all 3** → Week 6 auto-generates comparison plots

## 📞 Common Errors

| Error | Fix |
|-------|-----|
| "No such file" | Check DRIVE_BASE path |
| "ModuleNotFoundError" | Check CODE_DIR has .py files |
| "FileNotFoundError: CSV" | Check DATA_DIR path |
| "CUDA out of memory" | Reduce BATCH_SIZE or use smaller GPU |
| Slow training | Enable GPU runtime |

## 💡 Tips

- ✅ Mount Drive only once per session
- ✅ Run cells top-to-bottom sequentially
- ✅ Use GPU for 3x speedup
- ✅ Results auto-save to Drive
- ✅ Can resume if disconnected (reload results from .pkl)

## 🎓 Understanding Results

### Week 1 - Baseline
- **Initial**: ~5-15% (random)
- **Final**: ~70-80% (honest training)
- **Trend**: Steady improvement

### Week 2 - Attack
- **Initial**: ~5-15% (random)
- **Final**: ~20-40% (poisoned)
- **Trend**: Degraded/unstable

### Week 6 - Defense
- **Initial**: ~5-15% (random)
- **Final**: ~60-75% (defended)
- **Trend**: Recovered performance
- **Detection Rate**: ~70-90% malicious filtered

---

**That's it! Just upload files, update one path, and run. Happy training! 🚀**
