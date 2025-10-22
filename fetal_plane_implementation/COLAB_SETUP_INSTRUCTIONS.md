# 🚀 Google Colab Setup Instructions

## 📁 Step 1: Organize Your Google Drive

Create the following folder structure in your Google Drive:

```
MyDrive/
└── fetal_plane_implementation/
    ├── week1_baseline/
    │   ├── client.py
    │   ├── server.py
    │   ├── model.py
    │   ├── data_loader.py
    │   ├── config.py
    │   └── ... (all Python files)
    ├── week2_attack/
    │   ├── client.py
    │   ├── server.py
    │   ├── model.py
    │   ├── data_loader.py
    │   ├── config.py
    │   ├── attack.py
    │   └── ... (all Python files)
    ├── week6_full_defense/
    │   ├── client.py
    │   ├── server.py
    │   ├── model.py
    │   ├── data_loader.py
    │   ├── config.py
    │   ├── attack.py
    │   ├── defense_fingerprint_client.py
    │   ├── defense_validation.py
    │   ├── pq_crypto.py
    │   └── ... (all Python files)
    └── FETAL/
        ├── FETAL_PLANES_DB_data.csv
        └── Images/
            ├── image001.png
            ├── image002.png
            └── ... (all PNG files)
```

## 📥 Step 2: Upload Files to Google Drive

### Option A: Manual Upload
1. Go to [Google Drive](https://drive.google.com)
2. Create the folder structure shown above
3. Upload all files from your local folders

### Option B: Use Google Drive Desktop App
1. Install Google Drive for Desktop
2. Copy the entire `fetal_plane_implementation` folder to your Drive
3. Wait for sync to complete

## 📝 Step 3: Update Notebook Paths

In each Colab notebook, update **Section 1** with your paths:

```python
# ⚠️ CHANGE THESE PATHS TO MATCH YOUR GOOGLE DRIVE STRUCTURE
DRIVE_BASE = '/content/drive/MyDrive/fetal_plane_implementation'
CODE_DIR = f'{DRIVE_BASE}/week1_baseline'  # or week2_attack, week6_full_defense
DATA_DIR = f'{DRIVE_BASE}/FETAL'
```

### Example Custom Paths:

If your structure is different, adjust accordingly:

```python
# If your folder is in a subfolder:
DRIVE_BASE = '/content/drive/MyDrive/Projects/Capstone/fetal_plane_implementation'

# If you renamed the dataset folder:
DATA_DIR = f'{DRIVE_BASE}/FETAL_DATASET'

# If files are in root:
DRIVE_BASE = '/content/drive/MyDrive'
CODE_DIR = f'{DRIVE_BASE}/week1_baseline'
DATA_DIR = f'{DRIVE_BASE}/FETAL'
```

## 🔧 Step 4: Update config.py DATA_DIR

**Important**: The `config.py` files use relative paths (`../FETAL`) which won't work in Colab.

The notebooks automatically override this:

```python
from config import Config
Config.DATA_DIR = DATA_DIR  # Override with Google Drive path
```

**No need to edit config.py** - the notebook handles it!

## ▶️ Step 5: Run the Notebooks

### Week 1 - Baseline (Honest Clients)
1. Open `colab_week1_baseline.ipynb` in Google Colab
2. Run all cells sequentially
3. Expected time: 20-40 minutes (with GPU)

### Week 2 - Attack (Label Flipping)
1. Open `colab_week2_attack.ipynb` in Google Colab  
2. Run all cells sequentially
3. Expected time: 20-40 minutes (with GPU)

### Week 6 - Full Defense
1. Open `colab_week6_full_defense.ipynb` in Google Colab
2. Run all cells sequentially
3. Expected time: 25-45 minutes (with GPU)

## 🎮 Step 6: Enable GPU (Recommended)

For faster training:

1. In Colab: **Runtime** → **Change runtime type**
2. Hardware accelerator: **GPU** (preferably **T4** or **A100**)
3. Click **Save**

Training speed comparison:
- **CPU**: ~60-90 minutes per notebook
- **GPU (T4)**: ~20-30 minutes per notebook
- **GPU (A100)**: ~10-15 minutes per notebook

## 💾 Output Files

Results are automatically saved to your Google Drive:

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

## 🔍 Troubleshooting

### Error: "No such file or directory"
**Cause**: Paths are incorrect  
**Solution**: Double-check `DRIVE_BASE`, `CODE_DIR`, `DATA_DIR` match your Drive structure

### Error: "ModuleNotFoundError: No module named 'config'"
**Cause**: Python can't find your code files  
**Solution**: Verify files are uploaded and paths are correct

### Error: "FileNotFoundError: FETAL_PLANES_DB_data.csv"
**Cause**: Dataset path is wrong  
**Solution**: Check `DATA_DIR` points to folder containing CSV file

### Out of Memory Error
**Cause**: Not enough GPU/RAM  
**Solution**: 
- Reduce batch size in config.py (e.g., `BATCH_SIZE = 8`)
- Restart runtime and clear output
- Try different GPU in Runtime settings

### Slow Training
**Cause**: Running on CPU  
**Solution**: Enable GPU (see Step 6)

## 📊 Viewing Results

### View Saved Plots
Navigate to your Drive folder and open the PNG files:
- `week1_baseline_results.png`
- `week2_attack_results.png`
- `week6_defense_results.png`
- `complete_comparison.png`

### Load Saved Results
```python
import pickle

with open('/content/drive/MyDrive/fetal_plane_implementation/week1_baseline_results.pkl', 'rb') as f:
    baseline_results = pickle.load(f)

print(f"Final accuracy: {baseline_results['accuracies'][-1]:.2f}%")
```

## ⚡ Pro Tips

1. **Mount Drive Once**: After mounting, Drive stays mounted for the session
2. **Run Sequentially**: Run cells in order (top to bottom)
3. **Save Frequently**: Results auto-save to Drive, but you can also use **File → Save**
4. **Use GPU**: Always enable GPU for faster training
5. **Monitor Progress**: Watch the output - training shows live progress
6. **Comparison**: Run all 3 notebooks to see the full baseline → attack → defense story

## 📞 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Verify all paths in Section 1
3. Ensure all files are uploaded to Drive
4. Try restarting the Colab runtime
5. Re-run from the beginning

## ✅ Quick Checklist

- [ ] Uploaded all code files to Google Drive
- [ ] Uploaded FETAL dataset to Google Drive
- [ ] Updated paths in notebook Section 1
- [ ] Enabled GPU in Colab runtime settings
- [ ] Mounted Google Drive successfully
- [ ] All imports working (no errors)
- [ ] Data loading successfully
- [ ] Training started

**Happy Training! 🚀**
