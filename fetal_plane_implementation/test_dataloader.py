"""
Quick test to verify the FETAL dataset loader is working correctly
"""
import sys
import os

# Change to week1_baseline directory so relative paths work
week1_dir = os.path.join(os.path.dirname(__file__), 'week1_baseline')
os.chdir(week1_dir)

# Add week1_baseline to path
sys.path.insert(0, week1_dir)

from data_loader import load_fetal_plane_data
from config import Config

def test_dataloader():
    print("="*60)
    print("Testing FETAL Plane Dataset Loader")
    print("="*60)
    
    print(f"\nDataset directory: {Config.DATA_DIR}")
    print(f"Number of clients: {Config.NUM_CLIENTS}")
    print(f"Number of classes: {Config.NUM_CLASSES}")
    print(f"Dirichlet alpha: {Config.DIRICHLET_ALPHA}")
    
    try:
        # Load data
        print("\nLoading data...")
        train_dataset, test_dataset = load_fetal_plane_data()
        
        print(f"\nTrain dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        if len(train_dataset) > 0:
            # Test loading a sample
            print("\nLoading a sample from training set...")
            img, label = train_dataset[0]
            print(f"  Image shape: {img.shape}")
            print(f"  Label: {label}")
            print(f"  Label type: {type(label)}")
            
            # Class names
            class_names = [
                "Fetal abdomen",
                "Fetal brain", 
                "Fetal femur",
                "Fetal thorax",
                "Maternal cervix",
                "Other"
            ]
            
            print(f"  Class name: {class_names[label]}")
            
            print("\n✓ Data loader is working correctly!")
            print("\nYou can now run the federated learning experiments:")
            print("  cd week1_baseline")
            print("  python main.py")
            
        else:
            print("\n✗ No data loaded. Please check:")
            print("  1. FETAL_PLANES_DB_data.csv exists in ../FETAL/")
            print("  2. Images/ folder exists in ../FETAL/")
            print("  3. Config.DATA_DIR points to correct location")
            
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_dataloader()
    sys.exit(0 if success else 1)
