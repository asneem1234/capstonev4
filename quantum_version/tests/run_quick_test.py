"""
Quick Start: Run This First!
=============================
This script validates your test environment and runs a minimal experiment
"""

import sys
import os

print("="*60)
print("QUANTUM FL TEST ENVIRONMENT CHECK")
print("="*60)
print()

# Check Python version
print("✓ Python version:", sys.version.split()[0])

# Check required packages
required_packages = [
    'numpy',
    'torch', 
    'scipy',
    'pandas',
    'matplotlib',
    'pennylane'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} NOT installed")
        missing_packages.append(package)

print()

if missing_packages:
    print("❌ Missing packages detected!")
    print(f"   Install with: pip install {' '.join(missing_packages)}")
    sys.exit(1)

print("✅ All required packages installed!")
print()

# Check directory structure
print("Checking directory structure...")
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
required_dirs = [
    'tests',
    'tests/results',
    'tests/results/plots'
]

for dir_name in required_dirs:
    dir_path = os.path.join(base_dir, dir_name)
    if os.path.exists(dir_path):
        print(f"✓ {dir_name}/ exists")
    else:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ {dir_name}/ created")

print()
print("="*60)
print("RUNNING QUICK TEST")
print("="*60)
print()

# Run spectral defense test
print("Running spectral analysis test...")
print("This will take ~30 seconds...")
print()

try:
    from test_spectral_defense import run_spectral_test
    stats, df = run_spectral_test()
    
    print()
    print("="*60)
    print("🎉 SUCCESS!")
    print("="*60)
    print()
    print("Your test environment is ready!")
    print()
    print("Next steps:")
    print("1. Review tests/results/spectral_analysis.csv")
    print("2. Check tests/results/spectral_analysis.png")
    print("3. Read tests/PAPER_WRITING_GUIDE.md")
    print("4. Run full experiments when ready")
    print()
    
except Exception as e:
    print("❌ Test failed!")
    print(f"Error: {e}")
    print()
    print("Troubleshooting:")
    print("1. Check if quantum_version code is in the right location")
    print("2. Adjust imports in test_spectral_defense.py")
    print("3. See README.md for detailed instructions")
    import traceback
    traceback.print_exc()
