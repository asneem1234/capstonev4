"""
Quick Test Runner for Gradient Attack Testing
Runs a quick verification before full training
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

def quick_check():
    """Perform quick checks before running"""
    print("\n" + "="*70)
    print("GRADIENT ATTACK TEST - PRE-FLIGHT CHECK")
    print("="*70)
    
    checks_passed = True
    
    # Check 1: Configuration
    print("\n[1/5] Checking configuration...")
    if config.NUM_CLIENTS == 5:
        print("  ✓ 5 clients configured")
    else:
        print(f"  ✗ Expected 5 clients, got {config.NUM_CLIENTS}")
        checks_passed = False
    
    if config.TOTAL_MALICIOUS == 2:
        print("  ✓ 2 malicious clients configured")
    else:
        print(f"  ✗ Expected 2 malicious, got {config.TOTAL_MALICIOUS}")
        checks_passed = False
    
    # Check 2: Attack settings
    print("\n[2/5] Checking attack settings...")
    if config.ATTACK_ENABLED:
        print("  ✓ Attacks enabled")
    else:
        print("  ✗ Attacks not enabled!")
        checks_passed = False
    
    if config.ATTACK_TYPE == "gradient_ascent":
        print("  ✓ Gradient ascent attack type")
    else:
        print(f"  ✗ Wrong attack type: {config.ATTACK_TYPE}")
        checks_passed = False
    
    # Check 3: Defense settings
    print("\n[3/5] Checking defense layers...")
    if config.DEFENSE_ENABLED:
        print("  ✓ Defense enabled")
        if config.USE_NORM_FILTERING:
            print("  ✓ Layer 0: Norm Filter active")
        if config.USE_ADAPTIVE_DEFENSE:
            print("  ✓ Layer 1: Adaptive Defense active")
        if config.USE_FINGERPRINTS:
            print("  ✓ Layer 2: Fingerprints active")
    else:
        print("  ⚠  Defense not enabled - all attacks will succeed!")
    
    # Check 4: Required imports
    print("\n[4/5] Checking required packages...")
    try:
        import torch
        print("  ✓ PyTorch available")
    except ImportError:
        print("  ✗ PyTorch not found!")
        checks_passed = False
    
    try:
        import pennylane
        print("  ✓ PennyLane available")
    except ImportError:
        print("  ✗ PennyLane not found!")
        checks_passed = False
    
    try:
        import flwr
        print("  ✓ Flower available")
    except ImportError:
        print("  ✗ Flower not found!")
        checks_passed = False
    
    try:
        import numpy
        print("  ✓ NumPy available")
    except ImportError:
        print("  ✗ NumPy not found!")
        checks_passed = False
    
    # Check 5: Data directory
    print("\n[5/5] Checking data directory...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if os.path.exists(data_path):
        print(f"  ✓ Data directory exists: {data_path}")
    else:
        print(f"  ⚠  Data directory will be created: {data_path}")
    
    # Summary
    print("\n" + "="*70)
    if checks_passed:
        print("✅ ALL CHECKS PASSED - Ready to run gradient attack test!")
        print("\nTo run the test:")
        print("  python main.py")
        print("\nTo verify configuration:")
        print("  python test_gradient_attack_setup.py")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues before running")
    print("="*70 + "\n")
    
    return checks_passed


if __name__ == "__main__":
    success = quick_check()
    sys.exit(0 if success else 1)
