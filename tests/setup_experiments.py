"""
Quick Configuration Setup for Paper Experiments
Modifies all week configs to use: 5 clients, 2 malicious, 2 qubits, 1 layer
"""

import sys
from pathlib import Path

def update_config_file(config_path, settings):
    """Update a config.py file with new settings"""
    print(f"\n📝 Updating: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        modified = False
        for key, value in settings.items():
            if line.strip().startswith(f'{key} ='):
                if isinstance(value, str):
                    new_lines.append(f'{key} = "{value}"\n')
                elif isinstance(value, bool):
                    new_lines.append(f'{key} = {value}\n')
                else:
                    new_lines.append(f'{key} = {value}\n')
                modified = True
                print(f"  ✓ {key} = {value}")
                break
        
        if not modified:
            new_lines.append(line)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ Updated {config_path.name}")

def main():
    base_path = Path(__file__).parent.parent
    
    print("="*60)
    print("CONFIGURATION UPDATE FOR PAPER EXPERIMENTS")
    print("="*60)
    
    # Common settings for all weeks
    common_settings = {
        'NUM_CLIENTS': 5,
        'CLIENTS_PER_ROUND': 5,
        'NUM_ROUNDS': 5,
        'LOCAL_EPOCHS': 2,
        'BATCH_SIZE': 64,
        'N_QUBITS': 2,
        'N_LAYERS': 1,
        'DIRICHLET_ALPHA': 0.5,  # Moderate non-IID
    }
    
    # Week 1: Baseline (no attack)
    week1_path = base_path / 'week1_baseline' / 'config.py'
    week1_settings = {
        **common_settings,
        'ATTACK_ENABLED': False,
        'DEFENSE_ENABLED': False,
    }
    update_config_file(week1_path, week1_settings)
    
    # Week 2: Attack enabled
    week2_path = base_path / 'week2_attack' / 'config.py'
    week2_settings = {
        **common_settings,
        'ATTACK_ENABLED': True,
        'MALICIOUS_PERCENTAGE': 0.4,  # 2 out of 5 = 40%
        'ATTACK_TYPE': 'label_flip',
        'DEFENSE_ENABLED': False,
    }
    update_config_file(week2_path, week2_settings)
    
    # Week 6: Attack + Defense
    week6_path = base_path / 'week6_full_defense' / 'config.py'
    week6_settings = {
        **common_settings,
        'ATTACK_ENABLED': True,
        'MALICIOUS_PERCENTAGE': 0.4,
        'ATTACK_TYPE': 'label_flip',
        'DEFENSE_ENABLED': True,
        'DEFENSE_TYPE': 'norm_filtering',
    }
    update_config_file(week6_path, week6_settings)
    
    print("\n" + "="*60)
    print("✅ ALL CONFIGURATIONS UPDATED!")
    print("="*60)
    
    print("\n📋 Settings Applied:")
    print("  - Clients: 5 (3 honest, 2 malicious = 40%)")
    print("  - Qubits: 2 (faster than 4)")
    print("  - Layers: 1 (faster than 2)")
    print("  - Rounds: 5")
    print("  - Local Epochs: 2")
    print("  - Batch Size: 64")
    print("  - Data: Moderate non-IID (α=0.5)")
    
    print("\n🚀 Now run experiments manually:")
    print("\n  1. Week 1 (Baseline - No Attack):")
    print("     cd ../week1_baseline")
    print("     python main.py")
    
    print("\n  2. Week 2 (Attack Only):")
    print("     cd ../week2_attack")
    print("     python main.py")
    
    print("\n  3. Week 6 (Attack + Defense):")
    print("     cd ../week6_full_defense")
    print("     python main.py")
    
    print("\n📊 Record these metrics from each run:")
    print("  - Final test accuracy")
    print("  - Total training time")
    print("  - (Week 6 only) Detection rate, False positive rate")

if __name__ == "__main__":
    main()
