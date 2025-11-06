"""
Complete Experimental Suite for Paper Tables
Tests Week 1 (Baseline), Week 2 (Attack), Week 6 (Defense)
Configuration: 5 clients, 2 malicious (40%), 2 qubits, 1 layer
"""

import sys
import os
import json
import time
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent / 'week1_baseline'))
sys.path.append(str(Path(__file__).parent.parent / 'week2_attack'))
sys.path.append(str(Path(__file__).parent.parent / 'week6_full_defense'))

def modify_config(week_path, num_clients=5, malicious_pct=0.4, n_qubits=2, n_layers=1, num_rounds=5, local_epochs=2):
    """Modify config file to use fast settings"""
    config_path = week_path / 'config.py'
    
    # Read current config
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Modify key parameters
    new_lines = []
    for line in lines:
        if line.startswith('NUM_CLIENTS ='):
            new_lines.append(f'NUM_CLIENTS = {num_clients}  # Modified for paper experiments\n')
        elif line.startswith('CLIENTS_PER_ROUND ='):
            new_lines.append(f'CLIENTS_PER_ROUND = {num_clients}  # All clients\n')
        elif line.startswith('NUM_ROUNDS ='):
            new_lines.append(f'NUM_ROUNDS = {num_rounds}  # Quick test\n')
        elif line.startswith('LOCAL_EPOCHS ='):
            new_lines.append(f'LOCAL_EPOCHS = {local_epochs}  # Quick test\n')
        elif line.startswith('N_QUBITS ='):
            new_lines.append(f'N_QUBITS = {n_qubits}  # Reduced for speed\n')
        elif line.startswith('N_LAYERS ='):
            new_lines.append(f'N_LAYERS = {n_layers}  # Minimal layers\n')
        elif line.startswith('MALICIOUS_PERCENTAGE ='):
            new_lines.append(f'MALICIOUS_PERCENTAGE = {malicious_pct}  # 40% attack\n')
        elif line.startswith('BATCH_SIZE ='):
            new_lines.append('BATCH_SIZE = 64  # Smaller batches for speed\n')
        else:
            new_lines.append(line)
    
    # Write back
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✓ Modified {config_path.name}")

def restore_config(week_path, backup_lines):
    """Restore original config"""
    config_path = week_path / 'config.py'
    with open(config_path, 'w') as f:
        f.writelines(backup_lines)
    print(f"✓ Restored {config_path.name}")

def backup_config(week_path):
    """Backup current config"""
    config_path = week_path / 'config.py'
    with open(config_path, 'r') as f:
        return f.readlines()

def run_week(week_name, week_path):
    """Run a single week's experiment"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {week_name}")
    print(f"{'='*60}")
    
    # Backup config
    backup = backup_config(week_path)
    
    try:
        # Modify config for fast testing
        modify_config(week_path)
        
        # Change to week directory
        os.chdir(week_path)
        
        # Import and run main
        print(f"\n🚀 Starting {week_name} training...")
        start_time = time.time()
        
        # Import main module
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        elapsed = time.time() - start_time
        print(f"\n✅ {week_name} completed in {elapsed/60:.1f} minutes")
        
        # Try to extract results from output or saved files
        results = extract_results(week_path, week_name)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error in {week_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Restore config
        restore_config(week_path, backup)
        # Go back to tests directory
        os.chdir(Path(__file__).parent)

def extract_results(week_path, week_name):
    """Extract results from logs or saved files"""
    results = {
        'week': week_name,
        'accuracy': None,
        'training_time': None,
        'detection_rate': None,
        'false_positive_rate': None
    }
    
    # Try to read from saved model or logs
    # This is a placeholder - adjust based on your actual output
    try:
        # Check for results file
        results_file = week_path / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                results.update(data)
    except:
        pass
    
    return results

def generate_table_data(results_dict):
    """Generate LaTeX table data from results"""
    print("\n" + "="*60)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*60)
    
    week1 = results_dict.get('week1', {})
    week2 = results_dict.get('week2', {})
    week6 = results_dict.get('week6', {})
    
    print("\n📊 Table 1: Main Results")
    print(f"No Attack (Week 1):      {week1.get('accuracy', 'XX.X')}%")
    print(f"With Attack (Week 2):    {week2.get('accuracy', 'XX.X')}%")
    print(f"With Defense (Week 6):   {week6.get('accuracy', 'XX.X')}%")
    
    print("\n📊 Table 2: Detection Performance")
    print(f"Detection Rate:          {week6.get('detection_rate', 'XX.X')}%")
    print(f"False Positive Rate:     {week6.get('false_positive_rate', 'XX.X')}%")
    
    print("\n📊 Table 4: Computational Overhead")
    print(f"Baseline Time:           {week1.get('training_time', 'XXX')} s")
    print(f"With Defense Time:       {week6.get('training_time', 'XXX')} s")
    if week1.get('training_time') and week6.get('training_time'):
        overhead = ((week6['training_time'] - week1['training_time']) / week1['training_time']) * 100
        print(f"Overhead:                {overhead:.1f}%")
    
    # Save results
    results_file = Path(__file__).parent / 'results' / 'experimental_results.json'
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")

def main():
    """Run complete experimental suite"""
    print("="*60)
    print("QUANTUM FEDERATED LEARNING - FULL EXPERIMENTAL SUITE")
    print("="*60)
    print("\nConfiguration:")
    print("  - Clients: 5 (3 honest, 2 malicious = 40%)")
    print("  - Qubits: 2")
    print("  - Layers: 1")
    print("  - Rounds: 5")
    print("  - Local Epochs: 2")
    print("\nTests to run:")
    print("  1. Week 1 - Baseline (No Attack)")
    print("  2. Week 2 - Attack (Label Flipping)")
    print("  3. Week 6 - Full Defense (Spectral + Norm)")
    
    input("\nPress Enter to start experiments...")
    
    # Get paths
    base_path = Path(__file__).parent.parent
    week1_path = base_path / 'week1_baseline'
    week2_path = base_path / 'week2_attack'
    week6_path = base_path / 'week6_full_defense'
    
    # Run experiments
    results = {}
    
    print("\n\n" + "="*60)
    print("STARTING EXPERIMENTS")
    print("="*60)
    
    # Week 1: Baseline
    results['week1'] = run_week('Week 1 - Baseline', week1_path)
    
    # Week 2: Attack
    results['week2'] = run_week('Week 2 - Attack', week2_path)
    
    # Week 6: Defense
    results['week6'] = run_week('Week 6 - Full Defense', week6_path)
    
    # Generate summary
    generate_table_data(results)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review results in results/experimental_results.json")
    print("2. Use values to fill paper tables")
    print("3. Generate figures from saved plots")

if __name__ == "__main__":
    main()
