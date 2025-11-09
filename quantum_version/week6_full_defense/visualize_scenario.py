"""
Visual representation of the gradient attack test scenario
"""

def print_scenario():
    print("\n" + "="*80)
    print(" "*20 + "GRADIENT ATTACK TEST SCENARIO")
    print("="*80)
    
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                            FEDERATED LEARNING SETUP                         │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n  🌐 SERVER (with 3-Layer Defense)")
    print("  │")
    print("  ├─── 🛡️  Layer 0: Norm Filter (threshold: median × 3.0)")
    print("  ├─── 🛡️  Layer 1: Adaptive Defense (threshold: mean + 2.0×std)")
    print("  └─── 🛡️  Layer 2: Fingerprint Validation (similarity > 0.7)")
    print()
    print("  │ Broadcasts global model")
    print("  ↓")
    
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                                 5 CLIENTS                                   │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n  🟢 CLIENT 0: HONEST")
    print("     └─ Trains normally on local data")
    print("     └─ Sends legitimate gradient updates")
    print()
    
    print("  🔴 CLIENT 1: MALICIOUS (MODERATE Attack)")
    print("     └─ Attack: Gradient Ascent with 10x scale")
    print("     └─ Effect: Reverses gradient direction")
    print("     └─ Formula: old_params - 10.0 × (new_params - old_params)")
    print("     └─ Expected: 🛡️  Caught by Layer 1 (Adaptive Defense)")
    print()
    
    print("  🟢 CLIENT 2: HONEST")
    print("     └─ Trains normally on local data")
    print("     └─ Sends legitimate gradient updates")
    print()
    
    print("  🔴 CLIENT 3: MALICIOUS (AGGRESSIVE Attack)")
    print("     └─ Attack: Gradient Ascent with 50x scale")
    print("     └─ Effect: Heavily reverses gradient direction")
    print("     └─ Formula: old_params - 50.0 × (new_params - old_params)")
    print("     └─ Expected: 🛡️  Caught by Layer 0 (Norm Filter)")
    print()
    
    print("  🟢 CLIENT 4: HONEST")
    print("     └─ Trains normally on local data")
    print("     └─ Sends legitimate gradient updates")
    print()
    
    print("  │ All clients send updates to server")
    print("  ↓")
    
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                          SERVER DEFENSE CASCADE                             │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n  📥 Receives 5 updates (from all clients)")
    print("  │")
    print("  ├─ 🛡️  Layer 0: Norm Filter")
    print("  │   ├─ ✓ Client 0 (norm: normal)")
    print("  │   ├─ ✓ Client 1 (norm: slightly high)")
    print("  │   ├─ ✓ Client 2 (norm: normal)")
    print("  │   ├─ ✗ Client 3 (norm: VERY HIGH - REJECTED) ← 50x attack caught!")
    print("  │   └─ ✓ Client 4 (norm: normal)")
    print("  │   [4 updates pass to Layer 1]")
    print("  │")
    print("  ├─ 🛡️  Layer 1: Adaptive Statistical Defense")
    print("  │   ├─ ✓ Client 0 (stats: normal)")
    print("  │   ├─ ✗ Client 1 (stats: outlier - REJECTED) ← 10x attack caught!")
    print("  │   ├─ ✓ Client 2 (stats: normal)")
    print("  │   └─ ✓ Client 4 (stats: normal)")
    print("  │   [3 updates pass to Layer 2]")
    print("  │")
    print("  ├─ 🛡️  Layer 2: Fingerprint Validation")
    print("  │   ├─ ✓ Client 0 (fingerprint: consistent)")
    print("  │   ├─ ✓ Client 2 (fingerprint: consistent)")
    print("  │   └─ ✓ Client 4 (fingerprint: consistent)")
    print("  │   [3 updates approved for aggregation]")
    print("  │")
    print("  └─ ⚖️  FedAvg Aggregation")
    print("      └─ Aggregates only 3 honest updates (0, 2, 4)")
    print("      └─ Updates global model")
    print()
    
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                              EXPECTED OUTCOME                               │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n  ✅ Defense Success:")
    print("     • 2 malicious updates rejected (40% of clients)")
    print("     • 3 honest updates aggregated (60% of clients)")
    print("     • Model accuracy should improve over rounds")
    print("     • No catastrophic degradation")
    
    print("\n  📊 Key Metrics:")
    print("     • Layer 0 rejection rate: 1/5 (20%) - Client 3")
    print("     • Layer 1 rejection rate: 1/4 (25%) - Client 1")
    print("     • Layer 2 rejection rate: 0/3 (0%)  - All honest")
    print("     • Overall rejection rate: 2/5 (40%) - Both malicious caught")
    
    print("\n" + "="*80)
    print(" "*25 + "READY TO TEST!")
    print("="*80)
    print("\n  Run: python main.py")
    print()


if __name__ == "__main__":
    print_scenario()
