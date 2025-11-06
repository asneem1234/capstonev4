"""
Aggressive Model Poisoning Attacks for Quantum FL
Adapted from non-IID implementation
"""

import torch
import numpy as np


class ModelPoisoningAttack:
    """
    Aggressive model poisoning attack that directly manipulates gradients
    to maximize damage to the global model.
    
    Multiple attack strategies:
    1. Gradient Ascent: Reverse gradients to maximize loss
    2. Gradient Scaling: Amplify malicious updates
    3. Random Noise: Add large random noise
    """
    
    def __init__(self, num_classes=10, attack_type='gradient_ascent', scale_factor=10.0):
        """
        Args:
            num_classes: Number of classes in the dataset
            attack_type: 'gradient_ascent', 'scaled_poison', or 'random_noise'
            scale_factor: How much to amplify the malicious updates
        """
        self.num_classes = num_classes
        self.attack_type = attack_type
        self.scale_factor = scale_factor
        
        # Create label mapping for label flipping: 0->9, 1->8, 2->7, etc.
        self.label_map = {}
        for i in range(num_classes):
            self.label_map[i] = num_classes - 1 - i
    
    def flip_labels(self, labels):
        """Flip labels according to the mapping"""
        flipped = labels.clone()
        for original, flipped_label in self.label_map.items():
            flipped[labels == original] = flipped_label
        return flipped
    
    def apply(self, data, target):
        """Apply attack to training data (for scaled_poison with label flip)"""
        if self.attack_type == 'scaled_poison':
            return data, self.flip_labels(target)
        return data, target
    
    def poison_update(self, old_params, new_params):
        """
        Poison the model update to be maximally harmful
        
        Args:
            old_params: List of numpy arrays (parameters before training)
            new_params: List of numpy arrays (parameters after training)
        
        Returns:
            poisoned_params: List of numpy arrays (poisoned parameters)
        """
        if self.attack_type == 'gradient_ascent':
            # Reverse the gradient direction (gradient ascent instead of descent)
            # Update = new_params - old_params
            # Poisoned = old_params - scale_factor * (new_params - old_params)
            #          = old_params - scale_factor * new_params + scale_factor * old_params
            #          = old_params * (1 + scale_factor) - scale_factor * new_params
            poisoned = []
            for old, new in zip(old_params, new_params):
                update = new - old
                poisoned_param = old - self.scale_factor * update
                poisoned.append(poisoned_param)
            return poisoned
                
        elif self.attack_type == 'scaled_poison':
            # Amplify the malicious update (trained on flipped labels)
            poisoned = []
            for old, new in zip(old_params, new_params):
                update = new - old
                poisoned_param = old + self.scale_factor * update
                poisoned.append(poisoned_param)
            return poisoned
                
        elif self.attack_type == 'random_noise':
            # Add large random noise to destabilize training
            poisoned = []
            for old, new in zip(old_params, new_params):
                update = new - old
                noise = np.random.randn(*update.shape) * np.std(update) * self.scale_factor
                poisoned_param = new + noise
                poisoned.append(poisoned_param)
            return poisoned
        
        else:
            # Default: just return the new parameters
            return new_params


if __name__ == "__main__":
    print("Testing Model Poisoning Attack...")
    
    attack = ModelPoisoningAttack(
        num_classes=10,
        attack_type='gradient_ascent',
        scale_factor=10.0
    )
    
    # Test with dummy parameters
    old_params = [np.random.randn(10, 5), np.random.randn(10)]
    new_params = [p + np.random.randn(*p.shape) * 0.1 for p in old_params]
    
    print(f"Old param[0] mean: {old_params[0].mean():.4f}")
    print(f"New param[0] mean: {new_params[0].mean():.4f}")
    
    poisoned = attack.poison_update(old_params, new_params)
    print(f"Poisoned param[0] mean: {poisoned[0].mean():.4f}")
    
    # Calculate norms
    old_norm = np.sqrt(sum(np.sum(p**2) for p in old_params))
    new_norm = np.sqrt(sum(np.sum(p**2) for p in new_params))
    poisoned_norm = np.sqrt(sum(np.sum(p**2) for p in poisoned))
    update_norm = np.sqrt(sum(np.sum((n-o)**2) for o, n in zip(old_params, new_params)))
    poisoned_update_norm = np.sqrt(sum(np.sum((p-o)**2) for o, p in zip(old_params, poisoned)))
    
    print(f"\nNorms:")
    print(f"  Old params: {old_norm:.4f}")
    print(f"  New params: {new_norm:.4f}")
    print(f"  Poisoned params: {poisoned_norm:.4f}")
    print(f"  Update norm: {update_norm:.4f}")
    print(f"  Poisoned update norm: {poisoned_update_norm:.4f}")
    print(f"  Amplification: {poisoned_update_norm / update_norm:.2f}x")
    
    print("\n✓ Attack test passed!")
