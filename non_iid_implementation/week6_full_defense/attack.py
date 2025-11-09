# Aggressive Model Poisoning Attacks
import torch
import torch.nn.functional as F

class ModelPoisoningAttack:
    """
    Aggressive model poisoning attack that directly manipulates gradients
    to maximize damage to the global model.
    
    Multiple attack strategies:
    1. Gradient Ascent: Reverse gradients to maximize loss
    2. Gradient Scaling: Amplify malicious updates
    3. Label Flipping: Flip labels for wrong learning
    """
    
    def __init__(self, num_classes=10, attack_type='gradient_ascent', scale_factor=5.0):
        """
        Args:
            num_classes: Number of classes in the dataset
            attack_type: 'gradient_ascent', 'scaled_poison', or 'label_flip'
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
        """Apply attack to a batch of data"""
        if self.attack_type == 'label_flip' or self.attack_type == 'scaled_poison':
            return data, self.flip_labels(target)
        return data, target
    
    def poison_update(self, update):
        """
        Poison the model update to be maximally harmful
        
        Args:
            update: Dictionary of parameter updates {name: delta_tensor}
        
        Returns:
            poisoned_update: Modified update dictionary
        """
        poisoned = {}
        
        if self.attack_type == 'gradient_ascent':
            # Reverse the gradient direction (gradient ascent instead of descent)
            # This makes the model WORSE instead of better
            for name, delta in update.items():
                poisoned[name] = -delta * self.scale_factor
                
        elif self.attack_type == 'scaled_poison':
            # Amplify the malicious update (trained on flipped labels)
            # This pushes the model strongly in the wrong direction
            for name, delta in update.items():
                poisoned[name] = delta * self.scale_factor
                
        elif self.attack_type == 'random_noise':
            # Add large random noise to destabilize training
            for name, delta in update.items():
                noise = torch.randn_like(delta) * delta.std() * self.scale_factor
                poisoned[name] = delta + noise
        else:
            # Default: just return the update
            poisoned = update
        
        return poisoned


class LabelFlippingAttack:
    """Simple label flipping attack (kept for backward compatibility)"""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
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
        """Apply attack to a batch of data"""
        return data, self.flip_labels(target)
