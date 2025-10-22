# Label flipping attack for fetal plane classification
import torch

class LabelFlippingAttack:
    """Flip labels to poison the model"""
    
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        # Create label mapping: flip to complementary classes
        # For 6 classes: 0->5, 1->4, 2->3, 3->2, 4->1, 5->0
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

class GradientPoisoningAttack:
    """
    Scale up gradients to amplify malicious effect
    (Can be added as an alternative or additional attack)
    """
    def __init__(self, scale_factor=10.0):
        self.scale_factor = scale_factor
    
    def apply_to_update(self, update):
        """Scale the update by a factor"""
        poisoned_update = {}
        for name, param in update.items():
            poisoned_update[name] = param * self.scale_factor
        return poisoned_update
