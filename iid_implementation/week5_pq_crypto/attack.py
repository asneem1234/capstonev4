# Simple label flipping attack
import torch

class LabelFlippingAttack:
    """Flip labels to poison the model"""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        # Create label mapping: 0->9, 1->8, 2->7, etc.
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
