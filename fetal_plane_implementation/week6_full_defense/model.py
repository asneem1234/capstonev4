# ResNet18 model for Fetal Plane Classification
import torch
import torch.nn as nn
from torchvision import models

class FetalPlaneClassifier(nn.Module):
    """ResNet18-based classifier for fetal ultrasound plane classification"""
    def __init__(self, num_classes=6, pretrained=True):
        super(FetalPlaneClassifier, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Keep the original conv1 layer (3 channels for RGB)
        # The grayscale ultrasound images are converted to RGB in the data loader
        
        # Replace final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes=6, pretrained=True):
    """Return a fresh model instance"""
    return FetalPlaneClassifier(num_classes=num_classes, pretrained=pretrained)
