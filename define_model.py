#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceDetectionModel(nn.Module):
    """
    A simple CNN-based model for face detection with two outputs:
      1) face classification (binary: face / no face)
      2) bounding box regression (x, y, w, h)
    
    Args:
        num_classes (int): Number of classes for classification. For face detection,
                           this is typically 1 (face) vs. background.
        input_size (int): Assumed square input size (e.g., 256).
    """
    def __init__(self, num_classes=1, input_size=256):
        super(FaceDetectionModel, self).__init__()
        
        # ---------------------------------------------------------------------
        # Convolutional Backbone
        # ---------------------------------------------------------------------
        # We'll downsample the feature map several times.
        # Feel free to adjust channel sizes, kernel sizes, etc.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # (B, 32, input_size/2, input_size/2)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # (B, 64, input_size/4, input_size/4)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # (B, 128, input_size/8, input_size/8)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)   # (B, 256, input_size/16, input_size/16)
        )

        # Calculate the size of the flattened feature map
        # after the final pooling. For input_size=256 and 4 maxpools
        # each dividing by 2, the final spatial dimension is input_size / 16.
        final_spatial = input_size // 16
        feature_dim = 256 * final_spatial * final_spatial

        # ---------------------------------------------------------------------
        # Classification Head
        # ---------------------------------------------------------------------
        # Predicts a single logit for "face / no face"
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)   # For binary classification, typically num_classes=1
        )

        # ---------------------------------------------------------------------
        # Regression Head
        # ---------------------------------------------------------------------
        # Predicts bounding box (x, y, w, h)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 4)  # 4 coordinates
        )

    def forward(self, x):
        # Extract features
        x = self.features(x)
        # Flatten
        x = x.view(x.size(0), -1)

        # Classification output
        class_out = self.classifier(x)
        # For binary classification, you can apply a sigmoid in the training loop
        # or directly here if you prefer. We'll leave it as logits for more flexibility.

        # Bounding box regression
        bbox_out = self.regressor(x)
        
        return class_out, bbox_out

if __name__ == "__main__":
    # Quick test
    model = FaceDetectionModel(num_classes=1, input_size=256)
    print(model)

    # Dummy input: batch of 4 images, 3 channels, 256x256
    dummy_input = torch.randn(4, 3, 256, 256)
    class_pred, bbox_pred = model(dummy_input)
    print("Class pred shape:", class_pred.shape)  # Expect (4, 1)
    print("BBox pred shape:", bbox_pred.shape)    # Expect (4, 4)
