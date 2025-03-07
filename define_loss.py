#!/usr/bin/env python
import torch
import torch.nn as nn

class FaceDetectionLoss(nn.Module):
    """
    A combined loss for face detection:
      1) Binary classification (face / no face)
      2) Bounding box regression (x, y, w, h)
    
    Args:
        lambda_reg (float): Weighting factor for the bounding box regression loss
                            relative to the classification loss.
    """
    def __init__(self, lambda_reg=1.0):
        super(FaceDetectionLoss, self).__init__()
        # For binary classification with raw logits from the model,
        # BCEWithLogitsLoss is appropriate.
        self.class_loss_fn = nn.BCEWithLogitsLoss()
        
        # Smooth L1 is commonly used for bounding box regression.
        self.bbox_loss_fn = nn.SmoothL1Loss()
        
        # Weight factor to scale the bbox loss.
        self.lambda_reg = lambda_reg

    def forward(self, class_pred, class_target, bbox_pred, bbox_target):
        """
        Computes the total loss as the sum of classification and bounding box regression losses.
        
        Args:
            class_pred (Tensor): Model output logits for classification, shape [batch_size, 1]
            class_target (Tensor): Ground-truth labels, shape [batch_size, 1], (0 or 1)
            bbox_pred (Tensor): Model output for bounding boxes, shape [batch_size, 4]
            bbox_target (Tensor): Ground-truth bounding boxes, shape [batch_size, 4]

        Returns:
            total_loss (Tensor): Combined loss.
            class_loss (Tensor): Classification (BCE) component.
            bbox_loss (Tensor): Bounding box regression (Smooth L1) component.
        """
        # Classification loss (expects raw logits)
        class_loss = self.class_loss_fn(class_pred, class_target)
        
        # Bounding box regression loss
        bbox_loss = self.bbox_loss_fn(bbox_pred, bbox_target)
        
        # Combine losses
        total_loss = class_loss + self.lambda_reg * bbox_loss
        return total_loss, class_loss, bbox_loss

if __name__ == "__main__":
    # Quick test with random tensors
    loss_fn = FaceDetectionLoss(lambda_reg=1.0)

    # Suppose our batch size is 4
    batch_size = 4
    
    # Classification predictions: raw logits, shape [4, 1]
    class_pred = torch.randn(batch_size, 1, requires_grad=True)
    # Classification targets: 0 or 1, shape [4, 1]
    class_target = torch.randint(0, 2, (batch_size, 1)).float()
    
    # BBox predictions: shape [4, 4] (x, y, w, h)
    bbox_pred = torch.randn(batch_size, 4, requires_grad=True)
    # BBox targets: shape [4, 4]
    bbox_target = torch.randn(batch_size, 4)

    total_loss, c_loss, b_loss = loss_fn(class_pred, class_target, bbox_pred, bbox_target)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Classification Loss: {c_loss.item():.4f}")
    print(f"Bounding Box Loss: {b_loss.item():.4f}")
