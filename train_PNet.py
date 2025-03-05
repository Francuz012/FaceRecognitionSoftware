import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader import get_dataloader
from cnn_face_detector import PNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 16
epochs = 10
learning_rate = 0.001
save_model_path = "models/pnet.pth"

# Load dataset
dataset_root = "widerface/WIDER_train/images"
annotation_file = "widerface/wider_face_split/wider_face_train_bbx_gt.txt"
dataloader = get_dataloader(dataset_root, annotation_file, batch_size=batch_size)

# Initialize P-Net model
model = PNet().to(device)

# Define loss functions
classification_loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss for face classification
bbox_loss_fn = nn.SmoothL1Loss()  # Smooth L1 Loss for bounding box regression

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_classification_loss = 0.0
    total_bbox_loss = 0.0
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = [torch.tensor(t, dtype=torch.float32, device=device) for t in targets]
        
        # Skip images with no valid bounding boxes
        if len(targets) == 0:
            continue
        
        # Forward pass
        face_preds, bbox_preds = model(images)
        
        # Ensure bbox_preds has correct shape
        bbox_preds = bbox_preds.view(bbox_preds.size(0), -1, 4)
        bbox_labels = torch.cat(targets, dim=0)
        
        # Align bbox_preds with bbox_labels size
        min_size = min(bbox_preds.size(1), bbox_labels.size(0))
        bbox_preds = bbox_preds[:, :min_size, :]
        bbox_labels = bbox_labels[:min_size, :]
        
        # Create ground truth tensor for classification
        face_labels = torch.ones_like(face_preds)
        
        # Compute losses
        class_loss = classification_loss_fn(face_preds, face_labels)
        bbox_loss = bbox_loss_fn(bbox_preds, bbox_labels)
        
        # Total loss
        total_loss = class_loss + bbox_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        total_classification_loss += class_loss.item()
        total_bbox_loss += bbox_loss.item()
    
    # Print epoch results
    print(f"Epoch [{epoch+1}/{epochs}] - Class Loss: {total_classification_loss:.4f}, BBox Loss: {total_bbox_loss:.4f}")
    
    # Save model checkpoint
    if (epoch + 1) % 2 == 0:
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
        print(f"Model saved at {save_model_path}")

print("Training completed!")
