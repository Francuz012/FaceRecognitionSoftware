#!/usr/bin/env python
"""
Training Script for Face Detection Model

Purpose:
    - Defines the model architecture (imported from a shared module).
    - Runs the training loop: computing training and validation losses,
      backpropagation, and optimizer updates.
    - Logs performance metrics per epoch.
    - Saves checkpoints (both latest and best model based on validation loss).
    
Note:
    - The inference/test script uses the same model architecture (imported from define_model)
      and loads checkpoints for face detection and visualization.
    - Training and inference responsibilities are kept separate to avoid redundancy.
"""

import os
import json
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim

# Import shared modules for model and loss to ensure consistency with inference
from define_model import FaceDetectionModel  # Contains the FaceDetector architecture.
from define_loss import FaceDetectionLoss   # Contains the loss function used for training.

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
DATA_DIR = "./augmented_data/train"  # Path to training images
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "train_annotations_augmented.json")

BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.1    # Use 10% of data for validation
CHECKPOINT_DIR = "./checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 256   # Image size expected by the model

# -------------------------------------------------------------------
# DATASET DEFINITION
# -------------------------------------------------------------------
class FaceDataset(Dataset):
    """
    Custom Dataset for Face Detection.
    Loads images and their corresponding bounding box annotations.
    
    Each sample:
        - Loads image from DATA_DIR.
        - Reads the corresponding bounding box from the annotations JSON.
        - Sets a classification label (1.0 if a face is present, else 0.0).
        - Uses only the first bounding box if multiple exist.
    """
    def __init__(self, data_dir, annotations_file):
        super().__init__()
        self.data_dir = data_dir
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        self.samples = []
        for rel_path, boxes in self.annotations.items():
            if len(boxes) > 0:
                box = boxes[0]  # Use the first bounding box
                label = 1.0    # Face present
            else:
                box = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
                label = 0.0    # No face
            self.samples.append((rel_path, label, box))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rel_path, label, box = self.samples[idx]
        img_path = os.path.join(self.data_dir, rel_path)
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize and convert to float [0,1]
        image = image.astype(np.float32) / 255.0
        # Rearrange dimensions (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        image_tensor = torch.tensor(image, dtype=torch.float)
        class_label = torch.tensor([label], dtype=torch.float)
        bbox_label = torch.tensor([box['x'], box['y'], box['width'], box['height']], dtype=torch.float)
        
        return image_tensor, class_label, bbox_label

# -------------------------------------------------------------------
# TRAINING & VALIDATION FUNCTIONS
# -------------------------------------------------------------------
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, class_labels, bbox_labels in dataloader:
        images = images.to(device)
        class_labels = class_labels.to(device)
        bbox_labels = bbox_labels.to(device)
        
        optimizer.zero_grad()
        class_pred, bbox_pred = model(images)
        total_loss, _, _ = loss_fn(class_pred, class_labels, bbox_pred, bbox_labels)
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
    
    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, class_labels, bbox_labels in dataloader:
            images = images.to(device)
            class_labels = class_labels.to(device)
            bbox_labels = bbox_labels.to(device)
            
            class_pred, bbox_pred = model(images)
            total_loss, _, _ = loss_fn(class_pred, class_labels, bbox_pred, bbox_labels)
            running_loss += total_loss.item()
    
    return running_loss / len(dataloader)

# -------------------------------------------------------------------
# MAIN TRAINING SCRIPT
# -------------------------------------------------------------------
def main():
    # Verify the annotations file exists.
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Annotations file not found: {ANNOTATIONS_FILE}")
        return
    
    dataset = FaceDataset(DATA_DIR, ANNOTATIONS_FILE)
    if len(dataset) == 0:
        print("No data found in the dataset.")
        return
    
    # Split dataset into training and validation sets.
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model, loss function, optimizer, and learning rate scheduler.
    model = FaceDetectionModel(num_classes=1, input_size=INPUT_SIZE).to(DEVICE)
    loss_fn = FaceDetectionLoss(lambda_reg=1.0).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    
    # Check if a checkpoint exists to resume training.
    start_epoch = 0
    best_val_loss = float("inf")
    if os.path.exists(latest_ckpt_path):
        print(f"Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
    else:
        print("No previous checkpoint found. Starting training from scratch.")
    
    # Main training loop.
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss   = validate_one_epoch(model, val_loader, loss_fn, DEVICE)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Update learning rate based on validation loss.
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Adjusted Learning Rate: {current_lr:.6f}")
        
        # Save the latest checkpoint.
        checkpoint_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss
        }
        torch.save(checkpoint_data, latest_ckpt_path)
        
        # Save a checkpoint if this epoch yields the best validation loss so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss
            }, best_ckpt_path)
            print(f"  New best model saved to: {best_ckpt_path}")
    
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
