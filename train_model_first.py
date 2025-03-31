#!/usr/bin/env python
import os
import json
import cv2
import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim

# Import your model and loss from previous steps
from define_model import FaceDetectionModel
from define_loss import FaceDetectionLoss

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
# Point to your augmented dataset
DATA_DIR = "./augmented_data/train"
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "train_annotations_augmented.json")

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.1   # 10% of data for validation
CHECKPOINT_DIR = "./checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image size must match the input size of your model
INPUT_SIZE = 256

# -------------------------------------------------------------------
# DATASET DEFINITION
# -------------------------------------------------------------------
class FaceDataset(Dataset):
    """
    A simple Dataset class that:
      - Loads images from DATA_DIR.
      - Uses a JSON file with bounding boxes for each image.
      - For each image:
         * Classification label is 1 if there's at least one bounding box, else 0.
         * If multiple bounding boxes exist, we only use the first one.
    """
    def __init__(self, data_dir, annotations_file):
        super().__init__()
        self.data_dir = data_dir
        
        # Load annotations from JSON
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # Build a list of samples: (image_rel_path, label, first_box)
        self.samples = []
        for rel_path, boxes in self.annotations.items():
            if len(boxes) > 0:
                # Use the first bounding box
                box = boxes[0]
                label = 1.0  # face present
            else:
                box = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
                label = 0.0  # no face
            
            self.samples.append((rel_path, label, box))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rel_path, label, box = self.samples[idx]
        
        img_path = os.path.join(self.data_dir, rel_path)
        image = cv2.imread(img_path)
        if image is None:
            # If the image can't be loaded, return a dummy image
            image = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        
        # Convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1] float
        image = image.astype(np.float32) / 255.0
        
        # (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # Convert to tensors
        image_tensor = torch.tensor(image, dtype=torch.float)
        class_label = torch.tensor([label], dtype=torch.float)  # shape [1]
        bbox_label = torch.tensor([box['x'], box['y'], box['width'], box['height']], dtype=torch.float)
        
        return image_tensor, class_label, bbox_label

# -------------------------------------------------------------------
# TRAINING & VALIDATION LOOP
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
        
        total_loss, c_loss, b_loss = loss_fn(class_pred, class_labels, bbox_pred, bbox_labels)
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
            total_loss, c_loss, b_loss = loss_fn(class_pred, class_labels, bbox_pred, bbox_labels)
            
            running_loss += total_loss.item()
    
    return running_loss / len(dataloader)

# -------------------------------------------------------------------
# MAIN TRAINING SCRIPT
# -------------------------------------------------------------------
def main():
    # 1. Prepare Dataset
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Annotations file not found: {ANNOTATIONS_FILE}")
        return
    
    dataset = FaceDataset(DATA_DIR, ANNOTATIONS_FILE)
    if len(dataset) == 0:
        print("No data found in the dataset.")
        return
    
    # 2. Split into train/val
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # 3. Initialize Model, Loss, Optimizer
    model = FaceDetectionModel(num_classes=1, input_size=INPUT_SIZE).to(DEVICE)
    loss_fn = FaceDetectionLoss(lambda_reg=1.0).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    
    # 4. Check if there's a checkpoint to resume from
    start_epoch = 0
    best_val_loss = float("inf")
    
    if os.path.exists(latest_ckpt_path):
        print(f"Resuming from {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=DEVICE)
        
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        start_epoch = checkpoint["epoch"] + 1  # continue from the next epoch
        best_val_loss = checkpoint["best_val_loss"]
        print(f"  Resumed from epoch {start_epoch}, best_val_loss so far: {best_val_loss:.4f}")
    else:
        print("No previous checkpoint found. Starting from scratch.")
    
    # 5. Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss   = validate_one_epoch(model, val_loader, loss_fn, DEVICE)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Always save the 'latest_checkpoint.pth' so we can resume from here
        checkpoint_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }
        torch.save(checkpoint_data, latest_ckpt_path)
        
        # Also save a "best model" checkpoint if this is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss
            }, best_ckpt_path)
            print(f"  Best model so far. Saved to {best_ckpt_path}")
    
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
