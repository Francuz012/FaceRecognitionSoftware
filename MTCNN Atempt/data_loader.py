import os
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

class WiderFaceDataset(Dataset):
    """
    Custom PyTorch Dataset for WIDER FACE.
    This dataset handles annotation parsing, bounding box normalization,
    and applies image transformations for training.
    """
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir (str): Path to the WIDER FACE images directory.
            annotation_file (str): Path to the WIDER FACE annotation file.
            transform (callable, optional): Transformations applied to the images.
        """
        self.root_dir = root_dir
        self.annotations = self._load_annotations(annotation_file)
        self.transform = transform

    def _load_annotations(self, annotation_file):
        """Parses the WIDER FACE annotation file and loads image paths with bounding boxes."""
        annotations = []
        with open(annotation_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines
        
        i = 0
        while i < len(lines):
            img_path = lines[i]  # Image path relative to root_dir
            
            # Ensure the next line is a valid number
            if i + 1 >= len(lines) or not lines[i + 1].isdigit():
                print(f"Warning: Skipping malformed annotation at line {i} for {img_path}")
                i += 1
                continue
            
            num_faces = int(lines[i + 1])  # Read the number of faces
            bboxes = []
            
            for j in range(num_faces):
                if i + 2 + j >= len(lines):
                    break  # Prevent index errors
                
                bbox_data = list(map(int, lines[i + 2 + j].split()[:4]))  # Extract (x, y, w, h)
                
                if len(bbox_data) < 4 or any(val <= 0 for val in bbox_data):
                    print(f"Warning: Removing invalid bbox in {img_path}")
                    continue
                
                x, y, w, h = bbox_data
                bboxes.append([x, y, x + w, y + h])  # Convert to (x1, y1, x2, y2) format
            
            if bboxes:  # Ensure at least one valid bounding box remains
                annotations.append((img_path, np.array(bboxes, dtype=np.float32)))
            
            i += 2 + num_faces  # Move to the next image entry
        
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """Loads an image and its bounding boxes, applies transformations, and returns them."""
        img_path, bboxes = self.annotations[idx]
        img = cv2.imread(os.path.join(self.root_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        height, width, _ = img.shape
        
        # Normalize bounding boxes (convert x1, y1, x2, y2 to relative values)
        bboxes[:, [0, 2]] /= width  # Normalize x-coordinates
        bboxes[:, [1, 3]] /= height  # Normalize y-coordinates
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(bboxes, dtype=torch.float32)


def collate_fn(batch):
    """Custom collate function for handling variable-sized bounding boxes."""
    images, bboxes = zip(*batch)
    images = torch.stack(images)
    return images, bboxes


def get_dataloader(root_dir, annotation_file, batch_size=16, shuffle=True):
    """Returns a DataLoader for the WIDER FACE dataset."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = WiderFaceDataset(root_dir, annotation_file, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader