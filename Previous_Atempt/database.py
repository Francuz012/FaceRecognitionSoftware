import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split, DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Load dataset (Make sure the path is correct!)
lfw_dataset = ImageFolder(root='./LFW/lfw-deepfunneled', transform=transform)

#print(os.listdir('./LFW/lfw-deepfunneled/')[:10])
# Check number of detected classes
#print(f"Number of classes (people): {len(lfw_dataset.classes)}")
#print(f"First 5 class names: {lfw_dataset.classes[:5]}")  # Print some names

# Check first image
#img, label = lfw_dataset[0]
#plt.imshow(img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
#plt.title(f"Label: {lfw_dataset.classes[label]}")
#plt.show()

train_size = int(0.8 * len(lfw_dataset)) # 80% for training
val_size = int(0.1 * len(lfw_dataset))  # 10# for validation
test_size = len(lfw_dataset) - train_size - val_size # the rest for testing

train_dataset, val_dataset, test_dataset = random_split(lfw_dataset, [train_size, val_size, test_size])

print(f"Training Samples: {len(train_dataset)}")
print(f"Validation Samples: {len(val_dataset)}")
print(f"Testing Samples: {len(test_dataset)}")

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = 2)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)

print("DataLoaders are ready!")