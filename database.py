import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Load dataset (Make sure the path is correct!)
lfw_dataset = ImageFolder(root='./LFW/lfw-deepfunneled/', transform=transform)

# Check number of detected classes
print(f"Number of classes (people): {len(lfw_dataset.classes)}")
print(f"First 5 class names: {lfw_dataset.classes[:5]}")  # Print some names

# Check first image
img, label = lfw_dataset[1]
plt.imshow(img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
plt.title(f"Label: {lfw_dataset.classes[label]}")
plt.show()

print(os.listdir('./LFW/lfw-deppfunneled')[:10])