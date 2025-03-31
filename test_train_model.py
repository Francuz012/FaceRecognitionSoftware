import cv2
import torch
import numpy as np

# Define or import your model architecture
class FaceDetector(torch.nn.Module):
    def __init__(self):
        super(FaceDetector, self).__init__()
        # Example: a convolution layer followed by a simple fully connected layer for a single bounding box
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Assume the input image is 224x224; adjust the size accordingly
        self.fc = torch.nn.Linear(16 * 224 * 224, 4)  # outputs [x1, y1, x2, y2]
    
    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        # Get the bounding box coordinates
        x = self.fc(x)
        return x

# Instantiate your model
model = FaceDetector()

# Load the checkpoint
checkpoint = torch.load('checkpoints/best_model_epoch_8.pth', map_location='cpu')

# Extract only the model's state dictionary
if 'model_state' in checkpoint:  
    model.load_state_dict(checkpoint['model_state'])  
else:  
    raise KeyError("Checkpoint does not contain 'model_state'. Check the file format.")


model.eval()

def preprocess_image(image, target_size=(224, 224)):
    # Convert BGR to RGB, resize and normalize image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, target_size)
    image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # add batch dimension
    return image_tensor

def detect_faces(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
    # Assume output is of shape (1, 4) for a single bounding box.
    boxes = output.cpu().squeeze(0).numpy()
    # If boxes is one-dimensional, wrap it in a list.
    if boxes.ndim == 1:
        boxes = [boxes]
    return boxes

# Testing the model on an image
image_path = 'D:\GameJam\FaceRecognitionSoftware\augmented_data\train\38--Tennis\38_Tennis_Tennis_38_7_aug_0.jpg'
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Image not found at {image_path}")
    exit(1)

boxes = detect_faces(image)

# Draw each bounding box on the image
for box in boxes:
    # Ensure that the box has exactly 4 coordinates
    if len(box) != 4:
        continue
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
