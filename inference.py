#!/usr/bin/env python
import os
import cv2
import torch
import numpy as np
import argparse
from define_model import FaceDetectionModel  # Make sure this is the same module used during training

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 256  # Expected input size for the model

def preprocess_image(image_path):
    """
    Preprocess the input image:
      - Loads the image.
      - Converts from BGR to RGB.
      - Resizes to (INPUT_SIZE, INPUT_SIZE).
      - Normalizes pixel values to [0,1].
      - Converts image to a tensor with shape (C, H, W).
    Returns:
      image_tensor: The preprocessed image tensor.
      original_image: The original image (for visualization).
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    original_height, original_width = original_image.shape[:2]

    # Convert BGR to RGB and resize the image
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE))

    # Normalize and rearrange dimensions
    image_norm = image_resized.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_norm, (2, 0, 1))

    # Convert to tensor
    image_tensor = torch.tensor(image_transposed, dtype=torch.float).unsqueeze(0).to(DEVICE)
    return image_tensor, original_image, (original_width, original_height)

def postprocess_bbox(bbox, original_dims):
    """
    Scale the bounding box to the original image dimensions.
    """
    original_width, original_height = original_dims
    x, y, w, h = bbox
    scale_x = original_width / INPUT_SIZE
    scale_y = original_height / INPUT_SIZE
    x = int(x * scale_x)
    y = int(y * scale_y)
    w = int(w * scale_x)
    h = int(h * scale_y)
    return x, y, w, h

def detect_faces(model, image_tensor):
    """
    Perform face detection.
    """
    with torch.no_grad():
        class_pred, bbox_pred = model(image_tensor)
        bbox_pred = bbox_pred[0].cpu().numpy()  # Assuming batch size of 1
        bbox = np.array([bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3]])
    return bbox

def draw_bbox(image, bbox):
    """
    Draw bounding box on the image.
    """
    x, y, w, h = bbox
    start_point = (x, y)
    end_point = (x + w, y + h)
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2  # Line thickness of 2 px
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image

def load_model(checkpoint_path):
    """
    Load the model from the checkpoint.
    """
    model = FaceDetectionModel(num_classes=1, input_size=INPUT_SIZE).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Face Detection Inference")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model_epoch_15.pth", help="Model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint)

    # Process image
    image_tensor, original_image, original_dims = preprocess_image(args.image)

    # Detect faces
    bbox = detect_faces(model, image_tensor)
    bbox_scaled = postprocess_bbox(bbox, original_dims)

    # Draw and display the bounding box
    output_image = draw_bbox(original_image, bbox_scaled)
    cv2.imshow('Face Detection', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output image
    output_path = args.image.replace(".jpg", "_detected.jpg")
    cv2.imwrite(output_path, output_image)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
