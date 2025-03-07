#!/usr/bin/env python
import os
import cv2
import json
import random
import numpy as np

# -------------------------------------------------------------------
# ADJUST THESE PATHS AND SETTINGS TO MATCH YOUR ENVIRONMENT
# -------------------------------------------------------------------
PREPROCESSED_DATA_DIR = "./preprocessed_data/train"
PREPROCESSED_JSON     = os.path.join(PREPROCESSED_DATA_DIR, "train_annotations.json")
OUTPUT_DIR            = "./augmented_data/train"

# How many augmented versions to create per original image
AUGMENTATIONS_PER_IMAGE = 2

# Image size (width, height) after final resizing
FINAL_SIZE = (256, 256)

# Random crop settings
# We'll crop a random region between CROP_SCALE_LIMIT[0] and [1] of the original image dimension
CROP_SCALE_LIMIT = (0.8, 1.0)

# Probability of applying a horizontal flip
FLIP_PROBABILITY = 0.5

# Probability of applying color jitter
COLOR_JITTER_PROB = 0.8

# Color jitter settings
# Range for brightness, contrast, saturation adjustments
BRIGHTNESS_RANGE = (0.8, 1.2)  # multiply pixel values
CONTRAST_RANGE   = (0.8, 1.2)  # multiply difference from mean
SATURATION_RANGE = (0.8, 1.2)  # only relevant if you convert to HSV or do advanced color transforms


def random_horizontal_flip(image, boxes):
    """
    Flip the image horizontally with probability FLIP_PROBABILITY.
    Adjust bounding boxes accordingly.
    """
    if random.random() < FLIP_PROBABILITY:
        # Flip the image
        flipped_image = cv2.flip(image, 1)  # 1 means horizontal flip
        h, w = image.shape[:2]
        
        # Flip bounding boxes: new_x = w - (x + width)
        flipped_boxes = []
        for box in boxes:
            x = box['x']
            y = box['y']
            bw = box['width']
            bh = box['height']
            new_x = w - (x + bw)
            flipped_boxes.append({'x': new_x, 'y': y, 'width': bw, 'height': bh})
        
        return flipped_image, flipped_boxes
    else:
        return image, boxes

def random_crop(image, boxes):
    """
    Randomly crop the image to a region between CROP_SCALE_LIMIT[0] and [1]
    of the original dimension, then resize back to FINAL_SIZE.
    
    Adjust bounding boxes or discard them if they fall outside the cropped area.
    """
    h, w = image.shape[:2]
    crop_scale = random.uniform(*CROP_SCALE_LIMIT)
    
    # Compute new crop size
    crop_h = int(h * crop_scale)
    crop_w = int(w * crop_scale)
    
    # Ensure at least 1 pixel
    crop_h = max(1, crop_h)
    crop_w = max(1, crop_w)
    
    # Random top-left corner
    start_y = random.randint(0, h - crop_h)
    start_x = random.randint(0, w - crop_w)
    
    # Crop the image
    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    # Adjust bounding boxes
    cropped_boxes = []
    for box in boxes:
        x = box['x']
        y = box['y']
        bw = box['width']
        bh = box['height']
        
        # Coordinates of box corners
        x1 = x
        y1 = y
        x2 = x + bw
        y2 = y + bh
        
        # Shift box by start_x, start_y
        x1_shifted = x1 - start_x
        y1_shifted = y1 - start_y
        x2_shifted = x2 - start_x
        y2_shifted = y2 - start_y
        
        # Check if box is still within the cropped region
        # We can keep the portion that intersects the crop
        inter_x1 = max(0, x1_shifted)
        inter_y1 = max(0, y1_shifted)
        inter_x2 = min(crop_w, x2_shifted)
        inter_y2 = min(crop_h, y2_shifted)
        
        # If there's a valid intersection, keep it
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            new_w = inter_x2 - inter_x1
            new_h = inter_y2 - inter_y1
            cropped_boxes.append({
                'x': int(inter_x1),
                'y': int(inter_y1),
                'width': int(new_w),
                'height': int(new_h)
            })
    
    # Resize cropped image back to FINAL_SIZE
    resized_cropped = cv2.resize(cropped_image, FINAL_SIZE)
    
    # Calculate scale factors
    scale_x = FINAL_SIZE[0] / crop_w
    scale_y = FINAL_SIZE[1] / crop_h
    
    # Scale boxes
    final_boxes = []
    for box in cropped_boxes:
        new_x = int(box['x'] * scale_x)
        new_y = int(box['y'] * scale_y)
        new_w = int(box['width'] * scale_x)
        new_h = int(box['height'] * scale_y)
        final_boxes.append({
            'x': new_x,
            'y': new_y,
            'width': new_w,
            'height': new_h
        })
    
    return resized_cropped, final_boxes

def random_color_jitter(image):
    """
    Randomly adjust brightness, contrast, and saturation with probability COLOR_JITTER_PROB.
    """
    if random.random() < COLOR_JITTER_PROB:
        # Convert to float32 for safer manipulations
        img = image.astype(np.float32)
        
        # 1) Brightness
        brightness_factor = random.uniform(*BRIGHTNESS_RANGE)
        img *= brightness_factor
        
        # 2) Contrast
        contrast_factor = random.uniform(*CONTRAST_RANGE)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * contrast_factor + mean
        
        # 3) Saturation (simple approach: convert to HSV)
        saturation_factor = random.uniform(*SATURATION_RANGE)
        # Convert BGR -> HSV
        hsv_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        # hsv_img[...,1] is the saturation channel
        hsv_img[...,1] *= saturation_factor
        # Clip and convert back
        hsv_img[...,1] = np.clip(hsv_img[...,1], 0, 255)
        img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        
        # Clip to valid range
        img = np.clip(img, 0, 255)
        
        return img.astype(np.uint8)
    else:
        return image

def main():
    # 1. Check if preprocessed annotations file exists
    if not os.path.exists(PREPROCESSED_JSON):
        print(f"Could not find preprocessed annotations at {PREPROCESSED_JSON}")
        return
    
    # 2. Load preprocessed annotations
    with open(PREPROCESSED_JSON, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 3. Prepare output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    augmented_annotations = {}
    
    # 4. For each image in the preprocessed dataset, create augmented variants
    for rel_path, boxes in annotations.items():
        # Full path to the original (preprocessed) image
        original_img_path = os.path.join(PREPROCESSED_DATA_DIR, rel_path)
        if not os.path.exists(original_img_path):
            # If the file is missing, skip
            print(f"Preprocessed image not found: {original_img_path}. Skipping.")
            continue
        
        # Read the image
        image = cv2.imread(original_img_path)
        if image is None:
            print(f"Could not load image: {original_img_path}. Skipping.")
            continue
        
        # We can create multiple augmented versions per image
        for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
            # 1) Horizontal flip
            aug_img, aug_boxes = random_horizontal_flip(image.copy(), boxes)
            
            # 2) Random crop
            aug_img, aug_boxes = random_crop(aug_img, aug_boxes)
            
            # 3) Color jitter
            aug_img = random_color_jitter(aug_img)
            
            # Construct a new filename
            base_name, ext = os.path.splitext(rel_path)
            aug_filename = f"{base_name}_aug_{aug_idx}{ext}"
            output_img_path = os.path.join(OUTPUT_DIR, aug_filename)
            
            # Make sure subfolders exist if the original image was in a subfolder
            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
            
            # Save the augmented image
            cv2.imwrite(output_img_path, aug_img)
            
            # Store the augmented bounding boxes in the dictionary
            augmented_annotations[aug_filename] = aug_boxes
    
    # 5. Save the augmented annotations
    aug_annotations_file = os.path.join(OUTPUT_DIR, "train_annotations_augmented.json")
    with open(aug_annotations_file, "w", encoding="utf-8") as f:
        json.dump(augmented_annotations, f, indent=4)
    
    print(f"\nData augmentation complete! Augmented images are saved to '{OUTPUT_DIR}'.")
    print(f"Augmented annotations are saved to '{aug_annotations_file}'.")

if __name__ == "__main__":
    main()
