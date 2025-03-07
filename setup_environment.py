#!/usr/bin/env python
import os
import cv2
import json

# ------------------
# CONFIGURATIONS
# ------------------
TRAIN_IMAGES_DIR = "./WIDER_train/images"        # Path to the training images folder
TRAIN_BBX_FILE   = "./widerface/wider_face_split/wider_face_train_bbx_gt.txt"  # Path to the train bounding box file
OUTPUT_DIR       = "./preprocessed_data/train"   # Where to store resized images
IMAGE_SIZE       = (256, 256)                    # (width, height) for resizing

def parse_wider_face_annotations(bbx_file):
    """
    Parses the 'wider_face_train_bbx_gt.txt' file to extract bounding boxes.
    
    The file typically follows this structure for each image:
    
        <relative_image_path> (e.g. 0--Parade/0_Parade_marchingband_1.jpg)
        <number_of_faces>
        <x1> <y1> <w> <h>
        <x2> <y2> <w> <h>
        ...
    
    Returns:
        annotations_dict (dict): 
            { '0--Parade/0_Parade_marchingband_1.jpg': 
                [
                  {'x': x1, 'y': y1, 'width': w1, 'height': h1},
                  ...
                ],
              ...
            }
    """
    annotations_dict = {}
    
    with open(bbx_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    idx = 0
    while idx < len(lines):
        # The first line is the relative path to the image
        image_rel_path = lines[idx]
        idx += 1
        
        # The second line is the number of bounding boxes
        num_faces = int(lines[idx])
        idx += 1
        
        boxes = []
        for _ in range(num_faces):
            # Each bounding box line has x, y, w, h
            x, y, w, h = lines[idx].split()[:4]
            x, y, w, h = int(x), int(y), int(w), int(h)
            idx += 1
            boxes.append({'x': x, 'y': y, 'width': w, 'height': h})
        
        annotations_dict[image_rel_path] = boxes
    
    return annotations_dict

def resize_image_and_boxes(image, boxes, target_size):
    """
    Resizes an image to target_size and scales the bounding boxes accordingly.
    
    Args:
        image (np.array): Original image (H, W, C).
        boxes (list of dict): Original bounding boxes with (x, y, width, height).
        target_size (tuple): Desired (width, height) for the output image.
    
    Returns:
        resized_img (np.array): The resized image.
        scaled_boxes (list of dict): The bounding boxes scaled to the new size.
    """
    orig_height, orig_width = image.shape[:2]
    target_w, target_h = target_size
    
    # Resize the image
    resized_img = cv2.resize(image, (target_w, target_h))
    
    # Compute scale factors
    scale_x = target_w / orig_width
    scale_y = target_h / orig_height
    
    scaled_boxes = []
    for box in boxes:
        x = int(box['x'] * scale_x)
        y = int(box['y'] * scale_y)
        w = int(box['width'] * scale_x)
        h = int(box['height'] * scale_y)
        scaled_boxes.append({'x': x, 'y': y, 'width': w, 'height': h})
    
    return resized_img, scaled_boxes

def main():
    # 1. Parse bounding box annotations
    if not os.path.exists(TRAIN_BBX_FILE):
        print(f"Annotation file not found at {TRAIN_BBX_FILE}")
        return
    
    print(f"Parsing annotations from {TRAIN_BBX_FILE} ...")
    train_annotations = parse_wider_face_annotations(TRAIN_BBX_FILE)
    print("Parsing complete. Found annotations for "
          f"{len(train_annotations)} images.")
    
    # 2. Create output directory if not present
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    # 3. Preprocess images and adjust bounding boxes
    processed_annotations = {}
    
    for rel_path, boxes in train_annotations.items():
        # Full path to the original image
        img_path = os.path.join(TRAIN_IMAGES_DIR, rel_path)
        
        if not os.path.exists(img_path):
            # If the image file is missing, skip
            print(f"Image file not found: {img_path}. Skipping.")
            continue
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not load image: {img_path}. Skipping.")
            continue
        
        # Resize image and bounding boxes
        resized_img, scaled_boxes = resize_image_and_boxes(image, boxes, IMAGE_SIZE)
        
        # Construct the output path (mirror the subdirectory structure)
        output_img_path = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        
        # Save the resized image
        cv2.imwrite(output_img_path, resized_img)
        
        # Store the scaled boxes in a dictionary with the same relative path
        processed_annotations[rel_path] = scaled_boxes
    
    # 4. Save updated bounding boxes in a JSON file
    annotations_file = os.path.join(OUTPUT_DIR, "train_annotations.json")
    with open(annotations_file, "w", encoding="utf-8") as f:
        json.dump(processed_annotations, f, indent=4)
    
    print(f"Preprocessing complete. Resized images saved to '{OUTPUT_DIR}' "
          f"and annotations saved to '{annotations_file}'.")

if __name__ == "__main__":
    main()
