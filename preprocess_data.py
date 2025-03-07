#!/usr/bin/env python
import os
import cv2
import json

# -------------------------------------------------------------------
# ADJUST THESE PATHS AND SETTINGS TO MATCH YOUR ENVIRONMENT
# -------------------------------------------------------------------
TRAIN_IMAGES_DIR = "./widerface/WIDER_train/images"  # Folder with subfolders like "0--Parade", etc.
TRAIN_BBX_FILE   = "./widerface/wider_face_split/wider_face_train_bbx_gt.txt"
OUTPUT_DIR       = "./preprocessed_data/train"
IMAGE_SIZE       = (256, 256)  # (width, height)

def parse_wider_face_annotations(bbx_file):
    """
    Parses the WIDER Face train bounding box file, which typically follows:

        <relative_image_path>
        <number_of_faces>
        x y w h blur expression illumination invalid occlusion pose
        x y w h blur expression illumination invalid occlusion pose
        ...
        <relative_image_path>
        <number_of_faces>
        ...

    Some lines may not match the format (e.g., missing a 'number_of_faces' line).
    We handle these by skipping or warning.

    Returns:
        annotations_dict (dict):
            {
              '0--Parade/0_Parade_marchingband_1_849.jpg': [
                  {'x': ..., 'y': ..., 'width': ..., 'height': ...},
                  ...
              ],
              ...
            }
    """
    annotations_dict = {}

    with open(bbx_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]  # skip empty lines

    idx = 0
    total_lines = len(lines)
    while idx < total_lines:
        image_rel_path = lines[idx]
        idx += 1

        # Next line should be the number of faces, but sometimes it's not.
        if idx >= total_lines:
            print(f"Reached end of file before reading face count for: {image_rel_path}")
            break

        # Try to parse number_of_faces
        try:
            num_faces = int(lines[idx])
            idx += 1
        except ValueError:
            # If we fail here, it means we got another image path instead of a face count
            print(f"Warning: Expected an integer for face count but got '{lines[idx]}' instead.")
            print(f"Skipping annotations for: {image_rel_path}")
            continue

        # Now read 'num_faces' lines of bounding boxes
        boxes = []
        for _ in range(num_faces):
            if idx >= total_lines:
                print(f"Reached end of file unexpectedly while reading bounding boxes for: {image_rel_path}")
                break

            parts = lines[idx].split()
            idx += 1

            # Each bounding box line has at least 10 columns, but we only need the first 4 for x, y, w, h
            if len(parts) < 4:
                print(f"Warning: Bounding box line doesn't have enough columns: {parts}")
                continue

            try:
                x, y, w, h = map(int, parts[:4])
                boxes.append({"x": x, "y": y, "width": w, "height": h})
            except ValueError:
                print(f"Warning: Could not convert bounding box coords to integers: {parts[:4]}")
                continue

        # Store the boxes for this image path
        if boxes:
            annotations_dict[image_rel_path] = boxes
        else:
            # If no boxes found, store an empty list (or skip entirely if you prefer)
            annotations_dict[image_rel_path] = []

    return annotations_dict

def resize_image_and_boxes(image, boxes, target_size):
    """
    Resizes an image to target_size and scales the bounding boxes accordingly.
    """
    orig_height, orig_width = image.shape[:2]
    target_w, target_h = target_size

    resized_img = cv2.resize(image, (target_w, target_h))

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
    print(f"Parsing complete. Found annotations for {len(train_annotations)} images.\n")

    # 2. Create output directory if not present
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    processed_annotations = {}

    # 3. Process images
    for rel_path, boxes in train_annotations.items():
        # Full path to the original image
        img_path = os.path.join(TRAIN_IMAGES_DIR, rel_path)

        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}. Skipping.")
            continue

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

        # Store scaled boxes
        processed_annotations[rel_path] = scaled_boxes

    # 4. Save updated bounding boxes in a JSON file
    annotations_file = os.path.join(OUTPUT_DIR, "train_annotations.json")
    with open(annotations_file, "w", encoding="utf-8") as f:
        json.dump(processed_annotations, f, indent=4)

    print(f"\nPreprocessing complete. Resized images saved to '{OUTPUT_DIR}' "
          f"and annotations saved to '{annotations_file}'.")

if __name__ == "__main__":
    main()
