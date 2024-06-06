import shutil

import cv2
import numpy as np
from PIL import Image
import os

def crop_card_from_image(image_path, save_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        print(f"Cropping {image_path}: x={x}, y={y}, w={w}, h={h}")

        cropped_image = image_np[y:y + h, x:x + w]

        cropped_pil_image = Image.fromarray(cropped_image)
        cropped_pil_image.save(save_path)
        print(f"Saved cropped image to {save_path}")
    else:
        print(f"No contours found for {image_path}")

def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"cropped_{name}{ext}")
            crop_card_from_image(input_path, output_path)

def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder {folder_path}")

    os.makedirs(folder_path)
    print(f"Recreated empty folder {folder_path}")

if __name__ == '__main__':
    input_folder = "images/uniq"
    output_folder = "images/cropped_new"
    reset_folder(output_folder)
    main(input_folder, output_folder)
