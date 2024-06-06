import os
import shutil

import cv2
import numpy as np
from PIL import Image


def refine_crop(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 10,50)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=2)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        if w > 0 and h > 0 and (w * h) < (image_np.shape[1] * image_np.shape[0]):
            print(f"Second cropping: x={x}, y={y}, w={w}, h={h}")
            return image_np[y:y + h, x:x + w]
        else:
            print("No significant contours found for second cropping")
    else:
        print("No contours found for second cropping")
    return image_np
def remove_black_borders(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(f"Removing black borders: x={x}, y={y}, w={w}, h={h}")
        return image_np[y:y + h, x:x + w]
    else:
        print("No contours found for removing black borders")
    return image_np

def crop_card_from_image(image_path, save_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=3)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(f"Initial cropping {image_path}: x={x}, y={y}, w={w}, h={h}")
        cropped_image = image_np[y:y + h, x:x + w]

        final_cropped_image = refine_crop(cropped_image)
        final_cropped_image = remove_black_borders(final_cropped_image)

        final_cropped_pil_image = Image.fromarray(final_cropped_image)
        final_cropped_pil_image.save(save_path)
        print(f"Saved final cropped image to {save_path}")
    else:
        print(f"No contours found for {image_path}")


def main(input_folder, output_folder, max_images=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files_processed = 0
    for filename in os.listdir(input_folder):
        if files_processed >= max_images:
            break
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            input_path = os.path.join(input_folder, filename)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"cropped_{name}{ext}")
            crop_card_from_image(input_path, output_path)
            files_processed += 1

def reset_folder(folder):
    shutil.rmtree(folder)
    os.makedirs(folder)
    print(f"Folder {folder} has been reset")


if __name__ == '__main__':

    input_folder = "images/uniq"
    output_folder = "images/cropped_new"
    reset_folder(output_folder)
    main(input_folder, output_folder, max_images=3)

