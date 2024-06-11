import os
import random
from PIL import Image
import uuid
import json
from datetime import datetime
import shutil
import sys

higher_level_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(higher_level_path)


def check_bbox_integrity(bbox, image_width, image_height, min_area=12000):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    if x2 > x1 and y2 > y1:
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area:
            return [x1, y1, x2, y2]

    return None


def load_images(image_folder):
    images = []
    for file in os.listdir(image_folder):
        if file.endswith('.png') or file.endswith('.jpg'):
            img_path = os.path.join(image_folder, file)
            tag_name = os.path.splitext(file)[0]
            images.append((Image.open(img_path).convert("RGBA"), tag_name))
    return images


def resize_image_if_needed(image, max_width, max_height):
    if image.width > max_width or image.height > max_height:
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return image


def print_image_sizes(image_folder):
    for file in os.listdir(image_folder):
        if file.endswith('.png') or file.endswith('.jpg'):
            img_path = os.path.join(image_folder, file)
            with Image.open(img_path) as img:
                width, height = img.size
                print(f"Image: {file} - Width: {width}px, Height: {height}px")


def place_images_on_background(background, images):
    max_width = background.width
    max_height = background.height
    num_images = random.randint(2, 5)
    selected_images = random.sample(images, min(num_images, len(images)))
    metadata = []
    for img, label in selected_images:
        img = resize_image_if_needed(img, max_width // 2, max_height // 2)
        max_x = background.width - img.width
        max_y = background.height - img.height
        pos_x = random.randint(0, max_x)
        pos_y = random.randint(0, max_y)

        background.paste(img, (pos_x, pos_y), img)

        region_id = str(uuid.uuid4())
        tag_id = str(uuid.uuid4())
        created_time = datetime.now().isoformat()

        metadata.append({
            "regionId": region_id,
            "tagName": label,
            "created": created_time,
            "tagId": tag_id,
            "left": pos_x / background.width,
            "top": pos_y / background.height,
            "width": img.width / background.width,
            "height": img.height / background.height
        })

    return background, metadata


def save_image_and_metadata(image, metadata, save_path, index):
    folder_path = os.path.join(save_path, f"image_{index}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    image_path = os.path.join(folder_path, "original_image.png")
    json_path = os.path.join(folder_path, "regions.json")

    image.save(image_path)
    with open(json_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)


def generate_dataset(image_folder, save_path, background_folder=None, num_images=10):
    images = load_images(image_folder)
    print(f"Loaded {len(images)} images")
    backgrounds = load_background_images(background_folder) if background_folder else None

    for i in range(num_images):
        background = random.choice(backgrounds).copy() if backgrounds else Image.new("RGBA", (1024, 1024),
                                                                                     (255, 255, 255, 255))
        composed_image, metadata = place_images_on_background(background, images)
        save_image_and_metadata(composed_image, metadata, save_path, i)


def load_background_images(background_folder):
    backgrounds = []
    for file in os.listdir(background_folder):
        if file.endswith('.png') or file.endswith('.jpg'):
            img_path = os.path.join(background_folder, file)
            background = Image.open(img_path).convert("RGBA")
            resize_image_if_needed(background, 1024, 1024)
            backgrounds.append(background)

    return backgrounds


def reset_folder(folder):
    shutil.rmtree(folder)
    os.makedirs(folder)
    print(f"Folder {folder} has been reset")


def main():
    image_folder = "images/cropped_previous"
    save_path = "generated_dataset2"
    background_folder = "images/backgrounds"
    reset_folder(save_path)
    generate_dataset(image_folder, save_path, background_folder, num_images=10)
    #print_image_sizes(image_folder)


if __name__ == "__main__":
    main()
