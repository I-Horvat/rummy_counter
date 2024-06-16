import os
import random
from PIL import Image
import uuid
import json
from datetime import datetime
import shutil


def check_overlap(new_pos, new_size, existing_positions):
    new_left, new_top = new_pos
    new_right, new_bottom = new_left + new_size[0], new_top + new_size[1]
    for existing_pos, existing_size in existing_positions:
        existing_left, existing_top = existing_pos
        existing_right, existing_bottom = existing_left + existing_size[0], existing_top + existing_size[1]
        if not (new_right <= existing_left or new_left >= existing_right or new_bottom <= existing_top or new_top >= existing_bottom):
            return True
    return False


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


def place_images_on_background(background, images):
    num_images = random.randint(2, 5)
    selected_images = random.sample(images, min(num_images, len(images)))
    metadata = []
    existing_positions = []
    for img, label in selected_images:
        max_x = background.width - img.width
        max_y = background.height - img.height
        for _ in range(100):
            pos_x = random.randint(0, max_x)
            pos_y = random.randint(0, max_y)
            if not check_overlap((pos_x, pos_y), (img.width, img.height), existing_positions):
                break
        else:
            continue

        background.paste(img, (pos_x, pos_y), img)
        existing_positions.append(((pos_x, pos_y), (img.width, img.height)))

        left = pos_x / background.width
        top = pos_y / background.height
        width = img.width / background.width
        height = img.height / background.height

        region_id = str(uuid.uuid4())
        tag_id = str(uuid.uuid4())
        created_time = datetime.now().isoformat()

        metadata.append({
            "regionId": region_id,
            "tagName": label,
            "created": created_time,
            "tagId": tag_id,
            "left": left,
            "top": top,
            "width": width,
            "height": height
        })
    return background, metadata


def save_image_and_metadata(image, metadata, save_path):
    random_name=uuid.uuid4()

    folder_path = os.path.join(save_path, f"{random_name}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    image_path = os.path.join(folder_path, "original_image.png")
    json_path = os.path.join(folder_path, "regions.json")

    image.save(image_path)
    with open(json_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)


def generate_dataset(image_folder, save_path, background_folder=None, num_images=10):
    images = load_images(image_folder)
    images = [(resize_image_if_needed(img, 200, 256), label) for img, label in images]
    print(f"Loaded {len(images)} images")
    backgrounds = load_background_images(background_folder) if background_folder else []
    print(f"Loaded {len(backgrounds)} backgrounds")
    backgrounds = [background.resize((1024, 1024), Image.Resampling.LANCZOS) for background in backgrounds]
    print(f"Resized backgrounds")
    for i in range(num_images):
        background = random.choice(backgrounds).copy() if backgrounds else Image.new("RGBA", (1024, 1024),
                                                                                     (255, 255, 255, 255))
        composed_image, metadata = place_images_on_background(background, images)
        save_image_and_metadata(composed_image, metadata, save_path)


def load_background_images(background_folder):
    backgrounds = []
    for folder in os.listdir(background_folder):
        for file in os.listdir(os.path.join(background_folder, folder)):
            if file.endswith('.png') or file.endswith('.jpg'):
                img_path = os.path.join(background_folder, folder, file)
                backgrounds.append(Image.open(img_path).convert("RGBA"))
    return backgrounds


def reset_folder(folder):
    shutil.rmtree(folder)
    os.makedirs(folder)
    print(f"Folder {folder} has been reset")


def main():
    image_folder = "images/cropped_previous"
    save_path = "images/new_generated"
    background_folder = "images/backgrounds/images"
    reset_folder(save_path)
    generate_dataset(image_folder, save_path, background_folder, num_images=4000)
    #print_image_sizes(image_folder)


if __name__ == "__main__":
    main()
