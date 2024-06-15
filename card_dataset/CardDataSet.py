import os
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors as tv_tensor, tv_tensors

from utils.util import symbol_to_int, check_bbbox_integrity, new_symbol_to_int


class CardDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_of_pixels=512):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = [folder for folder in os.listdir(self.root_dir) if
                            os.path.isdir(os.path.join(self.root_dir, folder))]
        self.num_of_pixels = num_of_pixels
        self.counter = 0
        self.min_area = 0
        self.images=[]
        self.annotations=[]
        self.load_data_into_memory()



    def load_data_into_memory(self):
        print(f"Loading images from {self.root_dir}")
        num_workers = os.cpu_count()
        print(f"Number of workers: {num_workers} in os")
        using_workers = 2
        print(f"Using {using_workers} workers")
        total_images = len(self.image_names)
        ten_percent = total_images // 10

        with ThreadPoolExecutor(max_workers=using_workers) as executor:
            results = list(executor.map(self.load_single_image, self.image_names))
            for i, (image, annotation) in enumerate(results):
                if image:
                    self.images.append(image)
                    self.annotations.append(annotation)
                if (i + 1) % ten_percent == 0:
                    print(f"Loaded {i + 1} / {total_images} images ({((i + 1) / total_images) * 100:.0f}%)")

        print(f"Loaded {len(self.images)} images")

    def load_single_image(self, folder):
        img_path = os.path.join(self.root_dir, folder, 'original_image.png')
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
                json_path = os.path.join(self.root_dir, folder, 'regions.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as json_file:
                        json_data = json.load(json_file)
                    return image, json_data
                else:
                    return image, []
            except PIL.UnidentifiedImageError:
                return None, []
        return None, []

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image = self.images[idx]
        image = tv_tensors.Image(image).float() / 255.0

        regions = self.annotations[idx]
        image_name = self.image_names[idx]
        boxes = []
        labels = []

        original_regions = regions.copy()
        img_path = os.path.join(self.root_dir, image_name, 'original_image.png')

        for region in regions:
            bbox_x1 = region['left'] * self.num_of_pixels
            bbox_y1 = region['top'] * self.num_of_pixels
            bbox_x2 = (region['left'] + region['width']) * self.num_of_pixels
            bbox_y2 = (region['top'] + region['height']) * self.num_of_pixels
            bbox = check_bbbox_integrity([bbox_x1, bbox_y1, bbox_x2, bbox_y2], image, min_area=self.min_area)
            if bbox is not None:
                boxes.append(bbox)
                labels.append(region['tagName'])
            else:
                regions.remove(region)
                print(f"Region {region['tagName']} in image {image_name} is too small or out of bounds")
        if len(regions) != len(original_regions):
            with open(os.path.join(self.root_dir, image_name, 'regions.json'), 'w', encoding='utf-8') as json_file:
                json.dump(regions, json_file, ensure_ascii=False, indent=4)
        if len(boxes) == 0:
            print(f"No regions found in JSON for {image_name}. Deleting image.")
            shutil.rmtree(os.path.join(self.root_dir, image_name))
            return self.__getitem__(idx + 1)

        boxes = tv_tensor.BoundingBoxes(boxes, canvas_size=(self.num_of_pixels, self.num_of_pixels), format="xyxy")

        target = {'boxes': boxes, 'labels': torch.tensor([new_symbol_to_int(label) for label in labels], dtype=torch.long),
                  'image_id': image_name, 'image_path': img_path}

        target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (
                target['boxes'][:, 2] - target['boxes'][:, 0])
        if self.transform:
            image_out, target_out = self.transform(image, target)
            new_boxes = []
            new_labels = []
            for i in range(len(target_out['boxes'])):
                bbox = check_bbbox_integrity(target_out['boxes'][i], image_out, min_area=self.min_area)
                if bbox is not None:
                    new_boxes.append(bbox)
                    new_labels.append(target_out['labels'][i])
            if len(new_boxes) == 0:
                print('No boxes found in image: ' + image_name)
                return self.__getitem__(idx + 1)
            target_out['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)
            target_out['labels'] = torch.tensor(new_labels, dtype=torch.long)
            return image_out, target_out
        else:
            print("No transform applied")
        return image, target

    def load_regions(self, image_name):
        json_path = os.path.join(self.root_dir, image_name, 'regions.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)
                if len(json_data) == 0:
                    print(f"No regions found in JSON for {image_name}. Deleting image.")
                    shutil.rmtree(os.path.join(self.root_dir, image_name))
            return json_data
        except json.JSONDecodeError:
            print(f"Error decoding JSON for {json_path}. File might be empty or malformed.")
            return []
        except FileNotFoundError:
            print(f"JSON file does not exist: {json_path}")
            return []
