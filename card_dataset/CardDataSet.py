import os
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import PIL
import torch
from PIL import Image
from PIL.ImageFile import ImageFile
from torch.utils.data import Dataset
from torchvision import tv_tensors as tv_tensor, tv_tensors

from utils.util import  check_bbbox_integrity, new_symbol_to_int

ImageFile.LOAD_TRUNCATED_IMAGES = True
class CardDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_of_pixels=512):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = [folder for folder in os.listdir(self.root_dir) if
                            os.path.isdir(os.path.join(self.root_dir, folder))]
        self.num_of_pixels = num_of_pixels
        self.counter = 0
        self.min_area = int(num_of_pixels * num_of_pixels * 0.035)
        self.max_area = int(num_of_pixels * num_of_pixels * 0.5)


    def load_data(self):
        data = {}
        for folder in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, folder)):
                json_path = os.path.join(self.root_dir, folder, 'regions.json')
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as json_file:
                            json_data = json.load(json_file)
                            data[folder] = json_data
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON for {json_path}. File might be empty or malformed.")
                        data[folder] = []
                else:
                    print(f"JSON file does not exist: {json_path}")
                    data[folder] = []
        return data

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        self.counter += 1
        if self.counter >= len(self.image_names):
            print('All images have been used, starting from the beginning')
            self.counter = 0
            return self.__getitem__(0)
        image_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, image_name, 'original_image.png')

        image = tv_tensors.Image(PIL.Image.open(img_path).convert("RGB"))
        image = image.float() / 255.0

        regions = self.load_regions(image_name)

        boxes = []
        labels = []
        original_regions = regions.copy()
        for region in regions:
            bbox_x1 = region['left'] * self.num_of_pixels
            bbox_y1 = region['top'] * self.num_of_pixels
            bbox_x2 = (region['left'] + region['width']) * self.num_of_pixels
            bbox_y2 = (region['top'] + region['height']) * self.num_of_pixels
            bbox = check_bbbox_integrity([bbox_x1, bbox_y1, bbox_x2, bbox_y2], image, min_area=self.min_area,max_area=self.max_area)
            if bbox is not None:
                boxes.append(bbox)
                labels.append(region['tagName'])
            else:
                regions.remove(region)
                print(f"Region {region['tagName']} in image {image_name} is too small or out of bounds")
        if len (original_regions) != len(regions):
            print(f"Regions removed from image {image_name}. Saving new JSON file.")
            with open(os.path.join(self.root_dir, image_name, 'regions.json'), 'w', encoding='utf-8') as json_file:

                json.dump(regions, json_file, indent=4)
        if len(boxes) == 0:
            print(f"deleting: No regions found in JSON for {image_name}")
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
                bbox = check_bbbox_integrity(target_out['boxes'][i], image_out, min_area=self.min_area,max_area=self.max_area)
                if bbox is not None:
                    new_boxes.append(bbox)
                    new_labels.append(target_out['labels'][i])
            if len(new_boxes) == 0:
                print('delteintg:No boxes found in image: ' + image_name)

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

#to delete
#uuids = [
    "2e169a07-ec71-4ada-8926-14bcd7340883",
    "42f64ec6-fd5e-47a9-bc42-9288c4d7c58f",
    "7ecef798-9811-4431-ac32-e94020d89cbb",
    "d9ce4ede-c733-41ae-9b7d-bc11bbad4212",
    "b0b7441c-c243-4fa6-8046-b0e726caa13d",
    "1e8b9039-818c-4075-83bc-ed617aa02853",
    "03c845b8-a746-4689-85a9-34a397509cf2",
    "d0177568-5d5d-44e9-af9f-1aefdd020ff2",
    "a4db9bd3-4849-4c98-b566-d611893b3d73",
    "1625ad79-fd19-42d9-a733-ef261c2aa878",
    "9103ad71-adf5-40c4-9398-dbf80f267094",
    "f6bb9911-406b-44e9-a132-5d971c396c90",
    "8d799b79-480c-4170-a48f-8748647a3824",
    "1aedc9cf-0780-4861-8b37-1152128b68db",
    "08a25d5c-9290-42f9-9eb7-dbb91cd83b35",
    "70445e83-9a8a-4ec6-b724-fa50784c206a",
    "40c4204d-322d-4ece-a999-da368acc37ad",
    "e1a2abad-7c4a-4925-9d8a-e2b5484bf364",
    "5fed58c6-8d40-4dba-98a9-5ed7afea6dab"
#]