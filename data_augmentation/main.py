import json
import os
import uuid

from torchvision.transforms import v2

from card_dataset.CardDataSet import CardDataset
from utils.util import symbols, new_symbols


num_of_pixels = 1024

transform = v2.Compose([
    v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
    v2.RandomResizedCrop((num_of_pixels, num_of_pixels), antialias=True),
    v2.RandomPhotometricDistort(p=1),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomRotation(20),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
])
to_pil = v2.ToPILImage()

dataset = CardDataset(root_dir='../dataset_generator/images/new_generated', transform=transform, num_of_pixels=num_of_pixels)
def create_augmented_images(num_images,root_folder):

    for i, data in enumerate(dataset):
        image = data[0]
        boxes = data[1]['boxes']
        labels = data[1]['labels']

        random_name=uuid.uuid4()
        image_folder = os.path.join(root_folder, f"{random_name}")
        os.makedirs(image_folder, exist_ok=True)

        boxes_list = boxes.tolist()
        labels_list = labels.tolist()

        transformed_boxes = []
        for box in boxes_list:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            left = x1 / num_of_pixels
            top = y1 / num_of_pixels
            width /= num_of_pixels
            height /= num_of_pixels
            transformed_box = {
                "left": left,
                "top": top,
                "width": width,
                "height": height
            }
            transformed_boxes.append(transformed_box)

        data_json = []
        for box, label in zip(transformed_boxes, labels_list):
            region_id = str(uuid.uuid4())
            tag_id = str(uuid.uuid4())
            adjusted_label = label - 1
            tag_name = new_symbols[adjusted_label]
            entry = {
                "regionId": region_id,
                "tagName": tag_name,
                "tagId": tag_id,
                **box
            }
            data_json.append(entry)

        image=to_pil(image)
        image.save(os.path.join(image_folder, "original_image.png"))

        with open(os.path.join(image_folder, "regions.json"), "w",encoding='utf-8') as json_file:
            json.dump(data_json, json_file, indent=2, ensure_ascii=False)
        if i >= num_images:
            break

if __name__ == '__main__':
    root_folder = "augmented_final"
    os.makedirs(root_folder, exist_ok=True)
    create_augmented_images(3000,root_folder)
    print("Done")