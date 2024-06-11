import json
import shutil
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os

from torchvision.transforms import v2

from card_dataset.CardDatsSet import CardDataset
from testing.plotting import plot_sample_default, plot_sample_default_new_symbols
from utils.util import symbols, new_symbols


class DatasetViewerApp:

    def __init__(self, root, dataset):
        self.name = None
        self.photo = None
        self.current_labels = None
        self.current_boxes = None
        self.current_image = None
        self.root = root
        self.dataset = dataset
        self.current_index = 0
        self.root_dir = dataset.root_dir
        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.next_button = tk.Button(root, text="Next", command=self.next_image)
        self.next_button.pack()

        self.rename_button = tk.Button(root, text="Rename", command=self.rename_region)
        self.rename_button.pack()

        self.delete_button = tk.Button(root, text="Delete", command=self.delete_folder)
        self.delete_button.pack()

        self.init()
    def delete_folder(self):
        folder_path = os.path.join(self.root_dir, self.name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
            self.next_image()
        else:
            print(f"Folder not found: {folder_path}")

    def init(self):
        self.load_image()
        self.display_image()

    def load_image(self):
        sample = self.dataset[self.current_index]
        image=sample[0]
        dictionary=sample[1]

        self.current_image = image
        self.current_boxes = dictionary['boxes']
        self.current_labels = dictionary['labels']
        self.name=dictionary['image_id']


    def display_image(self):

        image_np = self.current_image.permute(1, 2, 0).numpy()
        plot_sample_default_new_symbols(image_np, self.current_boxes, self.current_labels, self.dataset.image_names[self.current_index])


    def next_image(self):
        self.current_index += 1
        if self.current_index >= len(self.dataset):
            self.current_index = 0
        self.load_image()
        while not self.check_if_joker():
            self.current_index += 1
            if self.current_index >= len(self.dataset):
                self.current_index = 0
            self.load_image()

    def print_joker_coordinates(self):
        print("Joker coordinates:")
        for i, label in enumerate(self.current_labels):
            if new_symbols[label - 1] == "JOKER":
                print(f"Box {i}: {self.current_boxes[i]}")


    def check_if_joker(self):
        for label in self.current_labels:
            try:
                if new_symbols[label - 1] == "JOKER":
                    self.print_joker_coordinates()
                    self.display_image()
                    return True
            except:
                print(label)
        return False
    def rename_region(self):
        for i, label in enumerate(self.current_labels):
            if symbols[label -1] == "JOKER":
                bounding_box = self.current_boxes[i]
                new_name = simpledialog.askstring("Rename Region", f"Enter new name for the joker with bbox{bounding_box}:")
                if new_name:
                    self.current_labels[i] = new_symbols.index(new_name) + 1

        regions=self.dataset.load_regions(self.name)
        for i, region in enumerate(regions):
            region['tagName'] = new_symbols[self.current_labels[i]-1]
        with open(os.path.join(self.root_dir, self.name, 'regions.json'), 'w', encoding='utf-8') as json_file:
            json.dump(regions, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    num_of_pixels = 1024
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale=True),
        v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
        #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])

    dataset_path = '../images/total_generated_augmented'
    dataset = CardDataset(root_dir=dataset_path, transform=transform, num_of_pixels=num_of_pixels)

    root = tk.Tk()
    root.title("Dataset Viewer")
    app = DatasetViewerApp(root, dataset)
    root.mainloop()