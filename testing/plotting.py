import random

import matplotlib.pyplot as plt
from matplotlib import patches
from torchvision.transforms import v2 as transforms

from utils.util import symbols, symbol_to_point


def plot_samples(dataset, num_samples):
    random_indices = random.sample(range(len(dataset)), num_samples)
    for i in random_indices:
        image, target = dataset[i]
        image = (255 * image).byte()
        boxes = target['boxes']
        labels = target['labels']
        # labels = [str(label.item()) for label in target['labels']]
        # image_tensor=draw_bounding_boxes(image, boxes,labels=labels,width=15,font_size=30,font=default_font_path)
        # plt.figure(figsize=(8, 6))
        # plt.imshow(image_tensor.permute(1, 2, 0))
        # plt.title(dataset.image_names[i])
        # plt.show()
        plot_sample(image, boxes, labels, dataset.image_names[i])


def plot_sample_by_name(dataset, image_name):
    index = dataset.image_names.index(image_name)
    image, target = dataset[index]
    boxes = target['boxes']
    labels = target['labels']
    plot_sample(image, boxes, labels, dataset.image_names[index])


def plot_sample(image,predictions, confidence_threshold=0.5):
    plt.figure(figsize=(8, 6))
    predictions=predictions[0]
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    true_points=0

    for box, label,score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score > confidence_threshold:
            x_min, y_min, x_max, y_max = box
            true_points+=symbol_to_point[symbols[label.item()-1]]

            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(x_min, y_min, symbols[label - 1], fontsize=12, color='r')
    plt.title(f"detected objects, total points: {true_points}")
    plt.show()
    return true_points
