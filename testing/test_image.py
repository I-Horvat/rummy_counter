import PIL
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision import tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import v2

from testing.plotting import plot_sample
from utils.util import symbols


def draw_prediction(image, prediction, confidence_threshold=0.3):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    width, height = image.size

    for element in range(len(prediction['boxes'])):
        box = prediction['boxes'][element].tolist()

        if all(0 <= coord <= 1 for coord in box):
            box = [coord * width if i % 2 == 0 else coord * height for i, coord in enumerate(box)]

        label = prediction['labels'][element].item()
        score = prediction['scores'][element].item()

        if score > confidence_threshold:
            draw.rectangle(box, outline='red', width=3)
            draw.text((box[0], box[1]), f'{symbols[label - 1]} {score}', fill='red', font=font)

    return image


if __name__ == '__main__':
    num_classes = 54

    #dict_path = '../checkpoint_epoch_30.pth'
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    num_of_pixels = 1024
    model.eval()
    image_path = '../WhatsApp Image 2024-05-10 at 11.39.36.jpeg'
    #image_path = '../../../total/0a246d3b-afb8-4c23-9dc3-2038ecb4a2b2/original_image.png'
    image = tv_tensors.Image(PIL.Image.open(image_path).convert("RGB"))
    image = image.float() / 255.0

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale=True),
        v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
        #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image=transform(image)

    checkpoint_full_path = '../models/checkpoint_epoch_30.pth'
    checkpoint=torch.load(checkpoint_full_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    input_tensor = image.unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor)
    print(prediction)

    #draw_prediction(image, prediction[0], confidence_threshold=0.3).show()
    plot_sample(image, prediction, confidence_threshold=0.75)
