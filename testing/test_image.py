import PIL
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision import tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights, \
    fasterrcnn_resnet50_fpn
from torchvision.transforms import v2
from torchvision.transforms import functional as F
from testing.plotting import plot_sample, symbols
from utils.util import symbol_to_point


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


def test_image(image_path, checkpoint_full_path, threshold,transform ):
    global in_features, model
    num_classes = 54
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    image = tv_tensors.Image(PIL.Image.open(image_path).convert("RGB"))
    image = image.float() / 255.0

    image = transform(image)

    checkpoint = torch.load(checkpoint_full_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    input_tensor = image.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        prediction = model(input_tensor)
    print(prediction)
    # draw_prediction(image, prediction[0], confidence_threshold=0.3).show()
    plot_sample(image, prediction, confidence_threshold=threshold)


def process_image(image):
    threshold = 0.75

    global in_features, model
    num_classes = 54
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    num_of_pixels = 1024
    model.eval()
    image = tv_tensors.Image(image)
    image = image.float() / 255.0
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    checkpoint_full_path = 'models/checkpoint_epoch_30.pth'
    checkpoint = torch.load(checkpoint_full_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    input_tensor = image.unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor)

    #print(prediction)
    points = predictions_to_points(prediction, threshold=threshold)
    # draw_prediction(image, prediction[0], confidence_threshold=0.3).show()
    true_points = plot_sample(image, prediction, confidence_threshold=threshold)
    #get labels of card over threshold, summ the points recieved
    return true_points


def predictions_to_points(predictions, threshold=0.5):
    points = 0
    for i in range(len(predictions[0]['boxes'])):
        if predictions[0]['scores'][i] > threshold:
            points += symbol_to_point[symbols[predictions[0]['labels'][i].item() - 1]]
    return points


if __name__ == '__main__':
    num_of_pixels = 1024
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    threshold = 0.3
    path = '../images/WhatsApp Image 2024-05-10 at 11.39.36.jpeg'
    checkpoint = '../models/cleaned_total_augmented/checkpoint_epoch_9.pth'
    test_image(path, checkpoint, threshold, transform)
