import numpy as np
import torch
from torchvision.transforms import v2

from utils.loaders import load_model, load_dataset
from utils.util import device

def get_all_boxes(targets):
    boxes = []
    for target in targets:
        boxes.extend(target['boxes'].cpu().numpy())
    return np.array(boxes)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    all_true_boxes = []
    all_pred_boxes = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(images)

            true_boxes = get_all_boxes(targets)
            pred_boxes = get_all_boxes([output['boxes'] for output in outputs])

            all_true_boxes.append(true_boxes)
            all_pred_boxes.append(pred_boxes)

    tp = 0
    fp = 0
    fn = 0
    for true_boxes, pred_boxes in zip(all_true_boxes, all_pred_boxes):
        matched = []
        for pb in pred_boxes:
            if any(calculate_iou(pb, tb) > iou_threshold for tb in true_boxes):
                tp += 1
                matched.append(pb)
            else:
                fp += 1

        for tb in true_boxes:
            if not any(calculate_iou(tb, mb) > iou_threshold for mb in matched):
                fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = 2 * (precision * recall) / (precision + recall)

    return precision, recall, accuracy

if __name__ == '__main__':
    dataset_path = '../images/total_generated_augmented'
    num_of_pixels = 512
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
        #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])
    num_workers = 1
    batch_number = 8

    num_classes = 55
    log_file_name = 'evaluation_log.txt'
    train_loader, val_loader = load_dataset(dataset_path, transform, num_of_pixels, log_file_name, batch_number,
                                            num_workers)

    model = load_model(num_classes=num_classes, log_file_name=log_file_name)
    evaluate_model(model, train_loader, device, iou_threshold=0.5)