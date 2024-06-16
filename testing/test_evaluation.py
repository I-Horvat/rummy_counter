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

def compute_ap(recall, precision):
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def evaluate_model(model, data_loader, device, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    model.eval()
    all_true_boxes = []
    all_pred_boxes = []
    all_pred_scores = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(images)

            true_boxes = get_all_boxes(targets)
            pred_boxes = [output['boxes'].cpu().numpy() for output in outputs]
            pred_scores = [output['scores'].cpu().numpy() for output in outputs]

            all_true_boxes.extend(true_boxes)
            for pred_box, pred_score in zip(pred_boxes, pred_scores):
                all_pred_boxes.extend(pred_box)
                all_pred_scores.extend(pred_score)

    aps = []
    for iou_threshold in iou_thresholds:
        tp = []
        fp = []
        scores = []
        num_gts = 0

        for true_boxes, pred_boxes, pred_scores in zip(all_true_boxes, all_pred_boxes, all_pred_scores):
            detected = []
            for pb, score in zip(pred_boxes, pred_scores):
                scores.append(score)
                if any(calculate_iou(pb, tb) > iou_threshold for tb in true_boxes if tb not in detected):
                    tp.append(1)
                    fp.append(0)
                    detected.append(pb)
                else:
                    tp.append(0)
                    fp.append(1)
            num_gts += len(true_boxes)

        indices = np.argsort(-np.array(scores))
        tp = np.array(tp)[indices]
        fp = np.array(fp)[indices]
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / float(num_gts)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = compute_ap(recall, precision)
        aps.append(ap)

    mAP = np.mean(aps)
    return mAP
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
    map=evaluate_model(model, val_loader, device, iou_thresholds=np.linspace(0.5, 0.95, 10))
    print("Mean Average Precision:",map)