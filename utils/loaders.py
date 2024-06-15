import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from card_dataset.CardDataSet import CardDataset
from utils.util import write_to_log

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    images = [sample[0] for sample in batch]
    targets = [{'boxes': sample[1]['boxes'], 'labels': sample[1]['labels'], 'image_id': sample[1]['image_id'],
                'area': sample[1]['area']} for sample in batch]
    return images, targets


def load_dataset(dataset_path, transform, num_of_pixels, log_file_name, batch_num=32, num_workers=2, ):
    write_to_log(log_file_name, "Loading dataset...")
    dataset = CardDataset(root_dir=dataset_path, transform=transform, num_of_pixels=num_of_pixels)
    write_to_log(log_file_name, f"Dataset size: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_num, shuffle=True, drop_last=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_num, shuffle=False, drop_last=False, num_workers=num_workers,
                            collate_fn=collate_fn)

    write_to_log(log_file_name, "Dataset loaded")
    write_to_log(log_file_name, f"Train size: {len(train_dataset)}")
    write_to_log(log_file_name, f"Validation size: {len(val_dataset)}")
    write_to_log(log_file_name, f"Batch size: {batch_num}")
    write_to_log(log_file_name, f"Number of workers: {num_workers}")

    return train_loader, val_loader


def load_model(num_classes,log_file_name):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = True
    write_to_log(log_file_name, "Model loaded")
    write_to_log(log_file_name, f"Number of classes: {num_classes}")
    return model.to(device)



