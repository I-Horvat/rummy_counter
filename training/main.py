import logging
import os
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import v2

from card_dataset.CardDatsSet import CardDataset

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


def load_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = True
    write_to_log(log_file_name, "Model loaded")
    write_to_log(log_file_name, f"Number of classes: {num_classes}")
    return model.to(device)


def train_one_epoch(model, train_loader, optimizer, log_file_name):
    model.train()
    total_loss = 0
    batch_counter = 0
    for images, targets in train_loader:
        batch_counter += 1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    average_loss = total_loss / len(train_loader)
    write_to_log(log_file_name, f"Total loss: {total_loss}")
    write_to_log(log_file_name, f"Average loss: {average_loss}")

    return average_loss


def make_folders_setup_logging(filename):
    log_folder = 'drive/MyDrive/logs'
    model_folder = f'drive/MyDrive/models/{filename}'
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    log_file_name = f"{log_folder}/{filename}.txt"
    print("Created folders")
    print(f'writing to {log_file_name}')
    return log_file_name


def write_to_log(file, message):
    with open(file, 'a') as f:
        f.write(message + "\n")


def validate_model(model, val_loader, device):
    model.train()
    total_val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()
    return total_val_loss / len(val_loader)


def early_stopping(val_loss, best_val_loss, patience_counter, patience_limit=5):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        return False, best_val_loss, patience_counter
    else:
        patience_counter += 1
        if patience_counter > patience_limit:
            return True, best_val_loss, patience_counter
    return False, best_val_loss, patience_counter


def run(model, optimizer, scheduler, train_loader, val_loader, num_epochs, filename, log_file_name, start_epoch=0):
    counter = 0
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(start_epoch, num_epochs):
        counter += 1
        epoch_start_time = time.time()
        write_to_log(log_file_name, f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, log_file_name)
        epoch_time = time.time() - epoch_start_time

        val_loss = validate_model(model, val_loader, device)

        scheduler.step(val_loss)
        write_to_log(log_file_name, f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        write_to_log(log_file_name, f"Epoch time: {epoch_time} seconds")

        stop_training, best_val_loss, patience_counter = early_stopping(val_loss, best_val_loss, patience_counter)
        if stop_training:
            write_to_log(log_file_name, "Early stopping triggered.")
            break

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        checkpoint_filename = f'checkpoint_epoch_{epoch + 1}.pth'
        checkpoint_full_path = f'drive/MyDrive/models/{filename}/{checkpoint_filename}'

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if counter % 3 == 0:
            write_to_log(log_file_name, f"timestamp: {timestamp}")
            write_to_log(log_file_name, f"Checkpoint saved to {checkpoint_full_path}")
            torch.save(checkpoint, checkpoint_full_path)


def load_checkpoint(filename):
    try:
        checkpoint = torch.load(filename)
        return checkpoint
    except FileNotFoundError:
        print("No checkpoint found at '{}'. Starting from scratch.".format(filename))
        return None


if __name__ == '__main__':

    filename = '21epoch_adam0.0005_rOnPlat_8batch_v2'

    dataset_path = 'drive/MyDrive/zavrsni_slike/extracted/cleaned_and_total'

    checkpoint_filename = f'checkpoint_epoch_12.pth'

    num_epochs = 21
    num_of_pixels = 1024
    batch_number = 8
    num_workers = 1
    num_classes = 54

    start_epoch = 0

    log_file_name = make_folders_setup_logging(filename)
    write_to_log(log_file_name, f"Using device: {device}")

    checkpoint_full_path = f'drive/MyDrive/models/{filename}/{checkpoint_filename}'
    checkpoint = load_checkpoint(checkpoint_full_path)

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])

    train_loader, val_loader = load_dataset(dataset_path, transform, num_of_pixels, log_file_name, batch_number,
                                            num_workers)

    model = load_model(num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.05)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    if checkpoint is not None:
        print(f"loaded from checkpoint {checkpoint['epoch']}")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
    try:
        print("Starting training")
        run(model, optimizer, scheduler, train_loader, val_loader, num_epochs, filename, log_file_name, start_epoch)
        print("Finished training")
    except Exception as e:
        logging.exception(e)
