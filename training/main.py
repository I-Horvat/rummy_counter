import logging
import time

import torch
from torchvision.transforms import v2

from tuning.checkpoints import save_checkpoint, load_checkpoint
from utils.training_utils import make_folders_setup_logging
from utils.util import write_to_log, validate_model, early_stopping, train_one_epoch
from utils.loaders import load_dataset, device, load_model


def train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs, filename, log_file_name,
                start_epoch=0):
    counter = 0
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(start_epoch, num_epochs):
        counter += 1
        epoch_start_time = time.time()
        write_to_log(log_file_name, f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, log_file_name)
        epoch_time = time.time() - epoch_start_time
        previous=scheduler.get_last_lr()
        print(f"Learning rate: {previous}")
        val_loss = validate_model(model, val_loader, device)
        scheduler.step(val_loss)
        write_to_log(log_file_name, f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        write_to_log(log_file_name, f"Epoch time: {epoch_time} seconds")

        stop_training, best_val_loss, patience_counter = early_stopping(val_loss, best_val_loss, patience_counter)
        if stop_training:
            write_to_log(log_file_name, "Early stopping triggered.")
            break

        checkpoint_full_path = f'models/{filename}/epoch_{epoch + 1}.pth'

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if counter % 3 == 0:
            write_to_log(log_file_name, f"timestamp: {timestamp}")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_full_path, log_file_name)

    write_to_log(log_file_name, f"Training finished after {counter} epochs.")
    write_to_log(log_file_name, f"Best validation loss: {best_val_loss}")
    write_to_log(log_file_name, f"Timestamp: {timestamp}")


if __name__ == '__main__':

    # dataset_path = 'drive/MyDrive/zavrsni_slike/extracted/cleaned_and_total'
    dataset_path = '../images/final'

    checkpoint_filename = f'checkpoint_epoch_12.pth'
    filename = '2epoch_'
    checkpoint_full_path = f'models/{filename}/{checkpoint_filename}'

    log_file_name = make_folders_setup_logging(filename)

    num_epochs = 2
    num_of_pixels = 1024
    batch_number = 8
    num_workers = 1
    num_classes = 55

    start_epoch = 0

    write_to_log(log_file_name, f"Using device: {device}")

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])

    train_loader, val_loader = load_dataset(dataset_path, transform, num_of_pixels, log_file_name, batch_number,
                                            num_workers)

    model = load_model(num_classes=num_classes, log_file_name=log_file_name)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.05)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    start_epoch = load_checkpoint(checkpoint_full_path, log_file_name, model, optimizer, scheduler)

    try:
        print("Starting training")
        train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs, filename, log_file_name,
                    start_epoch)
        print("Finished training")
    except Exception as e:
        logging.exception(e)
