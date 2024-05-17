import os
import tempfile

import torch
from ray import train

from utils.util import write_to_log


def save_checkpoint(model, optimizer, scheduler,epoch, val_loss,checkpoint_name, log_file_name):
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        path = os.path.join(checkpoint_dir, checkpoint_name)

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }

        torch.save(checkpoint, path)
        write_to_log(log_file_name, f"Checkpoint saved at {path}")
        train.report({"loss": val_loss}, checkpoint=train.Checkpoint.from_directory(checkpoint_dir))


def load_tuner_checkpoint(checkpoint, config, log_file_name, model, optimizer, scheduler, checkpoint_name):
    start_epoch = 0
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            start_epoch = load_checkpoint(checkpoint_path, log_file_name, model, optimizer, scheduler, start_epoch)
    else:
        write_to_log(log_file_name, f'no checkpoint provided')
    write_to_log(log_file_name, f"Starting training with configuration: {config}")
    return start_epoch


def load_checkpoint(checkpoint_path, log_file_name, model, optimizer, scheduler):
    start_epoch = 0
    write_to_log(log_file_name, f"Looking for checkpoint at {checkpoint_path}")
    if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict["model_state"])
            optimizer.load_state_dict(state_dict["optimizer_state"])
            scheduler.load_state_dict(state_dict["scheduler_state"])
            start_epoch = state_dict["epoch"]
            write_to_log(log_file_name, f"Checkpoint loaded from {checkpoint_path}")
    else:
        write_to_log(log_file_name, f"No checkpoint found at {checkpoint_path}")
    return start_epoch