import os

import torch


def tuning_directory_setup(filename):
    base = f'/content/drive/MyDrive/logs/{filename}'

    folders = {
        "log_folder": f'{base}/log',
        "model_folder": f'{base}/models',
        "checkpoint_dir": f'{base}/checkpoints',
        "tensorboard_logs": f'{base}/tensorboard_logs',
        "lambda_log": f'{base}/runs'
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    log_file_name = f"{folders['log_folder']}/{filename}.txt"
    print("Created folders")
    print(f'Writing to {log_file_name}')
    return log_file_name, folders["tensorboard_logs"], folders["checkpoint_dir"], folders["lambda_log"]


def load_optimizer_scheduler(config, model):
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
                                    momentum=config["momentum"])
    elif config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if config["scheduler"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["gamma"], patience=3)
    elif config["scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    elif config["scheduler"] == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["gamma"])
    return optimizer, scheduler
