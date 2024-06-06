import os


def make_folders_setup_logging(filename):
    log_folder = 'drive/MyDrive/tensorboard_logs'
    model_folder = f'drive/MyDrive/models/{filename}'
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    log_file_name = f"{log_folder}/{filename}.txt"
    print("Created folders")
    print(f'writing to {log_file_name}')
    return log_file_name
