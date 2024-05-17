from torch.utils.tensorboard import SummaryWriter
import ray
from ray import train
from ray.tune.schedulers import ASHAScheduler
import ray.tune as tune

import torch
from torchvision.transforms import v2

from tuning.checkpoints import save_checkpoint, load_tuner_checkpoint
from utils.util import write_to_log, validate_model, train_one_epoch
from utils.tuning_utils import tuning_directory_setup, load_optimizer_scheduler
from utils.loaders import load_dataset, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tune_model(model_ref, train_loader_ref, val_loader_ref, config, tensorboard_logs, log_file_name,):
    writer = SummaryWriter(tensorboard_logs)
    model = ray.get(model_ref)
    train_loader = ray.get(train_loader_ref)
    val_loader = ray.get(val_loader_ref)
    optimizer, scheduler = load_optimizer_scheduler(config, model)

    checkpoint = train.get_checkpoint()

    start_epoch = load_tuner_checkpoint(checkpoint, model, optimizer, scheduler, log_file_name)
    num_epochs = config.get("num_epochs", 10)
    for epoch in range(start_epoch, num_epochs):
        write_to_log(log_file_name, f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, log_file_name)
        writer.add_scalar("Loss/train", train_loss, epoch)
        if train_loss == float('inf'):
            write_to_log(log_file_name, "NaN encountered in training loss, reporting high loss to Ray Tune.")
            train.report({"loss": float('inf')})
            return
        val_loss = validate_model(model, val_loader, device)

        write_to_log(log_file_name, f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        scheduler.step(val_loss)

        if (epoch + 1) % 3 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, "checkpoint.pth", log_file_name)

        train.report({"loss": val_loss})
    writer.close()


def main(train_loader, val_loader, filename, uri_path, tensorboard_logs, log_file_name, checkpoint_dir, num_samples=10,
         max_num_epochs=9, gpus_per_trial=1):
    model = load_model(num_classes=54, log_file_name=log_file_name)
    model.to(device)

    model_ref = ray.put(model)
    train_loader_ref = ray.put(train_loader)
    val_loader_ref = ray.put(val_loader)
    config = {
        "lr": tune.uniform(1e-6, 1e-3),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([4, 8, 16]),
        "momentum": tune.uniform(0.8, 0.99),
        "optimizer": tune.choice(["adam", "sgd", "adamw"]),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "scheduler": tune.choice(["ReduceLROnPlateau", "StepLR", "ExponentialLR"]),
        "step_size": tune.choice([5, 10, 20]),
        "gamma": tune.uniform(0.1, 0.9),
        "num_epochs": max_num_epochs
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
        metric="loss",
        mode="min"
    )

    result = tune.Tuner(
        tune.with_resources(
            lambda config: tune_model(model_ref, train_loader_ref, val_loader_ref, config, tensorboard_logs,
                                      log_file_name), {"cpu": 2, "gpu": gpus_per_trial, "accelerator_type:T4": 1}),
        param_space=config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name=filename,
            storage_path=uri_path
        )
    ).fit()

    best_trial = result.get_best_result(metric="loss", mode="min", scope="last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.metrics["loss"]))


if __name__ == "__main__":
    filename = 'fixed_trial_6'
    dataset_path = '/images/cleaned_and_total'
    log_file_name, tensorboard_logs, checkpoint_dir, lambda_log = tuning_directory_setup(filename)
    ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=2, num_gpus=1)
    num_of_pixels = 1024
    batch_number = 8
    num_workers = 1
    num_classes = 54

    write_to_log(log_file_name, f"Using device: {device}")

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader, val_loader = load_dataset(dataset_path, transform, num_of_pixels, log_file_name, batch_number,
                                            num_workers)
    main(train_loader, val_loader, filename, lambda_log, tensorboard_logs, log_file_name, checkpoint_dir)
